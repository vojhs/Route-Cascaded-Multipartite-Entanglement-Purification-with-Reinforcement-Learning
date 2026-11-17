import os

os.environ["PYTHONHASHSEED"] = "random"  # 设为 "0" 或任意合法整数，或者直接设为 "random"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)  # 限制 PyTorch 内部也只用 1 个线程
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from torch.distributions import Categorical

import cirq
import qas_gym
from qas_gym import envs
from gymnasium.spaces import flatten
from torch.distributions.kl import kl_divergence
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
import torch
import time
import concurrent.futures
import random
import traceback
import threading

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import random

env_lock = threading.Lock()


def get_multi_ghz_state(num_ghz: int = 3, qubits_per_ghz: int = 3) -> np.ndarray:
    """
    生成多个独立 GHZ 态的张量积态矢量。
    例如 num_ghz=3, qubits_per_ghz=3 时，输出 9-qubit 的态矢量 (长度 512)。
    """
    single_ghz = get_ghz_state(qubits_per_ghz)
    state_vector = single_ghz
    for _ in range(num_ghz - 1):
        state_vector = np.kron(state_vector, single_ghz)
    return state_vector


@dataclass
class GRPOConfig:
    # --- 基础训练参数 ---
    num_iterations: int = 500  # <--- 控制训练总轮数
    lr: float = 2e-4  # LSTM 学习率 (5e-5)
    group_size: int = 16  # 每次采集 16 条完整的 Episode
    update_epochs: int = 4  # 每次采集后更新网络的次数

    # --- PPO/GRPO 核心参数 ---
    clip_epsilon: float = 0.2
    beta_kl: float = 0.01  # KL 散度惩罚系数
    entropy_coef: float = 0.05  # 熵正则化 (强制探索)
    gamma: float = 0.95  # 折扣因子 (解决延迟奖励)

    # --- 网络结构参数 ---
    hidden_size: int = 256  # LSTM 隐藏层大小

    # --- 硬件 ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RecurrentGRPOPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 1. 特征提取 (MLP)
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # 2. LSTM 层
        # batch_first=True: input shape (Batch, Seq_Len, Feature)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # 3. 动作头
        self.action_head = nn.Linear(hidden_size, action_dim)

    def forward(self, x, hidden_state=None):
        """
        x: [Batch, Seq_Len, Obs_Dim] 或 [Batch, Obs_Dim]
        """
        # 如果输入是单步数据 [Batch, Dim]，增加时间维度变成 [Batch, 1, Dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        features = self.feature_extractor(x)

        # LSTM 前向传播
        # output: [Batch, Seq_Len, Hidden]
        # next_hidden: (h_n, c_n)
        output, next_hidden = self.lstm(features, hidden_state)

        # 取所有时间步的 logits
        logits = self.action_head(output)
        return logits, next_hidden


class RecurrentGRPOAgent:
    def __init__(self, obs_dim, action_dim, config: GRPOConfig):
        self.cfg = config
        self.action_dim = action_dim

        self.policy = RecurrentGRPOPolicy(obs_dim, action_dim, config.hidden_size).to(config.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)

        # 用于计算 KL 的参考模型 (冻结权重)
        self.ref_policy = RecurrentGRPOPolicy(obs_dim, action_dim, config.hidden_size).to(config.device)
        self.ref_policy.load_state_dict(self.policy.state_dict())
        self.ref_policy.eval()

    def get_action(self, obs, hidden_state=None):
        """单步推理 (用于 Rollout)"""
        obs_tensor = torch.FloatTensor(obs).to(self.cfg.device).unsqueeze(0)  # [1, Dim]

        with torch.no_grad():
            # forward 会自动处理维度
            logits, next_hidden = self.policy(obs_tensor, hidden_state)
            # logits shape: [1, 1, Action_Dim] -> squeeze -> [Action_Dim]
            logits = logits.squeeze(0).squeeze(0)

            # Action Masking (可选：防止自杀动作)
            # logits[invalid_actions] = -1e9

            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), next_hidden

    def get_action_new(self, obs, hidden_state=None, last_action=None):
        """
        last_action: 上一步执行的动作 ID (新增参数)
        """
        obs_tensor = torch.FloatTensor(obs).to(self.cfg.device).unsqueeze(0)

        with torch.no_grad():
            logits, next_hidden = self.policy(obs_tensor, hidden_state)
            logits = logits.squeeze(0).squeeze(0)  # [Action_Dim]

            # === 【核心修改】Logit 惩罚 ===
            # 如果上一步有动作，我们就在 Logits 里把那个动作的分数减掉一大块
            if last_action is not None:
                # 惩罚系数：建议 1.0 到 5.0 之间
                # 越大，Agent 越不可能选重复动作
                penalty = 5.0
                logits[last_action] -= penalty

                # 进阶：如果你想禁止 A -> A 循环，但也想禁止 A -> B -> A
                # 你可以传一个 history_actions 列表进来，把最近 3 个动作的 logits 都减分

            # 重新计算分布（因为 Logits 变了）
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), next_hidden

    def compute_discounted_returns(self, rewards):
        """计算折扣回报 (代替 GAE)"""
        returns = []
        Gt = 0
        for r in reversed(rewards):
            Gt = r + self.cfg.gamma * Gt
            returns.insert(0, Gt)
        return returns

    def train_step_old(self, env_factory):
        self.policy.train()

        group_trajectories = []
        all_rewards = []
        env = env_factory()

        for _ in range(self.cfg.group_size):

            # --- [关键修改 1] 获取 Observation Space ---
            # 我们需要这个 space 对象来执行 flatten 操作
            obs_space = env.observation_space

            obs, _ = env.reset()

            # --- [关键修改 2] 立即展平 obs ---
            # 将字典 {key: val} 转换为 numpy array [v1, v2, ...]
            obs = flatten(obs_space, obs)

            hidden = None

            traj_obs = []
            traj_actions = []
            traj_rewards = []
            traj_log_probs = []

            done = False
            truncated = False
            last_action = None  # 初始化

            while not (done or truncated):
                # 现在传入 get_action 的 obs 已经是扁平数组了，不会报错
                action, log_prob, next_hidden = self.get_action(obs, hidden)

                next_obs, reward, done, truncated, _ = env.step(action)

                # --- [关键修改 3] 展平 next_obs ---
                next_obs = flatten(obs_space, next_obs)

                # 记录数据
                traj_obs.append(obs)
                traj_actions.append(action)
                traj_rewards.append(reward)
                traj_log_probs.append(log_prob)

                obs = next_obs
                hidden = next_hidden
                last_action = action  # 更新 last_action

            # env.close()
            all_rewards.append(sum(traj_rewards))

            returns = self.compute_discounted_returns(traj_rewards)

            group_trajectories.append({
                'obs': torch.FloatTensor(np.array(traj_obs)).to(self.cfg.device),
                'actions': torch.LongTensor(traj_actions).to(self.cfg.device),
                'returns': torch.FloatTensor(returns).to(self.cfg.device),
                'old_log_probs': torch.FloatTensor(traj_log_probs).to(self.cfg.device),
                'length': len(traj_obs),
                'total_reward': sum(traj_rewards)  # <--- 【就是新增这一行】
            })

            # --- 2. 优势计算与归一化 (Advantage Normalization) ---
            # 收集所有采样步的 Return，计算全局均值和标准差
            # --- 2. 优势计算与归一化 (Trajectory-Level GRPO) ---
            # 提取每条线路的无折扣总奖励（因为是完成才评估，也就是最后那一下的保真度得分）
            # 我们需要在上面组装 group_trajectories 时，把 sum(traj_rewards) 存进去，比如叫 'total_reward'

            # 假设你已经在上面的字典里加了 'total_reward': sum(traj_rewards)
            group_totals = torch.FloatTensor([t['total_reward'] for t in group_trajectories]).to(self.cfg.device)

            # 计算这一个 Group (16条线路) 内的均值和标准差
            adv_mean = group_totals.mean()
            # adv_std = group_totals.std() + 1e-8
            adv_std = group_totals.std(correction=0) + 1e-8

            # 归一化：算出每条线路的全局相对优势 (标量)
            group_advantages = (group_totals - adv_mean) / adv_std

            # 将这个统一的优势值，广播给该线路中的每一个时间步
            for i, t in enumerate(group_trajectories):
                # 一荣俱荣，一损俱损。这条线路好，里面所有的动作优势都是正的。
                # 形状与 t['returns'] 保持一致
                t['advantages'] = torch.full_like(t['returns'], group_advantages[i])
        env.close()

        # --- 3. PPO 更新循环 (支持 LSTM) ---
        loss_val = 0
        for _ in range(self.cfg.update_epochs):
            batch_loss = 0

            # LSTM 的特殊之处：我们不能把所有 step 打散混在一起 (Shuffle)
            # 必须保持 Trajectory 的顺序，因为 LSTM 需要上一步的 hidden state
            # 所以这里我们遍历每一条 Trajectory
            for traj in group_trajectories:
                # 增加 Batch 维度 [Seq_Len, Dim] -> [1, Seq_Len, Dim]
                obs_seq = traj['obs'].unsqueeze(0)
                actions = traj['actions'].unsqueeze(0)
                advantages = traj['advantages'].unsqueeze(0)
                old_log_probs = traj['old_log_probs'].unsqueeze(0)

                # Re-evaluate (重新前向传播)
                # 传入整个序列，LSTM 会自动 unroll
                # hidden_state=None 表示从头开始，这与 Rollout 时一致
                logits_seq, _ = self.policy(obs_seq, hidden_state=None)

                # 计算 Log Prob
                dist = Categorical(logits=logits_seq)
                curr_log_probs = dist.log_prob(actions)  # [1, Seq_Len]
                entropy = dist.entropy().mean()

                # 计算 KL 散度 (近似)
                with torch.no_grad():
                    ref_logits, _ = self.ref_policy(obs_seq, hidden_state=None)
                    ref_dist = Categorical(logits=ref_logits)

                # 计算精确的 KL 散度 [1, Seq_Len]
                kl_div = kl_divergence(ref_dist, dist)

                # PPO Loss 计算
                ratio = torch.exp(curr_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advantages

                # 综合 Loss (加权求和)
                traj_loss = -torch.min(surr1, surr2).mean() \
                            + self.cfg.beta_kl * kl_div.mean() \
                            - self.cfg.entropy_coef * entropy

                batch_loss += traj_loss

            # 对所有 Trajectory 的 Loss 取平均
            avg_loss = batch_loss / len(group_trajectories)

            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # 梯度裁剪
            self.optimizer.step()

            loss_val = avg_loss.item()

        # 同步参考模型 (硬更新)
        self.ref_policy.load_state_dict(self.policy.state_dict())

        # 返回 Loss 和 平均 Reward
        return loss_val, np.mean(all_rewards)

    def train_step_old2(self, envs):
        self.policy.train()
        num_envs = len(envs)  # 直接知道有多少个环境

        # =======================================================
        # 1. 直接复用传进来的环境，只进行 Reset！
        # =======================================================
        obs_space = envs[0].observation_space
        trajectories = [{"obs": [], "actions": [], "rewards": [], "log_probs": []} for _ in range(num_envs)]

        current_obs = []
        for env in envs:
            o, _ = env.reset()  # 极其迅速，因为不需要重新开辟内存
            current_obs.append(flatten(obs_space, o))

        # =======================================================
        # 2. LSTM 批量隐藏状态初始化
        # shape: [num_layers(1), Batch_Size(32), Hidden_Size(256)]
        # =======================================================
        h0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        c0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        hidden = (h0, c0)

        # 状态追踪标志
        active_mask = [True] * num_envs  # 记录环境是否跑完
        last_actions = [None] * num_envs  # 记录每个环境的上一步动作
        step_count = 0
        # =======================================================
        # 3. 核心并行推演循环 (榨干 GPU 并行算力)
        # =======================================================
        while any(active_mask):
            step_count += 1
            # 将 32 个环境的观测态拼成一个大矩阵，丢给 GPU
            # shape: [32, 1, Obs_Dim]
            obs_tensor = torch.FloatTensor(np.array(current_obs)).to(self.cfg.device)
            obs_tensor = obs_tensor.unsqueeze(1)
            t0 = time.time()
            with torch.no_grad():
                # GPU 瞬间算出 32 个环境的 logits
                logits, next_hidden = self.policy(obs_tensor, hidden)
                logits = logits.squeeze(1)  # shape: [32, Action_Dim]

                # --- 融入你设计的动作惩罚机制 (并行化处理) ---
                for i in range(num_envs):
                    if last_actions[i] is not None and active_mask[i]:
                        # 惩罚系数: 5.0，防止连续选相同动作
                        logits[i, last_actions[i]] -= 5.0

                dist = Categorical(logits=logits)
                actions = dist.sample()  # 瞬间采样 32 个动作
                log_probs = dist.log_prob(actions)
            gpu_time = time.time() - t0
            # 拉回 CPU，准备给量子环境执行
            actions_cpu = actions.cpu().tolist()
            log_probs_cpu = log_probs.cpu().tolist()

            next_obs_list = list(current_obs)  # 拷贝一份作为占位

            def step_single_env(idx):
                """定义单个环境的步进任务"""
                if not active_mask[idx]:
                    return idx, None
                # 执行极其耗时的物理模拟
                return idx, envs[idx].step(actions_cpu[idx])

            # 启动线程池（max_workers 可以设为你的电脑逻辑核心数，比如 16 或 32）
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                # 让所有存活的环境同时开始计算！
                results = list(executor.map(step_single_env, range(num_envs)))

            # 收集并发计算的结果
            for idx, res in results:
                if res is None:
                    continue
                nxt_o, reward, done, truncated, _ = res
                nxt_o_flat = flatten(obs_space, nxt_o)

                # 记录该线路的数据
                trajectories[idx]["obs"].append(current_obs[idx])
                trajectories[idx]["actions"].append(actions_cpu[idx])
                trajectories[idx]["rewards"].append(reward)
                trajectories[idx]["log_probs"].append(log_probs_cpu[idx])

                last_actions[idx] = actions_cpu[idx]
                next_obs_list[idx] = nxt_o_flat

                # 如果这条线路纯化结束，标记为 False
                if done or truncated:
                    active_mask[idx] = False
            # =======================================================

            current_obs = next_obs_list
            hidden = next_hidden

            # (顺便把你之前的硬编码 128 改成动态的 num_envs)
            active_num = sum(active_mask)
            # print(f"  >>> [推演心跳] 当前第 {step_count} 步 | 还在跑的量子线路剩余: {active_num}/{num_envs}    ", end='\r', flush=True)
        # print()
        # =======================================================
        # 4. 统一安全销毁环境，释放 CPU 内存
        # =======================================================
        for env in envs:
            env.close()

        # =======================================================
        # 5. 组装数据 (完全保留你原本的 GRPO 逻辑)
        # =======================================================
        group_trajectories = []
        for i in range(num_envs):
            traj = trajectories[i]

            # 安全校验：如果某条线路在环境崩溃等极端情况下长度为0，跳过它
            if len(traj["obs"]) == 0:
                continue

            # 计算无折扣的总保真度/得分
            total_reward = sum(traj["rewards"])
            returns = self.compute_discounted_returns(traj["rewards"])

            group_trajectories.append({
                'obs': torch.FloatTensor(np.array(traj["obs"])).to(self.cfg.device),
                'actions': torch.LongTensor(traj["actions"]).to(self.cfg.device),
                'returns': torch.FloatTensor(returns).to(self.cfg.device),
                'old_log_probs': torch.FloatTensor(traj["log_probs"]).to(self.cfg.device),
                'length': len(traj["obs"]),
                'total_reward': total_reward
            })

        # 如果所有线路都无效，直接跳过本次更新（防止崩溃）
        if len(group_trajectories) == 0:
            return 0.0, 0.0

        # =======================================================
        # 5.5 优势计算与归一化 (Trajectory-Level 一荣俱荣机制)
        # ⚠️ 注意：之前报 KeyError 就是因为少了这一段！
        # =======================================================
        group_totals = torch.FloatTensor([t['total_reward'] for t in group_trajectories]).to(self.cfg.device)
        adv_mean = group_totals.mean()
        adv_std = group_totals.std(correction=0) + 1e-8
        group_advantages = (group_totals - adv_mean) / adv_std

        for i, t in enumerate(group_trajectories):
            # 这里正式把 'advantages' 这个键和对应的值塞入字典
            t['advantages'] = torch.full_like(t['returns'], group_advantages[i])

        # =======================================================
        # 6. PPO/GRPO 网络更新循环 (LSTM) - 全 GPU 并行版
        # =======================================================
        # 提取所有 Tensor 并用 0 补齐长度 (Padding)
        obs_batch = pad_sequence([t['obs'] for t in group_trajectories], batch_first=True)
        actions_batch = pad_sequence([t['actions'] for t in group_trajectories], batch_first=True)
        old_lps_batch = pad_sequence([t['old_log_probs'] for t in group_trajectories], batch_first=True)
        advs_batch = pad_sequence([t['advantages'] for t in group_trajectories], batch_first=True)

        # 生成掩码 (Mask)，过滤掉补齐的无效部分
        lengths = torch.LongTensor([t['length'] for t in group_trajectories]).to(self.cfg.device)
        max_len = obs_batch.size(1)
        mask = torch.arange(max_len, device=self.cfg.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)

        loss_val = 0
        for _ in range(self.cfg.update_epochs):
            # 瞬间处理所有线路 (Batch 推理)
            logits_seq, _ = self.policy(obs_batch, hidden_state=None)

            dist = Categorical(logits=logits_seq)
            curr_log_probs = dist.log_prob(actions_batch)
            entropy = dist.entropy()

            with torch.no_grad():
                ref_logits, _ = self.ref_policy(obs_batch, hidden_state=None)
                ref_dist = Categorical(logits=ref_logits)
            kl_div = kl_divergence(ref_dist, dist)

            ratio = torch.exp(curr_log_probs - old_lps_batch)
            surr1 = ratio * advs_batch
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advs_batch

            # 计算每个 step 的 Loss 矩阵
            step_loss = -torch.min(surr1, surr2) \
                        + self.cfg.beta_kl * kl_div \
                        - self.cfg.entropy_coef * entropy

            # 过滤掉 Padding 部分的无效 Loss，求均值
            valid_loss = step_loss * mask
            avg_loss = valid_loss.sum() / mask.sum()

            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            loss_val = avg_loss.item()

        self.ref_policy.load_state_dict(self.policy.state_dict())

        return loss_val, group_totals.mean().item()

    def train_step3(self, envs):
        self.policy.train()
        num_envs = len(envs)

        # =======================================================
        # 1. 直接复用传进来的环境池，极其迅速地完成 Reset
        # =======================================================
        obs_space = envs[0].observation_space
        trajectories = [{"obs": [], "actions": [], "rewards": [], "log_probs": []} for _ in range(num_envs)]

        current_obs = []
        for env in envs:
            o, _ = env.reset()
            current_obs.append(flatten(obs_space, o))

        # =======================================================
        # 2. LSTM 批量隐藏状态初始化
        # shape: [num_layers(1), Batch_Size, Hidden_Size(256)]
        # =======================================================
        h0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        c0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        hidden = (h0, c0)

        active_mask = [True] * num_envs
        last_actions = [None] * num_envs
        step_count = 0

        # =======================================================
        # 3. 核心推演循环 (GPU 并行推理 + CPU 多线程推演)
        # =======================================================
        while any(active_mask):
            step_count += 1

            # --- [GPU 瞬间完成 32 条线路的动作预测] ---
            obs_tensor = torch.FloatTensor(np.array(current_obs)).to(self.cfg.device).unsqueeze(1)

            with torch.no_grad():
                logits, next_hidden = self.policy(obs_tensor, hidden)
                logits = logits.squeeze(1)

                # 动作惩罚机制：防止原位打转
                for i in range(num_envs):
                    if last_actions[i] is not None and active_mask[i]:
                        logits[i, last_actions[i]] -= 5.0

                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_cpu = actions.cpu().tolist()
            log_probs_cpu = log_probs.cpu().tolist()
            next_obs_list = list(current_obs)

            # --- [CPU 多线程并发推演物理环境] ---
            def step_single_env(idx):
                """定义单个环境的步进任务"""
                if not active_mask[idx]:
                    return idx, None
                # 执行耗时的物理模拟
                return idx, envs[idx].step(actions_cpu[idx])

            # 启动线程池让 CPU 所有核心一起干活 (max_workers=16 或你的 CPU 核心数)
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                results = list(executor.map(step_single_env, range(num_envs)))

            # --- [收集并发计算结果] ---
            for idx, res in results:
                if res is None:
                    continue
                nxt_o, reward, done, truncated, _ = res
                nxt_o_flat = flatten(obs_space, nxt_o)

                trajectories[idx]["obs"].append(current_obs[idx])
                trajectories[idx]["actions"].append(actions_cpu[idx])
                trajectories[idx]["rewards"].append(reward)
                trajectories[idx]["log_probs"].append(log_probs_cpu[idx])

                last_actions[idx] = actions_cpu[idx]
                next_obs_list[idx] = nxt_o_flat

                if done or truncated:
                    active_mask[idx] = False

            current_obs = next_obs_list
            hidden = next_hidden

            # 强力心跳打印
            active_num = sum(active_mask)
            print(f"  >>> [推演心跳] 当前第 {step_count} 步 | 还在跑的线路剩余: {active_num}/{num_envs}    ", end='\r',
                  flush=True)

        print()  # 换行，防止日志排版错乱

        # =======================================================
        # 4. 组装数据 (去除无效线路)
        # =======================================================
        group_trajectories = []
        for i in range(num_envs):
            traj = trajectories[i]
            if len(traj["obs"]) == 0:
                continue

            total_reward = sum(traj["rewards"])
            returns = self.compute_discounted_returns(traj["rewards"])

            group_trajectories.append({
                'obs': torch.FloatTensor(np.array(traj["obs"])).to(self.cfg.device),
                'actions': torch.LongTensor(traj["actions"]).to(self.cfg.device),
                'returns': torch.FloatTensor(returns).to(self.cfg.device),
                'old_log_probs': torch.FloatTensor(traj["log_probs"]).to(self.cfg.device),
                'length': len(traj["obs"]),
                'total_reward': total_reward
            })

        if len(group_trajectories) == 0:
            return 0.0, 0.0

        # =======================================================
        # 5. 优势计算与归一化 (Trajectory-Level 一荣俱荣机制)
        # =======================================================
        group_totals = torch.FloatTensor([t['total_reward'] for t in group_trajectories]).to(self.cfg.device)
        adv_mean = group_totals.mean()
        adv_std = group_totals.std(correction=0) + 1e-8
        group_advantages = (group_totals - adv_mean) / adv_std

        for i, t in enumerate(group_trajectories):
            t['advantages'] = torch.full_like(t['returns'], group_advantages[i])

        # =======================================================
        # 6. PPO/GRPO 网络更新循环 (LSTM) - 全 GPU 矩阵满载版
        # =======================================================
        obs_batch = pad_sequence([t['obs'] for t in group_trajectories], batch_first=True)
        actions_batch = pad_sequence([t['actions'] for t in group_trajectories], batch_first=True)
        old_lps_batch = pad_sequence([t['old_log_probs'] for t in group_trajectories], batch_first=True)
        advs_batch = pad_sequence([t['advantages'] for t in group_trajectories], batch_first=True)

        lengths = torch.LongTensor([t['length'] for t in group_trajectories]).to(self.cfg.device)
        max_len = obs_batch.size(1)
        mask = torch.arange(max_len, device=self.cfg.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)

        loss_val = 0
        for _ in range(self.cfg.update_epochs):
            logits_seq, _ = self.policy(obs_batch, hidden_state=None)

            dist = Categorical(logits=logits_seq)
            curr_log_probs = dist.log_prob(actions_batch)
            entropy = dist.entropy()

            with torch.no_grad():
                ref_logits, _ = self.ref_policy(obs_batch, hidden_state=None)
                ref_dist = Categorical(logits=ref_logits)
            kl_div = kl_divergence(ref_dist, dist)

            ratio = torch.exp(curr_log_probs - old_lps_batch)
            surr1 = ratio * advs_batch
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advs_batch

            step_loss = -torch.min(surr1, surr2) \
                        + self.cfg.beta_kl * kl_div \
                        - self.cfg.entropy_coef * entropy

            valid_loss = step_loss * mask
            avg_loss = valid_loss.sum() / mask.sum()

            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            loss_val = avg_loss.item()

        self.ref_policy.load_state_dict(self.policy.state_dict())

        return loss_val, group_totals.mean().item()

        # 修复 1：将参数名由 env_factory 改为 envs（接收外部传入的环境列表）

    def train_step0330(self, envs):
        self.policy.train()
        num_envs = len(envs)

        # =======================================================
        # 1. 极其迅速地完成 Reset，并加入统一起跑线 (common_seed)
        # =======================================================
        obs_space = envs[0].observation_space
        trajectories = [{"obs": [], "actions": [], "rewards": [], "log_probs": []} for _ in range(num_envs)]

        # 🌟 核心新增：生成一个统一的种子，统一起跑线
        common_seed = random.randint(0, 9999999)

        current_obs = []
        for env in envs:
            # 所有环境复用同一个 seed
            o, _ = env.reset(seed=common_seed)
            current_obs.append(flatten(obs_space, o))

        # =======================================================
        # 2. LSTM 批量隐藏状态初始化
        # =======================================================
        h0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        c0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        hidden = (h0, c0)

        active_mask = [True] * num_envs
        last_actions = [None] * num_envs
        step_count = 0

        # =======================================================
        # 3. 核心推演循环 (GPU 并行推理 + CPU 多线程推演)
        # =======================================================
        while any(active_mask):
            step_count += 1

            # ---[GPU 瞬间完成所有线路的动作预测] ---
            obs_tensor = torch.FloatTensor(np.array(current_obs)).to(self.cfg.device).unsqueeze(1)

            with torch.no_grad():
                logits, next_hidden = self.policy(obs_tensor, hidden)
                logits = logits.squeeze(1)

                # 动作惩罚机制：防止原位打转
                # for i in range(num_envs):
                # if last_actions[i] is not None and active_mask[i]:
                # logits[i, last_actions[i]] -= 5.0

                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_cpu = actions.cpu().tolist()
            log_probs_cpu = log_probs.cpu().tolist()
            next_obs_list = list(current_obs)

            # ---[CPU 多线程并发推演物理环境] ---
            def step_single_env(idx):
                if not active_mask[idx]:
                    return idx, None
                return idx, envs[idx].step(actions_cpu[idx])

            # 启动线程池让 CPU 所有核心一起干活 (量子模拟很吃 CPU)
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                results = list(executor.map(step_single_env, range(num_envs)))

            # --- [收集并发计算结果] ---
            for idx, res in results:
                if res is None:
                    continue
                nxt_o, reward, done, truncated, _ = res
                nxt_o_flat = flatten(obs_space, nxt_o)

                trajectories[idx]["obs"].append(current_obs[idx])
                trajectories[idx]["actions"].append(actions_cpu[idx])
                trajectories[idx]["rewards"].append(reward)
                trajectories[idx]["log_probs"].append(log_probs_cpu[idx])

                last_actions[idx] = actions_cpu[idx]
                next_obs_list[idx] = nxt_o_flat

                if done or truncated:
                    active_mask[idx] = False

            current_obs = next_obs_list
            hidden = next_hidden

            # 强力心跳打印 (如果在 Pycharm 运行，\r 会在同一行覆盖)
            active_num = sum(active_mask)
            print(f"  >>>[推演心跳] 当前第 {step_count} 步 | 还在跑的线路剩余: {active_num}/{num_envs}    ", end='\r',
                  flush=True)

        print()  # 换行，防止日志排版错乱

        # =======================================================
        # 4. 组装数据
        # =======================================================
        group_trajectories = []
        all_total_rewards = []  # 专门记录一下每条线路的总得分用于 logging

        for i in range(num_envs):
            traj = trajectories[i]
            if len(traj["obs"]) == 0:
                continue

            total_reward = sum(traj["rewards"])
            all_total_rewards.append(total_reward)
            # 这里调用了你的折扣回报逻辑！
            returns = self.compute_discounted_returns(traj["rewards"])

            group_trajectories.append({
                'obs': torch.FloatTensor(np.array(traj["obs"])).to(self.cfg.device),
                'actions': torch.LongTensor(traj["actions"]).to(self.cfg.device),
                'returns': torch.FloatTensor(returns).to(self.cfg.device),
                'old_log_probs': torch.FloatTensor(traj["log_probs"]).to(self.cfg.device),
                'length': len(traj["obs"]),
                'total_reward': total_reward
            })

        if len(group_trajectories) == 0:
            return 0.0, 0.0

        # =======================================================
        # 🌟 5. 优势计算与归一化 (完全同步你最新的 Step-Level 折扣逻辑)
        # =======================================================
        all_returns = torch.cat([t['returns'] for t in group_trajectories])

        # 计算当前批次所有量子态的“绝对价值度量衡”
        returns_mean = all_returns.mean()
        returns_std = all_returns.std(correction=0) + 1e-8

        # 凭绝对实力说话，无视时间轴的强行对齐
        for t in group_trajectories:
            t['advantages'] = (t['returns'] - returns_mean) / returns_std

        # =======================================================
        # 6. PPO/GRPO 网络更新循环 (LSTM) - 全 GPU 矩阵满载版
        # =======================================================
        # 使用 pad_sequence 将长短不一的线路对齐，GPU 瞬间算完！
        obs_batch = pad_sequence([t['obs'] for t in group_trajectories], batch_first=True)
        actions_batch = pad_sequence([t['actions'] for t in group_trajectories], batch_first=True)
        old_lps_batch = pad_sequence([t['old_log_probs'] for t in group_trajectories], batch_first=True)
        advs_batch = pad_sequence([t['advantages'] for t in group_trajectories], batch_first=True)

        # 生成 Mask 掩码，过滤掉补齐的空数据
        lengths = torch.LongTensor([t['length'] for t in group_trajectories]).to(self.cfg.device)
        max_len = obs_batch.size(1)
        mask = torch.arange(max_len, device=self.cfg.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)

        loss_val = 0
        for _ in range(self.cfg.update_epochs):
            # LSTM 并行处理全部 batch
            logits_seq, _ = self.policy(obs_batch, hidden_state=None)

            dist = Categorical(logits=logits_seq)
            curr_log_probs = dist.log_prob(actions_batch)
            entropy = dist.entropy()

            with torch.no_grad():
                ref_logits, _ = self.ref_policy(obs_batch, hidden_state=None)
                ref_dist = Categorical(logits=ref_logits)
            kl_div = kl_divergence(ref_dist, dist)

            ratio = torch.exp(curr_log_probs - old_lps_batch)
            surr1 = ratio * advs_batch
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advs_batch

            # 【重要】使用 mask 过滤掉无效时间的 loss
            step_loss = -torch.min(surr1, surr2) \
                        + self.cfg.beta_kl * kl_div \
                        - self.cfg.entropy_coef * entropy

            valid_loss = step_loss * mask
            avg_loss = valid_loss.sum() / mask.sum()

            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            loss_val = avg_loss.item()

        self.ref_policy.load_state_dict(self.policy.state_dict())

        # 返回 Loss 和本轮平均 Reward
        return loss_val, np.mean(all_total_rewards)

    def train_step_fast2(self, envs):
        self.policy.train()
        num_envs = len(envs)
        obs_space = envs[0].observation_space

        common_seed = random.randint(0, 9999999)
        print(f"  >>> [系统心跳] 正在单线程 Reset {num_envs} 个量子环境...", end='\r', flush=True)

        current_obs = []
        for env in envs:
            o, _ = env.reset(seed=common_seed)
            current_obs.append(flatten(obs_space, o))

        print(f"  >>> [系统心跳] 环境 Reset 完毕！开始物理推演...                 ")

        trajectories = [{"obs": [], "actions": [], "rewards": [], "log_probs": []} for _ in range(num_envs)]

        h0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        c0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        hidden = (h0, c0)

        active_mask = [True] * num_envs
        last_actions = [None] * num_envs
        step_count = 0

        while any(active_mask):
            step_count += 1
            obs_tensor = torch.FloatTensor(np.array(current_obs)).to(self.cfg.device).unsqueeze(1)

            with torch.no_grad():
                logits, next_hidden = self.policy(obs_tensor, hidden)
                logits = logits.squeeze(1)

                # for i in range(num_envs):
                #    if last_actions[i] is not None and active_mask[i]:
                #        logits[i, last_actions[i]] -= 5.0

                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_cpu = actions.cpu().tolist()
            log_probs_cpu = log_probs.cpu().tolist()
            next_obs_list = list(current_obs)

            # =======================================================
            # 🚀 绝对安全的单线程物理演化 (杜绝所有底层 C 语言崩溃)
            # =======================================================
            for idx in range(num_envs):
                if not active_mask[idx]:
                    continue

                # 打印当前计算进度，确保没有卡死
                print(f"  >>> [物理推演] 第 {step_count} 步 | 正在计算线路 {idx + 1}/{num_envs}...    ", end='\r',
                      flush=True)

                nxt_o, reward, done, truncated, _ = envs[idx].step(actions_cpu[idx])

                trajectories[idx]["obs"].append(current_obs[idx])
                trajectories[idx]["actions"].append(actions_cpu[idx])
                trajectories[idx]["rewards"].append(reward)
                trajectories[idx]["log_probs"].append(log_probs_cpu[idx])

                last_actions[idx] = actions_cpu[idx]
                next_obs_list[idx] = flatten(obs_space, nxt_o)

                if done or truncated:
                    active_mask[idx] = False
            # =======================================================

            current_obs = next_obs_list
            hidden = next_hidden
            print(f"  >>> [物理推演] 第 {step_count} 步 | 存活线路: {sum(active_mask)}/{num_envs} 完成!       ",
                  end='\r')
        print()

        group_trajectories = []
        all_total_rewards = []
        for i in range(num_envs):
            traj = trajectories[i]
            if len(traj["obs"]) == 0: continue

            total_reward = sum(traj["rewards"])
            all_total_rewards.append(total_reward)
            returns = self.compute_discounted_returns(traj["rewards"])

            group_trajectories.append({
                'obs': torch.FloatTensor(np.array(traj["obs"])).to(self.cfg.device),
                'actions': torch.LongTensor(traj["actions"]).to(self.cfg.device),
                'returns': torch.FloatTensor(returns).to(self.cfg.device),
                'old_log_probs': torch.FloatTensor(traj["log_probs"]).to(self.cfg.device),
                'length': len(traj["obs"]),
            })

        if len(group_trajectories) == 0: return 0.0, 0.0

        all_returns = torch.cat([t['returns'] for t in group_trajectories])
        returns_mean = all_returns.mean()
        returns_std = all_returns.std(correction=0) + 1e-8
        for t in group_trajectories:
            t['advantages'] = (t['returns'] - returns_mean) / returns_std

        obs_batch = pad_sequence([t['obs'] for t in group_trajectories], batch_first=True)
        actions_batch = pad_sequence([t['actions'] for t in group_trajectories], batch_first=True)
        old_lps_batch = pad_sequence([t['old_log_probs'] for t in group_trajectories], batch_first=True)
        advs_batch = pad_sequence([t['advantages'] for t in group_trajectories], batch_first=True)

        lengths = torch.LongTensor([t['length'] for t in group_trajectories]).to(self.cfg.device)
        max_len = obs_batch.size(1)
        mask = torch.arange(max_len, device=self.cfg.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)

        loss_val = 0
        for _ in range(self.cfg.update_epochs):
            logits_seq, _ = self.policy(obs_batch, hidden_state=None)

            dist = Categorical(logits=logits_seq)
            curr_log_probs = dist.log_prob(actions_batch)
            entropy = dist.entropy()

            with torch.no_grad():
                ref_logits, _ = self.ref_policy(obs_batch, hidden_state=None)
                ref_dist = Categorical(logits=ref_logits)
            kl_div = kl_divergence(ref_dist, dist)

            ratio = torch.exp(curr_log_probs - old_lps_batch)
            surr1 = ratio * advs_batch
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advs_batch

            step_loss = -torch.min(surr1, surr2) + self.cfg.beta_kl * kl_div - self.cfg.entropy_coef * entropy
            valid_loss = step_loss * mask
            avg_loss = valid_loss.sum() / mask.sum()

            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            loss_val = avg_loss.item()

        self.ref_policy.load_state_dict(self.policy.state_dict())
        return loss_val, np.mean(all_total_rewards)

    def train_step_fast_DGPO(self, envs):
        from threadpoolctl import threadpool_limits
        import concurrent.futures
        import threading
        import traceback
        import os
        from gymnasium.spaces import flatten

        # 定义一个函数内局部锁，确保物理模拟串行进入 C 库，但 Python 逻辑并行
        if not hasattr(self, '_env_lock'):
            self._env_lock = threading.Lock()

        self.policy.train()
        num_envs = len(envs)
        obs_space = envs[0].observation_space

        common_seed = random.randint(0, 9999999)
        print(f"  >>> [系统心跳] 正在单线程 Reset {num_envs} 个量子环境...", end='\r', flush=True)

        current_obs = []
        for env in envs:
            o, _ = env.reset(seed=common_seed)
            current_obs.append(flatten(obs_space, o))

        print(f"  >>> [系统心跳] 环境池就绪！启动安全多线程推演...                 ")

        trajectories = [{"obs": [], "actions": [], "rewards": [], "log_probs": []} for _ in range(num_envs)]
        h0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        c0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        hidden = (h0, c0)

        active_mask = [True] * num_envs
        last_actions = [None] * num_envs
        step_count = 0

        while any(active_mask):
            step_count += 1
            # 神经网络批处理：一次性算出所有环境的动作
            obs_tensor = torch.FloatTensor(np.array(current_obs)).to(self.cfg.device).unsqueeze(1)
            with torch.no_grad():
                logits, next_hidden = self.policy(obs_tensor, hidden)
                logits = logits.squeeze(1)

                # 动作惩罚逻辑（防止原位打转，建议保留）
                for i in range(num_envs):
                    if last_actions[i] is not None and active_mask[i]:
                        logits[i, last_actions[i]] -= 5.0

                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_cpu = actions.cpu().tolist()
            log_probs_cpu = log_probs.cpu().tolist()
            next_obs_list = list(current_obs)

            # --- 核心：安全的多线程推演任务 ---
            def step_single_env(idx):
                if not active_mask[idx]:
                    return idx, None, None
                try:
                    # 使用锁保护：防止 OpenBLAS 在 512x512 矩阵运算时发生内存冲突
                    with self._env_lock:
                        res = envs[idx].step(actions_cpu[idx])
                    return idx, res, None
                except Exception:
                    return idx, None, traceback.format_exc()

            # 建议 max_workers 设为 12 或 16，不要超过 32 以防 OpenBLAS 报警
            max_workers = min(num_envs, 44)

            with threadpool_limits(limits=1):
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(step_single_env, range(num_envs)))

            for idx, res, error_msg in results:
                if error_msg:
                    print(f"\n❌ 环境 {idx} 崩溃，动作 {actions_cpu[idx]} 报错:\n{error_msg}")
                    import sys;
                    sys.exit(1)
                if res is None: continue

                nxt_o, reward, done, truncated, _ = res
                trajectories[idx]["obs"].append(current_obs[idx])
                trajectories[idx]["actions"].append(actions_cpu[idx])
                trajectories[idx]["rewards"].append(reward)
                trajectories[idx]["log_probs"].append(log_probs_cpu[idx])

                last_actions[idx] = actions_cpu[idx]
                next_obs_list[idx] = flatten(obs_space, nxt_o)
                if done or truncated:
                    active_mask[idx] = False

            current_obs = next_obs_list
            hidden = next_hidden
            print(f"  >>> [物理推演] 第 {step_count} 步 | 存活: {sum(active_mask)}/{num_envs}       ", end='\r')

        print("\n  >>> [系统心跳] 推演结束，开始计算 Advantage 并更新网络...")

        # =======================================================
        # 🌟 恢复：全局展平归一化 (Global-Flattened Normalization)
        # =======================================================
        group_trajectories = []
        all_total_rewards = []
        for i in range(num_envs):
            traj = trajectories[i]
            if len(traj["obs"]) == 0: continue
            all_total_rewards.append(sum(traj["rewards"]))
            returns = self.compute_discounted_returns(traj["rewards"])
            group_trajectories.append({
                'obs': torch.FloatTensor(np.array(traj["obs"])).to(self.cfg.device),
                'actions': torch.LongTensor(traj["actions"]).to(self.cfg.device),
                'returns': torch.FloatTensor(returns).to(self.cfg.device),
                'old_log_probs': torch.FloatTensor(traj["log_probs"]).to(self.cfg.device),
                'length': len(traj["obs"]),
            })

        if not group_trajectories: return 0.0, 0.0

        all_returns = torch.cat([t['returns'] for t in group_trajectories])
        ret_mean, ret_std = all_returns.mean(), all_returns.std(correction=0) + 1e-8
        for t in group_trajectories:
            t['advantages'] = (t['returns'] - ret_mean) / ret_std

        # =======================================================
        # 🚀 终极网络更新逻辑 (动态非对称裁剪 Adaptive Clipping)
        # =======================================================
        obs_batch = pad_sequence([t['obs'] for t in group_trajectories], batch_first=True)
        actions_batch = pad_sequence([t['actions'] for t in group_trajectories], batch_first=True)
        old_lps_batch = pad_sequence([t['old_log_probs'] for t in group_trajectories], batch_first=True)
        advs_batch = pad_sequence([t['advantages'] for t in group_trajectories], batch_first=True)

        lengths = torch.LongTensor([t['length'] for t in group_trajectories]).to(self.cfg.device)
        mask = torch.arange(obs_batch.size(1), device=self.cfg.device).expand(len(lengths),
                                                                              obs_batch.size(1)) < lengths.unsqueeze(1)

        loss_val = 0
        for _ in range(self.cfg.update_epochs):
            logits_seq, _ = self.policy(obs_batch, None)
            dist = Categorical(logits=logits_seq)
            curr_log_probs = dist.log_prob(actions_batch)

            with torch.no_grad():
                ref_logits, _ = self.ref_policy(obs_batch, None)
                ref_dist = Categorical(logits=ref_logits)

            kl_div = kl_divergence(ref_dist, dist)
            ratio = torch.exp(curr_log_probs - old_lps_batch)

            # -----------------------------------------------------------------
            # 🚀 严谨的数学逻辑：上方软衰减 (保护顿悟) + 下方硬截断 (底线探索)
            # -----------------------------------------------------------------
            eps = self.cfg.clip_epsilon
            upper_bound = 1.0 + eps
            lower_bound = 1.0 - eps
            decay_alpha = 0.2  # 软衰减系数 (控制极品线路的宽容度)

            # 1. 好动作 (advs_batch > 0)：只有冲破上限时才做 log 软衰减
            ratio_up = torch.where(
                ratio > upper_bound,
                upper_bound + decay_alpha * torch.log(1.0 + ratio - upper_bound),
                ratio
            )

            # 2. 坏动作 (advs_batch <= 0)：跌破下限时强行锁死在 0.8 (防止探索欲枯竭)
            ratio_down = torch.max(ratio, torch.tensor(lower_bound, device=ratio.device))

            # 3. 动态融合：根据优势的正负号发牌
            effective_ratio = torch.where(advs_batch > 0, ratio_up, ratio_down)

            # 4. 计算最终的 Surrogate 优势
            surr_soft = effective_ratio * advs_batch
            # -----------------------------------------------------------------

            # 最终的 Step Loss，结合了 KL 惩罚和策略熵
            step_loss = -surr_soft + self.cfg.beta_kl * kl_div - self.cfg.entropy_coef * dist.entropy()

            # 使用 Mask 过滤无效时间步，并严谨求平均
            avg_loss = (step_loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            loss_val = avg_loss.item()

        # 同步参考模型
        self.ref_policy.load_state_dict(self.policy.state_dict())
        return loss_val, np.mean(all_total_rewards)

    def train_step_fast(self, envs):
        from threadpoolctl import threadpool_limits
        import concurrent.futures
        import threading
        import traceback
        import os
        from gymnasium.spaces import flatten

        # 定义一个函数内局部锁，确保物理模拟串行进入 C 库，但 Python 逻辑并行
        if not hasattr(self, '_env_lock'):
            self._env_lock = threading.Lock()

        self.policy.train()
        num_envs = len(envs)
        obs_space = envs[0].observation_space

        common_seed = random.randint(0, 9999999)
        print(f"  >>> [系统心跳] 正在单线程 Reset {num_envs} 个量子环境...", end='\r', flush=True)

        current_obs = []
        for env in envs:
            o, _ = env.reset(seed=common_seed)
            current_obs.append(flatten(obs_space, o))

        print(f"  >>> [系统心跳] 环境池就绪！启动安全多线程推演...                 ")

        trajectories = [{"obs": [], "actions": [], "rewards": [], "log_probs": []} for _ in range(num_envs)]
        h0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        c0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        hidden = (h0, c0)

        active_mask = [True] * num_envs
        last_actions = [None] * num_envs
        step_count = 0

        while any(active_mask):
            step_count += 1
            # 神经网络批处理：一次性算出所有环境的动作
            obs_tensor = torch.FloatTensor(np.array(current_obs)).to(self.cfg.device).unsqueeze(1)
            with torch.no_grad():
                logits, next_hidden = self.policy(obs_tensor, hidden)
                logits = logits.squeeze(1)
                # 动作惩罚逻辑（如果需要可以取消注释）
                # for i in range(num_envs):
                #     if last_actions[i] is not None and active_mask[i]:
                #         logits[i, last_actions[i]] -= 5.0
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_cpu = actions.cpu().tolist()
            log_probs_cpu = log_probs.cpu().tolist()
            next_obs_list = list(current_obs)

            # --- 核心：安全的多线程推演任务 ---
            def step_single_env(idx):
                if not active_mask[idx]:
                    return idx, None, None
                try:
                    # 使用锁保护：防止 OpenBLAS 在 512x512 矩阵运算时发生内存冲突
                    with self._env_lock:
                        res = envs[idx].step(actions_cpu[idx])
                    return idx, res, None
                except Exception:
                    return idx, None, traceback.format_exc()

            # 建议 max_workers 设为 12 或 16，不要超过 32 以防 OpenBLAS 报警
            max_workers = min(num_envs, 44)

            with threadpool_limits(limits=1):
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(step_single_env, range(num_envs)))

            for idx, res, error_msg in results:
                if error_msg:
                    print(f"\n❌ 环境 {idx} 崩溃，动作 {actions_cpu[idx]} 报错:\n{error_msg}")
                    import sys;
                    sys.exit(1)
                if res is None: continue

                nxt_o, reward, done, truncated, _ = res
                trajectories[idx]["obs"].append(current_obs[idx])
                trajectories[idx]["actions"].append(actions_cpu[idx])
                trajectories[idx]["rewards"].append(reward)
                trajectories[idx]["log_probs"].append(log_probs_cpu[idx])

                last_actions[idx] = actions_cpu[idx]
                next_obs_list[idx] = flatten(obs_space, nxt_o)
                if done or truncated:
                    active_mask[idx] = False

            current_obs = next_obs_list
            hidden = next_hidden
            print(f"  >>> [物理推演] 第 {step_count} 步 | 存活: {sum(active_mask)}/{num_envs}       ", end='\r')

        print("\n  >>> [系统心跳] 推演结束，开始计算 Advantage 并更新网络...")

        # =======================================================
        # 🌟 恢复：全局展平归一化 (Global-Flattened Normalization)
        # =======================================================
        group_trajectories = []
        all_total_rewards = []
        for i in range(num_envs):
            traj = trajectories[i]
            if len(traj["obs"]) == 0: continue
            all_total_rewards.append(sum(traj["rewards"]))
            returns = self.compute_discounted_returns(traj["rewards"])
            group_trajectories.append({
                'obs': torch.FloatTensor(np.array(traj["obs"])).to(self.cfg.device),
                'actions': torch.LongTensor(traj["actions"]).to(self.cfg.device),
                'returns': torch.FloatTensor(returns).to(self.cfg.device),
                'old_log_probs': torch.FloatTensor(traj["log_probs"]).to(self.cfg.device),
                'length': len(traj["obs"]),
            })

        if not group_trajectories: return 0.0, 0.0

        all_returns = torch.cat([t['returns'] for t in group_trajectories])
        ret_mean, ret_std = all_returns.mean(), all_returns.std(correction=0) + 1e-8
        for t in group_trajectories:
            t['advantages'] = (t['returns'] - ret_mean) / ret_std

        # =======================================================
        # 网络更新逻辑 (保持 PPO/GRPO 核心不变)
        # =======================================================
        obs_batch = pad_sequence([t['obs'] for t in group_trajectories], batch_first=True)
        actions_batch = pad_sequence([t['actions'] for t in group_trajectories], batch_first=True)
        old_lps_batch = pad_sequence([t['old_log_probs'] for t in group_trajectories], batch_first=True)
        advs_batch = pad_sequence([t['advantages'] for t in group_trajectories], batch_first=True)

        lengths = torch.LongTensor([t['length'] for t in group_trajectories]).to(self.cfg.device)
        mask = torch.arange(obs_batch.size(1), device=self.cfg.device).expand(len(lengths),
                                                                              obs_batch.size(1)) < lengths.unsqueeze(1)

        loss_val = 0
        for _ in range(self.cfg.update_epochs):
            logits_seq, _ = self.policy(obs_batch, None)
            dist = Categorical(logits=logits_seq)
            curr_log_probs = dist.log_prob(actions_batch)

            with torch.no_grad():
                ref_logits, _ = self.ref_policy(obs_batch, None)
                ref_dist = Categorical(logits=ref_logits)

            kl_div = kl_divergence(ref_dist, dist)
            ratio = torch.exp(curr_log_probs - old_lps_batch)
            surr1 = ratio * advs_batch
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advs_batch

            step_loss = -torch.min(surr1, surr2) + self.cfg.beta_kl * kl_div - self.cfg.entropy_coef * dist.entropy()
            avg_loss = (step_loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            loss_val = avg_loss.item()

        self.ref_policy.load_state_dict(self.policy.state_dict())
        return loss_val, np.mean(all_total_rewards)

    def train_step(self, env_factory):
        self.policy.train()

        group_trajectories = []
        all_rewards = []

        # =======================================================
        # 🌟 核心新增：生成一个统一的种子，统一起跑线
        # =======================================================
        common_seed = random.randint(0, 9999999)

        for _ in range(self.cfg.group_size):
            env = env_factory()

            # --- [关键修改 1] 获取 Observation Space ---
            obs_space = env.observation_space

            # =======================================================
            # 🌟 核心修改：所有环境在 reset 时强制使用同一个 common_seed
            # =======================================================
            obs, _ = env.reset(seed=common_seed)

            # --- [关键修改 2] 立即展平 obs ---
            obs = flatten(obs_space, obs)

            hidden = None

            traj_obs = []
            traj_actions = []
            traj_rewards = []
            traj_log_probs = []

            done = False
            truncated = False
            last_action = None  # 初始化

            while not (done or truncated):
                action, log_prob, next_hidden = self.get_action(obs, hidden)
                next_obs, reward, done, truncated, _ = env.step(action)

                # --- [关键修改 3] 展平 next_obs ---
                next_obs = flatten(obs_space, next_obs)

                # 记录数据
                traj_obs.append(obs)
                traj_actions.append(action)
                traj_rewards.append(reward)
                traj_log_probs.append(log_prob)

                obs = next_obs
                hidden = next_hidden
                last_action = action  # 更新 last_action

            env.close()
            all_rewards.append(sum(traj_rewards))

            returns = self.compute_discounted_returns(traj_rewards)

            group_trajectories.append({
                'obs': torch.FloatTensor(np.array(traj_obs)).to(self.cfg.device),
                'actions': torch.LongTensor(traj_actions).to(self.cfg.device),
                'returns': torch.FloatTensor(returns).to(self.cfg.device),
                'old_log_probs': torch.FloatTensor(traj_log_probs).to(self.cfg.device),
                'length': len(traj_obs),
                'total_reward': sum(traj_rewards)  # <--- 【就是新增这一行】
            })

        all_returns = torch.cat([t['returns'] for t in group_trajectories])

        # 计算全局的均值和标准差
        returns_mean = all_returns.mean()
        returns_std = all_returns.std(correction=0) + 1e-8

        # 给每一个具体的动作，分配它应得的标准化 Advantage
        for t in group_trajectories:
            t['advantages'] = (t['returns'] - returns_mean) / returns_std

        # --- 3. PPO 更新循环 (支持 LSTM) ---
        loss_val = 0
        for _ in range(self.cfg.update_epochs):

            # 【⭐修改点 1】：改用 batch_loss_sum 记录所有 step 的 Loss 总和，以及总有效步数
            batch_loss_sum = 0.0
            total_steps = 0

            # LSTM 的特殊之处：我们不能把所有 step 打散混在一起 (Shuffle)
            # 必须保持 Trajectory 的顺序，因为 LSTM 需要上一步的 hidden state
            # 所以这里我们遍历每一条 Trajectory
            for traj in group_trajectories:
                # 增加 Batch 维度[Seq_Len, Dim] -> [1, Seq_Len, Dim]
                obs_seq = traj['obs'].unsqueeze(0)
                actions = traj['actions'].unsqueeze(0)
                advantages = traj['advantages'].unsqueeze(0)
                old_log_probs = traj['old_log_probs'].unsqueeze(0)

                # Re-evaluate (重新前向传播)
                # 传入整个序列，LSTM 会自动 unroll
                # hidden_state=None 表示从头开始，这与 Rollout 时一致
                logits_seq, _ = self.policy(obs_seq, hidden_state=None)

                # 计算 Log Prob
                dist = Categorical(logits=logits_seq)
                curr_log_probs = dist.log_prob(actions)  # [1, Seq_Len]

                # 【⭐修改点 2】：去掉这里的 .mean()，保留原本的形状 [1, Seq_Len]
                entropy_seq = dist.entropy()

                # 计算 KL 散度 (近似)
                with torch.no_grad():
                    ref_logits, _ = self.ref_policy(obs_seq, hidden_state=None)
                    ref_dist = Categorical(logits=ref_logits)

                # 计算精确的 KL 散度 [1, Seq_Len]
                kl_div = kl_divergence(ref_dist, dist)

                # 🌟 之前的 GRPO / PPO Hard Clipping 逻辑 (会扼杀罕见量子捷径)
                # =========================================================
                # ratio = torch.exp(curr_log_probs - old_log_probs)
                # surr1 = ratio * advantages
                # surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advantages
                # traj_loss = (-torch.min(surr1, surr2) + ...).mean()

                ratio = torch.exp(curr_log_probs - old_log_probs)

                eps = self.cfg.clip_epsilon
                upper_bound = 1.0 + eps
                lower_bound = 1.0 - eps
                decay_alpha = 0.2  # 软衰减系数 (建议 0.1 ~ 0.3)

                # =========================================================
                # 🚀 组合拳：上方软裁剪 (Decay)，下方硬裁剪 (Hard Clip)
                # =========================================================

                # 1. 处理好动作 (advantages > 0) 的比值：
                # 只有当 ratio 冲破 upper_bound 时，才进行 log 软衰减；否则保持原 ratio！
                ratio_up = torch.where(
                    ratio > upper_bound,
                    upper_bound + decay_alpha * torch.log(1.0 + ratio - upper_bound),
                    ratio
                )

                # 2. 处理坏动作 (advantages <= 0) 的比值：
                # 跌破 lower_bound 时，强行截断在 lower_bound (标准 PPO 限制坏动作过度打压的逻辑)
                # 因为 advantages < 0，对乘积取 min(r*A, clip*A) 等价于对比值取 max(r, 1-eps)
                ratio_down = torch.max(ratio, torch.tensor(lower_bound, device=ratio.device))

                # 3. 动态融合：根据优势的正负号，选择使用哪一套比值
                effective_ratio = torch.where(advantages > 0, ratio_up, ratio_down)

                # 4. 计算最终的 Surrogate Loss 矩阵
                surr_soft = effective_ratio * advantages
                # =========================================================

                # 5. 综合 Loss 计算 (KL + Entropy)
                traj_loss = (-surr_soft
                             + self.cfg.beta_kl * kl_div
                             - self.cfg.entropy_coef * entropy_seq).sum()

                batch_loss_sum += traj_loss
                total_steps += traj['length']

                # 这里除以 total_steps 才是真正完美的 Step-Level Average
            avg_loss = batch_loss_sum / total_steps

            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # 梯度裁剪
            self.optimizer.step()

            loss_val = avg_loss.item()

        # 同步参考模型 (硬更新)
        self.ref_policy.load_state_dict(self.policy.state_dict())

        # 返回 Loss 和 平均 Reward
        return loss_val, np.mean(all_rewards)

    def train_step_vec_old0415_v2(self, vec_envs):
        import numpy as np
        import torch
        from torch.nn.utils.rnn import pad_sequence
        from torch.distributions import Categorical
        from torch.distributions.kl import kl_divergence
        import random
        from gymnasium.spaces import flatten

        self.policy.train()
        num_envs = vec_envs.num_envs
        single_obs_space = vec_envs.single_observation_space

        # =======================================================
        # 🚀 极其坚固的展平函数 (完美兼容 One-Hot 编码与任何动态维度)
        # =======================================================
        def get_flat_obs(obs_batch, idx):
            if isinstance(obs_batch, dict):
                # 从批量字典中抽取第 idx 个环境的数据重组
                single_o = {k: v[idx] for k, v in obs_batch.items()}
                return flatten(single_obs_space, single_o)
            elif isinstance(obs_batch, tuple):
                single_o = tuple(v[idx] for v in obs_batch)
                return flatten(single_obs_space, single_o)
            else:
                return flatten(single_obs_space, obs_batch[idx])

        # 1. 多进程环境 Reset
        common_seed = random.randint(0, 9999999)
        obs_dict, _ = vec_envs.reset(seed=common_seed)

        # 动态获取当前所有环境的观测态
        current_obs = np.array([get_flat_obs(obs_dict, i) for i in range(num_envs)])
        trajectories = [{"obs": [], "actions": [], "rewards": [], "log_probs": []} for _ in range(num_envs)]

        # 初始化 LSTM 隐藏状态
        h0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        c0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        hidden = (h0, c0)

        active_mask = [True] * num_envs
        last_actions = [None] * num_envs
        step_count = 0

        while any(active_mask):
            step_count += 1

            # GPU 一次性算出所有环境的动作
            obs_tensor = torch.FloatTensor(current_obs).to(self.cfg.device).unsqueeze(1)
            with torch.no_grad():
                logits, next_hidden = self.policy(obs_tensor, hidden)
                logits = logits.squeeze(1)

                for i in range(num_envs):
                    if last_actions[i] is not None and active_mask[i]:
                        logits[i, last_actions[i]] -= 5.0

                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_cpu = actions.cpu().numpy()
            log_probs_cpu = log_probs.cpu().tolist()

            safe_actions = [a if active else 0 for a, active in zip(actions_cpu, active_mask)]

            # 🚀 真正的多进程推演！C库底层火力全开，不再卡死！
            next_obs_dict, rewards, terminations, truncations, _ = vec_envs.step(safe_actions)

            # 动态获取下一步的所有观测态
            next_obs_batch = np.array([get_flat_obs(next_obs_dict, i) for i in range(num_envs)])

            for i in range(num_envs):
                if not active_mask[i]:
                    continue

                done = terminations[i] or truncations[i]

                trajectories[i]["obs"].append(current_obs[i].copy())
                trajectories[i]["actions"].append(actions_cpu[i])
                trajectories[i]["rewards"].append(rewards[i])
                trajectories[i]["log_probs"].append(log_probs_cpu[i])

                last_actions[i] = actions_cpu[i]

                if done:
                    active_mask[i] = False
                else:
                    current_obs[i] = next_obs_batch[i]

            hidden = next_hidden
            print(f"  >>>[多进程并行推演] 第 {step_count} 步 | 存活线路: {sum(active_mask)}/{num_envs}    ", end='\r',
                  flush=True)
        print()

        group_trajectories = []
        all_total_rewards = []

        for i in range(num_envs):
            traj = trajectories[i]
            if len(traj["obs"]) == 0:
                continue

            total_reward = sum(traj["rewards"])
            all_total_rewards.append(total_reward)
            returns = self.compute_discounted_returns(traj["rewards"])

            group_trajectories.append({
                'obs': torch.FloatTensor(np.array(traj["obs"])).to(self.cfg.device),
                'actions': torch.LongTensor(traj["actions"]).to(self.cfg.device),
                'returns': torch.FloatTensor(returns).to(self.cfg.device),
                'old_log_probs': torch.FloatTensor(traj["log_probs"]).to(self.cfg.device),
                'length': len(traj["obs"]),
            })

        if len(group_trajectories) == 0:
            return 0.0, 0.0

        all_returns = torch.cat([t['returns'] for t in group_trajectories])
        returns_mean = all_returns.mean()
        returns_std = all_returns.std(correction=0) + 1e-8

        for t in group_trajectories:
            t['advantages'] = (t['returns'] - returns_mean) / returns_std

        obs_batch = pad_sequence([t['obs'] for t in group_trajectories], batch_first=True)
        actions_batch = pad_sequence([t['actions'] for t in group_trajectories], batch_first=True)
        old_lps_batch = pad_sequence([t['old_log_probs'] for t in group_trajectories], batch_first=True)
        advs_batch = pad_sequence([t['advantages'] for t in group_trajectories], batch_first=True)

        lengths = torch.LongTensor([t['length'] for t in group_trajectories]).to(self.cfg.device)
        max_len = obs_batch.size(1)
        mask = torch.arange(max_len, device=self.cfg.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)

        loss_val = 0
        for _ in range(self.cfg.update_epochs):
            logits_seq, _ = self.policy(obs_batch, hidden_state=None)

            dist = Categorical(logits=logits_seq)
            curr_log_probs = dist.log_prob(actions_batch)
            entropy = dist.entropy()

            with torch.no_grad():
                ref_logits, _ = self.ref_policy(obs_batch, hidden_state=None)
                ref_dist = Categorical(logits=ref_logits)
            kl_div = kl_divergence(ref_dist, dist)

            ratio = torch.exp(curr_log_probs - old_lps_batch)
            surr1 = ratio * advs_batch
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advs_batch

            step_loss = -torch.min(surr1, surr2) \
                        + self.cfg.beta_kl * kl_div \
                        - self.cfg.entropy_coef * entropy

            valid_loss = step_loss * mask
            avg_loss = valid_loss.sum() / mask.sum()

            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            loss_val = avg_loss.item()

        self.ref_policy.load_state_dict(self.policy.state_dict())
        return loss_val, np.mean(all_total_rewards)

    def train_step_vec(self, vec_envs):

        self.policy.train()
        num_envs = vec_envs.num_envs
        single_obs_space = vec_envs.single_observation_space

        # =======================================================
        # 1. 极其坚固的展平函数 (完美兼容 One-Hot 编码与任何动态维度)
        # =======================================================
        def get_flat_obs(obs_batch, idx):
            if isinstance(obs_batch, dict):
                single_o = {k: v[idx] for k, v in obs_batch.items()}
                return flatten(single_obs_space, single_o)
            elif isinstance(obs_batch, tuple):
                single_o = tuple(v[idx] for v in obs_batch)
                return flatten(single_obs_space, single_o)
            else:
                return flatten(single_obs_space, obs_batch[idx])

        # =======================================================
        # 2. 多进程环境 Reset (生成统一 common_seed 起跑线)
        # =======================================================
        common_seed = random.randint(0, 9999999)
        obs_dict, _ = vec_envs.reset(seed=common_seed)

        current_obs = np.array([get_flat_obs(obs_dict, i) for i in range(num_envs)])
        trajectories = [{"obs": [], "actions": [], "rewards": [], "log_probs": [], "action_masks": []} for _ in range(num_envs)]

        # 初始化 LSTM 隐藏状态
        h0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        c0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
        hidden = (h0, c0)

        active_mask = [True] * num_envs
        last_actions = [None] * num_envs
        step_count = 0
        iteration_success_count = 0

        def get_vec_action_masks():
            try:
                raw_masks = vec_envs.call("get_action_mask")
            except Exception:
                raw_masks = [np.ones(self.action_dim, dtype=np.float32) for _ in range(num_envs)]

            masks_np = np.asarray(raw_masks, dtype=np.float32)
            if masks_np.ndim == 1:
                masks_np = np.repeat(masks_np[None, :], num_envs, axis=0)
            if masks_np.shape != (num_envs, self.action_dim):
                raise ValueError(f"Action mask shape {masks_np.shape} != {(num_envs, self.action_dim)}")

            masks_np = masks_np > 0.5
            for i, active in enumerate(active_mask):
                if not active:
                    masks_np[i, :] = True
            return masks_np

        # =======================================================
        # 3. 核心推演循环 (全矩阵运算 + 物理多进程并发)
        # =======================================================
        while any(active_mask):
            step_count += 1

            # GPU 一次性算出所有环境的动作
            obs_tensor = torch.FloatTensor(current_obs).to(self.cfg.device).unsqueeze(1)
            with torch.no_grad():
                logits, next_hidden = self.policy(obs_tensor, hidden)
                logits = logits.squeeze(1)

                # 防止原位打转惩罚机制
                for i in range(num_envs):
                    if last_actions[i] is not None and active_mask[i]:
                        logits[i, last_actions[i]] -= 5.0

                action_masks_np = get_vec_action_masks()
                action_masks_tensor = torch.as_tensor(action_masks_np, dtype=torch.bool, device=self.cfg.device)
                logits = logits.masked_fill(~action_masks_tensor, -1e9)

                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_cpu = actions.cpu().numpy()
            log_probs_cpu = log_probs.cpu().tolist()

            # 对于已经跑完的线路，发一个占位动作 0 给多进程环境
            safe_actions = [a if active else 0 for a, active in zip(actions_cpu, active_mask)]

            # 🚀【绝对核心】一键通知后台所有子进程并行执行物理模拟！
            next_obs_dict, rewards, terminations, truncations, infos = vec_envs.step(safe_actions)

            # --- 统计成功率 ---
            if isinstance(infos, (list, tuple)):
                for info in infos:
                    if isinstance(info, dict):
                        if info.get('goal_achieved', False):
                            iteration_success_count += 1
                        elif 'final_info' in info and isinstance(info['final_info'], dict):
                            if info['final_info'].get('goal_achieved', False):
                                iteration_success_count += 1
            elif isinstance(infos, dict) and "final_info" in infos:
                for final_info in infos["final_info"]:
                    if final_info and isinstance(final_info, dict) and final_info.get("goal_achieved", False):
                        iteration_success_count += 1

            # 动态获取下一步的所有观测态
            next_obs_batch = np.array([get_flat_obs(next_obs_dict, i) for i in range(num_envs)])

            # 处理收集到的数据
            for i in range(num_envs):
                if not active_mask[i]:
                    continue

                done = terminations[i] or truncations[i]

                trajectories[i]["obs"].append(current_obs[i].copy())
                trajectories[i]["actions"].append(actions_cpu[i])
                trajectories[i]["rewards"].append(rewards[i])
                trajectories[i]["log_probs"].append(log_probs_cpu[i])
                trajectories[i]["action_masks"].append(action_masks_np[i].copy())

                last_actions[i] = actions_cpu[i]

                if done:
                    active_mask[i] = False
                else:
                    current_obs[i] = next_obs_batch[i]

            hidden = next_hidden
            print(f"  >>>[多进程并行推演] 第 {step_count} 步 | 存活线路: {sum(active_mask)}/{num_envs}    ", end='\r',
                  flush=True)
        print()

        if iteration_success_count > 0:
            print(f"  🌟 物理突破！本轮成功纯化线路数: {iteration_success_count}/{num_envs} 🌟")

        # =======================================================
        # 4. 组装数据 & 单步折扣奖励计算 (Step-Level Discounted Returns)
        # =======================================================
        group_trajectories = []
        all_total_rewards = []

        for i in range(num_envs):
            traj = trajectories[i]
            if len(traj["obs"]) == 0: continue

            # 无折扣的总物理得分（用于 logging 打印）
            total_reward = sum(traj["rewards"])
            all_total_rewards.append(total_reward)

            # 计算带 Gamma 的折扣回报数组 G_t = r_t + γ * G_{t+1}
            returns = self.compute_discounted_returns(traj["rewards"])

            group_trajectories.append({
                'obs': torch.FloatTensor(np.array(traj["obs"])).to(self.cfg.device),
                'actions': torch.LongTensor(traj["actions"]).to(self.cfg.device),
                'returns': torch.FloatTensor(returns).to(self.cfg.device),
                'old_log_probs': torch.FloatTensor(traj["log_probs"]).to(self.cfg.device),
                'action_masks': torch.BoolTensor(np.array(traj["action_masks"])).to(self.cfg.device),
                'length': len(traj["obs"]),
            })

        if len(group_trajectories) == 0: return 0.0, 0.0

        # =======================================================
        # 🌟 5. 带 Gamma 的纯正 GRPO 优势计算 (Trajectory-Level)
        # =======================================================
        # 核心物理意义：G_0 (returns[0]) 代表了站在起跑线看待整条线路的“带折扣总身价”
        group_G0 = torch.FloatTensor([t['returns'][0] for t in group_trajectories]).to(self.cfg.device)

        # 在这 32 个并行环境构成的 Group 中求均值和标准差 (内卷机制)
        G0_mean = group_G0.mean()
        G0_std = group_G0.std(correction=0) + 1e-8

        # 计算每条线路的全局相对优势 (标量)
        group_advantages = (group_G0 - G0_mean) / G0_std

        # 广播优势：将这条线路的全局优势，像刷漆一样覆盖给里面的每一个 Step
        for i, t in enumerate(group_trajectories):
            t['advantages'] = torch.full_like(t['returns'], group_advantages[i])

        # =======================================================
        # 6. GRPO 更新循环 (硬裁剪 + 全局有效步求平均)
        # =======================================================
        obs_batch = pad_sequence([t['obs'] for t in group_trajectories], batch_first=True)
        actions_batch = pad_sequence([t['actions'] for t in group_trajectories], batch_first=True)
        old_lps_batch = pad_sequence([t['old_log_probs'] for t in group_trajectories], batch_first=True)
        advs_batch = pad_sequence([t['advantages'] for t in group_trajectories], batch_first=True)
        action_masks_batch = pad_sequence(
            [t['action_masks'] for t in group_trajectories],
            batch_first=True,
            padding_value=True
        ).bool()

        lengths = torch.LongTensor([t['length'] for t in group_trajectories]).to(self.cfg.device)
        max_len = obs_batch.size(1)

        # 生成 Mask 掩码，过滤掉 Padding 的 0
        mask = torch.arange(max_len, device=self.cfg.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)

        loss_val = 0
        for _ in range(self.cfg.update_epochs):
            logits_seq, _ = self.policy(obs_batch, hidden_state=None)
            logits_seq = logits_seq.masked_fill(~action_masks_batch, -1e9)

            dist = Categorical(logits=logits_seq)
            curr_log_probs = dist.log_prob(actions_batch)
            entropy = dist.entropy()

            with torch.no_grad():
                ref_logits, _ = self.ref_policy(obs_batch, hidden_state=None)
                ref_logits = ref_logits.masked_fill(~action_masks_batch, -1e9)
                ref_dist = Categorical(logits=ref_logits)
            kl_div = kl_divergence(ref_dist, dist)

            # GRPO / PPO Hard Clipping 核心
            ratio = torch.exp(curr_log_probs - old_lps_batch)
            surr1 = ratio * advs_batch
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advs_batch

            # 计算所有步的 Loss 矩阵
            step_loss = -torch.min(surr1, surr2) \
                        + self.cfg.beta_kl * kl_div \
                        - self.cfg.entropy_coef * entropy

            # 用 mask 过滤无效数据，并严格按照真实 Step 数求平均
            valid_loss = step_loss * mask
            avg_loss = valid_loss.sum() / mask.sum()

            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            loss_val = avg_loss.item()

        # 同步参考模型
        self.ref_policy.load_state_dict(self.policy.state_dict())

        # 返回 Loss 和所有线路的真实平均保真度/得分
        return loss_val, np.mean(all_total_rewards)

    def save_checkpoint(self, path):
        """保存模型权重"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 如果需要，也可以保存 config
            'config': self.cfg
        }, path)
        # print(f"Model saved to {path}")

    def load_checkpoint(self, path):
        """加载模型权重"""
        if not os.path.exists(path):
            print(f"Error: Checkpoint {path} not found!")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.cfg.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except RuntimeError as exc:
            print(f"Warning: checkpoint {path} is incompatible with the current observation/action space.")
            print(f"         {exc}")
            return False

        # 同步更新参考模型
        self.ref_policy.load_state_dict(self.policy.state_dict())
        print(f"Loaded checkpoint from {path}")
        return True


def get_6bit_target_state():
    # 1. 初始化 64 维全零复数向量 (2^6 = 64)
    target_64d = np.zeros(64, dtype=complex)

    # 2. 计算非零分量的二进制索引
    # 物理位顺序: q0 q1 q2 q3 q4 q5
    # 对应权重:  32 16  8  4  2  1

    # 分量 1: |00>|00>|00> -> 索引 0
    idx1 = 0

    # 分量 2: |00>|00>|11> -> 索引 2^1 + 2^0 = 3
    idx2 = 3

    # 分量 3: |11>|00>|00> -> 索引 2^5 + 2^4 = 32 + 16 = 48
    idx3 = 48

    # 分量 4: |11>|00>|11> -> 索引 32 + 16 + 2 + 1 = 51
    idx4 = 51

    # 3. 赋值系数
    # 系数 = (1/sqrt(2)) * 1 * (1/sqrt(2)) = 0.5
    val = 0.5 + 0.j
    target_64d[idx1] = val
    target_64d[idx2] = val
    target_64d[idx3] = val
    target_64d[idx4] = val

    return target_64d


def train_recurrent_grpo(env_factory, config: GRPOConfig, agent=None):
    """
    LSTM 版 GRPO 专用训练循环 (极致提速版)
    """
    # 1. 初始化环境与 Agent (如果没传 agent)
    if agent is None:
        temp_env = env_factory()
        try:
            obs_dim = flatten_space(temp_env.unwrapped.observation_space).shape[0]
        except:
            obs_dim = temp_env.observation_space.shape[0]

        action_dim = temp_env.action_space.n
        temp_env.close()

        print(f"Initializing Recurrent GRPO Agent (Obs: {obs_dim}, Act: {action_dim})...")
        agent = RecurrentGRPOAgent(obs_dim, action_dim, config)

    # 2. 准备保存目录
    save_dir = "saved_models/recurrent_grpo"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pt")

    print(f"Start Training on {config.device} | Group Size: {config.group_size}")

    # ================= 【核心提速魔法】 =================
    # 在 2500 轮循环开始前，一口气建好所有的环境！
    # 这些环境会作为“传家宝”一直复用，彻底消灭初始化带来的卡顿。
    print(f"🚀 正在初始化 {config.group_size} 个并行量子环境，这可能需要几十秒，请耐心等待...")
    t0 = time.time()

    # active_envs = [env_factory() for _ in range(config.group_size)]
    def create_single_env(idx):
        return env_factory()

        # 使用多线程瞬间铺开 256 个环境的创建任务
        # max_workers 可以设为你的 CPU 核心数，比如 16 或 32

    with concurrent.futures.ThreadPoolExecutor(max_workers=44) as executor:
        active_envs = list(executor.map(create_single_env, range(config.group_size)))

    print(f"✅ 环境池初始化完成！共耗时: {time.time() - t0:.2f} 秒")
    # print(f"✅ 环境池初始化完成！共耗时: {time.time() - t0:.2f} 秒")
    # ===================================================

    best_reward = -float('inf')

    # 3. 训练迭代
    for iteration in range(config.num_iterations):
        iter_start_time = time.time()

        # ⚠️ 核心修改：直接把建好的环境池列表 (active_envs) 传给 agent！
        loss, avg_reward = agent.train_step(active_envs)

        iter_time = time.time() - iter_start_time

        # 4. 日志与保存逻辑
        print(f"Iter {iteration + 1}/{config.num_iterations} | "
              f"Avg Reward: {avg_reward:.4f} | Loss: {loss:.4f} | 本轮耗时: {iter_time:.2f}s")

        # 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save_checkpoint(best_model_path)
            print(f"   >>> New Best Reward! Saved to {best_model_path}")

        # 定期保存 (例如每 50 轮)
        if (iteration + 1) % 50 == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_iter_{iteration + 1}.pt")
            agent.save_checkpoint(ckpt_path)
            print(f"   --- Checkpoint saved to {ckpt_path}")

    print("Training Finished.")

    # 5. 训练彻底结束后，安全打扫战场
    print("清理环境内存中...")
    for env in active_envs:
        env.close()


def train_step_vec_260415(self, vec_envs):
    self.policy.train()
    num_envs = vec_envs.num_envs
    single_obs_space = vec_envs.single_observation_space

    # =======================================================
    # 1. 多进程环境 Reset (生成统一 common_seed 起跑线)
    # =======================================================
    common_seed = random.randint(0, 9999999)
    obs_dict, _ = vec_envs.reset(seed=common_seed)

    # 提取单个环境 flat_obs 的辅助函数
    def get_flat_obs(obs_batch, idx):
        if isinstance(obs_batch, dict):
            single_o = {k: v[idx] for k, v in obs_batch.items()}
            return flatten(single_obs_space, single_o)
        else:
            return flatten(single_obs_space, obs_batch[idx])

    current_obs = np.array([get_flat_obs(obs_dict, i) for i in range(num_envs)])
    trajectories = [{"obs": [], "actions": [], "rewards": [], "log_probs": []} for _ in range(num_envs)]

    # =======================================================
    # 2. LSTM 批量隐藏状态初始化
    # =======================================================
    h0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
    c0 = torch.zeros(1, num_envs, self.cfg.hidden_size).to(self.cfg.device)
    hidden = (h0, c0)

    active_mask = [True] * num_envs
    last_actions = [None] * num_envs
    step_count = 0

    # =======================================================
    # 3. 核心推演循环 (全矩阵运算 + 物理多进程并发)
    # =======================================================
    while any(active_mask):
        step_count += 1

        # GPU 一次性算出 256 个环境的动作
        obs_tensor = torch.FloatTensor(current_obs).to(self.cfg.device).unsqueeze(1)
        with torch.no_grad():
            logits, next_hidden = self.policy(obs_tensor, hidden)
            logits = logits.squeeze(1)

            # 防止原位打转惩罚机制
            for i in range(num_envs):
                if last_actions[i] is not None and active_mask[i]:
                    logits[i, last_actions[i]] -= 5.0

            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        actions_cpu = actions.cpu().numpy()
        log_probs_cpu = log_probs.cpu().tolist()

        # 对于已经跑完的线路，发一个占位动作 0 给多进程环境
        safe_actions = [a if active else 0 for a, active in zip(actions_cpu, active_mask)]

        # 🚀【绝对核心】一键通知后台所有子进程并行执行物理模拟！
        next_obs_dict, rewards, terminations, truncations, _ = vec_envs.step(safe_actions)

        # 处理收集到的数据
        for i in range(num_envs):
            if not active_mask[i]:
                continue  # 这条线路上一步已经结束了，忽略新数据

            done = terminations[i] or truncations[i]

            trajectories[i]["obs"].append(current_obs[i])
            trajectories[i]["actions"].append(actions_cpu[i])
            trajectories[i]["rewards"].append(rewards[i])
            trajectories[i]["log_probs"].append(log_probs_cpu[i])

            last_actions[i] = actions_cpu[i]

            if done:
                active_mask[i] = False
            else:
                # 更新下一步观测
                current_obs[i] = get_flat_obs(next_obs_dict, i)

        hidden = next_hidden
        print(f"  >>>[多进程并行推演] 第 {step_count} 步 | 存活线路: {sum(active_mask)}/{num_envs}    ", end='\r')
    print()

    # =======================================================
    # 4. 组装数据 & 单步折扣奖励优势计算 (Step-Level Discounted Returns)
    # =======================================================
    group_trajectories = []
    all_total_rewards = []
    for i in range(num_envs):
        traj = trajectories[i]
        if len(traj["obs"]) == 0: continue

        total_reward = sum(traj["rewards"])
        all_total_rewards.append(total_reward)

        # 使用经典的折扣回报计算方法: G_t = r_t + γ * G_{t+1}
        returns = self.compute_discounted_returns(traj["rewards"])

        group_trajectories.append({
            'obs': torch.FloatTensor(np.array(traj["obs"])).to(self.cfg.device),
            'actions': torch.LongTensor(traj["actions"]).to(self.cfg.device),
            'returns': torch.FloatTensor(returns).to(self.cfg.device),
            'old_log_probs': torch.FloatTensor(traj["log_probs"]).to(self.cfg.device),
            'length': len(traj["obs"]),
        })

    if len(group_trajectories) == 0: return 0.0, 0.0

    # 🌟 恢复你原本的计算全局均值、标准差，并分配给每一步的逻辑
    all_returns = torch.cat([t['returns'] for t in group_trajectories])
    returns_mean = all_returns.mean()
    returns_std = all_returns.std(correction=0) + 1e-8

    for t in group_trajectories:
        t['advantages'] = (t['returns'] - returns_mean) / returns_std

    # =======================================================
    # 5. 原版 PPO 更新循环 (硬裁剪 + 全局有效步求平均)
    # =======================================================
    obs_batch = pad_sequence([t['obs'] for t in group_trajectories], batch_first=True)
    actions_batch = pad_sequence([t['actions'] for t in group_trajectories], batch_first=True)
    old_lps_batch = pad_sequence([t['old_log_probs'] for t in group_trajectories], batch_first=True)
    advs_batch = pad_sequence([t['advantages'] for t in group_trajectories], batch_first=True)

    lengths = torch.LongTensor([t['length'] for t in group_trajectories]).to(self.cfg.device)
    max_len = obs_batch.size(1)
    # 生成 Mask 掩码，过滤掉 Padding 的 0
    mask = torch.arange(max_len, device=self.cfg.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    loss_val = 0
    for _ in range(self.cfg.update_epochs):
        logits_seq, _ = self.policy(obs_batch, hidden_state=None)

        dist = Categorical(logits=logits_seq)
        curr_log_probs = dist.log_prob(actions_batch)
        entropy = dist.entropy()

        with torch.no_grad():
            ref_logits, _ = self.ref_policy(obs_batch, hidden_state=None)
            ref_dist = Categorical(logits=ref_logits)
        kl_div = kl_divergence(ref_dist, dist)

        # 🌟 恢复经典原版的 Hard Clipping (硬截断)
        ratio = torch.exp(curr_log_probs - old_lps_batch)
        surr1 = ratio * advs_batch
        surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advs_batch

        # 计算所有步的 Loss 矩阵
        step_loss = -torch.min(surr1, surr2) \
                    + self.cfg.beta_kl * kl_div \
                    - self.cfg.entropy_coef * entropy

        # 用 mask 过滤无效数据，并严格按照真实 Step 数求平均
        valid_loss = step_loss * mask
        avg_loss = valid_loss.sum() / mask.sum()

        self.optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        loss_val = avg_loss.item()

    self.ref_policy.load_state_dict(self.policy.state_dict())

    # 返回 Loss 和所有线路的真实平均保真度/得分
    return loss_val, np.mean(all_total_rewards)


def continue_train():
    LOAD_PATH = "saved_models/recurrent_grpo/best_model.pt"  # 或者 checkpoint_iter_X.pt
    HIDDEN_SIZE = 256
    temp_env = env_factory()
    try:
        obs_dim = flatten_space(temp_env.observation_space).shape[0]
    except:
        obs_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    temp_env.close()

    print(f"Initializing Agent for Resume Training...")
    # 3. 手动实例化 Agent
    agent = RecurrentGRPOAgent(obs_dim, action_dim, cfg)

    # 4. 【核心步骤】加载旧权重
    if os.path.exists(LOAD_PATH):
        print(f"Loading checkpoint from {LOAD_PATH}...")
        if not agent.load_checkpoint(LOAD_PATH):
            print("Checkpoint is incompatible with the current observation/action space. Starting from scratch.")
        # 注意：load_checkpoint 也会加载优化器状态(Adam momentum)，
        # 这意味着它会完美地接上之前的梯度轨迹，而不是冷启动。
    else:
        print(f"⚠️ Warning: Checkpoint {LOAD_PATH} not found! Starting from scratch.")

    # 5. 传入 Agent 开始训练
    # train_recurrent_grpo 检测到 agent 不为 None，就会跳过初始化，直接用你传入的这个
    train_recurrent_grpo(env_factory, cfg, agent=agent)


def load_and_expand_policy(agent, checkpoint_path):
    """
    智能加载权重：支持动作空间扩张 (Network Surgery)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=agent.cfg.device)
    state_dict = checkpoint['policy_state_dict']

    # 1. 获取新旧维度
    # 假设 action_head 是最后一个 Linear 层
    # state_dict['action_head.weight'] 的 shape 是 [New_Action_Dim, Hidden_Size]
    old_weight = state_dict['action_head.weight']
    old_bias = state_dict['action_head.bias']

    new_action_dim = agent.policy.action_head.out_features
    old_action_dim = old_weight.shape[0]

    print(f"Auto-detected dimensions: Old={old_action_dim}, New={new_action_dim}")

    # 2. 判断是否需要手术
    if old_action_dim == new_action_dim:
        # 维度一致，直接加载
        agent.policy.load_state_dict(state_dict)
        print("Dimensions match. Loaded directly.")

    elif old_action_dim < new_action_dim:
        # === 手术开始 ===
        print("Dimensions mismatch (Expansion). Performing surgery...")

        # A. 加载除了 action_head 以外的所有层 (LSTM, Feature Extractor)
        # 过滤掉 action_head 的键
        partial_state_dict = {k: v for k, v in state_dict.items() if 'action_head' not in k}
        agent.policy.load_state_dict(partial_state_dict, strict=False)

        # B. 手动移植 output layer 的权重
        # 取出当前新网络的权重 (随机初始化的)
        new_weight = agent.policy.action_head.weight.data
        new_bias = agent.policy.action_head.bias.data

        # 将旧权重覆盖到对应位置 (假设旧动作在索引 0 ~ old_dim-1)
        # 这样 Agent 就“记得”旧动作的效果
        new_weight[:old_action_dim, :] = old_weight
        new_bias[:old_action_dim] = old_bias

        # 新动作 (索引 old_dim ~ end) 保持随机初始化，等待学习

        # C. 强制赋值回去
        agent.policy.action_head.weight.data = new_weight
        agent.policy.action_head.bias.data = new_bias

        print(f"  - Transplanted {old_action_dim} old actions.")
        print(f"  - Initialized {new_action_dim - old_action_dim} new actions randomly.")

    else:
        raise ValueError("Error: Old model has MORE actions than new model. Cannot shrink automatically.")

    # 3. 同步 Reference Model
    agent.ref_policy.load_state_dict(agent.policy.state_dict())

    # 4. 重置优化器 (因为参数形状变了，旧优化器状态通常无法复用，建议重置)
    # 如果你想保留 Feature Extractor 的动量，比较复杂。
    # 对于微调来说，新建一个优化器通常效果更好。
    agent.optimizer = torch.optim.Adam(agent.policy.parameters(), lr=agent.cfg.lr)
    print("✅ Pretrained weights loaded successfully.")


def get_ghz_state(n_qubits: int = 3) -> np.ndarray:
    target = np.zeros(2 ** n_qubits, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    return target


def ex_train():
    def env_factory_phase2():
        target = get_6bit_target_state()
        env_name = 'BasicNQubit-v0'
        fidelity_threshold = 0.95
        reward_penalty = 0.01
        max_timesteps = 20
        seed = 42
        env = gym.make(
            env_name,
            target=target,
            fidelity_threshold=fidelity_threshold,
            reward_penalty=reward_penalty,
            max_timesteps=max_timesteps, )
        # 可以在这里加 seed，或者让 reset 处理
        # 这里定义包含【所有动作】的环境
        # 比如 include_catalyst_gates=True)
        return env

    # if __name__ == "__main__":
    # 1. 准备新环境的维度
    temp_env = env_factory_phase2()
    # 此时 Action Dim 应该是 17 (14旧 + 3新)
    new_obs_dim = flatten_space(temp_env.unwrapped.observation_space).shape[0]
    new_action_dim = temp_env.unwrapped.action_space.n
    temp_env.close()

    # 2. 初始化全维度 Agent
    cfg1 = GRPOConfig(
        group_size=16,
        lr=1e-5,  # 微调时学习率调低！别把旧知识洗没了
        entropy_coef=0.05
    )
    agent = RecurrentGRPOAgent(new_obs_dim, new_action_dim, cfg1)

    # 3. 【核心】加载旧模型并扩张
    # 指向你 Phase 1 训练好的模型
    PHASE1_MODEL = "saved_models/recurrent_grpo/best_model.pt"
    load_and_expand_policy(agent, PHASE1_MODEL)

    # 4. 开始训练
    # 此时 Agent 已经会做基础纯化了，它会慢慢尝试那几个新加的随机动作
    train_recurrent_grpo(env_factory_phase2, cfg1, agent=agent)


def get_ancilla_zero_state(n_qubits: int = 3) -> np.ndarray:
    """生成全 0 的辅助比特态 |00...0>"""
    ancilla = np.zeros(2 ** n_qubits, dtype=complex)
    ancilla[0] = 1.0 + 0.j  # 只有 |000> 分量为 1
    return ancilla


def generate_ghz4_state(F: float):
    """
    生成精确保真度为 F 的 4-qubit 带噪 GHZ 态 (Werner State)。
    维度: 16 x 16
    """
    # import numpy as np
    dim = 16

    # 物理学换算：保证最终测量的保真度严格等于 F
    # 根据 F = p * 1 + (1-p) * (1/d)，解出 p
    p = (16.0 * F - 1.0) / 15.0
    p = max(0.0, min(1.0, p))  # 防止传入的 F 太低导致 p 为负数

    # 构造理想的 4-qubit GHZ 态: (|0000> + |1111>) / sqrt(2)
    # |0000> 的索引是 0，|1111> 的索引是 15 (即二进制 1111)
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2)
    psi[15] = 1.0 / np.sqrt(2)
    rho_ideal = np.outer(psi, psi.conj())

    # 完全混合态: I / 16
    rho_mixed = np.eye(dim, dtype=complex) / dim

    # 组装最终密度矩阵
    rho_noisy = p * rho_ideal + (1.0 - p) * rho_mixed

    return rho_noisy


def get_ideal_9q_target_state():
    """
    生成完美的 9-qubit 目标密度矩阵 (512 x 512)。
    拓扑: Target(0,1,2,3) ⊗ Resource(4,5,6,7) ⊗ Ancilla(8)
    """
    # 1. 完美的 4-qubit Target GHZ 态
    rho_ideal_target = generate_ghz4_state(F=1.0)

    # 2. 完美的 4-qubit Resource GHZ 态
    rho_ideal_resource = generate_ghz4_state(F=1.0)

    # 3. 完美的辅助比特 |0><0|
    psi_anc = np.array([1.0, 0.0], dtype=complex)
    rho_anc = np.outer(psi_anc, psi_anc.conj())

    # 4. 张量积拼接 (Kron 乘积顺序必须严格遵守比特索引 0->8)
    rho_8q = np.kron(rho_ideal_target, rho_ideal_resource)
    rho_9q_ideal = np.kron(rho_8q, rho_anc)

    return rho_9q_ideal


def generate_ghz4_state_util(F: float) -> np.ndarray:
    """生成精确保真度为 F 的 4-qubit 带噪 GHZ 态 (16x16)."""
    dim = 16
    p = (16.0 * F - 1.0) / 15.0
    p = max(0.0, min(1.0, p))
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2)
    psi[15] = 1.0 / np.sqrt(2)
    rho_ideal = np.outer(psi, psi.conj())
    rho_noise = np.eye(dim, dtype=complex) / dim
    return p * rho_ideal + (1 - p) * rho_noise


def get_ideal_10q_target_state() -> np.ndarray:
    """
    生成完美的 10-qubit 目标密度矩阵 (1024 x 1024).
    拓扑: Target(0,1,2,3) x Resource(4,5,6,7) x Ancilla Z-check(8) x Ancilla X-check(9)
    """
    rho_ideal_target = generate_ghz4_state_util(F=1.0)
    rho_ideal_resource = generate_ghz4_state_util(F=1.0)
    # 双辅助比特 |00>
    psi_anc = np.zeros(4, dtype=complex)
    psi_anc[0] = 1.0
    rho_anc = np.outer(psi_anc, psi_anc.conj())
    rho_8q = np.kron(rho_ideal_target, rho_ideal_resource)
    return np.kron(rho_8q, rho_anc)



if __name__ == "__main__":
    from gymnasium.spaces import flatten_space
    import gymnasium as gym
    import time

    cfg = GRPOConfig(
        group_size=32,  # 32
        lr=2e-4,  # 2e-4 5e-5 1e-4
        entropy_coef=0.06,
        gamma=0.97,
    )

    GHZ6bit = np.kron(get_ghz_state(), get_ghz_state())
    ancilla_3bit = get_ancilla_zero_state(3)
    target_9bit = np.kron(GHZ6bit, ancilla_3bit)
    # target = target_9bit
    target = get_ideal_10q_target_state()

    env_name = 'BasicNQubit-v0'  # BasicNQubit NoisyNQubit
    fidelity_threshold = 0.95
    reward_penalty = 0.01
    max_timesteps = 9
    fixed_topology = "1|1|2"
    ancilla_count = 2
    apply_topology_conversion_noise = False
    error_rate = 0.0
    error_single = 0.01
    error_multi = 0.01


    # ================= 改造点 1：闭包工厂函数 =================
    # gym.vector.AsyncVectorEnv 需要一个返回 env 的无参函数
    def make_env_fn():
        def _init():
            return gym.make(
                env_name,
                target=target,
                fidelity_threshold=fidelity_threshold,
                reward_penalty=reward_penalty,
                max_timesteps=max_timesteps,
                fixed_topology=fixed_topology,
                ancilla_count=ancilla_count,
                apply_topology_conversion_noise=apply_topology_conversion_noise,
                error_rate=error_rate,
                error_single=error_single,
                error_multi=error_multi,
            )

        return _init


    print("🚀 正在启动真实多进程量子模拟器集群...")
    t0 = time.time()

    # ================= 改造点 2：启动真·多进程器 =================
    # AsyncVectorEnv 会在后台默默拉起 group_size (64) 个物理 CPU 进程
    vec_envs = gym.vector.AsyncVectorEnv([make_env_fn() for _ in range(cfg.group_size)])

    print(f"✅ 多进程集群就绪！耗时: {time.time() - t0:.2f} 秒")

    # 获取维度并初始化 Agent
    obs_dim = flatten_space(vec_envs.single_observation_space).shape[0]
    action_dim = vec_envs.single_action_space.n
    agent = RecurrentGRPOAgent(obs_dim, action_dim, cfg)

    # =======================================================
    # 🌟 改造点：加载历史存档进行断点续训
    # =======================================================
    save_dir = "saved_models/recurrent_grpo"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir,
                                   "checkpoint_iter_50_0708_REP4.pt")  # best_model_3CNOT_P10086 checkpoint_iter_500_PRE_10BIT checkpoint_iter_800_PRE_9BIT_0519_NONE

    # 定义你要加载的存档路径（可以是 best_model.pt，也可以是某个具体的 checkpoint_iter_X.pt）
    LOAD_PATH = best_model_path

    if os.path.exists(LOAD_PATH):
        print(f"🔄 发现历史存档！正在从 {LOAD_PATH} 恢复网络权重与优化器状态...")
        if agent.load_checkpoint(LOAD_PATH):
            print("✅ 恢复成功！无缝接轨之前的物理推演。")
        else:
            print("⚠️ 存档与当前 observation/action space 不兼容，Agent 将从零开始随机探索。")
    else:
        print(f"⚠️ 未找到存档文件 {LOAD_PATH}，Agent 将从零开始随机探索。")

    # ================= 改造点 3：调用向量化训练函数 =================
    best_reward = -float('inf')
    save_dir = "saved_models/recurrent_grpo"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pt")

    print(f"🚀 开始训练！模型将保存在: {save_dir}")

    # 2. 初始化最高分记录
    best_reward = -float('inf')

    # 3. 开启训练大循环
    for iteration in range(cfg.num_iterations):
        t_start = time.time()

        # 调用多进程极速步进
        loss, avg_reward = agent.train_step_vec(vec_envs)

        print(
            f"Iter {iteration + 1}/{cfg.num_iterations} | Avg Reward: {avg_reward:.4f} | Loss: {loss:.4f} | 耗时: {time.time() - t_start:.2f}s")

        # =======================================================
        # 🌟 核心：记录并保存最佳模型
        # =======================================================
        if avg_reward > best_reward:
            best_reward = avg_reward
            # 调用你写好的 agent.save_checkpoint
            agent.save_checkpoint(best_model_path)
            print(f"  >>> 🏆 突破历史最高分! New Best Reward: {best_reward:.4f} | 模型已保存！")

        # =======================================================
        # 可选：定期保存检查点 (防止意外断电，比如每 50 轮存一个备份)
        # =======================================================
        if (iteration + 1) % 50 == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_iter_{iteration + 1}.pt")
            agent.save_checkpoint(ckpt_path)
            print(f"  >>> 💾 自动备份完成: {ckpt_path}")

    # 训练结束，安全关闭后台的 32 个物理进程
    vec_envs.close()
    print("✅ 训练完美结束！")

