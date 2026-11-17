import sys
from contextlib import closing
from io import StringIO
from typing import Dict, List, Optional, Union, Tuple

from collections import deque
import cirq
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
import os
from cirq.devices import InsertionNoiseModel
import scipy

from qas_gym.utils import get_default_gates, get_default_gates_13_two_ancillas, get_bell_state
from qas_gym.utils import FGate, CXRVGate
import warnings


class QuantumArchSearchEnv(gym.Env):
    metadata = {'render_modes': ['ansi', 'human']}

    def __init__(
            self,
            target: np.ndarray,
            qubits: List[cirq.LineQubit],
            state_observables: List[cirq.GateOperation],
            action_gates: List[cirq.GateOperation],
            fidelity_threshold: float,
            reward_penalty: float,
            max_timesteps: int,
            fixed_topology: Optional[str] = None,
            apply_topology_conversion_noise: bool = True,
            error_observables: Optional[float] = None,
            error_gates: Optional[float] = None,
            seed: Optional[int] = None,  # 添加seed参数
            error_single=None, error_multi=None,  # 噪声
            ancilla_count: int = 1,
    ):
        super(QuantumArchSearchEnv, self).__init__()

        # set parameters
        self.used_cnot = False
        self.target = target
        self.qubits = qubits
        self.qubit_index = {qubit: idx for idx, qubit in enumerate(qubits)}
        self.state_observables = state_observables
        self.action_gates = action_gates
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps
        self.error_observables = error_observables
        self.error_gates = error_gates
        # 设置种子
        self.seed(seed)  # 在这里调用seed方法
        # 窗口设置
        self.min_gate_count = max_timesteps

        self.state = None
        self.window_size = 6  # 设置时间窗口大小
        self.operation_history = deque(maxlen=self.window_size)  # 记录操作历史
        self.previous_action = None
        self.penalty_weight = 1.  # 惩罚权重
        self.state_history = None
        self.previous_state = None
        self.previous_fidelity = None
        self.previous_entanglement = None
        self.reward = 0

        self.error_single = error_single  # 噪声添加
        self.error_multi = error_multi
        self.ancilla_count = int(ancilla_count)
        if self.ancilla_count not in (1, 2):
            raise ValueError("ancilla_count must be either 1 or 2.")

        # self.simulator = cirq.Simulator()
        self.simulator = cirq.DensityMatrixSimulator()

        #self.s_tot_norm_tar, self.s_a_norm_tar, self.s_b_norm_tar = self._get_renyi_target()
        #self.s_tot_norm_tar, self.s_a_norm_tar, self.s_b_norm_tar = self._get_renyi_target9bit()
        init_rho_tar = self._generate_ghz4_state(F=1)
        self.s_tot_norm_tar, self.s_a_norm_tar, self.s_b_norm_tar = self._get_renyi(init_rho_tar)

        self.finish_action_map = {
            "FINISH_MASK_ANC": (None, "ANC"),

            "FINISH_P1_MASK_RES": ("P1", "RES"),
            "FINISH_P1_MASK_BOTH": ("P1", "BOTH"),

            "FINISH_P2_MASK_RES": ("P2", "RES"),
            "FINISH_P2_MASK_BOTH": ("P2", "BOTH"),
        }

        self.topology_action_map = {
            "SELECT_TOPO_13": "1|3",
            "SELECT_TOPO_112": "1|1|2",
            "SELECT_TOPO_1111": "1|1|1|1",
            "SELECT_TOPO_22": "2|2",
        }
        self.topology_names = ["1|3", "1|1|2", "1|1|1|1", "2|2"]
        self.topology_groups = {
            "1|3": [{0}, {1, 2, 3}],
            "1|1|2": [{0}, {1}, {2, 3}],
            "1|1|1|1": [{0}, {1}, {2}, {3}],
            "2|2": [{0, 1}, {2, 3}],
        }
        self.topology_conversion_cost = {
            "1|3": 2.0,
            "1|1|2": 1.0,
            "1|1|1|1": 0.0,
            "2|2": 2.0,
        }
        self.topology_conversion_penalty_weight = 0.0
        self.current_topology = None
        self.topology_selected = False
        self.fixed_topology = fixed_topology
        if self.fixed_topology == "":
            self.fixed_topology = None
        if self.fixed_topology is not None and self.fixed_topology not in self.topology_names:
            raise ValueError(
                f"Unknown fixed_topology={self.fixed_topology!r}; "
                f"expected one of {self.topology_names}"
            )
        if self.ancilla_count == 2:
            if len(self.qubits) != 10:
                raise ValueError("Two ancillas require a 10-qubit target state.")
            if self.fixed_topology != "1|3":
                raise ValueError(
                    "The two-ancilla mode is defined only for fixed_topology='1|3'."
                )
        self.apply_topology_conversion_noise = bool(apply_topology_conversion_noise)
        self.source_logical = 0
        self.topology_total_noise_min = 0.1
        self.topology_total_noise_max = 0.3
        self.topology_initial_noise_min = 0.1
        self.topology_initial_to_conversion_min_ratio = 2.5
        self.topology_total_noise_p = 0.0
        self.topology_initial_noise_by_logical = {}
        self.topology_conversion_noise_by_logical = {}
        self.topology_noise_report = {}

        if self.ancilla_count == 2:
            self.action_gates = get_default_gates_13_two_ancillas(qubits)
        else:
            self.action_gates = get_default_gates(qubits)


        # set spaces
        self.observation_space = spaces.Dict({
            "expectations": spaces.Box(low=-1., high=1., shape=(len(state_observables),), dtype=np.float32),
            "gate_count": spaces.Discrete(self.max_timesteps + 1),
            # "entanglement": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32),
            # "entanglement_t": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32),
            "entanglement_a": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32),
            "topology": spaces.Box(low=0., high=1., shape=(len(self.topology_names),), dtype=np.float32),
            "noise": spaces.Box(low=0., high=1., shape=(4,), dtype=np.float32),
            # "entanglement_b": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32),
            # 🚀 新增：三个辅助比特的测量反馈 (q6, q7, q8)
            #"syndromes": spaces.Box(low=0., high=1., shape=(3,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(n=len(self.action_gates))
        # 混合动作空间定义
        # self.action_space = spaces.Tuple([
        #    spaces.Discrete(len(self.action_gates)),  # 离散动作（门类型选择）
        #    spaces.Box(low=-np.pi, high=np.pi, shape=(1,))  # 连续动作（角度参数）
        # ])

        # 之后为TD3DM空间
        # self.action_gates, self.num_fixed_actions = get_default_gates(self.qubits)

        # 离散部分：固定门个数 + 1（新增参数化门）
        # discrete_space = spaces.Discrete(self.num_fixed_actions + 1)
        # 连续部分：如果选择参数化门时使用的参数（比如旋转角度），范围 [-π, π]
        # continuous_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)

        # 定义混合动作空间
        # self.action_space = spaces.Tuple((discrete_space, continuous_space))
        # --- 修改动作空间定义 ---
        # 离散部分：固定门个数 + 1（新增参数化门）
        # discrete_space = spaces.Discrete(self.num_fixed_actions + 1)
        # ___________________________________到此为止

        # 连续部分：如果选择参数化门时使用的参数（比如旋转角度），范围 [-π, π]
        # 注意：参数维度可能不止 1，取决于你的参数化门，这里假设是 1
        # parameter_action_dim = 1  # 或者根据你的实际情况设定
        # continuous_space = spaces.Box(low=-np.pi, high=np.pi, shape=(parameter_action_dim,), dtype=np.float32)

        # 将 Tuple 改为 Dict，并使用 'discrete' 和 'continuous' 作为键
        # self.action_space = spaces.Dict({
        #    "discrete": discrete_space,
        #    "continuous": continuous_space
        # })

        self.seed()
        self.render_mode = None

        # self.rho_coherent = self.get_general_coherent_state(
        #     weights=[0.7, 0.1, 0.12, 0.08],
        #     coherences={(1, 3): 0.15}  # Phi- 和 Psi- 之间的实数相干
        # )

    def __str__(self):
        desc = 'QuantumArchSearch-v0('
        desc += '{}={}, '.format('Qubits', len(self.qubits))
        desc += '{}={}, '.format('Target', self.target)
        desc += '{}=[{}], '.format(
            'Gates', ', '.join(gate.__str__() for gate in self.action_gates))
        desc += '{}=[{}])'.format(
            'Observables',
            ', '.join(gate.__str__() for gate in self.state_observables))
        return desc

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        #os.environ['PYTHONHASHSEED'] = str(seed)  # config for python scripts
        return [seed]

    def _topology_one_hot(self):
        one_hot = np.zeros(len(self.topology_names), dtype=np.float32)
        if self.current_topology in self.topology_names:
            one_hot[self.topology_names.index(self.current_topology)] = 1.0
        return one_hot

    def _noise_observation(self):
        initial_noise = getattr(self, "topology_initial_noise_by_logical", {}) or {}
        noise = np.array(
            [
                float(getattr(self, "topology_total_noise_p", 0.0)),
                float(initial_noise.get(1, 0.0)),
                float(initial_noise.get(2, 0.0)),
                float(initial_noise.get(3, 0.0)),
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(np.clip(noise, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)

    def _select_topology(self, action_gate):
        self.current_topology = self.topology_action_map[action_gate]
        self.topology_selected = True
        self._sample_topology_noise_schedule()
        self._apply_topology_initial_noise()

    def _select_topology_by_name(self, topology_name):
        if topology_name not in self.topology_names:
            raise ValueError(f"Unknown topology_name={topology_name!r}")
        self.current_topology = topology_name
        self.topology_selected = True
        self._sample_topology_noise_schedule()
        self._apply_topology_initial_noise()

    def _finalize_reset_observation(self):
        if self.fixed_topology is not None:
            self._select_topology_by_name(self.fixed_topology)
        return self._get_obs(), {}

    def _reset_13_two_ancilla_state(self):
        """Prepare GHZ4 target/resource copies and two clean remote ancillas."""
        f_param = self.np_random.uniform(0.95, 1.0)
        rho_target = self._generate_ghz4_state(F=f_param)
        rho_resource = self._generate_ghz4_state(F=f_param)
        ancilla_state = np.zeros(4, dtype=complex)
        ancilla_state[0] = 1.0
        rho_ancillas = np.outer(ancilla_state, ancilla_state.conj())

        self.rho = np.kron(np.kron(rho_target, rho_resource), rho_ancillas)
        self.rho_initial = self.rho
        self.rho_4q = rho_target
        self.F_in = self._get_ghz4_fidelity(rho_target)
        self.en_in = self._get_logarithmic_negativity_4q(rho_target)
        self.F_temp = self.F_in
        return self._finalize_reset_observation()

    def _compose_depolarizing_p(self, p_first, p_second):
        return p_first + p_second - (4.0 / 3.0) * p_first * p_second

    def _remaining_depolarizing_p(self, total_p, used_p):
        if used_p <= 0:
            return total_p
        if used_p >= total_p:
            return 0.0
        denom = max(1e-12, 1.0 - (4.0 / 3.0) * used_p)
        return float(np.clip((total_p - used_p) / denom, 0.0, 1.0))

    def _minimum_initial_noise_for_p1_p2_ratio(self, total_p, ratio):
        total_p = max(0.0, float(total_p))
        ratio = max(0.0, float(ratio))
        if total_p <= 0 or ratio <= 0:
            return 0.0
        depol_cross = 4.0 / 3.0
        discriminant = max(
            0.0,
            (ratio + 1.0) ** 2 - 4.0 * depol_cross * ratio * total_p,
        )
        lower = ((ratio + 1.0) - np.sqrt(discriminant)) / (2.0 * depol_cross)
        return float(np.clip(lower, 0.0, total_p))

    def _sample_initial_noise_with_ratio_bound(self, total_p):
        total_p = max(0.0, float(total_p))
        if total_p <= 0:
            return 0.0
        lower = max(
            0.25 * total_p,
            self.topology_initial_noise_min,
            self._minimum_initial_noise_for_p1_p2_ratio(
                total_p,
                self.topology_initial_to_conversion_min_ratio,
            ),
        )
        upper = 0.75 * total_p
        if lower > upper:
            upper = total_p
        lower = min(lower, upper)
        return float(self.np_random.uniform(lower, upper))

    def _sample_topology_noise_schedule(self):
        total_p = float(getattr(self, "topology_total_noise_p", 0.0))
        initial_by_logical = {logical: 0.0 for logical in range(4)}
        conversion_by_logical = {logical: 0.0 for logical in range(4)}

        for group in self.topology_groups[self.current_topology]:
            group = set(group)
            non_source_logicals = [logical for logical in group if logical != self.source_logical]

            if not non_source_logicals:
                continue
            if self.current_topology == "1|1|1|1":
                group_initial_p = total_p
            elif self.current_topology == "1|1|2" and group == {1}:
                group_initial_p = total_p
            else:
                group_initial_p = self._sample_initial_noise_with_ratio_bound(total_p)

            for logical in non_source_logicals:
                initial_by_logical[logical] = group_initial_p

        for logical in range(4):
            if logical == self.source_logical:
                continue
            if self.apply_topology_conversion_noise:
                conversion_by_logical[logical] = self._remaining_depolarizing_p(
                    total_p,
                    initial_by_logical[logical],
                )

        self.topology_initial_noise_by_logical = initial_by_logical
        self.topology_conversion_noise_by_logical = conversion_by_logical
        effective_by_logical = {
            logical: self._compose_depolarizing_p(
                initial_by_logical[logical],
                conversion_by_logical[logical],
            )
            for logical in range(4)
        }
        p1_ge_p2_by_logical = {
            logical: initial_by_logical[logical] + 1e-12 >= conversion_by_logical[logical]
            for logical in range(4)
        }
        p1_ge_ratio_p2_by_logical = {
            logical: (
                initial_by_logical[logical] + 1e-12
                >= self.topology_initial_to_conversion_min_ratio * conversion_by_logical[logical]
            )
            for logical in range(4)
        }
        self.topology_noise_report = {
            "total_noise_p": total_p,
            "initial_to_conversion_min_ratio": self.topology_initial_to_conversion_min_ratio,
            "apply_topology_conversion_noise": self.apply_topology_conversion_noise,
            "initial_noise_by_logical": dict(initial_by_logical),
            "conversion_noise_by_logical": dict(conversion_by_logical),
            "effective_noise_by_logical": effective_by_logical,
            "p1_ge_p2_by_logical": p1_ge_p2_by_logical,
            "p1_ge_ratio_p2_by_logical": p1_ge_ratio_p2_by_logical,
        }

    def _apply_depolarizing_noise_to_rho(self, rho, num_qubits, noise_by_index):
        rho = self._clean_density_matrix(rho)
        if rho is None:
            return None

        circuit = cirq.Circuit()
        qs = cirq.LineQubit.range(num_qubits)
        for index, p_noise in noise_by_index.items():
            if p_noise and p_noise > 0:
                circuit.append(cirq.depolarize(float(p_noise)).on(qs[index]))
        if not circuit:
            return rho
        res = self.simulator.simulate(circuit, initial_state=rho, qubit_order=qs)
        return self._clean_density_matrix(res.final_density_matrix)

    def _apply_topology_initial_noise(self):
        noise_by_index = {}
        for logical, p_noise in self.topology_initial_noise_by_logical.items():
            if p_noise and p_noise > 0:
                noise_by_index[logical] = p_noise
                noise_by_index[logical + 4] = p_noise
        num_qubits = len(self.qubits)
        cleaned = self._apply_depolarizing_noise_to_rho(self.rho, num_qubits, noise_by_index)
        if cleaned is not None:
            self.rho = cleaned
            self.rho_initial = self.rho
            self.rho_4q = self.manual_partial_trace(
                self.rho,
                keep_indices=[0, 1, 2, 3],
                dims=[2] * num_qubits,
            )
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)
            self.F_temp = self.F_in

    def _apply_final_conversion_noise(self, target_rho):
        if not self.apply_topology_conversion_noise:
            return target_rho
        noise_by_index = {
            logical: p_noise
            for logical, p_noise in self.topology_conversion_noise_by_logical.items()
            if p_noise and p_noise > 0
        }
        cleaned = self._apply_depolarizing_noise_to_rho(target_rho, 4, noise_by_index)
        return cleaned if cleaned is not None else target_rho

    def _ancilla_logicals_for_topology(self):
        if self.current_topology is None:
            return set()
        groups = self.topology_groups[self.current_topology]
        max_size = max(len(group) for group in groups)
        largest_groups = [group for group in groups if len(group) == max_size]
        return set(largest_groups[-1])

    def _same_topology_lab(self, logical_a, logical_b):
        if self.current_topology is None:
            return False
        for group in self.topology_groups[self.current_topology]:
            if logical_a in group and logical_b in group:
                return True
        return False

    def _is_two_qubit_op_allowed_by_topology(self, op):
        indices = [self.qubit_index.get(qubit) for qubit in op.qubits]
        if any(index is None for index in indices):
            return False

        ancilla_indices = [index for index in indices if index >= 8]
        data_indices = [index for index in indices if 0 <= index < 8]

        if ancilla_indices:
            if len(ancilla_indices) != 1 or not data_indices:
                return False
            ancilla_lab = self._ancilla_logicals_for_topology()
            return all((index % 4) in ancilla_lab for index in data_indices)

        logicals = [index % 4 for index in data_indices]
        return all(
            self._same_topology_lab(logicals[i], logicals[j])
            for i in range(len(logicals))
            for j in range(i + 1, len(logicals))
        )

    def _is_action_allowed_by_topology(self, action_gate):
        if isinstance(action_gate, str):
            return True

        if isinstance(action_gate, cirq.Circuit):
            ops = list(action_gate.all_operations())
        else:
            ops = [action_gate]

        for op in ops:
            if len(op.qubits) <= 1:
                continue
            if not self._is_two_qubit_op_allowed_by_topology(op):
                return False
        return True

    def get_action_mask(self):
        mask = np.zeros(len(self.action_gates), dtype=np.float32)

        for index, action_gate in enumerate(self.action_gates):
            if isinstance(action_gate, str) and action_gate in self.topology_action_map:
                mask[index] = 1.0 if not self.topology_selected else 0.0
                continue

            if not self.topology_selected:
                mask[index] = 0.0
                continue

            if isinstance(action_gate, str) and action_gate.startswith("FINISH"):
                mask[index] = 1.0 if self.step_counter > 0 and self.used_cnot else 0.0
                continue

            mask[index] = 1.0 if self._is_action_allowed_by_topology(action_gate) else 0.0

        if not np.any(mask):
            mask[:] = 1.0
        return mask

    def action_masks(self):
        return self.get_action_mask()

    def _topology_conversion_penalty(self):
        if self.current_topology is None:
            return 0.0
        cost = self.topology_conversion_cost.get(self.current_topology, 0.0)
        return self.topology_conversion_penalty_weight * cost

    def reset_qas(self, seed=None, options=None):
        super().reset(seed=seed)
        self.circuit_gates = []
        self.state = None
        self.state_history = None
        self.operation_history.clear()  # 清空操作历史
        self.previous_action = None  # 重置上一个动作
        self.previous_fidelity = None
        self.previous_entanglement = None
        # self.circuit_gates.append(cirq.X(self.qubits[2]))#for w-state
        self.circuit_gates.append(cirq.H(self.qubits[0]))
        self.circuit_gates.append(cirq.CNOT(self.qubits[0], self.qubits[1]))
        self.circuit_gates.append(cirq.CNOT(self.qubits[1], self.qubits[2]))
        self.circuit_gates.append(cirq.CNOT(self.qubits[2], self.qubits[3]))
        # self.circuit_gates.append(cirq.ry(np.pi/2.)(self.qubits[0]))
        # self.circuit_gates.append(cirq.Y(self.qubits[1]))
        # self.circuit_gates.append(cirq.H(self.qubits[2]))
        # self.circuit_gates.append(cirq.CNOT(self.qubits[0],self.qubits[1]))
        # self.circuit_gates.append(FGate(n=3, k=2)(self.qubits[1], self.qubits[2]))
        # self.circuit_gates.append(cirq.CNOT(self.qubits[2], self.qubits[0]))

        return self._get_obs(), {}

    # 在 qas_env.py 的 reset 中修改
    #def reset_purify(self, seed=None, options=None):
        #super().reset(seed=seed)
        #self.circuit_gates = []
        #self.step_counter = 0
        #self.used_cnot = False

        # --- 1. 定义初始非对称噪声态 (DEJMPS 专用) ---
        #f_in = 0.73
        #l11, l01, l10 = 0.15, 0.06, 0.06  # 非对称分布：相位错误多

        # 生成单对非对称贝尔对角态
        #rho_single = self.get_bell_diagonal_state(f_in, l11, l01, l10)

        # 构造 4-qubit 初始总态：rho ⊗ rho
        # 这是 Agent 每一轮纯化时面对的“原材料”
        #self.rho_initial = np.kron(rho_single, rho_single)

        # --- 2. 获取初始观察值 ---
        # 注意：现在的状态是混合态，不能用原来的 simulate
        #return self._get_obs(), {}

    """def reset_bell(self, seed=None, options=None):
        super().reset(seed=seed)
        self.circuit_gates = []
        self.step_counter = 0
        self.used_cnot = False
        self.prob_sum = 1.

        f_noisy = 0.73
        #p_cata = 0.6  # 催化剂系数
        p_cata = (np.cos(np.pi / 8)) ** 2

        # --- 关键：同时生成 向量 和 矩阵 ---
        # 向量用于计算 vdot (cata_f)，矩阵用于演化 (rho_initial)
        psi_cata = np.zeros(4, dtype=complex)
        psi_cata[0] = np.sqrt(p_cata)
        psi_cata[3] = np.sqrt(1 - p_cata)

        #self.rho_catalyst_initial_vec = psi_cata  # <--- 这行解决了你的 AttributeError
        #self.rho_catalyst_initial = np.outer(psi_cata, psi_cata.conj())

        self.rho_catalyst_initial = self.rho_coherent#self.get_bell_diagonal_state(f_noisy, 0.15, 0.06, 0.06)

        # 目标对和资源对
        #rho_target = self.get_bell_diagonal_state(f_noisy, 0.15, 0.06, 0.06)
        #rho_res = self.get_bell_diagonal_state(f_noisy, 0.15, 0.06, 0.06)
        #rho_target = self.get_dejmps_limit_state()
        #rho_res = self.get_dejmps_limit_state()
        rho_target = self.rho_coherent
        rho_res =  self.rho_coherent

        # 组合成 6-bit 初始态
        self.rho_initial = np.kron(np.kron(rho_target, rho_res), self.rho_catalyst_initial)
        #self.rho_initial = self._clean_density_matrix(self.rho_initial)
        return self._get_obs(), {}"""



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #reward = 0
        #self.last_syndromes = [0.0, 0.0, 0.0]
        self.circuit_gates = []
        self.step_counter = 0
        self.used_cnot = False
        self.prob_sum = 1.
        #self.F_in = None
        self.flage_F = False
        self.goal_achieved=False
        self.is_entangling = False  # 建议将变量名改为更通用的“是否包含纠缠门/多比特门”
        self.current_topology = None
        self.topology_selected = False
        self.topology_total_noise_p = float(self.np_random.uniform(
            self.topology_total_noise_min,
            self.topology_total_noise_max,
        ))
        self.topology_initial_noise_by_logical = {}
        self.topology_conversion_noise_by_logical = {}
        self.topology_noise_report = {"total_noise_p": self.topology_total_noise_p}

        if self.ancilla_count == 2:
            return self._reset_13_two_ancilla_state()

        # === 核心修改：状态随机化 ===
        rand_scenario = self.np_random.random()

        if rand_scenario < 0:
            # 【模式 1: 标准 Werner 态】 -> 训练 P1 (CNOT)
            # 模拟“刚领到的脏原材料”
            f_param = self.np_random.uniform(0.55, 0.95)
            self.rho_3q = self._generate_ghz_state(F=f_param)

        elif rand_scenario < 0:
            # 【模式 2: 严重相位错误态】 -> 训练 P2 (H + CNOT + H)
            f_param = self.np_random.uniform(0.70, 0.85)
            rho_init = self._generate_ghz_state(F=f_param)

            # 2. 模拟一次真实的 P1 纯化过程
            # (使用你内核里现成的逻辑，但不计入步数和奖励)
            rho_input_6q = np.kron(rho_init, rho_init)

            # 构建 P1 电路 (CNOTs)
            q = self.qubits
            p1_circuit = cirq.Circuit(cirq.CNOT(q[0], q[3]), cirq.CNOT(q[1], q[4]), cirq.CNOT(q[2], q[5]))

            # 演化并后选择
            sim = cirq.DensityMatrixSimulator()
            result = sim.simulate(p1_circuit, initial_state=rho_input_6q, qubit_order=self.qubits)
            rho_post, p_succ = self._post_select_ghz_ancilla(result.final_density_matrix)

            # 3. 提取系统态作为这一局的“初态”
            if p_succ > 1e-4:
                self.rho_3q = self.manual_partial_trace(rho_post, keep_indices=[0, 1, 2], dims=[2] * 6)
            else:
                # 防止极端情况失败，回退到 Werner
                self.rho_3q = self._generate_ghz_state(F=f_param)
        elif rand_scenario < 0:  # 3-qubit 异构网络裸考版 (9-qubit 空间：3Q+3Q+3Anc)

            # =========================================================
            # 1. 初始化 9-qubit 完美总态 (512 x 512)
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)  # 初始纯度

            # 生成 3-qubit 初始态 (8 x 8 密度矩阵)
            # ⚠️ 确保你的环境类中有 _generate_ghz3_state 方法
            rho_sys_base = self._generate_ghz_state(F=f_param)
            rho_res_base = self._generate_ghz_state(F=f_param)

            # 张量积拼接出 6-qubit 核心态 (64 x 64)
            rho_6q_input = np.kron(rho_sys_base, rho_res_base)

            # 🚀 准备 3 个干净的辅助比特 (q6, q7 为 Alice 侧; q8 为 Bob 侧)
            # 3个辅助比特的初始态为 |000> (8 x 8 密度矩阵)
            psi_anc = np.zeros(8, dtype=complex)
            psi_anc[0] = 1.0
            rho_anc = np.outer(psi_anc, psi_anc.conj())

            # 当前总态：9-qubit 密度矩阵 (512 x 512)
            current_9q_state = np.kron(rho_6q_input, rho_anc)

            # =========================================================
            # 🚀 2. 异构信道加噪 (Heterogeneous Noise Injection)
            # 物理拓扑: Target(0,1,2), Resource(3,4,5)
            # Alice 节点拥有: 0, 1, 3, 4 (外加探针 6, 7)
            # Bob   节点拥有: 2, 5 (外加探针 8)
            # =========================================================
            qc_channel_noise = cirq.Circuit()

            # --- Bob 侧长途光纤信道噪声 (q2, q5) ---
            # 遭受极其严重的去极化噪声
            noise_p_bob = self.np_random.uniform(0, 0)
            self.niose_bob = noise_p_bob
            for q_idx in [2, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            # --- Alice 侧本地轻微退相干噪声 (q1, q4) ---
            # 模拟节点内部的不完美存储与局域门噪声
            noise_p_alice = self.np_random.uniform(0.05, 0.5)
            self.niose_alice = noise_p_alice
            for q_idx in [1, 4]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_alice).on(self.qubits[q_idx]))

            # 注：q0 和 q3 作为绝对基准位，假定其受到完美的本地保护，不加额外噪声

            # =========================================================
            # 3. 演化初始信道加噪 (最纯粹的起点)
            # =========================================================
            sim = cirq.DensityMatrixSimulator()
            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])
            rho_after_noise = self._clean_density_matrix(res_noise.final_density_matrix)

            # =========================================================
            # 4. 直接移交 RL Agent (终极裸考模式)
            # =========================================================
            # 覆写环境真实初始状态为加噪后的原始状态
            self.rho = rho_after_noise
            self.rho_initial = self.rho

            # 提取 3-qubit Target 的保真度和纠缠度基线
            # ⚠️ 确保环境类中包含 _get_ghz3_fidelity 和 _get_logarithmic_negativity_3q 方法
            self.rho_3q = self.manual_partial_trace(self.rho, keep_indices=[0, 1, 2], dims=[2] * 9)
            self.F_in = self._get_ghz_fidelity(self.rho_3q)
            self.en_in = self._get_logarithmic_negativity(self.rho_3q)
        elif rand_scenario < 0:
            f_param = self.np_random.uniform(0.55, 0.95)
            rho_init = self._generate_ghz_state(F=f_param)
            q = self.qubits
            p2_circuit = cirq.Circuit(cirq.CNOT(q[0], q[3]),cirq.CNOT(q[1], q[3]))#cirq.CNOT(q[4], q[5]),
            for i in range(2):
                # 2. 模拟一次真实的 非对称GHZ 纯化过程
                # (使用你内核里现成的逻辑，但不计入步数和奖励)
                if self.flage_F:
                    rho_init = self.rho_3q
                rho_input_6q = np.kron(rho_init, rho_init)

                # 构建 P2 电路 (CNOTs)
                sim = cirq.DensityMatrixSimulator()
                result = sim.simulate(p2_circuit, initial_state=rho_input_6q, qubit_order=self.qubits)
                rho_post, p_succ = self._post_select_ghz_ancilla(result.final_density_matrix)

                # 3. 提取系统态作为这一局的“初态”
                if p_succ > 1e-4:
                    self.rho_3q = self.manual_partial_trace(rho_post, keep_indices=[0, 1, 2], dims=[2] * 6)
                else:
                    # 防止极端情况失败，回退到 Werner
                    self.rho_3q = self._generate_ghz_state(F=f_param)
                self.flage_F = True
        elif rand_scenario < 0:  # 你可以调整概率范围来触发它
            # 1. 生成基础状态 (两对标准的 Werner GHZ)
            f_param = self.np_random.uniform(0.65, 0.95)
            rho_sys_base = self._generate_ghz_state(F=f_param)
            rho_res_base = self._generate_ghz_state(F=f_param)

            # 初始 6-qubit 总态
            self.rho = np.kron(rho_sys_base, rho_res_base)

            # 2. 注入针对 B 方比特的局部噪声
            qc_b_noise = cirq.Circuit()

            # 对 B 方的 Target (q2) 注入噪声
            # 如果 q2 发生了位翻转 (X)，那么 A 侧的协议如果不动 B 方，保真度绝对升不上去
            if self.np_random.random() < 0.6:#0.6
                qc_b_noise.append(cirq.X(self.qubits[2]))

                # 对 B 方的 Resource (q4, q5) 注入噪声 (模拟 B 的资源本身也是坏的)
            for b_res_q in [self.qubits[4], self.qubits[5]]:
                if self.np_random.random() < 0.4:#0.4
                    qc_b_noise.append(cirq.Z(b_res_q))

                    # 3. 运行物理演化，得到注入噪声后的总态
            sim = cirq.DensityMatrixSimulator()
            res = sim.simulate(qc_b_noise, initial_state=self.rho, qubit_order=self.qubits)
            self.rho = self._clean_density_matrix(res.final_density_matrix)
            self.rho_initial = self.rho  # 保存为初始态

            # 4. 【核心步骤】提取 Target 系统的初始保真度
            # 必须追踪掉 q3, q4, q5 (Resource)，保留 q0, q1, q2 (Target)
            # 这样 self.F_in 才能准确反映出 q2 被注入噪声后的惨状
            rho_sys_only = self.manual_partial_trace(self.rho, keep_indices=[0, 1, 2], dims=[2] * 6)
            self.F_in = self._get_ghz_fidelity(rho_sys_only)
        elif rand_scenario < 0:#含噪
            # 1. 生成基础状态：本地制备的高保真 3-qubit GHZ 态
            # 假设 Alice 在本地制备了接近完美的纠缠对
            f_param = self.np_random.uniform(0.95, 1.0)
            rho_sys_base = self._generate_ghz_state(F=f_param)
            rho_res_base = self._generate_ghz_state(F=f_param)

            # 初始 6-qubit 总态
            self.rho = np.kron(rho_sys_base, rho_res_base)

            # 2. 【灵活配置】定义哪些比特通过了噪声信道
            # 你可以根据需要修改这个列表，例如 [2, 5], [2, 4, 5], 甚至 [1, 2, 4, 5]
            channel_qubits_indices = [0,1,3]

            qc_channel_noise = cirq.Circuit()

            # 设置信道噪声强度 p (去极化概率)
            # 参考 Wallnöfer 论文，p 值越大，受损越严重
            noise_p = self.np_random.uniform(0.1, 0.5)

            for q_idx in channel_qubits_indices:
                # 为列表中的每一个比特应用去极化噪声
                # 这模拟了这些比特在传输过程中与环境发生了非受控的相互作用
                qc_channel_noise.append(cirq.depolarize(p=noise_p).on(self.qubits[q_idx]))

            # 3. 运行物理演化
            sim = cirq.DensityMatrixSimulator()
            res = sim.simulate(qc_channel_noise, initial_state=self.rho, qubit_order=self.qubits)

            # 【关键参数：self.rho_initial】
            # 保存注入信道噪声后的密度矩阵作为 RL 搜索的起点
            self.rho = self._clean_density_matrix(res.final_density_matrix)
            self.rho_initial = self.rho

            # 4. 【关键参数：self.F_in】
            # 提取 Target 系统 (q0, q1, q2) 的初始保真度
            # 注意：即便 q4, q5 坏了，self.F_in 也只反映 Target 系统本身受到的直接损伤 (q2)
            rho_sys_only = self.manual_partial_trace(self.rho, keep_indices=[0, 1, 2], dims=[2] * 6)
            self.F_in = self._get_ghz_fidelity(rho_sys_only)

            # 可选：打印一下哪些比特被注入了噪声
            # print(f"Noise injected into qubits: {channel_qubits_indices}, F_in: {self.F_in:.4f}")


        elif rand_scenario < 0:  # 课程学习：人为制造 Z 错误主导态

            # 1. 生成基础 Werner 态

            f_param = self.np_random.uniform(0.55, 0.8)

            rho_current = self._generate_ghz_state(F=f_param)

            q = self.qubits

            p2_circuit = cirq.Circuit(cirq.CNOT(q[0], q[3]), cirq.CNOT(q[1], q[3]))

            #p3_circuit = cirq.Circuit(cirq.CNOT(q[1], q[3]), cirq.CNOT(q[2], q[4]), cirq.CNOT(q[2], q[5]))
            p3_circuit = cirq.Circuit(cirq.CNOT(q[0], q[4]), cirq.CNOT(q[2], q[4]))

            sim = cirq.DensityMatrixSimulator()

            # 2. 模拟真实 P2 纯化过程，人为制造“纯化天花板”

            num_pre_iterations = 1

            for i in range(num_pre_iterations):

                rho_input_6q = np.kron(rho_current, rho_current)

                # 运行 2-CNOT

                result = sim.simulate(p2_circuit, initial_state=rho_input_6q, qubit_order=self.qubits)

                rho_post, p_succ = self._post_select_ghz_ancilla(result.final_density_matrix)

                if p_succ > 1e-4:

                    # 提取系统态，准备进入下一轮

                    rho_current = self.manual_partial_trace(rho_post, keep_indices=[0, 1, 2], dims=[2] * 6)

                else:

                    # 失败则保留上一轮的 rho_current 直接退出循环

                    break

            for i in range(num_pre_iterations):

                rho_input_6q = np.kron(rho_current, rho_current)

                # 运行 2-CNOT

                result = sim.simulate(p3_circuit, initial_state=rho_input_6q, qubit_order=self.qubits)

                rho_post, p_succ = self._post_select_ghz_ancilla(result.final_density_matrix)

                if p_succ > 1e-4:

                    # 提取系统态，准备进入下一轮

                    rho_current = self.manual_partial_trace(rho_post, keep_indices=[0, 1, 2], dims=[2] * 6)

                else:

                    # 失败则保留上一轮的 rho_current 直接退出循环

                    break

            # 3. 【核心修复区】组装 RL 需要的各项关键参数

            self.rho_3q = self._clean_density_matrix(rho_current)

            # RL 环境需要 6-qubit 的总态 (Target ⊗ Resource)！

            self.rho = np.kron(self.rho_3q, self.rho_3q)

            # 保存注入“Z错误”后的 6-qubit 密度矩阵作为 RL 搜索的起点

            self.rho_initial = self.rho

            # 提取 Target 系统 (q0, q1, q2) 的初始保真度 F_in

            self.F_in = self._get_ghz_fidelity(self.rho_3q)

        elif rand_scenario < 0:  # 课程学习：基于真实非对称信道，人为制造 Z 错误主导态

            # --- 1. 搬用情况 0：生成真实信道噪声下的 6-qubit 初态 ---
            f_param = self.np_random.uniform(0.95, 1.0)
            rho_sys_base = self._generate_ghz_state(F=f_param)
            rho_res_base = self._generate_ghz_state(F=f_param)

            # 初始完美的 6-qubit 总态
            rho_6q_input = np.kron(rho_sys_base, rho_res_base)

            channel_qubits_indices = [0, 1, 3]
            qc_channel_noise = cirq.Circuit()

            # 建议将课程学习的噪声上限稍微调低一点(比如0.4)，防止预处理时成功率直接归零
            noise_p = self.np_random.uniform(0.1, 0.4)

            for q_idx in channel_qubits_indices:
                qc_channel_noise.append(cirq.depolarize(p=noise_p).on(self.qubits[q_idx]))

            sim = cirq.DensityMatrixSimulator()
            #res = sim.simulate(qc_channel_noise, initial_state=rho_6q_input, qubit_order=self.qubits)
            res = sim.simulate(qc_channel_noise, initial_state=rho_6q_input, qubit_order=self.qubits[:6])

            # 拿到刚过完光纤、充满所有错误的 6-qubit 输入态
            current_6q_state = self._clean_density_matrix(res.final_density_matrix)

            # --- 2. 模拟真实 P2 纯化过程，过滤 X 错误，逼出 Z 错误天花板 ---
            q = self.qubits
            def _build_noisy_circuit(ops):
                noisy_circuit = cirq.Circuit()
                for op in ops:
                    noisy_circuit.append(op)  # 先加理想门
                    op_qubits = op.qubits  # 获取这个门作用的比特

                    # 注入单比特噪声
                    if len(op_qubits) == 1 and self.error_single is not None and self.error_single > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))
                    # 注入多比特 (双比特) 噪声
                    elif len(op_qubits) > 1 and self.error_multi is not None and self.error_multi > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))
                return noisy_circuit

            # 把原本理想的门写成列表，然后通过组装函数变成带噪声的物理电路
            p2_ops = [cirq.CNOT(q[0], q[3]), cirq.CNOT(q[1], q[3])]
            p2_circuit = _build_noisy_circuit(p2_ops)

            p3_ops = [cirq.H(q[0]), cirq.H(q[1]), cirq.H(q[2]), cirq.H(q[3]), cirq.H(q[4]), cirq.H(q[5]),
                      cirq.CNOT(q[2], q[5]), cirq.CNOT(q[1], q[4]), cirq.CNOT(q[0], q[3]),
                      cirq.H(q[0]), cirq.H(q[1]), cirq.H(q[2])]
            p3_circuit = _build_noisy_circuit(p3_ops)

            p4_ops = [cirq.CNOT(q[0], q[3]), cirq.CNOT(q[1], q[4]), cirq.CNOT(q[2], q[5])]
            p4_circuit = _build_noisy_circuit(p4_ops)

            num_pre_iterations = 1#3  # 跑 1~2 轮就能过滤掉大部分 X/Y 错误了
            num_pre_iterations2 = 1#1
            num_pre_iterations3 = 0#1
            rho_3q_current = None  # 用于保存提纯后的 3-qubit 态
            num_pre_iterations_total = 1#18

            for i in range(num_pre_iterations):
                # 运行 2-CNOT
                #result = sim.simulate(p2_circuit, initial_state=current_6q_state, qubit_order=self.qubits)
                result = sim.simulate(p2_circuit, initial_state=current_6q_state, qubit_order=self.qubits[:6])

                # 【关键点】：显式使用 'P2' 掩码规则 (000/111) 进行后选择
                rho_post, p_succ = self._post_select_ghz_ancilla(result.final_density_matrix)

                if p_succ > 1e-4:
                    # 成功：提取并清洗系统态
                    rho_3q_current = self.manual_partial_trace(rho_post, keep_indices=[0, 1, 2], dims=[2] * 6)
                    rho_3q_current = self._clean_density_matrix(rho_3q_current)

                    # 为下一轮准备 6-qubit 态（如果是最后一轮，这行算完也没关系）
                    current_6q_state = np.kron(rho_3q_current, rho_3q_current)
                else:
                    # 失败：如果运气极差导致成功率极低，直接从当前的 6-qubit 态中提取系统态作为保底
                    rho_3q_current = self.manual_partial_trace(current_6q_state, keep_indices=[0, 1, 2], dims=[2] * 6)
                    rho_3q_current = self._clean_density_matrix(rho_3q_current)
                    break
            for j in range (num_pre_iterations_total):
                for i in range(num_pre_iterations2):
                    #
                    #result = sim.simulate(p3_circuit, initial_state=current_6q_state, qubit_order=self.qubits)
                    result = sim.simulate(p3_circuit, initial_state=current_6q_state, qubit_order=self.qubits[:6])

                    # 【关键点】：显式使用 'P1' 掩码规则进行后选择
                    rho_post, p_succ = self._post_select_ghz_ancilla_even(result.final_density_matrix, mode='P1')

                    if p_succ > 1e-4:
                        # 成功：提取并清洗系统态
                        rho_3q_current = self.manual_partial_trace(rho_post, keep_indices=[0, 1, 2], dims=[2] * 6)
                        rho_3q_current = self._clean_density_matrix(rho_3q_current)

                        # 为下一轮准备 6-qubit 态（如果是最后一轮，这行算完也没关系）
                        current_6q_state = np.kron(rho_3q_current, rho_3q_current)
                    else:
                        # 失败：如果运气极差导致成功率极低，直接从当前的 6-qubit 态中提取系统态作为保底
                        rho_3q_current = self.manual_partial_trace(current_6q_state, keep_indices=[0, 1, 2], dims=[2] * 6)
                        rho_3q_current = self._clean_density_matrix(rho_3q_current)
                        break

                for i in range(num_pre_iterations3):
                    #result = sim.simulate(p4_circuit, initial_state=current_6q_state, qubit_order=self.qubits)
                    result = sim.simulate(p4_circuit, initial_state=current_6q_state, qubit_order=self.qubits[:6])

                    # 【关键点】：显式使用 'P2' 掩码规则进行后选择
                    rho_post, p_succ = self._post_select_ghz_ancilla(result.final_density_matrix)

                    if p_succ > 1e-4:
                        # 成功：提取并清洗系统态
                        rho_3q_current = self.manual_partial_trace(rho_post, keep_indices=[0, 1, 2], dims=[2] * 6)
                        rho_3q_current = self._clean_density_matrix(rho_3q_current)

                        # 为下一轮准备 6-qubit 态（如果是最后一轮，这行算完也没关系）
                        current_6q_state = np.kron(rho_3q_current, rho_3q_current)
                    else:
                        # 失败：如果运气极差导致成功率极低，直接从当前的 6-qubit 态中提取系统态作为保底
                        rho_3q_current = self.manual_partial_trace(current_6q_state, keep_indices=[0, 1, 2], dims=[2] * 6)
                        rho_3q_current = self._clean_density_matrix(rho_3q_current)
                        break


            # --- 3. 【核心组装区】将提纯后的态喂给 RL ---
            # rho_3q_current = self.manual_partial_trace(current_6q_state, keep_indices=[0, 1, 2], dims=[2] * 6)
            # rho_3q_current = self._clean_density_matrix(rho_3q_current)
            self.rho_3q = rho_3q_current

            # 将被 P2 蹂躏过（此时 X 错误被大幅消灭，Z 错误极度富集）的态，组装成 6-qubit
            self.rho = np.kron(self.rho_3q, self.rho_3q)
            self.rho_initial = self.rho
            self.F_in = self._get_ghz_fidelity(self.rho_3q)
            self.en_in = self._get_logarithmic_negativity(self.rho_3q)

            # === 新增：生成 3 个辅助比特的完美初态 |000> ===
            psi_anc = np.zeros(8, dtype=complex)
            psi_anc[0] = 1.0  # 若想初始为 |+++>，可改为 np.ones(8)/np.sqrt(8)
            rho_anc = np.outer(psi_anc, psi_anc.conj())

            # === 组装 9-qubit 总态 ===
            self.rho = np.kron(self.rho, rho_anc)
            self.rho_initial = self.rho

        elif rand_scenario < 0:  # 课程学习：基于真实非对称信道

            # =========================================================
            # 1. 物理拓扑初始化：直接在 9-qubit 空间构建完美初态
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)
            rho_sys_base = self._generate_ghz_state(F=f_param)
            rho_res_base = self._generate_ghz_state(F=f_param)

            # Target(0,1,2) ⊗ Resource(3,4,5)
            rho_6q_input = np.kron(rho_sys_base, rho_res_base)

            # Ancilla(6,7,8) 初始化为 |000>
            psi_anc = np.zeros(8, dtype=complex)
            psi_anc[0] = 1.0
            rho_anc = np.outer(psi_anc, psi_anc.conj())

            # 当前总态：真正的 9-qubit 密度矩阵 (512 x 512)
            current_9q_state = np.kron(rho_6q_input, rho_anc)

            # =========================================================
            # 2. 信道加噪 (仅作用于 0, 1, 3 号比特)
            # =========================================================
            channel_qubits_indices = [0, 1,3,4]
            qc_channel_noise = cirq.Circuit()
            noise_p = self.np_random.uniform(0.1, 0.4)

            for q_idx in channel_qubits_indices:
                qc_channel_noise.append(cirq.depolarize(p=noise_p).on(self.qubits[q_idx]))

            sim = cirq.DensityMatrixSimulator()
            # 既然是 9-qubit 总态，simulator 必须跑全部 self.qubits
            res = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits)
            current_9q_state = self._clean_density_matrix(res.final_density_matrix)

            # =========================================================
            # 3. 预处理线路组装器 (附带硬件物理噪声)
            # =========================================================
            def _build_noisy_circuit(ops):
                noisy_circuit = cirq.Circuit()
                for op in ops:
                    noisy_circuit.append(op)  # 加理想门
                    op_qubits = op.qubits  # 提取门作用的比特

                    # 注入硬件噪声
                    if len(op_qubits) == 1 and self.error_single is not None and self.error_single > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))
                    elif len(op_qubits) > 1 and self.error_multi is not None and self.error_multi > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))
                return noisy_circuit

            q = self.qubits

            # --- P2: X 错预处理 (使用 q6 做安全探针，保护 Resource) ---
            p2_ops = [cirq.CNOT(q[0], q[6]), cirq.CNOT(q[1], q[6])]
            p2_circuit = _build_noisy_circuit(p2_ops)

            # --- P3: Z 错预处理 (Murao 协议优化版) ---
            # 核心 1：反向 CNOT (Resource -> Target) 抓取 Z 错
            # 核心 2：紧接着对 Resource (q3, q4, q5) 打 H 门，将其翻转到 X 基底，准备测偶宇称！
            p3_ops = [
                cirq.CNOT(q[3], q[0]), cirq.CNOT(q[4], q[1]), cirq.CNOT(q[5], q[2]),
                cirq.H(q[3]), cirq.H(q[4]), cirq.H(q[5])
            ]
            p3_ops_ori = [cirq.H(q[0]), cirq.H(q[1]), cirq.H(q[2]),cirq.H(q[3]), cirq.H(q[4]), cirq.H(q[5]),
                cirq.CNOT(q[0], q[3]), cirq.CNOT(q[1], q[4]), cirq.CNOT(q[2], q[5]),
                cirq.H(q[0]), cirq.H(q[1]), cirq.H(q[2])
            ]
            p3_circuit = _build_noisy_circuit(p3_ops)

            # --- P4: X 错预处理 (Murao 协议标准版) ---
            # 核心：前向 CNOT (Target -> Resource)
            p4_ops = [cirq.CNOT(q[0], q[3]), cirq.CNOT(q[1], q[4]), cirq.CNOT(q[2], q[5])]
            p4_circuit = _build_noisy_circuit(p4_ops)

            # =========================================================
            # 4. 迭代提纯与后选择 (Post-selection) 循环
            # =========================================================
            num_pre_iterations = 1  # 可调节: P2 (单探针抓 X 错)
            num_pre_iterations_total = 14  # 可调节: Murao 总轮数
            num_pre_iterations2 = 1  # 可调节: P3 (抓 Z 错)
            num_pre_iterations3 = 1  # 可调节: P4 (抓 X 错)

            # --- P2 预处理循环 (测量 q6) ---
            for i in range(num_pre_iterations):
                res = sim.simulate(p2_circuit, initial_state=current_9q_state, qubit_order=self.qubits)
                rho_after_p2 = res.final_density_matrix

                # 手动构建 q6=|0> 的投影算符
                I_6 = np.eye(2 ** 6, dtype=complex)
                proj_0 = np.array([[1, 0], [0, 0]], dtype=complex)
                I_2 = np.eye(2 ** 2, dtype=complex)
                P_q6_0 = np.kron(np.kron(I_6, proj_0), I_2)

                # 投影并计算存活率
                rho_proj = P_q6_0 @ rho_after_p2 @ P_q6_0.T
                p_succ = np.trace(rho_proj).real

                if p_succ > 1e-4:
                    # 1. 先得到投影后的 9-qubit 态 (此时 Target 干净，Resource 脏)
                    temp_9q_state = self._clean_density_matrix(rho_proj / p_succ)

                    # 2. 提取出干净的 Target
                    rho_3q_clean = self.manual_partial_trace(temp_9q_state, keep_indices=[0, 1, 2], dims=[2] * 9)
                    rho_3q_clean = self._clean_density_matrix(rho_3q_clean)

                    # =========================================================
                    # 🚀 核心补丁：让 Res = Target，消除"脏水污染"！
                    # 物理意义：我们假定网络调度器拿了另一对同样通过了 P2 校验的态来做 Resource
                    # =========================================================
                    rho_6q_new = np.kron(rho_3q_clean, rho_3q_clean)

                    # 4. 重新挂上干净的 Ancilla (q6, q7, q8)，重置为 9-qubit 态
                    current_9q_state = np.kron(rho_6q_new, rho_anc)
                else:
                    break  # 概率极低，保留上一步态直接退出

            # --- Murao 纯化循环 (测量并消耗 Resource q3,q4,q5) ---
            for j in range(num_pre_iterations_total):

                # P3 (Z 错纯化) 循环
                for i in range(num_pre_iterations2):
                    res = sim.simulate(p3_circuit, initial_state=current_9q_state, qubit_order=self.qubits)

                    # ✅ 先偏迹掉闲置 ancilla，回到 6‑qubit 系统空间 (q0‑q5)
                    rho_6q_after_p3 = self.manual_partial_trace(res.final_density_matrix,
                                                                keep_indices=[0, 1, 2, 3, 4, 5],
                                                                dims=[2] * 9)
                    # ✅ 在 6‑qubit 上做偶宇称后选择，target_mask 正确指向 q3,q4,q5
                    rho_post, p_succ = self._post_select_ghz_ancilla_even(rho_6q_after_p3, mode='P1')

                    if p_succ > 1e-4:
                        # 注意：rho_post 现在是 64×64 矩阵，dims 必须为 [2]*6
                        rho_3q_current = self.manual_partial_trace(rho_post, keep_indices=[0, 1, 2], dims=[2] * 6)
                        rho_3q_current = self._clean_density_matrix(rho_3q_current)

                        # 复制 clean Target 作为新 Resource，并重新挂上 ancilla
                        rho_6q_new = np.kron(rho_3q_current, rho_3q_current)
                        current_9q_state = np.kron(rho_6q_new, rho_anc)
                    else:
                        # 失败时从当前的 9‑qubit 态提取系统态（保持 dims=[2]*9）
                        rho_3q_current = self.manual_partial_trace(current_9q_state, keep_indices=[0, 1, 2],
                                                                   dims=[2] * 9)
                        rho_3q_current = self._clean_density_matrix(rho_3q_current)
                        break

                # P4 (X 错纯化) 循环
                for i in range(num_pre_iterations3):
                    res = sim.simulate(p4_circuit, initial_state=current_9q_state, qubit_order=self.qubits)

                    # 同样先偏迹到 6‑qubit
                    rho_6q_after_p4 = self.manual_partial_trace(res.final_density_matrix,
                                                                keep_indices=[0, 1, 2, 3, 4, 5],
                                                                dims=[2] * 9)
                    rho_post, p_succ = self._post_select_ghz_ancilla(rho_6q_after_p4)

                    if p_succ > 1e-4:
                        rho_3q_current = self.manual_partial_trace(rho_post, keep_indices=[0, 1, 2], dims=[2] * 6)
                        rho_3q_current = self._clean_density_matrix(rho_3q_current)
                        rho_6q_new = np.kron(rho_3q_current, rho_3q_current)
                        current_9q_state = np.kron(rho_6q_new, rho_anc)
                    else:
                        rho_3q_current = self.manual_partial_trace(current_9q_state, keep_indices=[0, 1, 2],
                                                                   dims=[2] * 9)
                        rho_3q_current = self._clean_density_matrix(rho_3q_current)
                        break

            # =========================================================
            # 5. 【核心组装区】整理并输出给 RL Agent
            # =========================================================
            self.rho = current_9q_state
            self.rho_initial = self.rho

            # 提取 0,1,2 比特计算保底数值，存入环境中供后续奖励计算参考
            self.rho_3q = self.manual_partial_trace(self.rho, keep_indices=[0, 1, 2], dims=[2] * 9)
            self.F_in = self._get_ghz_fidelity(self.rho_3q)
            self.en_in = self._get_logarithmic_negativity(self.rho_3q)

        elif rand_scenario < 0:  # 4-qubit 裸考版 (10-qubit 物理空间)

            f_param = self.np_random.uniform(0.95, 1.0)
            rho_sys_base = self._generate_ghz4_state(F=f_param)
            rho_res_base = self._generate_ghz4_state(F=f_param)
            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            # =========================================================
            # 🚀 Charlie 有两个辅助比特 q8 (Z-check) 和 q9 (X-check)，都初始化为 |00>
            # =========================================================
            psi_anc2 = np.zeros(4, dtype=complex); psi_anc2[0] = 1.0  # |00>
            rho_anc2 = np.outer(psi_anc2, psi_anc2.conj())

            # 当前总态：10-qubit 密度矩阵 (1024 x 1024)
            current_10q_state = np.kron(rho_8q_input, rho_anc2)

            # --- 信道加噪 (Alice 和 Bob 的光纤存在噪声) ---
            channel_qubits_indices = [1,2,3,5,6,7]
            qc_channel_noise = cirq.Circuit()
            noise_p = self.np_random.uniform(0.08, 0.35)

            for q_idx in channel_qubits_indices:
                qc_channel_noise.append(cirq.depolarize(p=noise_p).on(self.qubits[q_idx]))

            sim = cirq.DensityMatrixSimulator()
            res = sim.simulate(qc_channel_noise, initial_state=current_10q_state,
                               qubit_order=self.qubits)

            self.rho = self._clean_density_matrix(res.final_density_matrix)
            self.rho_initial = self.rho

            # 提取 Baseline 保真度
            self.rho_4q = self.manual_partial_trace(self.rho, keep_indices=[0, 1, 2, 3], dims=[2] * 10)
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = 0.0

        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (10-qubit 空间)

            # =========================================================
            # 1. 初始化 10-qubit 完美总态 (1024 x 1024)
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)  # 初始纯度

            rho_sys_base = self._generate_ghz4_state(F=f_param)
            rho_res_base = self._generate_ghz4_state(F=f_param)
            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            # Charlie 的两个辅助比特 q8 (Z-check) 和 q9 (X-check) 初始化为 |00>
            psi_anc2 = np.zeros(4, dtype=complex); psi_anc2[0] = 1.0
            rho_anc2 = np.outer(psi_anc2, psi_anc2.conj())

            current_10q_state = np.kron(rho_8q_input, rho_anc2)

            # =========================================================
            # 🚀 2. 异构信道加噪 (Heterogeneous Noise Injection)
            # =========================================================
            qc_channel_noise = cirq.Circuit()

            # --- Bob 侧噪声 (q1, q5) ---
            noise_p_bob = self.np_random.uniform(0.05, 0.3)#0.05-0.10
            self.niose_bob=noise_p_bob
            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            # --- Charlie 侧噪声 (q2, q3, q6, q7) ---
            noise_p_charlie = self.np_random.uniform(0.05, 0.3)
            self.niose_charlie = noise_p_charlie
            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================
            # 3. 演化并提取基线 (Baseline)
            # =========================================================
            sim = cirq.DensityMatrixSimulator()
            res = sim.simulate(qc_channel_noise, initial_state=current_10q_state, qubit_order=self.qubits)

            self.rho = self._clean_density_matrix(res.final_density_matrix)
            self.rho_initial = self.rho

            # 提取 Baseline 保真度 (提取 Target 0,1,2,3)
            self.rho_4q = self.manual_partial_trace(self.rho, keep_indices=[0, 1, 2, 3], dims=[2] * 10)
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)

        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (10-qubit 空间)

            # =========================================================
            # 1. 初始化 10-qubit 完美总态 (1024 x 1024)
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)  # 初始纯度

            rho_sys_base = self._generate_ghz4_state(F=f_param)
            rho_res_base = self._generate_ghz4_state(F=f_param)
            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            # Charlie 的两个辅助比特 q8 (Z-check) 和 q9 (X-check) 初始化为 |00>
            psi_anc2 = np.zeros(4, dtype=complex); psi_anc2[0] = 1.0
            rho_anc2 = np.outer(psi_anc2, psi_anc2.conj())

            current_10q_state = np.kron(rho_8q_input, rho_anc2)

            # =========================================================
            # 🚀 2. 异构信道加噪 (Heterogeneous Noise Injection)
            # =========================================================
            qc_channel_noise = cirq.Circuit()

            # --- Bob 侧噪声 (q1, q5) ---
            noise_p_bob = self.np_random.uniform(0.05, 0.3)#0.05-0.10
            self.niose_bob=noise_p_bob
            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            # --- Charlie 侧噪声 (q2, q3, q6, q7) ---
            noise_p_charlie = self.np_random.uniform(0.05, 0.3)
            self.niose_charlie = noise_p_charlie
            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

                # =========================================================
                # 3. 演化噪声并执行预处理与后选择 (Baseline)
                # =========================================================
                sim = cirq.DensityMatrixSimulator()

                # --- 3.1 演化信道加噪 ---
                res_noise = sim.simulate(qc_channel_noise, initial_state=current_10q_state, qubit_order=self.qubits)
                rho_after_noise = self._clean_density_matrix(res_noise.final_density_matrix)

                # --- 3.2 构造并在 Charlie 端执行 CNOT28 和 CNOT38 ---
                qc_pre = cirq.Circuit()
                qc_pre.append(cirq.CNOT(self.qubits[2], self.qubits[8]))
                qc_pre.append(cirq.CNOT(self.qubits[3], self.qubits[8]))

                # 演化该预处理操作
                res_pre = sim.simulate(qc_pre, initial_state=rho_after_noise, qubit_order=self.qubits)
                rho_pre = res_pre.final_density_matrix

                # --- 3.3 物理测量与后选择 (保留 q8 = |0>) ---
                dim = 1024
                indices = np.arange(dim)

                # 在 Cirq 默认的大端序排列中，q8 是倒数第 2 个比特（即 bit 1）
                # 提取 q8 的位掩码 (Mask) 为 0b0000000010 = 2
                q8_mask = 2

                # 我们要求 q8 的测量结果严格为 0
                valid_indices_mask = (indices & q8_mask) == 0

                # 提取对角线元素计算测量出 |0> 的成功概率 P(0)
                diags = np.diagonal(rho_pre)
                p_succ = np.sum(diags[valid_indices_mask]).real

                if p_succ > 1e-6:
                    # 构造投影矩阵向量，物理上执行 P * rho * P
                    mask_vec = valid_indices_mask.astype(float).reshape(-1, 1)

                    # 投影出特定分支，同时抹除与 q8=1 分支的量子相干性
                    rho_post = rho_pre * (mask_vec @ mask_vec.T)

                    # 迹归一化
                    rho_post /= p_succ
                    self.rho = self._clean_density_matrix(rho_post)
                else:
                    # 科学性保底：若概率极低导致崩溃，直接按失败态或原态处理
                    self.rho = self._clean_density_matrix(rho_pre)

                self.rho_initial = self.rho

                # --- 3.4 提取 Baseline 保真度 (提取 Target q0, q1, q2, q3) ---
                self.rho_4q = self.manual_partial_trace(self.rho, keep_indices=[0, 1, 2, 3], dims=[2] * 10)
                self.F_in = self._get_ghz4_fidelity(self.rho_4q)
                self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)


        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版——预处理 (9-qubit 空间)

            # =========================================================

            # 1. 初始化 9-qubit 完美总态 (512 x 512)

            # =========================================================

            f_param = self.np_random.uniform(0.95, 1.0)  # 初始纯度

            rho_sys_base = self._generate_ghz4_state(F=f_param)

            rho_res_base = self._generate_ghz4_state(F=f_param)

            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            # 🚀 仅保留 Charlie 的 1 个辅助比特 q8 (Z-check) 初始化为 |0>

            psi_anc = np.zeros(2, dtype=complex)

            psi_anc[0] = 1.0

            rho_anc = np.outer(psi_anc, psi_anc.conj())

            # 当前总态：9-qubit 密度矩阵 (512 x 512)

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================

            # 🚀 2. 异构信道加噪 (Heterogeneous Noise Injection)

            # =========================================================

            qc_channel_noise = cirq.Circuit()

            # --- Bob 侧噪声 (q1, q5) ---

            noise_p_bob = self.np_random.uniform(0.05, 0.3)

            self.niose_bob = noise_p_bob

            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            # --- Charlie 侧噪声 (q2, q3, q6, q7) ---

            noise_p_charlie = self.np_random.uniform(0.05, 0.3)

            self.niose_charlie = noise_p_charlie

            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================

            # 3. 核心流水线：演化噪声 -> 初始预过滤 (Pre-selection)

            # =========================================================

            sim = cirq.DensityMatrixSimulator()

            # --- 3.1 演化初始信道加噪 ---

            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])

            rho_after_noise = self._clean_density_matrix(res_noise.final_density_matrix)

            rho_current = rho_after_noise

            # 🚀 鲁棒噪声组装器

            def _build_noisy_circuit(ops):

                noisy_circuit = cirq.Circuit()

                for op in ops:

                    noisy_circuit.append(op)

                    op_qubits = op.qubits

                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:

                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))

                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:

                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))

                return noisy_circuit

            dim = 512

            # =========================================================

            # 🚀 3.2 初始预处理 (Pre-selection): 确保 Target 和 Resource 同等过滤

            # =========================================================

            ops_pre = [

                cirq.CNOT(self.qubits[2], self.qubits[8]),

                cirq.CNOT(self.qubits[3], self.qubits[8])

            ]

            qc_pre = _build_noisy_circuit(ops_pre)

            res_pre = sim.simulate(qc_pre, initial_state=rho_current, qubit_order=self.qubits[:9])

            rho_pre = res_pre.final_density_matrix

            q8_mask = 1

            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0

            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:

                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)

                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T)

                rho_pre_post /= p_succ_pre

                rho_current = self._clean_density_matrix(rho_pre_post)

                # 提取预处理成功的 Target 态，并克隆给 Resource，保证两者同样纯净

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_target_pre = rho_4q_pre_success

                rho_resource_pre = rho_4q_pre_success

            else:

                # 保底机制

                rho_target_pre = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_resource_pre = rho_target_pre

            # =========================================================

            # 🚀 4. 为 RL Agent 装填起跑线！

            # =========================================================

            # 重新组装 8-qubit (Target ⊗ Resource)

            rho_8q_pre = np.kron(rho_target_pre, rho_resource_pre)

            # 重置干净的 q8 探针

            psi_anc = np.zeros(2, dtype=complex);
            psi_anc[0] = 1.0

            rho_anc_clean = np.outer(psi_anc, psi_anc.conj())

            # 覆写环境真实初始状态

            self.rho = np.kron(rho_8q_pre, rho_anc_clean)

            self.rho_initial = self.rho

            # 给 RL Agent 设立 Baseline 目标 (即预过滤后的基础保真度)

            self.rho_4q = rho_target_pre

            self.F_in = self._get_ghz4_fidelity(self.rho_4q)

            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)


        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)

            # =========================================================

            # 1. 初始化 9-qubit 完美总态 (512 x 512)

            # =========================================================

            f_param = self.np_random.uniform(0.95, 1.0)  # 初始纯度

            rho_sys_base = self._generate_ghz4_state(F=f_param)

            rho_res_base = self._generate_ghz4_state(F=f_param)

            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            # 🚀 仅保留 Charlie 的 1 个辅助比特 q8 (Z-check) 初始化为 |0>

            psi_anc = np.zeros(2, dtype=complex)

            psi_anc[0] = 1.0

            rho_anc = np.outer(psi_anc, psi_anc.conj())

            # 当前总态：9-qubit 密度矩阵 (512 x 512)

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================

            # 🚀 2. 异构信道加噪 (Heterogeneous Noise Injection)

            # =========================================================

            qc_channel_noise = cirq.Circuit()

            # --- Bob 侧噪声 (q1, q5) ---

            noise_p_bob = self.np_random.uniform(0.05, 0.3)

            self.niose_bob = noise_p_bob

            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            # --- Charlie 侧噪声 (q2, q3, q6, q7) ---

            noise_p_charlie = self.np_random.uniform(0.05, 0.3)

            self.niose_charlie = noise_p_charlie

            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================

            # 3. 核心流水线：演化噪声 -> 初始预过滤 (Pre-selection)

            # =========================================================

            sim = cirq.DensityMatrixSimulator()

            # --- 3.1 演化初始信道加噪 ---

            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])

            rho_after_noise = self._clean_density_matrix(res_noise.final_density_matrix)

            rho_current = rho_after_noise

            # 🚀 鲁棒噪声组装器

            def _build_noisy_circuit(ops):

                noisy_circuit = cirq.Circuit()

                for op in ops:

                    noisy_circuit.append(op)

                    op_qubits = op.qubits

                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:

                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))

                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:

                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))

                return noisy_circuit

            dim = 512

            # =========================================================

            # 🚀 3.2 初始预处理 (Pre-selection): 确保 Target 和 Resource 同等过滤

            # =========================================================

            ops_pre = [

                cirq.CNOT(self.qubits[2], self.qubits[8]),

                cirq.CNOT(self.qubits[3], self.qubits[8])

            ]

            qc_pre = _build_noisy_circuit(ops_pre)

            res_pre = sim.simulate(qc_pre, initial_state=rho_current, qubit_order=self.qubits[:9])

            rho_pre = res_pre.final_density_matrix

            q8_mask = 1

            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0

            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:

                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)

                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T)

                rho_pre_post /= p_succ_pre

                rho_current = self._clean_density_matrix(rho_pre_post)

                # 提取预处理成功的 Target 态，并克隆给 Resource，保证两者同样纯净

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_target_pre = rho_4q_pre_success

                rho_resource_pre = rho_4q_pre_success

            else:

                # 保底机制

                rho_target_pre = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_resource_pre = rho_target_pre

            # =========================================================

            # 🚀 4. 为 RL Agent 装填起跑线！

            # =========================================================

            # 重新组装 8-qubit (Target ⊗ Resource)

            rho_8q_pre = np.kron(rho_target_pre, rho_resource_pre)

            # 重置干净的 q8 探针

            psi_anc = np.zeros(2, dtype=complex);
            psi_anc[0] = 1.0

            rho_anc_clean = np.outer(psi_anc, psi_anc.conj())

            # 覆写环境真实初始状态

            self.rho = np.kron(rho_8q_pre, rho_anc_clean)

            self.rho_initial = self.rho

            # 给 RL Agent 设立 Baseline 目标 (即预过滤后的基础保真度)

            self.rho_4q = rho_target_pre

            self.F_in = self._get_ghz4_fidelity(self.rho_4q)

            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)

        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)

            # =========================================================
            # 1. 初始化 9-qubit 完美总态 (512 x 512)
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)  # 初始纯度

            rho_sys_base = self._generate_ghz4_state(F=f_param)
            rho_res_base = self._generate_ghz4_state(F=f_param)
            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            # 🚀 仅保留 Charlie 的 1 个辅助比特 q8 (Z-check) 初始化为 |0>
            psi_anc = np.zeros(2, dtype=complex)
            psi_anc[0] = 1.0
            rho_anc = np.outer(psi_anc, psi_anc.conj())

            # 当前总态：9-qubit 密度矩阵 (512 x 512)
            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================
            # 🚀 2. 异构信道加噪 (Heterogeneous Noise Injection)
            # =========================================================
            qc_channel_noise = cirq.Circuit()

            # --- Bob 侧噪声 (q1, q5) ---
            noise_p_bob = self.np_random.uniform(0.05, 0.3)
            self.niose_bob = noise_p_bob
            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            # --- Charlie 侧噪声 (q2, q3, q6, q7) ---
            noise_p_charlie = self.np_random.uniform(0.05, 0.3)
            self.niose_charlie = noise_p_charlie
            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================
            # 3. 演化初始信道加噪 (最纯粹的起点)
            # =========================================================
            sim = cirq.DensityMatrixSimulator()
            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])
            rho_after_noise = self._clean_density_matrix(res_noise.final_density_matrix)

            # =========================================================
            # 4. 直接移交 RL Agent (裸考模式)
            # =========================================================
            # 覆写环境真实初始状态为加噪后的原始状态
            self.rho = rho_after_noise
            self.rho_initial = self.rho

            # 给 RL Agent 设立 Baseline 目标 (即未做任何纯化的初始加噪保真度)
            self.rho_4q = self.manual_partial_trace(self.rho, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)


        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)

            # =========================================================

            # 1. 初始化 9-qubit 完美总态 (512 x 512)

            # =========================================================

            f_param = self.np_random.uniform(0.95, 1.0)  # 初始纯度

            rho_sys_base = self._generate_ghz4_state(F=f_param)

            rho_res_base = self._generate_ghz4_state(F=f_param)

            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            # 🚀 仅保留 Charlie 的 1 个辅助比特 q8 (Z-check) 初始化为 |0>

            psi_anc = np.zeros(2, dtype=complex)

            psi_anc[0] = 1.0

            rho_anc = np.outer(psi_anc, psi_anc.conj())

            # 当前总态：9-qubit 密度矩阵 (512 x 512)

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================

            # 🚀 2. 异构信道加噪 (Heterogeneous Noise Injection)

            # =========================================================

            qc_channel_noise = cirq.Circuit()

            # --- Bob 侧噪声 (q1, q5) ---

            noise_p_bob = self.np_random.uniform(0.05, 0.3)

            self.niose_bob = noise_p_bob

            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            # --- Charlie 侧噪声 (q2, q3, q6, q7) ---

            noise_p_charlie = self.np_random.uniform(0.05, 0.3)

            self.niose_charlie = noise_p_charlie

            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================

            # 3. 核心大流水线：演化噪声 -> 预过滤 -> 经典第一轮

            # =========================================================

            sim = cirq.DensityMatrixSimulator()

            # --- 3.1 演化初始信道加噪 ---

            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])

            rho_after_noise = self._clean_density_matrix(res_noise.final_density_matrix)

            rho_current = rho_after_noise

            # 🚀 鲁棒噪声组装器 (全局复用)

            def _build_noisy_circuit(ops):

                noisy_circuit = cirq.Circuit()

                for op in ops:

                    noisy_circuit.append(op)

                    op_qubits = op.qubits

                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:

                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))

                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:

                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))

                return noisy_circuit

            dim = 512

            # =========================================================

            # 🚀 3.2 初始预处理 (Pre-selection): Target 和 Resource 同等预过滤

            # =========================================================

            ops_pre = [

                cirq.CNOT(self.qubits[2], self.qubits[8]),

                cirq.CNOT(self.qubits[3], self.qubits[8])

            ]

            qc_pre = _build_noisy_circuit(ops_pre)

            res_pre = sim.simulate(qc_pre, initial_state=rho_current, qubit_order=self.qubits[:9])

            rho_pre = res_pre.final_density_matrix

            q8_mask = 1

            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0

            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:

                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)

                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T)

                rho_pre_post /= p_succ_pre

                rho_current = self._clean_density_matrix(rho_pre_post)

                # 提取预处理成功的 Target 态并克隆给 Resource

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_target_pre = rho_4q_pre_success

                rho_resource_pre = rho_4q_pre_success

            else:

                rho_target_pre = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_resource_pre = rho_target_pre

            # 重新组装并清空 q8 探针

            rho_8q_pre = np.kron(rho_target_pre, rho_resource_pre)

            rho_current = np.kron(rho_8q_pre, np.outer(psi_anc, psi_anc.conj()))

            # =========================================================

            # 🚀 3.3 修改后的第一轮：纯粹的 Action 48 + Action 126 (无第2轮)

            # =========================================================

            # --- Step 0: Action 48 (Resource 控制 Target 泵浦) ---

            ops_r1 = [

                cirq.CNOT(self.qubits[4], self.qubits[0]),

                cirq.CNOT(self.qubits[5], self.qubits[1]),

                cirq.CNOT(self.qubits[6], self.qubits[2]),

                cirq.CNOT(self.qubits[7], self.qubits[3])

            ]

            qc_r1 = _build_noisy_circuit(ops_r1)

            res_r1 = sim.simulate(qc_r1, initial_state=rho_current, qubit_order=self.qubits[:9])

            # --- Step 1: Action 126 (FINISH_P1_MASK_BOTH 结算) ---

            # P1 模式切换：Resource  qubit 施加 H 门

            basis_change_ops_r1 = [cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]

            qc_basis_r1 = _build_noisy_circuit(basis_change_ops_r1)

            res_basis_r1 = sim.simulate(qc_basis_r1, initial_state=res_r1.final_density_matrix,
                                        qubit_order=self.qubits[:9])

            rho_9q_r1 = res_basis_r1.final_density_matrix

            diag_elements_r1 = np.diagonal(rho_9q_r1).real

            final_valid_mask_r1 = np.ones(dim, dtype=bool)

            # MASK_BOTH 双重过滤：ANC 必须为 0 且 RES 必须为 X 基底偶宇称

            final_valid_mask_r1 &= np.array([(i & 0b1) == 0 for i in range(dim)])  # ANC Mask (q8=0)

            allowed_res_p1 = [0, 3, 5, 6, 9, 10, 12, 15]

            res_vals_r1 = np.array([((i >> 1) & 0b1111) for i in range(dim)])

            final_valid_mask_r1 &= np.isin(res_vals_r1, allowed_res_p1)  # RES Mask

            p_succ_r1 = np.sum(diag_elements_r1[final_valid_mask_r1])

            if p_succ_r1 > 1e-10:

                projected_rho_r1 = np.zeros_like(rho_9q_r1)

                valid_indices_r1 = np.where(final_valid_mask_r1)[0]

                for val in set(valid_indices_r1 % 32):

                    branch_mask = np.array([(i % 32) == val for i in range(dim)])

                    if final_valid_mask_r1[np.where(branch_mask)[0][0]]:
                        vec = branch_mask.astype(float).reshape(-1, 1)

                        projected_rho_r1 += rho_9q_r1 * (vec @ vec.T)

                rho_current = self._clean_density_matrix(projected_rho_r1 / p_succ_r1)

            else:

                rho_current = self._clean_density_matrix(rho_9q_r1)

            # =========================================================

            # 🚀 4. 提取第一轮战果，直接装填为 Agent 的起跑线！

            # =========================================================

            rho_4q_r1_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r1 > 1e-10:

                # 物理意义：两对同样挺过了 Action 48+126 洗礼的极品优质态

                rho_target_next = rho_4q_r1_success

                rho_resource_next = rho_4q_r1_success  # 复制一份作为下一轮的 Resource

            else:

                # 保底机制

                rho_target_next = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_resource_next = rho_target_next

            # 张量积重新组装 9-qubit 空间

            rho_8q_next = np.kron(rho_target_next, rho_resource_next)

            # 覆写环境真实初始状态，送给 RL Agent

            self.rho = np.kron(rho_8q_next, np.outer(psi_anc, psi_anc.conj()))

            self.rho_initial = self.rho

            # 刷新 Baseline 目标（现在的基线就是这套新策略纯化出来的保真度）

            self.rho_4q = rho_target_next

            self.F_in = self._get_ghz4_fidelity(self.rho_4q)

            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)



        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间) 找第三轮

            # =========================================================

            # 🚀 0. 安全过滤装甲 (防 NoneType 崩溃补丁)

            # =========================================================

            def _safe_clean(rho_in):

                if rho_in is None: return None

                _res = self._clean_density_matrix(rho_in)

                return _res if _res is not None else rho_in

            # =========================================================

            # 1. 初始化 9-qubit 完美总态 (512 x 512)

            # =========================================================

            f_param = self.np_random.uniform(0.95, 1.0)  # 初始纯度

            rho_sys_base = self._generate_ghz4_state(F=f_param)

            rho_res_base = self._generate_ghz4_state(F=f_param)

            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            # 仅保留 Charlie 的 1 个辅助比特 q8 (Z-check) 初始化为 |0>

            psi_anc = np.zeros(2, dtype=complex)

            psi_anc[0] = 1.0

            rho_anc = np.outer(psi_anc, psi_anc.conj())

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================

            # 🚀 2. 异构信道加噪

            # =========================================================

            qc_channel_noise = cirq.Circuit()

            noise_p_bob = self.np_random.uniform(0.05, 0.3)

            self.niose_bob = noise_p_bob

            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            noise_p_charlie = self.np_random.uniform(0.05, 0.3)

            self.niose_charlie = noise_p_charlie

            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================

            # 3. 核心大流水线

            # =========================================================

            sim = cirq.DensityMatrixSimulator()

            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])

            # 全局替换：使用 _safe_clean 替代 self._clean_density_matrix

            rho_after_noise = _safe_clean(res_noise.final_density_matrix)

            rho_current = rho_after_noise

            def _build_noisy_circuit(ops):

                noisy_circuit = cirq.Circuit()

                for op in ops:

                    noisy_circuit.append(op)

                    op_qubits = op.qubits

                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:

                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))

                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:

                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))

                return noisy_circuit

            dim = 512

            # =========================================================

            # 🚀 3.2 初始预处理 (Pre-selection)

            # =========================================================

            ops_pre = [

                cirq.CNOT(self.qubits[2], self.qubits[8]),

                cirq.CNOT(self.qubits[3], self.qubits[8])

            ]

            qc_pre = _build_noisy_circuit(ops_pre)

            res_pre = sim.simulate(qc_pre, initial_state=rho_current, qubit_order=self.qubits[:9])

            rho_pre = res_pre.final_density_matrix

            q8_mask = 1

            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0

            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:

                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)

                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T)

                rho_pre_post /= p_succ_pre

                # 🛡️ 安全过关！

                rho_current = _safe_clean(rho_pre_post)

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_target_pre = rho_4q_pre_success

                rho_resource_pre = rho_4q_pre_success

            else:

                rho_target_pre = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_resource_pre = rho_target_pre

            rho_8q_pre = np.kron(rho_target_pre, rho_resource_pre)

            rho_current = np.kron(rho_8q_pre, np.outer(psi_anc, psi_anc.conj()))

            # =========================================================

            # 🚀 3.3 拓扑感知第一轮：纯粹的 Action 97 + Action 129

            # =========================================================

            ops_r1 = [

                cirq.CNOT(self.qubits[2], self.qubits[8]),

                cirq.CNOT(self.qubits[3], self.qubits[8]),

                cirq.CNOT(self.qubits[6], self.qubits[8]),

                cirq.CNOT(self.qubits[7], self.qubits[8])

            ]

            qc_r1 = _build_noisy_circuit(ops_r1)

            res_r1 = sim.simulate(qc_r1, initial_state=rho_current, qubit_order=self.qubits[:9])

            rho_9q_r1 = res_r1.final_density_matrix

            diag_elements_r1 = np.diagonal(rho_9q_r1).real

            final_valid_mask_r1 = np.ones(dim, dtype=bool)

            final_valid_mask_r1 &= np.array([(i & 0b1) == 0 for i in range(dim)])  # ANC Mask (q8=0)

            allowed_res_p2 = [0, 15]

            res_vals_r1 = np.array([((i >> 1) & 0b1111) for i in range(dim)])

            final_valid_mask_r1 &= np.isin(res_vals_r1, allowed_res_p2)  # RES Mask

            p_succ_r1 = np.sum(diag_elements_r1[final_valid_mask_r1])

            if p_succ_r1 > 1e-10:

                projected_rho_r1 = np.zeros_like(rho_9q_r1)

                valid_indices_r1 = np.where(final_valid_mask_r1)[0]

                for val in set(valid_indices_r1 % 32):

                    branch_mask = np.array([(i % 32) == val for i in range(dim)])

                    if final_valid_mask_r1[np.where(branch_mask)[0][0]]:
                        vec = branch_mask.astype(float).reshape(-1, 1)

                        projected_rho_r1 += rho_9q_r1 * (vec @ vec.T)

                # 🛡️ 安全过关！

                rho_current = _safe_clean(projected_rho_r1 / p_succ_r1)

            else:

                rho_current = _safe_clean(rho_9q_r1)

            # =========================================================

            # 🚀 4. 提取战果，直接装填为 Agent 的起跑线！

            # =========================================================

            rho_4q_r1_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r1 > 1e-10:

                rho_target_next = rho_4q_r1_success

                rho_resource_next = rho_4q_r1_success

            else:

                rho_target_next = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_resource_next = rho_target_next

            rho_8q_next = np.kron(rho_target_next, rho_resource_next)

            self.rho = np.kron(rho_8q_next, np.outer(psi_anc, psi_anc.conj()))

            self.rho_initial = self.rho

            self.rho_4q = rho_target_next

            self.F_in = self._get_ghz4_fidelity(self.rho_4q)

            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)

        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)

            # =========================================================
            # 🚀 0. 安全过滤装甲 (防 NoneType 崩溃补丁)
            # =========================================================
            def _safe_clean(rho_in):
                if rho_in is None: return None
                _res = self._clean_density_matrix(rho_in)
                return _res if _res is not None else rho_in

            # =========================================================
            # 1. 初始化 9-qubit 完美总态 (512 x 512)
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)  # 初始纯度

            rho_sys_base = self._generate_ghz4_state(F=f_param)
            rho_res_base = self._generate_ghz4_state(F=f_param)
            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            # 仅保留 Charlie 的 1 个辅助比特 q8 初始化为 |0>
            psi_anc = np.zeros(2, dtype=complex)
            psi_anc[0] = 1.0
            rho_anc = np.outer(psi_anc, psi_anc.conj())

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================
            # 🚀 2. 异构信道加噪
            # =========================================================
            qc_channel_noise = cirq.Circuit()

            noise_p_bob = self.np_random.uniform(0.05, 0.3)
            self.niose_bob = noise_p_bob
            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            noise_p_charlie = self.np_random.uniform(0.05, 0.3)
            self.niose_charlie = noise_p_charlie
            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================
            # 3. 核心大流水线开始
            # =========================================================
            sim = cirq.DensityMatrixSimulator()

            # --- 3.1 演化初始信道加噪 ---
            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])
            rho_after_noise = _safe_clean(res_noise.final_density_matrix)
            rho_current = rho_after_noise

            def _build_noisy_circuit(ops):
                noisy_circuit = cirq.Circuit()
                for op in ops:
                    noisy_circuit.append(op)
                    op_qubits = op.qubits
                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))
                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))
                return noisy_circuit

            dim = 512

            # =========================================================
            # 🚀 3.2 初始预处理 (Pre-selection)
            # =========================================================
            ops_pre = [
                cirq.CNOT(self.qubits[2], self.qubits[8]),
                cirq.CNOT(self.qubits[3], self.qubits[8])
            ]
            qc_pre = _build_noisy_circuit(ops_pre)
            res_pre = sim.simulate(qc_pre, initial_state=rho_current, qubit_order=self.qubits[:9])
            rho_pre = res_pre.final_density_matrix

            q8_mask = 1
            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0
            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:
                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)
                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T)
                rho_pre_post /= p_succ_pre
                rho_current = _safe_clean(rho_pre_post)

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_target_pre = rho_4q_pre_success
                rho_resource_pre = rho_4q_pre_success
            else:
                rho_target_pre = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_pre = rho_target_pre

            rho_8q_pre = np.kron(rho_target_pre, rho_resource_pre)
            rho_current = np.kron(rho_8q_pre, np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 3.3 第一轮：拓扑感知纯化 (Action 97 + Action 129: P2_MASK)
            # =========================================================
            ops_r1 = [
                cirq.CNOT(self.qubits[2], self.qubits[8]),
                cirq.CNOT(self.qubits[3], self.qubits[8]),
                cirq.CNOT(self.qubits[6], self.qubits[8]),
                cirq.CNOT(self.qubits[7], self.qubits[8])
            ]
            qc_r1 = _build_noisy_circuit(ops_r1)
            res_r1 = sim.simulate(qc_r1, initial_state=rho_current, qubit_order=self.qubits[:9])
            rho_9q_r1 = res_r1.final_density_matrix

            # Action 129: P2 MASK BOTH (Z 基底偶宇称，不加 H 门)
            diag_elements_r1 = np.diagonal(rho_9q_r1).real
            final_valid_mask_r1 = np.ones(dim, dtype=bool)
            final_valid_mask_r1 &= np.array([(i & 0b1) == 0 for i in range(dim)])  # q8 = 0

            allowed_res_p2 = [0, 15]  # 0000 和 1111
            res_vals_r1 = np.array([((i >> 1) & 0b1111) for i in range(dim)])
            final_valid_mask_r1 &= np.isin(res_vals_r1, allowed_res_p2)

            p_succ_r1 = np.sum(diag_elements_r1[final_valid_mask_r1])
            if p_succ_r1 > 1e-10:
                projected_rho_r1 = np.zeros_like(rho_9q_r1)
                valid_indices_r1 = np.where(final_valid_mask_r1)[0]
                for val in set(valid_indices_r1 % 32):
                    branch_mask = np.array([(i % 32) == val for i in range(dim)])
                    if final_valid_mask_r1[np.where(branch_mask)[0][0]]:
                        vec = branch_mask.astype(float).reshape(-1, 1)
                        projected_rho_r1 += rho_9q_r1 * (vec @ vec.T)
                rho_current = _safe_clean(projected_rho_r1 / p_succ_r1)
            else:
                rho_current = _safe_clean(rho_9q_r1)

            # 装填第二轮起跑线
            rho_4q_r1_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            if p_succ_r1 > 1e-10:
                rho_target_R2 = rho_4q_r1_success
                rho_resource_R2 = rho_4q_r1_success
            else:
                rho_target_R2 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_R2 = rho_target_R2

            rho_8q_R2 = np.kron(rho_target_R2, rho_resource_R2)
            rho_current = np.kron(rho_8q_R2, np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 3.4 第二轮：精细相位嗅探 (Action 61 + Action 126: P1_MASK)
            # =========================================================
            ops_r2 = [
                cirq.CNOT(self.qubits[4], self.qubits[0]), cirq.CNOT(self.qubits[5], self.qubits[1]),
                cirq.CNOT(self.qubits[6], self.qubits[2]), cirq.CNOT(self.qubits[7], self.qubits[3]),
                cirq.CZ(self.qubits[0], self.qubits[4]), cirq.CZ(self.qubits[1], self.qubits[5])
            ]
            qc_r2 = _build_noisy_circuit(ops_r2)
            res_r2 = sim.simulate(qc_r2, initial_state=rho_current, qubit_order=self.qubits[:9])

            # Action 126: P1 MASK BOTH (X 基底偶宇称，需加 H 门)
            basis_change_ops_r2 = [cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]
            qc_basis_r2 = _build_noisy_circuit(basis_change_ops_r2)
            res_basis_r2 = sim.simulate(qc_basis_r2, initial_state=res_r2.final_density_matrix,
                                        qubit_order=self.qubits[:9])
            rho_9q_r2 = res_basis_r2.final_density_matrix

            diag_elements_r2 = np.diagonal(rho_9q_r2).real
            final_valid_mask_r2 = np.ones(dim, dtype=bool)
            final_valid_mask_r2 &= np.array([(i & 0b1) == 0 for i in range(dim)])  # q8 = 0

            allowed_res_p1 = [0, 3, 5, 6, 9, 10, 12, 15]  # X基底偶宇称
            res_vals_r2 = np.array([((i >> 1) & 0b1111) for i in range(dim)])
            final_valid_mask_r2 &= np.isin(res_vals_r2, allowed_res_p1)

            p_succ_r2 = np.sum(diag_elements_r2[final_valid_mask_r2])
            if p_succ_r2 > 1e-10:
                projected_rho_r2 = np.zeros_like(rho_9q_r2)
                valid_indices_r2 = np.where(final_valid_mask_r2)[0]
                for val in set(valid_indices_r2 % 32):
                    branch_mask = np.array([(i % 32) == val for i in range(dim)])
                    if final_valid_mask_r2[np.where(branch_mask)[0][0]]:
                        vec = branch_mask.astype(float).reshape(-1, 1)
                        projected_rho_r2 += rho_9q_r2 * (vec @ vec.T)
                rho_current = _safe_clean(projected_rho_r2 / p_succ_r2)
            else:
                rho_current = _safe_clean(rho_9q_r2)

            # =========================================================
            # 🚀 4. 装填终极起跑线，移交 RL Agent！
            # =========================================================
            rho_4q_r2_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r2 > 1e-10:
                rho_target_R3 = rho_4q_r2_success
                rho_resource_R3 = rho_4q_r2_success
            else:
                rho_target_R3 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_R3 = rho_target_R3

            rho_8q_R3 = np.kron(rho_target_R3, rho_resource_R3)

            self.rho = np.kron(rho_8q_R3, np.outer(psi_anc, psi_anc.conj()))
            self.rho_initial = self.rho

            self.rho_4q = rho_target_R3
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)


        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)

            # =========================================================

            # 🚀 0. 安全过滤装甲 (防 NoneType 崩溃补丁)

            # =========================================================

            def _safe_clean(rho_in):

                if rho_in is None: return None

                _res = self._clean_density_matrix(rho_in)

                return _res if _res is not None else rho_in

            # =========================================================

            # 1. 初始化 9-qubit 完美总态 (512 x 512)

            # =========================================================

            f_param = self.np_random.uniform(0.95, 1.0)  # 初始纯度

            rho_sys_base = self._generate_ghz4_state(F=f_param)

            rho_res_base = self._generate_ghz4_state(F=f_param)

            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            # 仅保留 Charlie 的 1 个辅助比特 q8 初始化为 |0>

            psi_anc = np.zeros(2, dtype=complex)

            psi_anc[0] = 1.0

            rho_anc = np.outer(psi_anc, psi_anc.conj())

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================

            # 🚀 2. 异构信道加噪

            # =========================================================

            qc_channel_noise = cirq.Circuit()

            noise_p_bob = self.np_random.uniform(0.05, 0.3)

            self.niose_bob = noise_p_bob

            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            noise_p_charlie = self.np_random.uniform(0.05, 0.3)

            self.niose_charlie = noise_p_charlie

            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================

            # 3. 核心大流水线开始

            # =========================================================

            sim = cirq.DensityMatrixSimulator()

            # --- 3.1 演化初始信道加噪 ---

            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])

            rho_after_noise = _safe_clean(res_noise.final_density_matrix)

            rho_current = rho_after_noise

            def _build_noisy_circuit(ops):

                noisy_circuit = cirq.Circuit()

                for op in ops:

                    noisy_circuit.append(op)

                    op_qubits = op.qubits

                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:

                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))

                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:

                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))

                return noisy_circuit

            dim = 512

            # =========================================================

            # 🚀 3.2 初始预处理 (Pre-selection)

            # =========================================================

            ops_pre = [

                cirq.CNOT(self.qubits[2], self.qubits[8]),

                cirq.CNOT(self.qubits[3], self.qubits[8])

            ]

            qc_pre = _build_noisy_circuit(ops_pre)

            res_pre = sim.simulate(qc_pre, initial_state=rho_current, qubit_order=self.qubits[:9])

            rho_pre = res_pre.final_density_matrix

            q8_mask = 1

            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0

            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:

                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)

                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T)

                rho_pre_post /= p_succ_pre

                rho_current = _safe_clean(rho_pre_post)

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_target_pre = rho_4q_pre_success

                rho_resource_pre = rho_4q_pre_success

            else:

                rho_target_pre = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_resource_pre = rho_target_pre

            rho_8q_pre = np.kron(rho_target_pre, rho_resource_pre)

            rho_current = np.kron(rho_8q_pre, np.outer(psi_anc, psi_anc.conj()))

            # =========================================================

            # 🚀 3.3 第一轮：拓扑感知粗筛 (Action 97 + Action 129: P2_MASK)

            # =========================================================

            ops_r1 = [

                cirq.CNOT(self.qubits[2], self.qubits[8]),

                cirq.CNOT(self.qubits[3], self.qubits[8]),

                cirq.CNOT(self.qubits[6], self.qubits[8]),

                cirq.CNOT(self.qubits[7], self.qubits[8])

            ]

            qc_r1 = _build_noisy_circuit(ops_r1)

            res_r1 = sim.simulate(qc_r1, initial_state=rho_current, qubit_order=self.qubits[:9])

            rho_9q_r1 = res_r1.final_density_matrix

            diag_elements_r1 = np.diagonal(rho_9q_r1).real

            final_valid_mask_r1 = np.ones(dim, dtype=bool)

            final_valid_mask_r1 &= np.array([(i & 0b1) == 0 for i in range(dim)])  # q8 = 0

            allowed_res_p2 = [0, 15]  # Z基底偶宇称 (0000, 1111)

            res_vals_r1 = np.array([((i >> 1) & 0b1111) for i in range(dim)])

            final_valid_mask_r1 &= np.isin(res_vals_r1, allowed_res_p2)

            p_succ_r1 = np.sum(diag_elements_r1[final_valid_mask_r1])

            if p_succ_r1 > 1e-10:

                projected_rho_r1 = np.zeros_like(rho_9q_r1)

                valid_indices_r1 = np.where(final_valid_mask_r1)[0]

                for val in set(valid_indices_r1 % 32):

                    branch_mask = np.array([(i % 32) == val for i in range(dim)])

                    if final_valid_mask_r1[np.where(branch_mask)[0][0]]:
                        vec = branch_mask.astype(float).reshape(-1, 1)

                        projected_rho_r1 += rho_9q_r1 * (vec @ vec.T)

                rho_current = _safe_clean(projected_rho_r1 / p_succ_r1)

            else:

                rho_current = _safe_clean(rho_9q_r1)

            # 装填第二轮起跑线

            rho_4q_r1_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r1 > 1e-10:

                rho_target_R2 = rho_4q_r1_success

                rho_resource_R2 = rho_4q_r1_success

            else:

                rho_target_R2 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_resource_R2 = rho_target_R2

            rho_8q_R2 = np.kron(rho_target_R2, rho_resource_R2)

            rho_current = np.kron(rho_8q_R2, np.outer(psi_anc, psi_anc.conj()))

            # =========================================================

            # 🚀 3.4 第二轮：精细相位嗅探 (Action 61 + Action 126: P1_MASK)

            # =========================================================

            ops_r2 = [

                cirq.CNOT(self.qubits[4], self.qubits[0]), cirq.CNOT(self.qubits[5], self.qubits[1]),

                cirq.CNOT(self.qubits[6], self.qubits[2]), cirq.CNOT(self.qubits[7], self.qubits[3]),

                cirq.CZ(self.qubits[0], self.qubits[4]), cirq.CZ(self.qubits[1], self.qubits[5])

            ]

            qc_r2 = _build_noisy_circuit(ops_r2)

            res_r2 = sim.simulate(qc_r2, initial_state=rho_current, qubit_order=self.qubits[:9])

            basis_change_ops_r2 = [cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]

            qc_basis_r2 = _build_noisy_circuit(basis_change_ops_r2)

            res_basis_r2 = sim.simulate(qc_basis_r2, initial_state=res_r2.final_density_matrix,
                                        qubit_order=self.qubits[:9])

            rho_9q_r2 = res_basis_r2.final_density_matrix

            diag_elements_r2 = np.diagonal(rho_9q_r2).real

            final_valid_mask_r2 = np.ones(dim, dtype=bool)

            final_valid_mask_r2 &= np.array([(i & 0b1) == 0 for i in range(dim)])  # q8 = 0

            allowed_res_p1 = [0, 3, 5, 6, 9, 10, 12, 15]  # X基底偶宇称

            res_vals_r2 = np.array([((i >> 1) & 0b1111) for i in range(dim)])

            final_valid_mask_r2 &= np.isin(res_vals_r2, allowed_res_p1)

            p_succ_r2 = np.sum(diag_elements_r2[final_valid_mask_r2])

            if p_succ_r2 > 1e-10:

                projected_rho_r2 = np.zeros_like(rho_9q_r2)

                valid_indices_r2 = np.where(final_valid_mask_r2)[0]

                for val in set(valid_indices_r2 % 32):

                    branch_mask = np.array([(i % 32) == val for i in range(dim)])

                    if final_valid_mask_r2[np.where(branch_mask)[0][0]]:
                        vec = branch_mask.astype(float).reshape(-1, 1)

                        projected_rho_r2 += rho_9q_r2 * (vec @ vec.T)

                rho_current = _safe_clean(projected_rho_r2 / p_succ_r2)

            else:

                rho_current = _safe_clean(rho_9q_r2)

            # 装填第三轮起跑线

            rho_4q_r2_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r2 > 1e-10:

                rho_target_R3 = rho_4q_r2_success

                rho_resource_R3 = rho_4q_r2_success

            else:

                rho_target_R3 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_resource_R3 = rho_target_R3

            rho_8q_R3 = np.kron(rho_target_R3, rho_resource_R3)

            rho_current = np.kron(rho_8q_R3, np.outer(psi_anc, psi_anc.conj()))

            # =========================================================

            # 🚀 3.5 第三轮：相干相位校准 (Action 43 + Action 129: P2_MASK)

            # =========================================================

            ops_r3 = [

                cirq.CNOT(self.qubits[0], self.qubits[4]), cirq.CNOT(self.qubits[1], self.qubits[5]),

                cirq.CNOT(self.qubits[2], self.qubits[6]), cirq.CNOT(self.qubits[3], self.qubits[7]),

                cirq.S(self.qubits[5]), cirq.S(self.qubits[7])

            ]

            qc_r3 = _build_noisy_circuit(ops_r3)

            res_r3 = sim.simulate(qc_r3, initial_state=rho_current, qubit_order=self.qubits[:9])

            rho_9q_r3 = res_r3.final_density_matrix

            # Action 129: P2 MASK BOTH (Z 基底偶宇称，不加 H 门)

            diag_elements_r3 = np.diagonal(rho_9q_r3).real

            final_valid_mask_r3 = np.ones(dim, dtype=bool)

            final_valid_mask_r3 &= np.array([(i & 0b1) == 0 for i in range(dim)])  # q8 = 0

            allowed_res_p2_r3 = [0, 15]  # 0000 和 1111

            res_vals_r3 = np.array([((i >> 1) & 0b1111) for i in range(dim)])

            final_valid_mask_r3 &= np.isin(res_vals_r3, allowed_res_p2_r3)

            p_succ_r3 = np.sum(diag_elements_r3[final_valid_mask_r3])

            if p_succ_r3 > 1e-10:

                projected_rho_r3 = np.zeros_like(rho_9q_r3)

                valid_indices_r3 = np.where(final_valid_mask_r3)[0]

                for val in set(valid_indices_r3 % 32):

                    branch_mask = np.array([(i % 32) == val for i in range(dim)])

                    if final_valid_mask_r3[np.where(branch_mask)[0][0]]:
                        vec = branch_mask.astype(float).reshape(-1, 1)

                        projected_rho_r3 += rho_9q_r3 * (vec @ vec.T)

                rho_current = _safe_clean(projected_rho_r3 / p_succ_r3)

            else:

                rho_current = _safe_clean(rho_9q_r3)

            # =========================================================

            # 🚀 4. 装填终极第四轮起跑线，移交 RL Agent！

            # =========================================================

            rho_4q_r3_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r3 > 1e-10:

                rho_target_R4 = rho_4q_r3_success

                rho_resource_R4 = rho_4q_r3_success

            else:

                rho_target_R4 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_resource_R4 = rho_target_R4

            rho_8q_R4 = np.kron(rho_target_R4, rho_resource_R4)

            self.rho = np.kron(rho_8q_R4, np.outer(psi_anc, psi_anc.conj()))

            self.rho_initial = self.rho

            self.rho_4q = rho_target_R4

            self.F_in = self._get_ghz4_fidelity(self.rho_4q)

            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)

        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)

            # =========================================================
            # 🚀 0. 安全过滤装甲 (防 NoneType 崩溃补丁)
            # =========================================================
            def _safe_clean(rho_in):
                if rho_in is None: return None
                _res = self._clean_density_matrix(rho_in)
                return _res if _res is not None else rho_in

            # =========================================================
            # 1. 初始化 9-qubit 完美总态 (512 x 512)
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)

            rho_sys_base = self._generate_ghz4_state(F=f_param)
            rho_res_base = self._generate_ghz4_state(F=f_param)
            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            psi_anc = np.zeros(2, dtype=complex)
            psi_anc[0] = 1.0
            rho_anc = np.outer(psi_anc, psi_anc.conj())

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================
            # 🚀 2. 异构信道加噪
            # =========================================================
            qc_channel_noise = cirq.Circuit()

            noise_p_bob = self.np_random.uniform(0.15, 0.35)
            self.niose_bob = noise_p_bob
            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            noise_p_charlie = self.np_random.uniform(0.15, 0.35)
            self.niose_charlie = noise_p_charlie
            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================
            # 3. 核心大流水线开始
            # =========================================================
            sim = cirq.DensityMatrixSimulator()

            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])
            rho_after_noise = _safe_clean(res_noise.final_density_matrix)
            rho_current = rho_after_noise

            def _build_noisy_circuit(ops):
                noisy_circuit = cirq.Circuit()
                for op in ops:
                    noisy_circuit.append(op)
                    op_qubits = op.qubits
                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))
                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))
                return noisy_circuit

            dim = 512

            # =========================================================
            # 🚀 3.2 初始预处理 (Pre-selection)
            # =========================================================
            ops_pre = [
                cirq.CNOT(self.qubits[2], self.qubits[8]),
                cirq.CNOT(self.qubits[3], self.qubits[8])
            ]
            qc_pre = _build_noisy_circuit(ops_pre)
            res_pre = sim.simulate(qc_pre, initial_state=rho_current, qubit_order=self.qubits[:9])
            rho_pre = res_pre.final_density_matrix

            q8_mask = 1
            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0
            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:
                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)
                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T)
                rho_pre_post /= p_succ_pre
                rho_current = _safe_clean(rho_pre_post)

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_target_pre = rho_4q_pre_success
                rho_resource_pre = rho_4q_pre_success
            else:
                rho_target_pre = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_pre = rho_target_pre

            rho_8q_pre = np.kron(rho_target_pre, rho_resource_pre)
            rho_current = np.kron(rho_8q_pre, np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R1：拓扑感知粗筛 (Action 97 + Action 129: P2_MASK)
            # =========================================================
            ops_r1 = [
                cirq.CNOT(self.qubits[2], self.qubits[8]), cirq.CNOT(self.qubits[3], self.qubits[8]),
                cirq.CNOT(self.qubits[6], self.qubits[8]), cirq.CNOT(self.qubits[7], self.qubits[8])
            ]
            res_r1 = sim.simulate(_build_noisy_circuit(ops_r1), initial_state=rho_current, qubit_order=self.qubits[:9])
            rho_9q_r1 = res_r1.final_density_matrix

            final_valid_mask_r1 = np.ones(dim, dtype=bool)
            final_valid_mask_r1 &= np.array([(i & 0b1) == 0 for i in range(dim)])
            final_valid_mask_r1 &= np.isin(np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 15])  # P2 Z-even

            p_succ_r1 = np.sum(np.diagonal(rho_9q_r1).real[final_valid_mask_r1])
            if p_succ_r1 > 1e-10:
                projected_rho_r1 = np.zeros_like(rho_9q_r1)
                for val in set(np.where(final_valid_mask_r1)[0] % 32):
                    branch_mask = np.array([(i % 32) == val for i in range(dim)])
                    if final_valid_mask_r1[np.where(branch_mask)[0][0]]:
                        vec = branch_mask.astype(float).reshape(-1, 1)
                        projected_rho_r1 += rho_9q_r1 * (vec @ vec.T)
                rho_current = _safe_clean(projected_rho_r1 / p_succ_r1)
            else:
                rho_current = _safe_clean(rho_9q_r1)

            rho_4q_r1_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_target_R2 = rho_4q_r1_success if p_succ_r1 > 1e-10 else self.manual_partial_trace(rho_after_noise,
                                                                                                  keep_indices=[0, 1, 2,
                                                                                                                3],
                                                                                                  dims=[2] * 9)
            rho_current = np.kron(np.kron(rho_target_R2, rho_target_R2), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R2：精细相位嗅探 (Action 61 + Action 126: P1_MASK)
            # =========================================================
            ops_r2 = [
                cirq.CNOT(self.qubits[4], self.qubits[0]), cirq.CNOT(self.qubits[5], self.qubits[1]),
                cirq.CNOT(self.qubits[6], self.qubits[2]), cirq.CNOT(self.qubits[7], self.qubits[3]),
                cirq.CZ(self.qubits[0], self.qubits[4]), cirq.CZ(self.qubits[1], self.qubits[5])
            ]
            res_r2 = sim.simulate(_build_noisy_circuit(ops_r2), initial_state=rho_current, qubit_order=self.qubits[:9])

            res_basis_r2 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                        initial_state=res_r2.final_density_matrix, qubit_order=self.qubits[:9])
            rho_9q_r2 = res_basis_r2.final_density_matrix

            final_valid_mask_r2 = np.ones(dim, dtype=bool)
            final_valid_mask_r2 &= np.array([(i & 0b1) == 0 for i in range(dim)])
            final_valid_mask_r2 &= np.isin(np.array([((i >> 1) & 0b1111) for i in range(dim)]),
                                           [0, 3, 5, 6, 9, 10, 12, 15])  # P1 X-even

            p_succ_r2 = np.sum(np.diagonal(rho_9q_r2).real[final_valid_mask_r2])
            if p_succ_r2 > 1e-10:
                projected_rho_r2 = np.zeros_like(rho_9q_r2)
                for val in set(np.where(final_valid_mask_r2)[0] % 32):
                    branch_mask = np.array([(i % 32) == val for i in range(dim)])
                    if final_valid_mask_r2[np.where(branch_mask)[0][0]]:
                        vec = branch_mask.astype(float).reshape(-1, 1)
                        projected_rho_r2 += rho_9q_r2 * (vec @ vec.T)
                rho_current = _safe_clean(projected_rho_r2 / p_succ_r2)
            else:
                rho_current = _safe_clean(rho_9q_r2)

            rho_4q_r2_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_target_R3 = rho_4q_r2_success if p_succ_r2 > 1e-10 else self.manual_partial_trace(rho_after_noise,
                                                                                                  keep_indices=[0, 1, 2,
                                                                                                                3],
                                                                                                  dims=[2] * 9)
            rho_current = np.kron(np.kron(rho_target_R3, rho_target_R3), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R3：相干相位校准 (Action 43 + Action 129: P2_MASK)
            # =========================================================
            ops_r3 = [
                cirq.CNOT(self.qubits[0], self.qubits[4]), cirq.CNOT(self.qubits[1], self.qubits[5]),
                cirq.CNOT(self.qubits[2], self.qubits[6]), cirq.CNOT(self.qubits[3], self.qubits[7]),
                cirq.S(self.qubits[5]), cirq.S(self.qubits[7])
            ]
            res_r3 = sim.simulate(_build_noisy_circuit(ops_r3), initial_state=rho_current, qubit_order=self.qubits[:9])
            rho_9q_r3 = res_r3.final_density_matrix

            final_valid_mask_r3 = np.ones(dim, dtype=bool)
            final_valid_mask_r3 &= np.array([(i & 0b1) == 0 for i in range(dim)])
            final_valid_mask_r3 &= np.isin(np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 15])  # P2 Z-even

            p_succ_r3 = np.sum(np.diagonal(rho_9q_r3).real[final_valid_mask_r3])
            if p_succ_r3 > 1e-10:
                projected_rho_r3 = np.zeros_like(rho_9q_r3)
                for val in set(np.where(final_valid_mask_r3)[0] % 32):
                    branch_mask = np.array([(i % 32) == val for i in range(dim)])
                    if final_valid_mask_r3[np.where(branch_mask)[0][0]]:
                        vec = branch_mask.astype(float).reshape(-1, 1)
                        projected_rho_r3 += rho_9q_r3 * (vec @ vec.T)
                rho_current = _safe_clean(projected_rho_r3 / p_succ_r3)
            else:
                rho_current = _safe_clean(rho_9q_r3)

            rho_4q_r3_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_target_R4 = rho_4q_r3_success if p_succ_r3 > 1e-10 else self.manual_partial_trace(rho_after_noise,
                                                                                                  keep_indices=[0, 1, 2,
                                                                                                                3],
                                                                                                  dims=[2] * 9)
            rho_current = np.kron(np.kron(rho_target_R4, rho_target_R4), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R4：噪声感知非对称探测 (最新! Action 63 + Action 126: P1_MASK)
            # =========================================================
            ops_r4 = [
                cirq.CNOT(self.qubits[4], self.qubits[0]), cirq.CNOT(self.qubits[5], self.qubits[1]),
                cirq.CNOT(self.qubits[6], self.qubits[2]), cirq.CNOT(self.qubits[7], self.qubits[3]),
                # 惊世骇俗的非对称探测：完美避开有毒的 1,5 和 2,6！
                cirq.CZ(self.qubits[0], self.qubits[4]), cirq.CZ(self.qubits[3], self.qubits[7])
            ]
            res_r4 = sim.simulate(_build_noisy_circuit(ops_r4), initial_state=rho_current, qubit_order=self.qubits[:9])

            res_basis_r4 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                        initial_state=res_r4.final_density_matrix, qubit_order=self.qubits[:9])
            rho_9q_r4 = res_basis_r4.final_density_matrix

            final_valid_mask_r4 = np.ones(dim, dtype=bool)
            final_valid_mask_r4 &= np.array([(i & 0b1) == 0 for i in range(dim)])
            final_valid_mask_r4 &= np.isin(np.array([((i >> 1) & 0b1111) for i in range(dim)]),
                                           [0, 3, 5, 6, 9, 10, 12, 15])  # P1 X-even

            p_succ_r4 = np.sum(np.diagonal(rho_9q_r4).real[final_valid_mask_r4])
            if p_succ_r4 > 1e-10:
                projected_rho_r4 = np.zeros_like(rho_9q_r4)
                for val in set(np.where(final_valid_mask_r4)[0] % 32):
                    branch_mask = np.array([(i % 32) == val for i in range(dim)])
                    if final_valid_mask_r4[np.where(branch_mask)[0][0]]:
                        vec = branch_mask.astype(float).reshape(-1, 1)
                        projected_rho_r4 += rho_9q_r4 * (vec @ vec.T)
                rho_current = _safe_clean(projected_rho_r4 / p_succ_r4)
            else:
                rho_current = _safe_clean(rho_9q_r4)

            # =========================================================
            # 🚀 装填终极起跑线，移交 RL Agent！
            # =========================================================
            rho_4q_r4_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r4 > 1e-10:
                rho_target_R5 = rho_4q_r4_success
                rho_resource_R5 = rho_4q_r4_success
            else:
                rho_target_R5 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_R5 = rho_target_R5

            rho_8q_R5 = np.kron(rho_target_R5, rho_resource_R5)

            self.rho = np.kron(rho_8q_R5, np.outer(psi_anc, psi_anc.conj()))
            self.rho_initial = self.rho

            self.rho_4q = rho_target_R5
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)

        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)

            # =========================================================
            # 🚀 0. 安全过滤装甲 (防 NoneType 崩溃补丁)
            # =========================================================
            def _safe_clean(rho_in):
                if rho_in is None: return None
                _res = self._clean_density_matrix(rho_in)
                return _res if _res is not None else rho_in

            # =========================================================
            # 1. 初始化 9-qubit 完美总态 (512 x 512)
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)

            rho_sys_base = self._generate_ghz4_state(F=f_param)
            rho_res_base = self._generate_ghz4_state(F=f_param)
            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            psi_anc = np.zeros(2, dtype=complex)
            psi_anc[0] = 1.0
            rho_anc = np.outer(psi_anc, psi_anc.conj())

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================
            # 🚀 2. 异构信道加噪
            # =========================================================
            qc_channel_noise = cirq.Circuit()

            noise_p_bob = self.np_random.uniform(0.05, 0.3)
            self.niose_bob = noise_p_bob
            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            noise_p_charlie = self.np_random.uniform(0.05, 0.3)
            self.niose_charlie = noise_p_charlie
            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================
            # 3. 核心大流水线开始
            # =========================================================
            sim = cirq.DensityMatrixSimulator()

            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])
            rho_current = _safe_clean(res_noise.final_density_matrix)

            def _build_noisy_circuit(ops):
                noisy_circuit = cirq.Circuit()
                for op in ops:
                    noisy_circuit.append(op)
                    op_qubits = op.qubits
                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))
                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))
                return noisy_circuit

            dim = 512

            # =========================================================
            # 🚀 R0: 初始预处理 (Pre-selection)
            # =========================================================
            ops_pre = [cirq.CNOT(self.qubits[2], self.qubits[8]), cirq.CNOT(self.qubits[3], self.qubits[8])]
            res_pre = sim.simulate(_build_noisy_circuit(ops_pre), initial_state=rho_current,
                                   qubit_order=self.qubits[:9])
            rho_pre = res_pre.final_density_matrix

            q8_mask = 1
            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0
            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:
                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)
                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T) / p_succ_pre
                rho_current = _safe_clean(rho_pre_post)

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_target_pre = rho_4q_pre_success;
                rho_resource_pre = rho_4q_pre_success
            else:
                rho_target_pre = self.manual_partial_trace(_safe_clean(res_noise.final_density_matrix),
                                                           keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_pre = rho_target_pre

            rho_current = np.kron(np.kron(rho_target_pre, rho_resource_pre), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R1: 拓扑感知粗筛 (Action 97 + 129: P2_MASK)
            # =========================================================
            ops_r1 = [cirq.CNOT(self.qubits[2], self.qubits[8]), cirq.CNOT(self.qubits[3], self.qubits[8]),
                      cirq.CNOT(self.qubits[6], self.qubits[8]), cirq.CNOT(self.qubits[7], self.qubits[8])]
            res_r1 = sim.simulate(_build_noisy_circuit(ops_r1), initial_state=rho_current, qubit_order=self.qubits[:9])
            rho_9q_r1 = res_r1.final_density_matrix

            mask_r1 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 15])
            p_succ_r1 = np.sum(np.diagonal(rho_9q_r1).real[mask_r1])
            if p_succ_r1 > 1e-10:
                proj_r1 = np.zeros_like(rho_9q_r1)
                for val in set(np.where(mask_r1)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r1[np.where(bm)[0][0]]: vec = bm.astype(float).reshape(-1, 1); proj_r1 += rho_9q_r1 * (
                                vec @ vec.T)
                rho_current = _safe_clean(proj_r1 / p_succ_r1)
            else:
                rho_current = _safe_clean(rho_9q_r1)

            r1_suc = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_target_R2 = r1_suc if p_succ_r1 > 1e-10 else self.manual_partial_trace(
                _safe_clean(res_noise.final_density_matrix), keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_current = np.kron(np.kron(rho_target_R2, rho_target_R2), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R2: 精细相位嗅探 (Action 61 + 126: P1_MASK)
            # =========================================================
            ops_r2 = [cirq.CNOT(self.qubits[4], self.qubits[0]), cirq.CNOT(self.qubits[5], self.qubits[1]),
                      cirq.CNOT(self.qubits[6], self.qubits[2]), cirq.CNOT(self.qubits[7], self.qubits[3]),
                      cirq.CZ(self.qubits[0], self.qubits[4]), cirq.CZ(self.qubits[1], self.qubits[5])]
            res_r2 = sim.simulate(_build_noisy_circuit(ops_r2), initial_state=rho_current, qubit_order=self.qubits[:9])
            rho_9q_r2 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                     initial_state=res_r2.final_density_matrix,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            mask_r2 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 3, 5, 6, 9, 10, 12, 15])
            p_succ_r2 = np.sum(np.diagonal(rho_9q_r2).real[mask_r2])
            if p_succ_r2 > 1e-10:
                proj_r2 = np.zeros_like(rho_9q_r2)
                for val in set(np.where(mask_r2)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r2[np.where(bm)[0][0]]: vec = bm.astype(float).reshape(-1, 1); proj_r2 += rho_9q_r2 * (
                                vec @ vec.T)
                rho_current = _safe_clean(proj_r2 / p_succ_r2)
            else:
                rho_current = _safe_clean(rho_9q_r2)

            r2_suc = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_target_R3 = r2_suc if p_succ_r2 > 1e-10 else self.manual_partial_trace(
                _safe_clean(res_noise.final_density_matrix), keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_current = np.kron(np.kron(rho_target_R3, rho_target_R3), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R3: 相干相位校准 (Action 43 + 129: P2_MASK)
            # =========================================================
            ops_r3 = [cirq.CNOT(self.qubits[0], self.qubits[4]), cirq.CNOT(self.qubits[1], self.qubits[5]),
                      cirq.CNOT(self.qubits[2], self.qubits[6]), cirq.CNOT(self.qubits[3], self.qubits[7]),
                      cirq.S(self.qubits[5]), cirq.S(self.qubits[7])]
            rho_9q_r3 = sim.simulate(_build_noisy_circuit(ops_r3), initial_state=rho_current,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            mask_r3 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 15])
            p_succ_r3 = np.sum(np.diagonal(rho_9q_r3).real[mask_r3])
            if p_succ_r3 > 1e-10:
                proj_r3 = np.zeros_like(rho_9q_r3)
                for val in set(np.where(mask_r3)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r3[np.where(bm)[0][0]]: vec = bm.astype(float).reshape(-1, 1); proj_r3 += rho_9q_r3 * (
                                vec @ vec.T)
                rho_current = _safe_clean(proj_r3 / p_succ_r3)
            else:
                rho_current = _safe_clean(rho_9q_r3)

            r3_suc = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_target_R4 = r3_suc if p_succ_r3 > 1e-10 else self.manual_partial_trace(
                _safe_clean(res_noise.final_density_matrix), keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_current = np.kron(np.kron(rho_target_R4, rho_target_R4), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R4: 噪声感知非对称探测 (Action 63 + 126: P1_MASK)
            # =========================================================
            ops_r4 = [cirq.CNOT(self.qubits[4], self.qubits[0]), cirq.CNOT(self.qubits[5], self.qubits[1]),
                      cirq.CNOT(self.qubits[6], self.qubits[2]), cirq.CNOT(self.qubits[7], self.qubits[3]),
                      cirq.CZ(self.qubits[0], self.qubits[4]), cirq.CZ(self.qubits[3], self.qubits[7])]
            res_r4 = sim.simulate(_build_noisy_circuit(ops_r4), initial_state=rho_current, qubit_order=self.qubits[:9])
            rho_9q_r4 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                     initial_state=res_r4.final_density_matrix,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            mask_r4 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 3, 5, 6, 9, 10, 12, 15])
            p_succ_r4 = np.sum(np.diagonal(rho_9q_r4).real[mask_r4])
            if p_succ_r4 > 1e-10:
                proj_r4 = np.zeros_like(rho_9q_r4)
                for val in set(np.where(mask_r4)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r4[np.where(bm)[0][0]]: vec = bm.astype(float).reshape(-1, 1); proj_r4 += rho_9q_r4 * (
                                vec @ vec.T)
                rho_current = _safe_clean(proj_r4 / p_succ_r4)
            else:
                rho_current = _safe_clean(rho_9q_r4)

            r4_suc = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_target_R5 = r4_suc if p_succ_r4 > 1e-10 else self.manual_partial_trace(
                _safe_clean(res_noise.final_density_matrix), keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_current = np.kron(np.kron(rho_target_R5, rho_target_R5), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R5: 终极微创本地体检 (最新! Action 96 + Action 127: P2_MASK_ANC)
            # =========================================================
            ops_r5 = [cirq.CNOT(self.qubits[2], self.qubits[8]), cirq.CNOT(self.qubits[3], self.qubits[8])]
            rho_9q_r5 = sim.simulate(_build_noisy_circuit(ops_r5), initial_state=rho_current,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            # Action 127: P2_MASK_ANC (仅校验 ancilla q8=0，完全解放 Resource！)
            final_valid_mask_r5 = np.ones(dim, dtype=bool)
            final_valid_mask_r5 &= np.array([(i & 0b1) == 0 for i in range(dim)])

            p_succ_r5 = np.sum(np.diagonal(rho_9q_r5).real[final_valid_mask_r5])
            if p_succ_r5 > 1e-10:
                projected_rho_r5 = np.zeros_like(rho_9q_r5)
                for val in set(np.where(final_valid_mask_r5)[0] % 32):
                    branch_mask = np.array([(i % 32) == val for i in range(dim)])
                    if final_valid_mask_r5[np.where(branch_mask)[0][0]]:
                        vec = branch_mask.astype(float).reshape(-1, 1)
                        projected_rho_r5 += rho_9q_r5 * (vec @ vec.T)
                rho_current = _safe_clean(projected_rho_r5 / p_succ_r5)
            else:
                rho_current = _safe_clean(rho_9q_r5)

            # =========================================================
            # 🚀 装填终极起跑线，移交 RL Agent！
            # =========================================================
            rho_4q_r5_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r5 > 1e-10:
                rho_target_R6 = rho_4q_r5_success
                rho_resource_R6 = rho_4q_r5_success
            else:
                rho_target_R6 = self.manual_partial_trace(_safe_clean(res_noise.final_density_matrix),
                                                          keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_R6 = rho_target_R6

            rho_8q_R6 = np.kron(rho_target_R6, rho_resource_R6)

            self.rho = np.kron(rho_8q_R6, np.outer(psi_anc, psi_anc.conj()))
            self.rho_initial = self.rho

            self.rho_4q = rho_target_R6
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)

        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)——RE VER.

            # =========================================================
            # 🚀 0. 安全过滤装甲 (防 NoneType 崩溃补丁)
            # =========================================================
            def _safe_clean(rho_in):
                if rho_in is None: return None
                _res = self._clean_density_matrix(rho_in)
                return _res if _res is not None else rho_in

            # =========================================================
            # 1. 初始化 9-qubit 完美总态 (512 x 512)
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)

            rho_sys_base = self._generate_ghz4_state(F=f_param)
            rho_res_base = self._generate_ghz4_state(F=f_param)
            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            psi_anc = np.zeros(2, dtype=complex)
            psi_anc[0] = 1.0
            rho_anc = np.outer(psi_anc, psi_anc.conj())

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================
            # 🚀 2. 异构信道加噪
            # =========================================================
            qc_channel_noise = cirq.Circuit()

            noise_p_bob = self.np_random.uniform(0.05, 0.3)
            self.niose_bob = noise_p_bob
            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            noise_p_charlie = self.np_random.uniform(0.05, 0.3)
            self.niose_charlie = noise_p_charlie
            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================
            # 3. 核心大流水线开始
            # =========================================================
            sim = cirq.DensityMatrixSimulator()

            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])
            rho_after_noise = _safe_clean(res_noise.final_density_matrix)
            rho_current = rho_after_noise

            def _build_noisy_circuit(ops):
                noisy_circuit = cirq.Circuit()
                for op in ops:
                    noisy_circuit.append(op)
                    op_qubits = op.qubits
                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))
                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))
                return noisy_circuit

            dim = 512

            # =========================================================
            # 🚀 R0: 初始预处理 (Pre-selection)
            # =========================================================
            ops_pre = [cirq.CNOT(self.qubits[2], self.qubits[8]), cirq.CNOT(self.qubits[3], self.qubits[8])]
            res_pre = sim.simulate(_build_noisy_circuit(ops_pre), initial_state=rho_current,
                                   qubit_order=self.qubits[:9])
            rho_pre = res_pre.final_density_matrix

            q8_mask = 1
            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0
            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:
                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)
                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T) / p_succ_pre
                rho_current = _safe_clean(rho_pre_post)

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_target_pre = rho_4q_pre_success
                rho_resource_pre = rho_4q_pre_success
            else:
                rho_target_pre = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_pre = rho_target_pre

            rho_current = np.kron(np.kron(rho_target_pre, rho_resource_pre), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 第一轮：精细相位嗅探 (即原来的 R2: Action 61 + 126 P1_MASK)
            # =========================================================
            ops_r1 = [
                cirq.CNOT(self.qubits[4], self.qubits[0]), cirq.CNOT(self.qubits[5], self.qubits[1]),
                cirq.CNOT(self.qubits[6], self.qubits[2]), cirq.CNOT(self.qubits[7], self.qubits[3]),
                cirq.CZ(self.qubits[0], self.qubits[4]), cirq.CZ(self.qubits[1], self.qubits[5])
            ]
            res_r1 = sim.simulate(_build_noisy_circuit(ops_r1), initial_state=rho_current, qubit_order=self.qubits[:9])
            rho_9q_r1 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                     initial_state=res_r1.final_density_matrix,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            mask_r1 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 3, 5, 6, 9, 10, 12, 15])
            p_succ_r1 = np.sum(np.diagonal(rho_9q_r1).real[mask_r1])

            if p_succ_r1 > 1e-10:
                proj_r1 = np.zeros_like(rho_9q_r1)
                for val in set(np.where(mask_r1)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r1[np.where(bm)[0][0]]: vec = bm.astype(float).reshape(-1, 1); proj_r1 += rho_9q_r1 * (
                                vec @ vec.T)
                rho_current = _safe_clean(proj_r1 / p_succ_r1)
            else:
                rho_current = _safe_clean(rho_9q_r1)

            # =========================================================
            # 🚀 装填终极起跑线，移交 RL Agent！
            # =========================================================
            rho_4q_r1_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r1 > 1e-10:
                rho_target_R2 = rho_4q_r1_success
                rho_resource_R2 = rho_4q_r1_success
            else:
                rho_target_R2 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_R2 = rho_target_R2

            rho_8q_R2 = np.kron(rho_target_R2, rho_resource_R2)

            self.rho = np.kron(rho_8q_R2, np.outer(psi_anc, psi_anc.conj()))
            self.rho_initial = self.rho

            self.rho_4q = rho_target_R2
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)


        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)

            # =========================================================

            # 🚀 0. 安全过滤装甲 (防 NoneType 崩溃补丁)

            # =========================================================

            def _safe_clean(rho_in):

                if rho_in is None: return None

                _res = self._clean_density_matrix(rho_in)

                return _res if _res is not None else rho_in

            # =========================================================

            # 1. 初始化 9-qubit 完美总态 (512 x 512)

            # =========================================================

            f_param = self.np_random.uniform(0.95, 1.0)

            rho_sys_base = self._generate_ghz4_state(F=f_param)

            rho_res_base = self._generate_ghz4_state(F=f_param)

            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            psi_anc = np.zeros(2, dtype=complex)

            psi_anc[0] = 1.0

            rho_anc = np.outer(psi_anc, psi_anc.conj())

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================

            # 🚀 2. 异构信道加噪

            # =========================================================

            qc_channel_noise = cirq.Circuit()

            noise_p_bob = self.np_random.uniform(0.01, 0.3)

            self.niose_bob = noise_p_bob

            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            noise_p_charlie = self.np_random.uniform(0.01, 0.3)

            self.niose_charlie = noise_p_charlie

            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================

            # 3. 核心大流水线开始

            # =========================================================

            sim = cirq.DensityMatrixSimulator()

            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])

            rho_after_noise = _safe_clean(res_noise.final_density_matrix)

            rho_current = rho_after_noise

            def _build_noisy_circuit(ops):

                noisy_circuit = cirq.Circuit()

                for op in ops:

                    noisy_circuit.append(op)

                    op_qubits = op.qubits

                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:

                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))

                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:

                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))

                return noisy_circuit

            dim = 512

            # =========================================================

            # 🚀 R0: 初始预处理 (Pre-selection)

            # =========================================================

            ops_pre = [cirq.CNOT(self.qubits[2], self.qubits[8]), cirq.CNOT(self.qubits[3], self.qubits[8])]

            res_pre = sim.simulate(_build_noisy_circuit(ops_pre), initial_state=rho_current,
                                   qubit_order=self.qubits[:9])

            rho_pre = res_pre.final_density_matrix

            q8_mask = 1

            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0

            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:

                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)

                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T) / p_succ_pre

                rho_current = _safe_clean(rho_pre_post)

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_target_pre = rho_4q_pre_success

                rho_resource_pre = rho_4q_pre_success

            else:

                rho_target_pre = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_resource_pre = rho_target_pre

            rho_current = np.kron(np.kron(rho_target_pre, rho_resource_pre), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================

            # 🚀 第一轮：标准纠缠泵浦 (Action 48 + 126 P1_MASK)

            # =========================================================

            ops_r1 = [

                cirq.CNOT(self.qubits[4], self.qubits[0]),

                cirq.CNOT(self.qubits[5], self.qubits[1]),

                cirq.CNOT(self.qubits[6], self.qubits[2]),

                cirq.CNOT(self.qubits[7], self.qubits[3])

            ]

            res_r1 = sim.simulate(_build_noisy_circuit(ops_r1), initial_state=rho_current, qubit_order=self.qubits[:9])

            # Action 126 P1 模式，需要先加 H 门切换到 X 基底

            rho_9q_r1 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                     initial_state=res_r1.final_density_matrix,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            # MASK_BOTH: ANC(q8)=0 且 RES(q4,q5,q6,q7) 为 X基底偶宇称

            mask_r1 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 3, 5, 6, 9, 10, 12, 15])

            p_succ_r1 = np.sum(np.diagonal(rho_9q_r1).real[mask_r1])

            if p_succ_r1 > 1e-10:

                proj_r1 = np.zeros_like(rho_9q_r1)

                for val in set(np.where(mask_r1)[0] % 32):

                    bm = np.array([(i % 32) == val for i in range(dim)])

                    if mask_r1[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1)

                        proj_r1 += rho_9q_r1 * (vec @ vec.T)

                rho_current = _safe_clean(proj_r1 / p_succ_r1)

            else:

                rho_current = _safe_clean(rho_9q_r1)

            # =========================================================

            # 🚀 装填终极起跑线，移交 RL Agent！

            # =========================================================

            rho_4q_r1_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r1 > 1e-10:

                rho_target_R2 = rho_4q_r1_success

                rho_resource_R2 = rho_4q_r1_success

            else:

                rho_target_R2 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

                rho_resource_R2 = rho_target_R2

            rho_8q_R2 = np.kron(rho_target_R2, rho_resource_R2)

            self.rho = np.kron(rho_8q_R2, np.outer(psi_anc, psi_anc.conj()))

            self.rho_initial = self.rho

            self.rho_4q = rho_target_R2

            self.F_in = self._get_ghz4_fidelity(self.rho_4q)

            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)

        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)

            # =========================================================
            # 🚀 0. 安全过滤装甲 (防 NoneType 崩溃补丁)
            # =========================================================
            def _safe_clean(rho_in):
                if rho_in is None: return None
                _res = self._clean_density_matrix(rho_in)
                return _res if _res is not None else rho_in

            # =========================================================
            # 1. 初始化 9-qubit 完美总态 (512 x 512)
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)

            rho_sys_base = self._generate_ghz4_state(F=f_param)
            rho_res_base = self._generate_ghz4_state(F=f_param)
            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            psi_anc = np.zeros(2, dtype=complex)
            psi_anc[0] = 1.0
            rho_anc = np.outer(psi_anc, psi_anc.conj())

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================
            # 🚀 2. 异构信道加噪
            # =========================================================
            qc_channel_noise = cirq.Circuit()

            noise_p_bob = self.np_random.uniform(0.05, 0.3)
            self.niose_bob = noise_p_bob
            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            noise_p_charlie = self.np_random.uniform(0.05, 0.3)
            self.niose_charlie = noise_p_charlie
            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================
            # 3. 核心大流水线开始
            # =========================================================
            sim = cirq.DensityMatrixSimulator()

            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])
            rho_after_noise = _safe_clean(res_noise.final_density_matrix)
            rho_current = rho_after_noise

            def _build_noisy_circuit(ops):
                noisy_circuit = cirq.Circuit()
                for op in ops:
                    noisy_circuit.append(op)
                    op_qubits = op.qubits
                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))
                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))
                return noisy_circuit

            dim = 512

            # =========================================================
            # 🚀 R0: 初始预处理 (Pre-selection)
            # =========================================================
            ops_pre = [cirq.CNOT(self.qubits[2], self.qubits[8]), cirq.CNOT(self.qubits[3], self.qubits[8])]
            res_pre = sim.simulate(_build_noisy_circuit(ops_pre), initial_state=rho_current,
                                   qubit_order=self.qubits[:9])
            rho_pre = res_pre.final_density_matrix

            q8_mask = 1
            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0
            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:
                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)
                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T) / p_succ_pre
                rho_current = _safe_clean(rho_pre_post)

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_target_pre = rho_4q_pre_success
                rho_resource_pre = rho_4q_pre_success
            else:
                rho_target_pre = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_pre = rho_target_pre

            rho_current = np.kron(np.kron(rho_target_pre, rho_resource_pre), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R1：标准纠缠泵浦 (Action 48 + 126 P1_MASK_BOTH)
            # =========================================================
            ops_r1 = [
                cirq.CNOT(self.qubits[4], self.qubits[0]),
                cirq.CNOT(self.qubits[5], self.qubits[1]),
                cirq.CNOT(self.qubits[6], self.qubits[2]),
                cirq.CNOT(self.qubits[7], self.qubits[3])
            ]
            res_r1 = sim.simulate(_build_noisy_circuit(ops_r1), initial_state=rho_current, qubit_order=self.qubits[:9])

            rho_9q_r1 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                     initial_state=res_r1.final_density_matrix,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            mask_r1 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 3, 5, 6, 9, 10, 12, 15])
            p_succ_r1 = np.sum(np.diagonal(rho_9q_r1).real[mask_r1])

            if p_succ_r1 > 1e-10:
                proj_r1 = np.zeros_like(rho_9q_r1)
                for val in set(np.where(mask_r1)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r1[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1)
                        proj_r1 += rho_9q_r1 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r1 / p_succ_r1)
            else:
                rho_current = _safe_clean(rho_9q_r1)

            rho_4q_r1_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r1 > 1e-10:
                rho_target_R2 = rho_4q_r1_success
                rho_resource_R2 = rho_4q_r1_success
            else:
                rho_target_R2 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_R2 = rho_target_R2

            rho_8q_R2 = np.kron(rho_target_R2, rho_resource_R2)
            rho_current = np.kron(rho_8q_R2, np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R2：相干相位校准 (Action 44 + 128 P2_MASK_RES)
            # =========================================================
            ops_r2 = [
                cirq.CNOT(self.qubits[0], self.qubits[4]),
                cirq.CNOT(self.qubits[1], self.qubits[5]),
                cirq.CNOT(self.qubits[2], self.qubits[6]),
                cirq.CNOT(self.qubits[3], self.qubits[7]),
                cirq.S(self.qubits[6]),
                cirq.S(self.qubits[7])
            ]
            res_r2 = sim.simulate(_build_noisy_circuit(ops_r2), initial_state=rho_current, qubit_order=self.qubits[:9])

            # P2: Z 基底直接测量，无需 H 门
            rho_9q_r2 = res_r2.final_density_matrix

            # MASK_RES: 忽略辅助比特 q8，仅校验 Resource (q4-q7) 的 Z 基底偶宇称
            mask_r2 = np.isin(np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 15])
            p_succ_r2 = np.sum(np.diagonal(rho_9q_r2).real[mask_r2])

            if p_succ_r2 > 1e-10:
                proj_r2 = np.zeros_like(rho_9q_r2)
                for val in set(np.where(mask_r2)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r2[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1)
                        proj_r2 += rho_9q_r2 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r2 / p_succ_r2)
            else:
                rho_current = _safe_clean(rho_9q_r2)

            # =========================================================
            # 🚀 装填终极起跑线，移交 RL Agent！
            # =========================================================
            rho_4q_r2_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r2 > 1e-10:
                rho_target_R3 = rho_4q_r2_success
                rho_resource_R3 = rho_4q_r2_success
            else:
                rho_target_R3 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_R3 = rho_target_R3

            rho_8q_R3 = np.kron(rho_target_R3, rho_resource_R3)

            self.rho = np.kron(rho_8q_R3, np.outer(psi_anc, psi_anc.conj()))
            self.rho_initial = self.rho

            self.rho_4q = rho_target_R3
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)

        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)

            # =========================================================
            # 🚀 0. 安全过滤装甲 (防 NoneType 崩溃补丁)
            # =========================================================
            def _safe_clean(rho_in):
                if rho_in is None: return None
                _res = self._clean_density_matrix(rho_in)
                return _res if _res is not None else rho_in

            # =========================================================
            # 1. 初始化 9-qubit 完美总态 (512 x 512)
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)

            rho_sys_base = self._generate_ghz4_state(F=f_param)
            rho_res_base = self._generate_ghz4_state(F=f_param)
            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            psi_anc = np.zeros(2, dtype=complex)
            psi_anc[0] = 1.0
            rho_anc = np.outer(psi_anc, psi_anc.conj())

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================
            # 🚀 2. 异构信道加噪
            # =========================================================
            qc_channel_noise = cirq.Circuit()

            noise_p_bob = self.np_random.uniform(0.05, 0.3)
            self.niose_bob = noise_p_bob
            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            noise_p_charlie = self.np_random.uniform(0.05, 0.3)
            self.niose_charlie = noise_p_charlie
            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================
            # 3. 核心流水线：加噪 -> 不对称预装填
            # =========================================================
            sim = cirq.DensityMatrixSimulator()

            # --- 3.1 演化初始信道加噪 ---
            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])
            rho_after_noise = _safe_clean(res_noise.final_density_matrix)
            rho_current = rho_after_noise

            def _build_noisy_circuit(ops):
                noisy_circuit = cirq.Circuit()
                for op in ops:
                    noisy_circuit.append(op)
                    op_qubits = op.qubits
                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))
                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))
                return noisy_circuit

            dim = 512

            # 🚀 提前提取未处理的原始加噪态 (作为一会儿的 Resource)
            rho_raw_4q = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            # =========================================================
            # 🚀 3.2 初始预处理 (Pre-selection)
            # =========================================================
            ops_pre = [cirq.CNOT(self.qubits[2], self.qubits[8]), cirq.CNOT(self.qubits[3], self.qubits[8])]
            res_pre = sim.simulate(_build_noisy_circuit(ops_pre), initial_state=rho_current,
                                   qubit_order=self.qubits[:9])
            rho_pre = res_pre.final_density_matrix

            q8_mask = 1
            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0
            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:
                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)
                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T) / p_succ_pre
                rho_current = _safe_clean(rho_pre_post)

                # Target 采用预过滤成功的干净态
                rho_target_pre = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                # Resource 采用刚刚提取的未处理原始脏态
                rho_resource_pre = rho_raw_4q
            else:
                rho_target_pre = rho_raw_4q
                rho_resource_pre = rho_raw_4q

            # =========================================================
            # 🚀 4. 装填不对称起跑线，移交 RL Agent！
            # =========================================================
            rho_8q_asymmetric = np.kron(rho_target_pre, rho_resource_pre)

            self.rho = np.kron(rho_8q_asymmetric, np.outer(psi_anc, psi_anc.conj()))
            self.rho_initial = self.rho

            # Baseline 以过滤过的 Target 为准
            self.rho_4q = rho_target_pre
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)

        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)

            # =========================================================
            # 🚀 0. 安全过滤装甲 (防 NoneType 崩溃补丁)
            # =========================================================
            def _safe_clean(rho_in):
                if rho_in is None: return None
                _res = self._clean_density_matrix(rho_in)
                return _res if _res is not None else rho_in

            # =========================================================
            # 1. 初始化 9-qubit 完美总态 (512 x 512)
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)

            rho_sys_base = self._generate_ghz4_state(F=f_param)
            rho_res_base = self._generate_ghz4_state(F=f_param)
            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            psi_anc = np.zeros(2, dtype=complex)
            psi_anc[0] = 1.0
            rho_anc = np.outer(psi_anc, psi_anc.conj())

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================
            # 🚀 2. 异构信道加噪
            # =========================================================
            qc_channel_noise = cirq.Circuit()

            noise_p_bob = self.np_random.uniform(0.01, 0.3)
            self.niose_bob = noise_p_bob
            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            noise_p_charlie = self.np_random.uniform(0.01, 0.3)
            self.niose_charlie = noise_p_charlie
            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================
            # 3. 核心大流水线开始
            # =========================================================
            sim = cirq.DensityMatrixSimulator()

            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])
            rho_after_noise = _safe_clean(res_noise.final_density_matrix)
            rho_current = rho_after_noise

            def _build_noisy_circuit(ops):
                noisy_circuit = cirq.Circuit()
                for op in ops:
                    noisy_circuit.append(op)
                    op_qubits = op.qubits
                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))
                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))
                return noisy_circuit

            dim = 512

            # =========================================================
            # 🚀 R0: 初始预处理 (Pre-selection)
            # =========================================================
            ops_pre = [cirq.CNOT(self.qubits[2], self.qubits[8]), cirq.CNOT(self.qubits[3], self.qubits[8])]
            res_pre = sim.simulate(_build_noisy_circuit(ops_pre), initial_state=rho_current,
                                   qubit_order=self.qubits[:9])
            rho_pre = res_pre.final_density_matrix

            q8_mask = 1
            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0
            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:
                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)
                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T) / p_succ_pre
                rho_current = _safe_clean(rho_pre_post)

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_target_pre = rho_4q_pre_success
                rho_resource_pre = rho_4q_pre_success
            else:
                rho_target_pre = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_pre = rho_target_pre

            rho_current = np.kron(np.kron(rho_target_pre, rho_resource_pre), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R1：标准纠缠泵浦 (Action 48 + 126 P1_MASK_BOTH)
            # =========================================================
            ops_r1 = [
                cirq.CNOT(self.qubits[4], self.qubits[0]),
                cirq.CNOT(self.qubits[5], self.qubits[1]),
                cirq.CNOT(self.qubits[6], self.qubits[2]),
                cirq.CNOT(self.qubits[7], self.qubits[3])
            ]
            res_r1 = sim.simulate(_build_noisy_circuit(ops_r1), initial_state=rho_current, qubit_order=self.qubits[:9])

            rho_9q_r1 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                     initial_state=res_r1.final_density_matrix,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            mask_r1 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 3, 5, 6, 9, 10, 12, 15])
            p_succ_r1 = np.sum(np.diagonal(rho_9q_r1).real[mask_r1])

            if p_succ_r1 > 1e-10:
                proj_r1 = np.zeros_like(rho_9q_r1)
                for val in set(np.where(mask_r1)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r1[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1)
                        proj_r1 += rho_9q_r1 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r1 / p_succ_r1)
            else:
                rho_current = _safe_clean(rho_9q_r1)

            rho_4q_r1_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r1 > 1e-10:
                rho_target_R2 = rho_4q_r1_success
                rho_resource_R2 = rho_4q_r1_success
            else:
                rho_target_R2 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_R2 = rho_target_R2

            rho_8q_R2 = np.kron(rho_target_R2, rho_resource_R2)
            rho_current = np.kron(rho_8q_R2, np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R2：反向宇称泵浦 (最新修改! Action 32 + Action 129 P2_MASK_BOTH)
            # =========================================================
            ops_r2 = [
                cirq.CNOT(self.qubits[0], self.qubits[4]),
                cirq.CNOT(self.qubits[1], self.qubits[5]),
                cirq.CNOT(self.qubits[2], self.qubits[6]),
                cirq.CNOT(self.qubits[3], self.qubits[7])
            ]
            res_r2 = sim.simulate(_build_noisy_circuit(ops_r2), initial_state=rho_current, qubit_order=self.qubits[:9])

            # P2 模式：Z 基底直接测量，无需施加 H 门
            rho_9q_r2 = res_r2.final_density_matrix

            # MASK_BOTH: 要求辅助比特 q8=0 且 Resource (q4-q7) 为 Z 基底偶宇称 (0000 或 1111)
            mask_r2 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 15])
            p_succ_r2 = np.sum(np.diagonal(rho_9q_r2).real[mask_r2])

            if p_succ_r2 > 1e-10:
                proj_r2 = np.zeros_like(rho_9q_r2)
                for val in set(np.where(mask_r2)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r2[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1)
                        proj_r2 += rho_9q_r2 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r2 / p_succ_r2)
            else:
                rho_current = _safe_clean(rho_9q_r2)

            # =========================================================
            # 🚀 4. 装填终极第三轮起跑线，移交 RL Agent！
            # =========================================================
            rho_4q_r2_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r2 > 1e-10:
                rho_target_R3 = rho_4q_r2_success
                rho_resource_R3 = rho_4q_r2_success
            else:
                rho_target_R3 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_R3 = rho_target_R3

            rho_8q_R3 = np.kron(rho_target_R3, rho_resource_R3)

            self.rho = np.kron(rho_8q_R3, np.outer(psi_anc, psi_anc.conj()))
            self.rho_initial = self.rho

            self.rho_4q = rho_target_R3
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)


        elif rand_scenario < 1:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)

            # =========================================================
            # 🚀 0. 安全过滤装甲 (防 NoneType 崩溃补丁)
            # =========================================================
            def _safe_clean(rho_in):
                if rho_in is None: return None
                _res = self._clean_density_matrix(rho_in)
                return _res if _res is not None else rho_in

            # =========================================================
            # 1. 初始化 9-qubit 完美总态 (512 x 512)
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)

            rho_sys_base = self._generate_ghz4_state(F=f_param)
            rho_res_base = self._generate_ghz4_state(F=f_param)
            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            psi_anc = np.zeros(2, dtype=complex)
            psi_anc[0] = 1.0
            rho_anc = np.outer(psi_anc, psi_anc.conj())

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================
            # 🚀 2. 异构信道加噪
            # =========================================================
            qc_channel_noise = cirq.Circuit()

            noise_p_bob = 0.0
            self.niose_bob = noise_p_bob
            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            noise_p_charlie = 0.0
            self.niose_charlie = noise_p_charlie
            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================
            # 3. 核心大流水线开始
            # =========================================================
            sim = cirq.DensityMatrixSimulator()

            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])
            rho_after_noise = _safe_clean(res_noise.final_density_matrix)
            rho_current = rho_after_noise

            # 1|1|2 的训练从级联协议的中间态开始：
            # local GHZ4 --d1--> 1|3 --双辅助比特纯化--> --d2--> 1|1|2。
            # 这里只在固定 1|1|2 拓扑时启用，避免改变其余动态/固定拓扑的 reset 行为。
            if self.fixed_topology == "1|1|2":
                route_delta1 = float(self.topology_total_noise_p)
                route_delta2 = float(self.np_random.uniform(
                    self.topology_total_noise_min,
                    self.topology_total_noise_max,
                ))

                # d1：从源节点分发至 1|3；q1、q2、q3 经过同一段信道。
                rho_13_input = self._apply_depolarizing_noise_to_rho(
                    rho_sys_base,
                    num_qubits=4,
                    noise_by_index={1: route_delta1, 2: route_delta1, 3: route_delta1},
                )
                if rho_13_input is None:
                    rho_13_input = rho_sys_base

                # 在 1|3 节点内用两个干净辅助比特分别测 Z1Z2、Z2Z3，
                # 仅保留双 syndrome 为 00 的分支。
                pre_qubits = cirq.LineQubit.range(6)
                pre_ancilla = np.zeros(4, dtype=complex)
                pre_ancilla[0] = 1.0
                rho_13_with_ancillas = np.kron(
                    rho_13_input,
                    np.outer(pre_ancilla, pre_ancilla.conj()),
                )

                purification_circuit = cirq.Circuit()
                for control, ancilla in ((1, 4), (2, 4), (2, 5), (3, 5)):
                    operation = cirq.CNOT(pre_qubits[control], pre_qubits[ancilla])
                    purification_circuit.append(operation)
                    if self.error_multi is not None and self.error_multi > 0:
                        purification_circuit.append(
                            cirq.depolarize(self.error_multi).on_each(*operation.qubits)
                        )

                pre_result = sim.simulate(
                    purification_circuit,
                    initial_state=rho_13_with_ancillas,
                    qubit_order=pre_qubits,
                )
                rho_13_evolved = _safe_clean(pre_result.final_density_matrix)

                syndrome_00_mask = (np.arange(64) & 0b11) == 0
                p_13 = float(np.sum(
                    np.diagonal(rho_13_evolved).real[syndrome_00_mask]
                ))
                if p_13 > 1e-10:
                    syndrome_vector = syndrome_00_mask.astype(float).reshape(-1, 1)
                    rho_13_post = rho_13_evolved * (
                        syndrome_vector @ syndrome_vector.T
                    ) / p_13
                    rho_13 = self.manual_partial_trace(
                        _safe_clean(rho_13_post),
                        keep_indices=[0, 1, 2, 3],
                        dims=[2] * 6,
                    )
                    rho_13 = _safe_clean(rho_13)
                else:
                    # 数值保护：极端门噪声下若后选择概率下溢，保留 d1 后的物理态。
                    rho_13 = rho_13_input

                # d2：从 1|3 继续分发 q2、q3，到达 1|1|2。
                rho_112 = self._apply_depolarizing_noise_to_rho(
                    rho_13,
                    num_qubits=4,
                    noise_by_index={2: route_delta2, 3: route_delta2},
                )
                if rho_112 is None:
                    rho_112 = rho_13
                rho_112 = _safe_clean(rho_112)

                # 环境仍保持 Target(4Q) + Resource(4Q) + 单 ancilla 的 9Q 接口；
                # 两份 rho_112 表示独立制备、具有相同密度矩阵的级联中间态。
                single_ancilla = np.zeros(2, dtype=complex)
                single_ancilla[0] = 1.0
                self.rho = np.kron(
                    np.kron(rho_112, rho_112),
                    np.outer(single_ancilla, single_ancilla.conj()),
                )
                self.rho_initial = self.rho
                self.rho_4q = rho_112
                self.F_in = self._get_ghz4_fidelity(self.rho_4q)
                self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)
                self.F_temp = self.F_in

                # rho_112 已经处于 1|1|2，直接标记拓扑，避免
                # _finalize_reset_observation 再次注入一遍初始分布噪声。
                cumulative_p2 = self._compose_depolarizing_p(
                    route_delta1,
                    route_delta2,
                )
                self.current_topology = "1|1|2"
                self.topology_selected = True
                self.topology_total_noise_p = cumulative_p2
                self.topology_initial_noise_by_logical = {
                    0: 0.0,
                    1: route_delta1,
                    2: cumulative_p2,
                    3: cumulative_p2,
                }
                self.topology_conversion_noise_by_logical = {
                    logical: 0.0 for logical in range(4)
                }
                cumulative_noise = dict(self.topology_initial_noise_by_logical)
                self.topology_noise_report = {
                    "total_noise_p": cumulative_p2,
                    "apply_topology_conversion_noise": False,
                    "initial_noise_by_logical": cumulative_noise,
                    "effective_noise_by_logical": cumulative_noise,
                    "preparation": "1|3 (2 ancillas) -> 1|1|2",
                    "route_delta1": route_delta1,
                    "route_delta2": route_delta2,
                    "cumulative_noise_by_logical": cumulative_noise,
                    "conversion_noise_by_logical": dict(
                        self.topology_conversion_noise_by_logical
                    ),
                    "p_13": p_13,
                    "prepared_pair_probability": p_13 ** 2,
                }
                return self._get_obs(), {
                    "preparation": self.topology_noise_report["preparation"],
                    "route_delta1": route_delta1,
                    "route_delta2": route_delta2,
                    "p_13": p_13,
                }

            self.rho = rho_current
            self.rho_initial = self.rho
            self.rho_4q = self.manual_partial_trace(self.rho, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)
            self.F_temp = self.F_in
            return self._finalize_reset_observation()

            def _build_noisy_circuit(ops):
                noisy_circuit = cirq.Circuit()
                for op in ops:
                    noisy_circuit.append(op)
                    op_qubits = op.qubits
                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))
                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))
                return noisy_circuit

            dim = 512

            # =========================================================
            # 🚀 R0: 初始预处理 (Pre-selection)
            # =========================================================
            ops_pre = [cirq.CNOT(self.qubits[2], self.qubits[8]), cirq.CNOT(self.qubits[3], self.qubits[8])]
            res_pre = sim.simulate(_build_noisy_circuit(ops_pre), initial_state=rho_current,
                                   qubit_order=self.qubits[:9])
            rho_pre = res_pre.final_density_matrix

            q8_mask = 1
            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0
            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:
                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)
                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T) / p_succ_pre
                rho_current = _safe_clean(rho_pre_post)

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_target_pre = rho_4q_pre_success
                rho_resource_pre = rho_4q_pre_success
            else:
                rho_target_pre = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_pre = rho_target_pre

            rho_current = np.kron(np.kron(rho_target_pre, rho_resource_pre), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R1：标准纠缠泵浦 (Action 48 + 126 P1_MASK_BOTH)
            # =========================================================
            ops_r1 = [
                cirq.CNOT(self.qubits[4], self.qubits[0]),
                cirq.CNOT(self.qubits[5], self.qubits[1]),
                cirq.CNOT(self.qubits[6], self.qubits[2]),
                cirq.CNOT(self.qubits[7], self.qubits[3])
            ]
            res_r1 = sim.simulate(_build_noisy_circuit(ops_r1), initial_state=rho_current, qubit_order=self.qubits[:9])

            rho_9q_r1 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                     initial_state=res_r1.final_density_matrix,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            mask_r1 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 3, 5, 6, 9, 10, 12, 15])
            p_succ_r1 = np.sum(np.diagonal(rho_9q_r1).real[mask_r1])

            if p_succ_r1 > 1e-10:
                proj_r1 = np.zeros_like(rho_9q_r1)
                for val in set(np.where(mask_r1)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r1[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1)
                        proj_r1 += rho_9q_r1 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r1 / p_succ_r1)
            else:
                rho_current = _safe_clean(rho_9q_r1)

            rho_4q_r1_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r1 > 1e-10:
                rho_target_R2 = rho_4q_r1_success
                rho_resource_R2 = rho_4q_r1_success
            else:
                rho_target_R2 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_R2 = rho_target_R2

            rho_8q_R2 = np.kron(rho_target_R2, rho_resource_R2)
            rho_current = np.kron(rho_8q_R2, np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R2：反向宇称泵浦 (Action 32 + 129 P2_MASK_BOTH)
            # =========================================================
            ops_r2 = [
                cirq.CNOT(self.qubits[0], self.qubits[4]),
                cirq.CNOT(self.qubits[1], self.qubits[5]),
                cirq.CNOT(self.qubits[2], self.qubits[6]),
                cirq.CNOT(self.qubits[3], self.qubits[7])
            ]
            res_r2 = sim.simulate(_build_noisy_circuit(ops_r2), initial_state=rho_current, qubit_order=self.qubits[:9])

            # P2 模式：直接 Z 基底测量
            rho_9q_r2 = res_r2.final_density_matrix

            mask_r2 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 15])
            p_succ_r2 = np.sum(np.diagonal(rho_9q_r2).real[mask_r2])

            if p_succ_r2 > 1e-10:
                proj_r2 = np.zeros_like(rho_9q_r2)
                for val in set(np.where(mask_r2)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r2[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1)
                        proj_r2 += rho_9q_r2 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r2 / p_succ_r2)
            else:
                rho_current = _safe_clean(rho_9q_r2)

            rho_4q_r2_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r2 > 1e-10:
                rho_target_R3 = rho_4q_r2_success
                rho_resource_R3 = rho_4q_r2_success
            else:
                rho_target_R3 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_R3 = rho_target_R3

            rho_8q_R3 = np.kron(rho_target_R3, rho_resource_R3)
            rho_current = np.kron(rho_8q_R3, np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R3：双向交织与非对称嗅探 (最新! Action 93 + Action 125 P1_MASK_RES)
            # =========================================================
            ops_r3 = [
                # 1. 抽 X 错误
                cirq.CNOT(self.qubits[4], self.qubits[0]), cirq.CNOT(self.qubits[5], self.qubits[1]),
                cirq.CNOT(self.qubits[6], self.qubits[2]), cirq.CNOT(self.qubits[7], self.qubits[3]),
                # 2. 抽 Z 错误 (组合形成 2/3 SWAP)
                #cirq.CNOT(self.qubits[0], self.qubits[4]), cirq.CNOT(self.qubits[1], self.qubits[5]),
                #cirq.CNOT(self.qubits[2], self.qubits[6]), cirq.CNOT(self.qubits[3], self.qubits[7]),
                # 3. 非对称探测
                cirq.CZ(self.qubits[0], self.qubits[4]), cirq.CZ(self.qubits[1], self.qubits[5])
            ]
            res_r3 = sim.simulate(_build_noisy_circuit(ops_r3), initial_state=rho_current, qubit_order=self.qubits[:9])

            # P1 模式：需要加 H 门
            rho_9q_r3 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                     initial_state=res_r3.final_density_matrix,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            # MASK_RES: 完全不管 q8，仅校验 Resource (q4-q7) 的 X 基底偶宇称
            mask_r3 = np.isin(np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 3, 5, 6, 9, 10, 12, 15])
            p_succ_r3 = np.sum(np.diagonal(rho_9q_r3).real[mask_r3])

            if p_succ_r3 > 1e-10:
                proj_r3 = np.zeros_like(rho_9q_r3)
                for val in set(np.where(mask_r3)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r3[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1)
                        proj_r3 += rho_9q_r3 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r3 / p_succ_r3)
            else:
                rho_current = _safe_clean(rho_9q_r3)

            # =========================================================
            # 🚀 装填终极第四轮起跑线，移交 RL Agent！
            # =========================================================
            rho_4q_r3_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r3 > 1e-10:
                rho_target_R4 = rho_4q_r3_success
                rho_resource_R4 = rho_4q_r3_success
            else:
                rho_target_R4 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_R4 = rho_target_R4

            rho_8q_R4 = np.kron(rho_target_R4, rho_resource_R4)

            self.rho = np.kron(rho_8q_R4, np.outer(psi_anc, psi_anc.conj()))
            self.rho_initial = self.rho

            self.rho_4q = rho_target_R4
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)

        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)

            # =========================================================
            # 🚀 0. 安全过滤装甲 (防 NoneType 崩溃补丁)
            # =========================================================
            def _safe_clean(rho_in):
                if rho_in is None: return None
                _res = self._clean_density_matrix(rho_in)
                return _res if _res is not None else rho_in

            # =========================================================
            # 1. 初始化 9-qubit 完美总态 (512 x 512)
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)

            rho_sys_base = self._generate_ghz4_state(F=f_param)
            rho_res_base = self._generate_ghz4_state(F=f_param)
            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            psi_anc = np.zeros(2, dtype=complex)
            psi_anc[0] = 1.0
            rho_anc = np.outer(psi_anc, psi_anc.conj())

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================
            # 🚀 2. 异构信道加噪
            # =========================================================
            qc_channel_noise = cirq.Circuit()

            noise_p_bob = self.np_random.uniform(0.05, 0.3)
            self.niose_bob = noise_p_bob
            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            noise_p_charlie = self.np_random.uniform(0.05, 0.3)
            self.niose_charlie = noise_p_charlie
            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================
            # 3. 核心大流水线开始
            # =========================================================
            sim = cirq.DensityMatrixSimulator()

            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])
            rho_after_noise = _safe_clean(res_noise.final_density_matrix)
            rho_current = rho_after_noise

            def _build_noisy_circuit(ops):
                noisy_circuit = cirq.Circuit()
                for op in ops:
                    noisy_circuit.append(op)
                    op_qubits = op.qubits
                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))
                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))
                return noisy_circuit

            dim = 512

            # =========================================================
            # 🚀 R0: 初始预处理 (Pre-selection)
            # =========================================================
            ops_pre = [cirq.CNOT(self.qubits[2], self.qubits[8]), cirq.CNOT(self.qubits[3], self.qubits[8])]
            res_pre = sim.simulate(_build_noisy_circuit(ops_pre), initial_state=rho_current,
                                   qubit_order=self.qubits[:9])
            rho_pre = res_pre.final_density_matrix

            q8_mask = 1
            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0
            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:
                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)
                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T) / p_succ_pre
                rho_current = _safe_clean(rho_pre_post)

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_target_pre = rho_4q_pre_success
                rho_resource_pre = rho_4q_pre_success
            else:
                rho_target_pre = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_pre = rho_target_pre

            rho_current = np.kron(np.kron(rho_target_pre, rho_resource_pre), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R1：标准纠缠泵浦 (Action 48 + 126 P1_MASK_BOTH)
            # =========================================================
            ops_r1 = [
                cirq.CNOT(self.qubits[4], self.qubits[0]), cirq.CNOT(self.qubits[5], self.qubits[1]),
                cirq.CNOT(self.qubits[6], self.qubits[2]), cirq.CNOT(self.qubits[7], self.qubits[3])
            ]
            res_r1 = sim.simulate(_build_noisy_circuit(ops_r1), initial_state=rho_current, qubit_order=self.qubits[:9])

            rho_9q_r1 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                     initial_state=res_r1.final_density_matrix,
                                     qubit_order=self.qubits[:9]).final_density_matrix
            mask_r1 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 3, 5, 6, 9, 10, 12, 15])
            p_succ_r1 = np.sum(np.diagonal(rho_9q_r1).real[mask_r1])

            if p_succ_r1 > 1e-10:
                proj_r1 = np.zeros_like(rho_9q_r1)
                for val in set(np.where(mask_r1)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r1[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1);
                        proj_r1 += rho_9q_r1 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r1 / p_succ_r1)
            else:
                rho_current = _safe_clean(rho_9q_r1)

            rho_4q_r1_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_target_R2 = rho_4q_r1_success if p_succ_r1 > 1e-10 else self.manual_partial_trace(rho_after_noise,
                                                                                                  keep_indices=[0, 1, 2,
                                                                                                                3],
                                                                                                  dims=[2] * 9)
            rho_current = np.kron(np.kron(rho_target_R2, rho_target_R2), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R2：反向宇称泵浦 (Action 32 + 129 P2_MASK_BOTH)
            # =========================================================
            ops_r2 = [
                cirq.CNOT(self.qubits[0], self.qubits[4]), cirq.CNOT(self.qubits[1], self.qubits[5]),
                cirq.CNOT(self.qubits[2], self.qubits[6]), cirq.CNOT(self.qubits[3], self.qubits[7])
            ]
            rho_9q_r2 = sim.simulate(_build_noisy_circuit(ops_r2), initial_state=rho_current,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            mask_r2 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 15])
            p_succ_r2 = np.sum(np.diagonal(rho_9q_r2).real[mask_r2])

            if p_succ_r2 > 1e-10:
                proj_r2 = np.zeros_like(rho_9q_r2)
                for val in set(np.where(mask_r2)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r2[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1);
                        proj_r2 += rho_9q_r2 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r2 / p_succ_r2)
            else:
                rho_current = _safe_clean(rho_9q_r2)

            rho_4q_r2_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_target_R3 = rho_4q_r2_success if p_succ_r2 > 1e-10 else self.manual_partial_trace(rho_after_noise,
                                                                                                  keep_indices=[0, 1, 2,
                                                                                                                3],
                                                                                                  dims=[2] * 9)
            rho_current = np.kron(np.kron(rho_target_R3, rho_target_R3), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R3：双向交织与非对称嗅探 (Action 93 + 125 P1_MASK_RES)
            # =========================================================
            ops_r3 = [
                cirq.CNOT(self.qubits[4], self.qubits[0]), cirq.CNOT(self.qubits[5], self.qubits[1]),
                cirq.CNOT(self.qubits[6], self.qubits[2]), cirq.CNOT(self.qubits[7], self.qubits[3]),
                cirq.CNOT(self.qubits[0], self.qubits[4]), cirq.CNOT(self.qubits[1], self.qubits[5]),
                cirq.CNOT(self.qubits[2], self.qubits[6]), cirq.CNOT(self.qubits[3], self.qubits[7]),
                cirq.CZ(self.qubits[0], self.qubits[4]), cirq.CZ(self.qubits[1], self.qubits[5])
            ]
            res_r3 = sim.simulate(_build_noisy_circuit(ops_r3), initial_state=rho_current, qubit_order=self.qubits[:9])

            rho_9q_r3 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                     initial_state=res_r3.final_density_matrix,
                                     qubit_order=self.qubits[:9]).final_density_matrix
            mask_r3 = np.isin(np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 3, 5, 6, 9, 10, 12, 15])
            p_succ_r3 = np.sum(np.diagonal(rho_9q_r3).real[mask_r3])

            if p_succ_r3 > 1e-10:
                proj_r3 = np.zeros_like(rho_9q_r3)
                for val in set(np.where(mask_r3)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r3[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1);
                        proj_r3 += rho_9q_r3 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r3 / p_succ_r3)
            else:
                rho_current = _safe_clean(rho_9q_r3)

            rho_4q_r3_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_target_R4 = rho_4q_r3_success if p_succ_r3 > 1e-10 else self.manual_partial_trace(rho_after_noise,
                                                                                                  keep_indices=[0, 1, 2,
                                                                                                                3],
                                                                                                  dims=[2] * 9)
            rho_current = np.kron(np.kron(rho_target_R4, rho_target_R4), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R4：极限非对称降维打击 (最新! Action 63 + Action 126 P1_MASK_BOTH)
            # =========================================================
            ops_r4 = [
                cirq.CNOT(self.qubits[4], self.qubits[0]), cirq.CNOT(self.qubits[5], self.qubits[1]),
                cirq.CNOT(self.qubits[6], self.qubits[2]), cirq.CNOT(self.qubits[7], self.qubits[3]),
                cirq.CZ(self.qubits[0], self.qubits[4]), cirq.CZ(self.qubits[3], self.qubits[7])
            ]
            res_r4 = sim.simulate(_build_noisy_circuit(ops_r4), initial_state=rho_current, qubit_order=self.qubits[:9])

            # P1 模式：需要加 H 门
            rho_9q_r4 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                     initial_state=res_r4.final_density_matrix,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            # MASK_BOTH: 全局大消杀！严格检查探针 q8=0 且 Resource 为 X 基底偶宇称
            mask_r4 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 3, 5, 6, 9, 10, 12, 15])
            p_succ_r4 = np.sum(np.diagonal(rho_9q_r4).real[mask_r4])

            if p_succ_r4 > 1e-10:
                proj_r4 = np.zeros_like(rho_9q_r4)
                for val in set(np.where(mask_r4)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r4[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1);
                        proj_r4 += rho_9q_r4 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r4 / p_succ_r4)
            else:
                rho_current = _safe_clean(rho_9q_r4)

            # =========================================================
            # 🚀 装填终极第五轮起跑线，移交 RL Agent！
            # =========================================================
            rho_4q_r4_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r4 > 1e-10:
                rho_target_R5 = rho_4q_r4_success
                rho_resource_R5 = rho_4q_r4_success
            else:
                rho_target_R5 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_R5 = rho_target_R5

            rho_8q_R5 = np.kron(rho_target_R5, rho_resource_R5)

            self.rho = np.kron(rho_8q_R5, np.outer(psi_anc, psi_anc.conj()))
            self.rho_initial = self.rho

            self.rho_4q = rho_target_R5
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)

        elif rand_scenario < 0:  # 4-qubit 不对称异构网络裸考版 (9-qubit 空间)

            # =========================================================
            # 🚀 0. 安全过滤装甲 (防 NoneType 崩溃补丁)
            # =========================================================
            def _safe_clean(rho_in):
                if rho_in is None: return None
                _res = self._clean_density_matrix(rho_in)
                return _res if _res is not None else rho_in

            # =========================================================
            # 1. 初始化 9-qubit 完美总态 (512 x 512)
            # =========================================================
            f_param = self.np_random.uniform(0.95, 1.0)

            rho_sys_base = self._generate_ghz4_state(F=f_param)
            rho_res_base = self._generate_ghz4_state(F=f_param)
            rho_8q_input = np.kron(rho_sys_base, rho_res_base)

            psi_anc = np.zeros(2, dtype=complex)
            psi_anc[0] = 1.0
            rho_anc = np.outer(psi_anc, psi_anc.conj())

            current_9q_state = np.kron(rho_8q_input, rho_anc)

            # =========================================================
            # 🚀 2. 异构信道加噪
            # =========================================================
            qc_channel_noise = cirq.Circuit()

            noise_p_bob = self.np_random.uniform(0.05, 0.3)
            self.niose_bob = noise_p_bob
            for q_idx in [1, 5]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_bob).on(self.qubits[q_idx]))

            noise_p_charlie = self.np_random.uniform(0.05, 0.3)
            self.niose_charlie = noise_p_charlie
            for q_idx in [2, 3, 6, 7]:
                qc_channel_noise.append(cirq.depolarize(p=noise_p_charlie).on(self.qubits[q_idx]))

            # =========================================================
            # 3. 核心大流水线开始
            # =========================================================
            sim = cirq.DensityMatrixSimulator()

            res_noise = sim.simulate(qc_channel_noise, initial_state=current_9q_state, qubit_order=self.qubits[:9])
            rho_after_noise = _safe_clean(res_noise.final_density_matrix)
            rho_current = rho_after_noise

            def _build_noisy_circuit(ops):
                noisy_circuit = cirq.Circuit()
                for op in ops:
                    noisy_circuit.append(op)
                    op_qubits = op.qubits
                    if len(op_qubits) == 1 and getattr(self, 'error_single',
                                                       None) is not None and self.error_single > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*op_qubits))
                    elif len(op_qubits) > 1 and getattr(self, 'error_multi', None) is not None and self.error_multi > 0:
                        noisy_circuit.append(cirq.depolarize(self.error_multi).on_each(*op_qubits))
                return noisy_circuit

            dim = 512

            # =========================================================
            # 🚀 R0: 初始预处理 (Pre-selection)
            # =========================================================
            ops_pre = [cirq.CNOT(self.qubits[2], self.qubits[8]), cirq.CNOT(self.qubits[3], self.qubits[8])]
            res_pre = sim.simulate(_build_noisy_circuit(ops_pre), initial_state=rho_current,
                                   qubit_order=self.qubits[:9])
            rho_pre = res_pre.final_density_matrix

            q8_mask = 1
            valid_indices_mask_pre = (np.arange(dim) & q8_mask) == 0
            p_succ_pre = np.sum(np.diagonal(rho_pre)[valid_indices_mask_pre]).real

            if p_succ_pre > 1e-10:
                mask_vec = valid_indices_mask_pre.astype(float).reshape(-1, 1)
                rho_pre_post = rho_pre * (mask_vec @ mask_vec.T) / p_succ_pre
                rho_current = _safe_clean(rho_pre_post)

                rho_4q_pre_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_target_pre = rho_4q_pre_success
                rho_resource_pre = rho_4q_pre_success
            else:
                rho_target_pre = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_pre = rho_target_pre

            rho_current = np.kron(np.kron(rho_target_pre, rho_resource_pre), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R1：标准纠缠泵浦 (Action 48 + 126 P1_MASK_BOTH)
            # =========================================================
            ops_r1 = [
                cirq.CNOT(self.qubits[4], self.qubits[0]), cirq.CNOT(self.qubits[5], self.qubits[1]),
                cirq.CNOT(self.qubits[6], self.qubits[2]), cirq.CNOT(self.qubits[7], self.qubits[3])
            ]
            res_r1 = sim.simulate(_build_noisy_circuit(ops_r1), initial_state=rho_current, qubit_order=self.qubits[:9])
            rho_9q_r1 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                     initial_state=res_r1.final_density_matrix,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            mask_r1 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 3, 5, 6, 9, 10, 12, 15])
            p_succ_r1 = np.sum(np.diagonal(rho_9q_r1).real[mask_r1])
            if p_succ_r1 > 1e-10:
                proj_r1 = np.zeros_like(rho_9q_r1)
                for val in set(np.where(mask_r1)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r1[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1);
                        proj_r1 += rho_9q_r1 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r1 / p_succ_r1)
            else:
                rho_current = _safe_clean(rho_9q_r1)

            r1_suc = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_target_R2 = r1_suc if p_succ_r1 > 1e-10 else self.manual_partial_trace(rho_after_noise,
                                                                                       keep_indices=[0, 1, 2, 3],
                                                                                       dims=[2] * 9)
            rho_current = np.kron(np.kron(rho_target_R2, rho_target_R2), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R2：反向宇称泵浦 (Action 32 + 129 P2_MASK_BOTH)
            # =========================================================
            ops_r2 = [
                cirq.CNOT(self.qubits[0], self.qubits[4]), cirq.CNOT(self.qubits[1], self.qubits[5]),
                cirq.CNOT(self.qubits[2], self.qubits[6]), cirq.CNOT(self.qubits[3], self.qubits[7])
            ]
            rho_9q_r2 = sim.simulate(_build_noisy_circuit(ops_r2), initial_state=rho_current,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            mask_r2 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 15])
            p_succ_r2 = np.sum(np.diagonal(rho_9q_r2).real[mask_r2])
            if p_succ_r2 > 1e-10:
                proj_r2 = np.zeros_like(rho_9q_r2)
                for val in set(np.where(mask_r2)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r2[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1);
                        proj_r2 += rho_9q_r2 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r2 / p_succ_r2)
            else:
                rho_current = _safe_clean(rho_9q_r2)

            r2_suc = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_target_R3 = r2_suc if p_succ_r2 > 1e-10 else self.manual_partial_trace(rho_after_noise,
                                                                                       keep_indices=[0, 1, 2, 3],
                                                                                       dims=[2] * 9)
            rho_current = np.kron(np.kron(rho_target_R3, rho_target_R3), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R3：双向交织与非对称嗅探 (Action 93 + 125 P1_MASK_RES)
            # =========================================================
            ops_r3 = [
                cirq.CNOT(self.qubits[4], self.qubits[0]), cirq.CNOT(self.qubits[5], self.qubits[1]),
                cirq.CNOT(self.qubits[6], self.qubits[2]), cirq.CNOT(self.qubits[7], self.qubits[3]),
                cirq.CNOT(self.qubits[0], self.qubits[4]), cirq.CNOT(self.qubits[1], self.qubits[5]),
                cirq.CNOT(self.qubits[2], self.qubits[6]), cirq.CNOT(self.qubits[3], self.qubits[7]),
                cirq.CZ(self.qubits[0], self.qubits[4]), cirq.CZ(self.qubits[1], self.qubits[5])
            ]
            res_r3 = sim.simulate(_build_noisy_circuit(ops_r3), initial_state=rho_current, qubit_order=self.qubits[:9])
            rho_9q_r3 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                     initial_state=res_r3.final_density_matrix,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            # P1_MASK_RES: 不检查 q8
            mask_r3 = np.isin(np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 3, 5, 6, 9, 10, 12, 15])
            p_succ_r3 = np.sum(np.diagonal(rho_9q_r3).real[mask_r3])
            if p_succ_r3 > 1e-10:
                proj_r3 = np.zeros_like(rho_9q_r3)
                for val in set(np.where(mask_r3)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r3[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1);
                        proj_r3 += rho_9q_r3 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r3 / p_succ_r3)
            else:
                rho_current = _safe_clean(rho_9q_r3)

            r3_suc = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_target_R4 = r3_suc if p_succ_r3 > 1e-10 else self.manual_partial_trace(rho_after_noise,
                                                                                       keep_indices=[0, 1, 2, 3],
                                                                                       dims=[2] * 9)
            rho_current = np.kron(np.kron(rho_target_R4, rho_target_R4), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R4：极限非对称降维打击 (Action 63 + 126 P1_MASK_BOTH)
            # =========================================================
            ops_r4 = [
                cirq.CNOT(self.qubits[4], self.qubits[0]), cirq.CNOT(self.qubits[5], self.qubits[1]),
                cirq.CNOT(self.qubits[6], self.qubits[2]), cirq.CNOT(self.qubits[7], self.qubits[3]),
                cirq.CZ(self.qubits[0], self.qubits[4]), cirq.CZ(self.qubits[3], self.qubits[7])
            ]
            res_r4 = sim.simulate(_build_noisy_circuit(ops_r4), initial_state=rho_current, qubit_order=self.qubits[:9])
            rho_9q_r4 = sim.simulate(_build_noisy_circuit([cirq.H(self.qubits[i]) for i in [4, 5, 6, 7]]),
                                     initial_state=res_r4.final_density_matrix,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            mask_r4 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 3, 5, 6, 9, 10, 12, 15])
            p_succ_r4 = np.sum(np.diagonal(rho_9q_r4).real[mask_r4])
            if p_succ_r4 > 1e-10:
                proj_r4 = np.zeros_like(rho_9q_r4)
                for val in set(np.where(mask_r4)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r4[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1);
                        proj_r4 += rho_9q_r4 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r4 / p_succ_r4)
            else:
                rho_current = _safe_clean(rho_9q_r4)

            r4_suc = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
            rho_target_R5 = r4_suc if p_succ_r4 > 1e-10 else self.manual_partial_trace(rho_after_noise,
                                                                                       keep_indices=[0, 1, 2, 3],
                                                                                       dims=[2] * 9)
            rho_current = np.kron(np.kron(rho_target_R5, rho_target_R5), np.outer(psi_anc, psi_anc.conj()))

            # =========================================================
            # 🚀 R5：返璞归真的反向 Z 泵浦 (最新! Action 32 + 129 P2_MASK_BOTH)
            # =========================================================
            ops_r5 = [
                cirq.CNOT(self.qubits[0], self.qubits[4]), cirq.CNOT(self.qubits[1], self.qubits[5]),
                cirq.CNOT(self.qubits[2], self.qubits[6]), cirq.CNOT(self.qubits[3], self.qubits[7])
            ]
            rho_9q_r5 = sim.simulate(_build_noisy_circuit(ops_r5), initial_state=rho_current,
                                     qubit_order=self.qubits[:9]).final_density_matrix

            # P2 模式：严格 Z 基底，双重过滤 (q8=0 且 q4-q7 为 0000 或 1111)
            mask_r5 = np.array([(i & 0b1) == 0 for i in range(dim)]) & np.isin(
                np.array([((i >> 1) & 0b1111) for i in range(dim)]), [0, 15])
            p_succ_r5 = np.sum(np.diagonal(rho_9q_r5).real[mask_r5])

            if p_succ_r5 > 1e-10:
                proj_r5 = np.zeros_like(rho_9q_r5)
                for val in set(np.where(mask_r5)[0] % 32):
                    bm = np.array([(i % 32) == val for i in range(dim)])
                    if mask_r5[np.where(bm)[0][0]]:
                        vec = bm.astype(float).reshape(-1, 1);
                        proj_r5 += rho_9q_r5 * (vec @ vec.T)
                rho_current = _safe_clean(proj_r5 / p_succ_r5)
            else:
                rho_current = _safe_clean(rho_9q_r5)

            # =========================================================
            # 🚀 装填终极第六轮起跑线，移交 RL Agent！
            # =========================================================
            rho_4q_r5_success = self.manual_partial_trace(rho_current, keep_indices=[0, 1, 2, 3], dims=[2] * 9)

            if p_succ_r5 > 1e-10:
                rho_target_R6 = rho_4q_r5_success
                rho_resource_R6 = rho_4q_r5_success
            else:
                rho_target_R6 = self.manual_partial_trace(rho_after_noise, keep_indices=[0, 1, 2, 3], dims=[2] * 9)
                rho_resource_R6 = rho_target_R6

            rho_8q_R6 = np.kron(rho_target_R6, rho_resource_R6)

            self.rho = np.kron(rho_8q_R6, np.outer(psi_anc, psi_anc.conj()))
            self.rho_initial = self.rho

            self.rho_4q = rho_target_R6
            self.F_in = self._get_ghz4_fidelity(self.rho_4q)
            self.en_in = self._get_logarithmic_negativity_4q(self.rho_4q)

        else:
            # 【模式 3: 极低保真度状态】 -> 训练“放弃”或强力纯化
            # 增加训练难度
            f_param = self.np_random.uniform(0.55, 0.65)
            self.rho_3q = self._generate_ghz_state(F=f_param)

            # 【核心修正】: 准确记录当前状态的保真度，供 Reward 函数使用
        #self.F_in = self._get_ghz_fidelity(self.rho_3q) #原

        # 构造 6-qubit 输入对
        #self.rho = np.kron(self.rho_3q, self.rho_3q) #原
        #self.rho_initial = self.rho #原
        self.F_temp = self.F_in
        return self._finalize_reset_observation()


    def _get_cirq2(self, maybe_add_noise=False):
        circuit = cirq.Circuit(cirq.I(qubit) for qubit in self.qubits)
        for gate in self.circuit_gates:
            circuit.append(gate)
            if maybe_add_noise and (self.error_gates is not None):
                noise_gate = cirq.depolarize(
                    self.error_gates).on_each(*gate._qubits)
                circuit.append(noise_gate)
        if maybe_add_noise and (self.error_observables is not None):
            noise_observable = cirq.bit_flip(
                self.error_observables).on_each(*self.qubits)
            circuit.append(noise_observable)
        return circuit

    def _get_cirq_noisy(self, maybe_add_noise=False):
        # 初始化电路时为每个量子比特添加单位操作（占位符）
        circuit = cirq.Circuit(cirq.I(qubit) for qubit in self.qubits)
        for gate in self.circuit_gates:
            circuit.append(gate)
            if maybe_add_noise:
                # 尝试获取门作用的量子比特列表，优先使用 _qubits 属性，否则使用 qubits 属性
                if hasattr(gate, "_qubits"):
                    qubits = gate._qubits
                else:
                    qubits = gate.qubits
                # 根据门作用的量子比特数量来区分噪声模型
                if len(qubits) == 1 and (self.error_single is not None):
                    # 对单比特门添加噪声（例如使用 depolarize 噪声）
                    noise_gate = cirq.depolarize(self.error_single).on_each(*qubits)
                    circuit.append(noise_gate)
                elif len(qubits) > 1 and (self.error_multi is not None):
                    # 对多比特门添加噪声
                    noise_gate = cirq.depolarize(self.error_multi).on_each(*qubits)
                    circuit.append(noise_gate)
        # 对观测操作也可以添加噪声
        if maybe_add_noise and (self.error_observables is not None):
            noise_observable = cirq.bit_flip(self.error_observables).on_each(*self.qubits)
            circuit.append(noise_observable)
        return circuit

    def _get_cirq(self, maybe_add_noise=False):
        # 初始化电路
        circuit = cirq.Circuit(cirq.I(qubit) for qubit in self.qubits)

        for action in self.circuit_gates:
            # --- 核心修改：处理双边门电路 ---
            # 如果 action 是 Circuit（双边门），将其转为操作列表
            # 如果 action 是 Operation（单比特门），也统一转为列表处理
            if isinstance(action, cirq.Circuit):
                ops = list(action.all_operations())
            else:
                ops = [action]

            for op in ops:
                circuit.append(op)

                if maybe_add_noise:
                    # 获取操作作用的量子比特（此时 op 一定是 Operation）
                    qubits = op.qubits

                    # 根据门作用的量子比特数量来区分噪声模型
                    if len(qubits) == 1 and (self.error_single is not None):
                        # 对单比特门添加噪声
                        noise_gate = cirq.depolarize(self.error_single).on_each(*qubits)
                        circuit.append(noise_gate)
                    elif len(qubits) > 1 and (self.error_multi is not None):
                        # 对多比特门添加噪声
                        # 注意：双边门里的两个 CNOT 会分别在这里被加上噪声
                        noise_gate = cirq.depolarize(self.error_multi).on_each(*qubits)
                        circuit.append(noise_gate)

        # 对观测操作添加噪声
        if maybe_add_noise and (self.error_observables is not None):
            noise_observable = cirq.bit_flip(self.error_observables).on_each(*self.qubits)
            circuit.append(noise_observable)

        return circuit

    def _get_cirq3(self, maybe_add_noise=False):
        # 1. 构建原始电路
        circuit = cirq.Circuit()
        circuit.append(self.circuit_gates)

        if maybe_add_noise:
            # 2. 分解复合门 (如果需要)
            # 假设 circuit_gates 可能包含复合门
            try:
                # 尝试分解，如果电路中没有可分解的门，会返回原电路
                decomposed_circuit = cirq.expand_composite(circuit,
                                                           no_decomp=lambda op: not hasattr(op.gate, '_decompose_'))
            except Exception as e:
                print(f"Warning: Failed to decompose circuit, proceeding with original. Error: {e}")
                decomposed_circuit = circuit  # 如果分解失败，使用原电路

            # 3. 对分解后的电路添加噪声
            noisy_moments = []
            for moment in decomposed_circuit:
                noisy_moments.append(moment)  # 先添加原始的 Moment

                added_noise_qubits_single = set()
                added_noise_qubits_two = set()  # 用于记录哪些 qubit pair 在本 moment 已处理

                for gate_op in moment.operations:
                    qubits_tuple = tuple(sorted(gate_op.qubits))

                    # 对单比特门添加噪声 (跳过 IdentityGate)
                    if len(gate_op.qubits) == 1 and not isinstance(gate_op.gate, cirq.ops.identity.IdentityGate):
                        qubit = gate_op.qubits[0]
                        if qubit not in added_noise_qubits_single:
                            noise_ops = cirq.bit_flip(self.error_single).on(qubit)  # bit_flip 本身是 GateOperation
                            noisy_moments.append(cirq.Moment([noise_ops]))
                            added_noise_qubits_single.add(qubit)

                    # 对两比特门添加噪声 (使用 .on_each)
                    elif len(gate_op.qubits) == 2:
                        if qubits_tuple not in added_noise_qubits_two:
                            # 使用 .on_each() 将单比特退极化噪声分别应用到两个 qubit 上
                            # .on_each() 返回一个操作的迭代器
                            noise_ops_iterable = cirq.depolarize(self.error_multi).on_each(*gate_op.qubits)
                            # 将这些独立的操作放入一个新的 Moment 中
                            noisy_moments.append(cirq.Moment(noise_ops_iterable))
                            added_noise_qubits_two.add(qubits_tuple)

            circuit = cirq.Circuit(noisy_moments)

        # --- 观测噪声添加 (保持不变) ---
        if maybe_add_noise and (self.error_observables is not None):
            noise_observable = cirq.bit_flip(
                self.error_observables).on_each(*self.qubits)
            circuit.append(noise_observable)
        return circuit

    def _get_obs_backup_0401(self):
        # ---------------------------------------------------------
        # 1. 运行模拟
        # ---------------------------------------------------------
        # 获取当前电路（包含噪声）
        circuit = self._get_cirq(maybe_add_noise=True)

        #purity_in = np.trace(self.rho @ self.rho).real
        #print(f"DEBUG: _get_obs Input Rho Purity: {purity_in:.4f}")

        # 【关键修正】initial_state 必须是 6-qubit 的 self.rho
        # self.rho 在 reset 中被定义为 np.kron(rho_3q, rho_3q)
        result = self.simulator.simulate(
            circuit,
            initial_state=self.rho,
            qubit_order=self.qubits
        )
        rho_final = result.final_density_matrix

        # ---------------------------------------------------------
        # 2. 物理清洗
        # ---------------------------------------------------------
        # 消除数值误差导致的负特征值，保证后续 log 计算不报错
        rho_final = self._clean_density_matrix(rho_final)

        # ---------------------------------------------------------
        # 3. 计算期望值 (Expectation Values)
        # ---------------------------------------------------------
        obs_exp = []

        # 【性能提示】如果训练慢，建议在 __init__ 中预先计算好这些矩阵并存为 self.obs_matrices
        for obs in self.state_observables:
            # 获取算符矩阵 (64x64)
            # 注意：确保 obs 是定义在 self.qubits (0-5) 上的
            obs_matrix = obs.matrix(qubits=self.qubits)

            # 计算 Trace(rho * O)
            # 使用 real 丢弃 1e-18 级别的虚部噪声
            val = np.real(np.trace(rho_final @ obs_matrix))
            obs_exp.append(val)

        # ---------------------------------------------------------
        # 4. 计算纠缠熵 (Renyi Entropy)
        # ---------------------------------------------------------
        # 【关键修正】必须把当前的 rho_final 传给计算函数
        # 假设你关注的是 Target 系统 (q0, q1, q2) 和 Resource (q3, q4, q5) 的纠缠
        # 或者 Alice (q0,3) vs Bob (q1,4) vs Charlie (q2,5)

        # 这里调用一个修改版的 _get_renyi，传入当前态
        s_tot, s_a, s_b = self._get_renyi(rho_final)

        #purity_out = np.trace(rho_final @ rho_final).real
        #print(f"DEBUG: _get_obs Final Rho Purity: {purity_out:.4f}")

        # ---------------------------------------------------------
        # 5. 格式化输出
        # ---------------------------------------------------------
        gate_count = len(self.circuit_gates)

        # 转换为 float32 并截断，防止神经网络数值不稳定
        expectations = np.array(obs_exp).real.astype(np.float32)
        expectations = np.clip(expectations, -1.0, 1.0)

        return {
            "expectations": expectations,
            "gate_count": int(gate_count),
            # 这里的 entropy 可以作为一个辅助特征，帮助 Agent 判断是否发生了解纠缠
            "entanglement_a": np.array([s_a], dtype=np.float32),
        }

    def _get_obs_0420(self):
        # 1. 此时的 self.rho 已经是 step 实时更新好的最新态！
        rho_final = self.rho

        # 2. 计算期望值 (极速矩阵乘法求迹)
        obs_exp = []
        for obs in self.state_observables:
            # 建议：如果你觉得这步还是慢，可以在 __init__ 里缓存 obs_matrix
            obs_matrix = obs.matrix(qubits=self.qubits)
            val = np.real(np.trace(rho_final @ obs_matrix))
            obs_exp.append(val)

        # 3. 计算纠缠熵
        # 传入当前最新的 rho_final 即可
        s_tot, s_a, s_b = self._get_renyi(rho_final)

        # 4. 格式化输出
        gate_count = self.step_counter
        expectations = np.array(obs_exp, dtype=np.float32)
        expectations = np.clip(expectations, -1.0, 1.0)
        # 获取刚才存储的 syndrome
        syndromes = getattr(self, 'last_syndromes', [0.0, 0.0, 0.0])

        return {
            "expectations": expectations,
            "gate_count": int(gate_count),
            "entanglement_a": np.array([s_a], dtype=np.float32),
            "syndromes": np.array(syndromes, dtype=np.float32)  # 新增反馈
        }

    def _get_obs(self):
        # 1. 获取当前最新态
        rho_final = self.rho

        # ==========================================
        # A. 计算 Expectations
        # ==========================================
        obs_exp = []
        for obs in self.state_observables:
            obs_matrix = obs.matrix(qubits=self.qubits)
            val = np.real(np.trace(rho_final @ obs_matrix))
            obs_exp.append(val)

        # 强制转换为 float32，处理 NaN，并严格截断在 [-1.0, 1.0] 之间
        expectations = np.array(obs_exp, dtype=np.float32)
        expectations = np.nan_to_num(expectations, nan=0.0, posinf=1.0, neginf=-1.0)
        expectations = np.clip(expectations, -1.0, 1.0)

        # ==========================================
        # B. 格式化 Gate Count (Step Counter)
        # ==========================================
        # 确保步数无论如何不会超过 Discrete 空间的最大值
        gate_count = int(self.step_counter)
        gate_count = min(gate_count, self.max_timesteps)

        # ==========================================
        # C. 计算 Entanglement (纠缠熵)
        # ==========================================
        s_tot, s_a, s_b = self._get_renyi(rho_final)

        # 强制转为数组 float32，处理 NaN，并严格截断在 [0.0, 1.0] 之间
        entanglement_a = np.array([s_a], dtype=np.float32)
        entanglement_a = np.nan_to_num(entanglement_a, nan=0.0, posinf=1.0, neginf=0.0)
        entanglement_a = np.clip(entanglement_a, 0.0, 1.0)


        # ==========================================
        # 返回严格匹配 Observation Space 的字典
        # ==========================================
        return {
            "expectations": expectations,
            "gate_count": gate_count,  # Discrete 接受标准 Python int 即可
            "entanglement_a": entanglement_a,
            "topology": self._topology_one_hot(),
            "noise": self._noise_observation(),
            #"syndromes": syndromes_arr
        }

    def _get_renyi2(self, rho_6q=None):
        """
        计算纠缠熵 (Renyi-2 Entropy) S = -ln(Tr(rho_A^2))
        """
        if rho_6q is None:
            rho_6q = self.rho  # Fallback

        # 假设我们要计算 Target 系统 (前3个 qubit) 的纠缠熵
        # 这反映了 Target 和 Resource 混合的程度
        # 纯化成功时，Target 应该尽可能纯，即 Entropy -> 0

        # 1. 计算 Target 子系统 (q0, q1, q2) 的约化密度矩阵
        rho_sys = self.manual_partial_trace(rho_6q, keep_indices=[0, 1, 2], dims=[2] * 6)

        # 2. 计算纯度 P = Tr(rho^2)
        purity = np.real(np.trace(rho_sys @ rho_sys))

        # 3. 计算 Renyi 熵
        # 加上 1e-9 防止 log(0)
        s_sys = -np.log(purity + 1e-9)

        # 这里你可以根据需要返回其他部分的熵
        # 为了兼容你的接口 s_tot, s_a, s_b
        return 0.0, s_sys, 0.0

    def _get_fidelity(self):
        circuit = self._get_cirq(maybe_add_noise=True)
        pred = self.simulator.simulate(circuit).final_state_vector
        inner = np.inner(np.conj(pred), self.target)
        fidelity = np.conj(inner) * inner
        return fidelity.real

    def set_attr(self, attr_name, value):
        setattr(self, attr_name, value)

    def step_qas(self, action):

        action_gate = self.action_gates[action]  # 原有，选择动作
        self.circuit_gates.append(action_gate)  # 原有，执行动作
        '''DICT空间
        # --- 你提供的正确代码段 开始 ---
        assert isinstance(action, dict), f"Step function received action of type {type(action)}, expected dict."
        assert "discrete" in action, "Action dictionary missing 'discrete' key."
        assert "continuous" in action, "Action dictionary missing 'continuous' key."

        discrete_action = action["discrete"]
        continuous_parameter_array = action["continuous"] # 重命名一下更清晰，这是个数组
        # --- 你提供的正确代码段 结束 ---

        # --- 后续逻辑 开始 ---
        if discrete_action < self.num_fixed_actions:
            # 选择固定门：忽略 continuous_parameter_array
            gate = self.action_gates[discrete_action]
            # (可选) 打印调试信息
            # print(f"Step: Chose fixed gate {discrete_action}. Ignored continuous param: {continuous_parameter_array}")

        else:
            # 选择参数化门：使用 continuous_parameter_array
            param_ry_func = self.action_gates[self.num_fixed_actions] # 获取参数化门函数本身

            # 从数组中提取标量角度 (假设参数只有一个)
            if isinstance(continuous_parameter_array, np.ndarray) and continuous_parameter_array.size == 1:
                 angle = continuous_parameter_array.item() # 或者 continuous_parameter_array[0]
            else:
                 # 处理预期之外的格式
                 print(f"Warning: Unexpected continuous_param format: {continuous_parameter_array} (type: {type(continuous_parameter_array)}). Using 0.0.")
                 angle = 0.0 # 或者抛出错误

            gate = param_ry_func(angle) # 调用参数化门函数并传入角度'''

        # TD3DM连续动作选择
        # discrete_action, continuous_param = action


        # 将选中的门操作添加到电路中
        # self.circuit_gates.append(gate)
        #    到此为止————————————————————————————————————————

        # 添加操作到历史记录  (添加的是 action 的索引！)
        # self.operation_history.append(action)#没必要去除
        # 执行动作后，更新状态
        # self.state = self._get_current_state()#没必要去除

        # compute observation
        observation = self._get_obs()

        # compute fidelity
        # fidelity = self._get_fidelity()
        fidelity = self.fidelity_measure()

        # diff_F = fidelity - self.previous_fidelity
        # cur_diff_E = abs(self.current_concurrence() - self.target_concurrence())
        # diff_E = -(cur_diff_E - pre_diff_E)
        self.previous_fidelity = fidelity
        # self.previous_entanglement = self.current_concurrence()

        # 更新状态
        # self.state_history.append(np.copy(self.state))
        # penalty = self.calc_history_fidelity()

        # target_con = self.target_concurrence()  # 3bit
        # current_con = self.current_concurrence()  # 3bit

        # 初始化或传递 entangle_list
        # if not hasattr(self, 'entangle_list'):
        #    self.entangle_list = []  # 如果不存在则初始化
        # 将 entangle_list 传递给 entangle 方法
        # self.entangle(self.entangle_list)

        # if not hasattr(self, 'fidelity_list'):
        #    self.fidelity_list = []  # 如果不存在则初始化
        # 将 fidelity_list 传递给 fidelity_getlist 方法
        # self.fidelity_getlist(self.fidelity_list)

        if not hasattr(self, 'fidelity_values'):
            self.fidelity_values = []
        if not hasattr(self, 'concurrence_values'):
            self.concurrence_values = []
        # if not hasattr(self, 'fidelity_values_mean_list'):
        # self.fidelity_values_mean_list = []
        # if not hasattr(self, 'concurrence_values_mean_list'):
        # self.concurrence_values_mean_list = []

        max_gate_count = self.max_timesteps
        gate_count = len(self.circuit_gates)
        # con_difference = abs(current_con - target_con)
        s_tot_norm, s_a_norm, s_b_norm = self._get_renyi()
        # con_difference = (abs(self.s_tot_norm_tar - s_tot_norm) + abs(self.s_a_norm_tar - s_a_norm) + abs(self.s_b_norm_tar - s_b_norm))/3.
        con_difference = abs(self.s_a_norm_tar - s_a_norm)
        # con_difference = self._mixed_state_ent_diff()

        gate_score = gate_count / max_gate_count

        exploration_weight = 0.2 * (1 - fidelity)  # 0.2
        exploration_threshold = 0.6
        gate_penalty_factor = 1.
        # reward = diff_F
        # compute reward
        if fidelity >= self.fidelity_threshold:  # 优先判断 fidelity > fidelity_threshold
            reward = (-1.5 * gate_score + fidelity * 10.2 + (1 - exploration_weight) * (
                    1 - con_difference))  # -exploration_weight * con_difference注重保真度和门数
        elif self.fidelity_threshold > fidelity >= exploration_threshold:
            reward = ((0.5 * (1 - exploration_weight) * fidelity - 1. * gate_score) - 1. + 0.5 * (
                    1 - exploration_weight) * (1 - con_difference)) * 0.5  # 平衡保真度和纠缠度
        elif exploration_threshold > fidelity >= 0.4:
            reward = ((- 0.75 * gate_score + exploration_weight * (fidelity)) - 1. + (exploration_weight) * (
                    1 - con_difference)) * 0.5  # +0.18*fidelity
            # reward = - 0.7 * gate_penalty + exploration_weight * (fidelity) + 0.3 * fidelity - (exploration_weight) * ( con_difference)
            # reward = - 0.7*gate_score + exploration_weight * (1-con_difference+fidelity)+0.18*fidelity#reward = - 0.7 * gate_score + exploration_weight * ( fidelity)+0.3*fidelity
        else:
            reward = ((- 1. * (exploration_weight) * (
                    1 - fidelity) - 0.5 * gate_score) - 1.) * 0.5  # - (1- exploration_weight) * (1-fidelity) # 注重纠缠度和门数
        terminal = (fidelity > self.fidelity_threshold) or (
                len(self.circuit_gates) >= self.max_timesteps)
        truncated = False  # (len(self.circuit_gates) >= self.max_timesteps)
        '''if fidelity > self.fidelity_threshold:
            reward = fidelity - 0.01
        else:
            reward = -0.01
        terminal = (reward > 0.) or (len(self.circuit_gates) >=
                                     self.max_timesteps)
        if fidelity >= self.fidelity_threshold:  # 优先判断 fidelity > fidelity_threshold
            reward = (-1.5 * gate_score + fidelity * 2.2 + (1 - exploration_weight) * (
                        1 - con_difference))  # -exploration_weight * con_difference注重保真度和门数
        elif self.fidelity_threshold > fidelity >= exploration_threshold:
            reward = ((0.5 * (1 - exploration_weight) * fidelity - 1. * gate_score) - 1. + 0.5 * (
                        1 - exploration_weight) * (1 - con_difference)) * 0.5  # 平衡保真度和纠缠度
        elif exploration_threshold > fidelity >= 0.4:
            reward = ((- 0.75 * gate_score + exploration_weight * (fidelity)) - 1. + (exploration_weight) * (
                        1 - con_difference)) * 0.5  # +0.18*fidelity
            # reward = - 0.7 * gate_penalty + exploration_weight * (fidelity) + 0.3 * fidelity - (exploration_weight) * ( con_difference)
            # reward = - 0.7*gate_score + exploration_weight * (1-con_difference+fidelity)+0.18*fidelity#reward = - 0.7 * gate_score + exploration_weight * ( fidelity)+0.3*fidelity
        else:
            reward = ((- 1. * (exploration_weight) * (
                    1 - fidelity) - 0.5 * gate_score) - 1.) * 0.5  # - (1- exploration_weight) * (1-fidelity) # 注重纠缠度和门数
        terminal = (fidelity > self.fidelity_threshold) or (
                len(self.circuit_gates) >= self.max_timesteps)
        truncated = False #(len(self.circuit_gates) >= self.max_timesteps)


        if fidelity >= self.fidelity_threshold:  # 优先判断 fidelity > fidelity_threshold
            reward = (-1.5 * gate_score + fidelity * 2.2 + (1 - exploration_weight) * (
                    1 - con_difference))-0.01  # -exploration_weight * con_difference注重保真度和门数
        elif self.fidelity_threshold > fidelity >= exploration_threshold:
            reward = (((1 - exploration_weight) * diff_F - 1.2 * gate_score) + (
                    1- exploration_weight) * (diff_E) ) -0.01 # 平衡保真度和纠缠度
        elif exploration_threshold > fidelity >= 0.4:
            reward = ((- 1 * gate_score + (1-exploration_weight) * (diff_F)) + (exploration_weight) * (
                    diff_E) ) -0.01  #
            # reward = - 0.7 * gate_penalty + exploration_weight * (fidelity) + 0.3 * fidelity - (exploration_weight) * ( con_difference)
            # reward = - 0.7*gate_score + exploration_weight * (1-con_difference+fidelity)+0.18*fidelity#reward = - 0.7 * gate_score + exploration_weight * ( fidelity)+0.3*fidelity
        else:
            reward = ((- 1. * (exploration_weight) * (
                    1 - fidelity) - 0.85 * gate_score) )  -0.01# - (1- exploration_weight) * (1-fidelity) # 注重纠缠度和门数
            # reward = 0.3*diff_F + reward
        terminal = (fidelity > self.fidelity_threshold) or (
                len(self.circuit_gates) >= self.max_timesteps)
        #terminal = (fidelity > 0.95)
        #truncated = (len(self.circuit_gates) >=
        #             self.max_timesteps)
        if fidelity >= self.fidelity_threshold:  # 优先判断 fidelity > fidelity_threshold
            reward = (-1.5 * gate_score + fidelity * 2.2 + (1 - exploration_weight) * (
                    1 - con_difference))  # -exploration_weight * con_difference注重保真度和门数
        elif self.fidelity_threshold > fidelity >= exploration_threshold:
            reward = (((1 - exploration_weight) * fidelity - 1.2 * gate_score) + (
                    1 - exploration_weight) * (1 - con_difference)-2)*0.7   # 平衡保真度和纠缠度
        elif exploration_threshold > fidelity >= 0.4:
            reward = ((- 1 * gate_score + exploration_weight * (fidelity))  + (exploration_weight) * (
                    1 - con_difference) + 0.18 * fidelity-2)*0.7  #
            # reward = - 0.7 * gate_penalty + exploration_weight * (fidelity) + 0.3 * fidelity - (exploration_weight) * ( con_difference)
            # reward = - 0.7*gate_score + exploration_weight * (1-con_difference+fidelity)+0.18*fidelity#reward = - 0.7 * gate_score + exploration_weight * ( fidelity)+0.3*fidelity
        else:
            reward = ((- 1. * (exploration_weight) * (
                    1 - fidelity) - 0.85 * gate_score) -2) *0.7  # - (1- exploration_weight) * (1-fidelity) # 注重纠缠度和门数
        #reward = 0.3*diff_F + reward
        terminal = (fidelity > self.fidelity_threshold) or (
                len(self.circuit_gates) >= self.max_timesteps)
        terminal = (fidelity > 0.95)
        truncated = (len(self.circuit_gates) >=
                                     self.max_timesteps)
        if fidelity >= self.fidelity_threshold:  # 优先判断 fidelity > fidelity_threshold
            reward = (-1.5 * gate_score + fidelity * 2.2 + (1 - exploration_weight) * (
                        1 - con_difference))  # -exploration_weight * con_difference注重保真度和门数
        elif self.fidelity_threshold > fidelity >= exploration_threshold:
            reward = (((1 - exploration_weight) * fidelity - 1.2 * gate_score) - 2. +(
                        1 - exploration_weight) * (1 - con_difference))*0.5   # 平衡保真度和纠缠度
        elif exploration_threshold > fidelity >= 0.4:
            reward = ((- 1 * gate_score + exploration_weight * (fidelity)) - 2. + (exploration_weight) * (
                        1 - con_difference)+0.18*fidelity)*0.5#
            # reward = - 0.7 * gate_penalty + exploration_weight * (fidelity) + 0.3 * fidelity - (exploration_weight) * ( con_difference)
            # reward = - 0.7*gate_score + exploration_weight * (1-con_difference+fidelity)+0.18*fidelity#reward = - 0.7 * gate_score + exploration_weight * ( fidelity)+0.3*fidelity
        else:
            reward = ((- 1. * (exploration_weight) * (
                    1 - fidelity) - 0.85 * gate_score) - 2.)*0.5  # - (1- exploration_weight) * (1-fidelity) # 注重纠缠度和门数
        #terminal = (fidelity > self.fidelity_threshold) or (
        #        len(self.circuit_gates) >= self.max_timesteps)
        terminal = (fidelity >= self.fidelity_threshold)
        truncated = (len(self.circuit_gates) >= self.max_timesteps)
        # compute reward
        if fidelity > self.fidelity_threshold:
            reward = fidelity - 0.01
        else:
            reward = -0.01
        terminal = (reward > 0.) or (len(self.circuit_gates) >=
                                     self.max_timesteps)
        if fidelity >= self.fidelity_threshold:  # 优先判断 fidelity > fidelity_threshold
                    reward = (-1.5* gate_score +fidelity*2.2+ (1-exploration_weight) * (1 - con_difference))# -exploration_weight * con_difference注重保真度和门数
                elif self.fidelity_threshold > fidelity >= exploration_threshold:
                    reward = ((0.5*(1 - exploration_weight) * fidelity  - 0.85 * gate_score ) -1.+ 0.5*(1-exploration_weight) * (1-con_difference)) *0.5# 平衡保真度和纠缠度
                elif exploration_threshold > fidelity >= 0.4:
                    reward = ((- 0.7 * gate_score + exploration_weight * (fidelity) )-1. + (exploration_weight) * (1 - con_difference))*0.5# +0.18*fidelity
                    # reward = - 0.7 * gate_penalty + exploration_weight * (fidelity) + 0.3 * fidelity - (exploration_weight) * ( con_difference)
                    # reward = - 0.7*gate_score + exploration_weight * (1-con_difference+fidelity)+0.18*fidelity#reward = - 0.7 * gate_score + exploration_weight * ( fidelity)+0.3*fidelity
                else:
                    reward = ((- 1. * (exploration_weight) * (
                            1 - fidelity) - 0.5 * gate_score )-1.)*0.5# - (1- exploration_weight) * (1-fidelity) # 注重纠缠度和门数
                terminal = (fidelity > self.fidelity_threshold) or (
                        len(self.circuit_gates) >= self.max_timesteps)

                #repetition_penalty = self.calculate_repetition_penalty()
                #reward += penalty  # 应用惩罚
                # 计算重复操作的惩罚
                repetition_penalty = self.calculate_repetition_penalty()
                #reward += repetition_penalty  # 应用惩罚

                penalty_transitions = self.calc_history_fidelity()
                reward += 0  # 占位符
                #到此处位置为新reward'''
        info = {'fidelity': fidelity, 'circuit': self._get_cirq(), 'concurrence_diff': con_difference,  # current_con,
                'goal_achieved': terminal, }  # 'max_steps_reached': truncated
        # if terminal:  # 优先判断是否达到目标
        # info['goal_achieved'] = True  # 设置目标完成信息
        # if truncated:  # 如果也达到了最大步数，则设置truncated为True
        truncated = False
        return observation, reward, terminal, truncated, info

    # 在 qas_env.py 的 step 中修改逻辑
    def step_4bit(self, action):
        self.step_counter += 1
        action_gate = self.action_gates[action]

        terminal = False
        truncated = False
        reward = 0
        info = {}

        # --- 1. 处理终止动作 (FINISH_ACTION) ---
        if action_gate == "FINISH_ACTION":
            terminal = True

            # 约束检查 A: 严禁第一步直接终止
            if self.step_counter <= 1:
                reward = -20.0  # 极重惩罚
                return self._get_obs(), reward, terminal, truncated, {'msg': 'Forbidden: Immediate termination'}

            # 约束检查 B: 必须使用过至少一个本地双比特门
            if not self.used_cnot:
                reward = -10.0  # 较重惩罚
                return self._get_obs(), reward, terminal, truncated, {'msg': 'Failed: No CNOT used'}

            # 如果满足以上约束，计算纯化收益
            reward, info = self._evaluate_purification_protocol()
            return self._get_obs(), reward, terminal, truncated, info

        # --- 2. 处理普通门动作 ---
        self.circuit_gates.append(action_gate)

        # 更新 CNOT 使用标记
        # 注意：根据 get_default_gates 定义，判断是否为 CNOT
        # if hasattr(action_gate, 'gate') and isinstance(action_gate.gate, cirq.CNotPowGate):
        #    self.used_cnot = True
        is_cnot_detected = False

        # 情况 1: 动作是一个普通的门操作 (Operation)
        if hasattr(action_gate, 'gate'):
            if isinstance(action_gate.gate, cirq.CNotPowGate):
                is_cnot_detected = True

        # 情况 2: 动作是一个双边门电路 (Circuit)
        elif isinstance(action_gate, cirq.Circuit):
            # 遍历电路中的所有操作，检查是否有 CNOT
            for op in action_gate.all_operations():
                if hasattr(op, 'gate') and isinstance(op.gate, cirq.CNotPowGate):
                    is_cnot_detected = True
                    break  # 只要发现一个 CNOT 就够了

        # 如果检测到 CNOT，更新标记
        if is_cnot_detected:
            self.used_cnot = True

        # 检查是否达到硬性最大步数
        if self.step_counter >= self.max_timesteps:
            terminal = True
            reward, info = self._evaluate_purification_protocol()
        else:
            # 每一步执行门的操作成本 (鼓励 Agent 寻找最短序列)
            reward = -0.1 - 1 * (self.step_counter / self.max_timesteps)
            info = {'msg': 'Step in progress'}

        return self._get_obs(), reward, terminal, truncated, info

    def step_backup0401(self, action):
        self.step_counter += 1
        #if self.step_counter == 1:

        action_gate = self.action_gates[action]

        terminal = False
        truncated = False


        # --- 1. 处理终止动作 ---
        if action_gate == "FINISH_ACTION":
            terminal = True
            # 禁止第一步直接终止
            if self.step_counter <= 1:
                return self._get_obs(), -0.2, True, False, {'msg': 'Forbidden immediate finish'}#-20
            # 必须使用过 CNOT
            if not self.used_cnot:
                return self._get_obs(), -0.1, True, False, {'msg': 'Failed: No CNOT used'}#-10

            # 调用你刚才写的 6-bit 评估逻辑
            reward, info = self._evaluate_purification_protocol()
            #reward += 0.5
            return self._get_obs(), reward, terminal, truncated, info


        #if is_cnot:
            #self.used_cnot = True

            # --- 2. 处理门操作：核心加固 (支持 CNOT 和 TOFFOLI) ---


            # 定义我们需要检测的门类型元组
            # CNotPowGate 对应 CNOT, CCXPowGate 对应 Toffoli
        target_gates = (cirq.CNotPowGate, cirq.CCXPowGate,cirq.CZPowGate,)

        if isinstance(action_gate, cirq.Circuit):
                # 检查电路里的每一个操作
            for op in action_gate.all_operations():
                if hasattr(op, 'gate') and isinstance(op.gate, target_gates):
                    self.is_entangling = True
                    break
        elif hasattr(action_gate, 'gate') and isinstance(action_gate.gate, target_gates):
            self.is_entangling = True

        if self.is_entangling:
                # 你可以保留 self.used_cnot 这个名字，
                # 但为了严谨，建议在类初始化时加一个 self.used_multi_qubit_gate = True
            self.used_cnot = True

        # 正式添加到电路记录中
        self.circuit_gates.append(action_gate)

        curr_potential_f, curr_p_succ = self._peek_potential_fidelity()
        # --- 3. 时间步判定 ---
        if self.step_counter >= self.max_timesteps or curr_potential_f > self.F_in + 0.01:
            # 必须使用过 CNOT
            if not self.used_cnot:
                truncated = True
                return self._get_obs(), -0.1, False, truncated, {'msg': 'Failed: No CNOT used'}#-10
            else:
                terminal = True
            reward, info = self._evaluate_purification_protocol()
        else:
            reward = 0#(-0.02 ) + r_c*0.5 #- (self.step_counter/self.max_timesteps)*0.25)
            info = {}

        return self._get_obs(), reward, terminal, truncated, info

    def step260420_9bit(self, action):
        self.step_counter += 1
        action_gate = self.action_gates[action]

        terminal = False
        truncated = False

        # --- 1. 处理终止动作 ---
        if action_gate == "FINISH_ACTION":
            terminal = True
            if self.step_counter <= 1:
                return self._get_obs(), -0.2, True, False, {'msg': 'Forbidden immediate finish'}
            if not self.used_cnot:
                return self._get_obs(), -0.1, True, False, {'msg': 'Failed: No CNOT used'}

            reward, info = self._evaluate_purification_protocol()
            return self._get_obs(), reward, terminal, truncated, info

        # =========================================================
        # 🚀 2. 马尔可夫极速状态演化 (只演化当前门)
        # =========================================================
        step_circuit = cirq.Circuit()

        # 拆解动作：处理双边门电路 (Circuit) 或单比特门 (Operation)
        if isinstance(action_gate, cirq.Circuit):
            ops = list(action_gate.all_operations())
        else:
            ops = [action_gate]

        # 挨个加门，并当场注入噪声
        for op in ops:
            step_circuit.append(op)

            # 获取操作作用的量子比特
            op_qubits = op.qubits

            # 注入单比特/多比特去极化噪声
            # 加上 > 0 的判断，防止传入 0.0 时依然强行生成空白信道
            if len(op_qubits) == 1 and self.error_single is not None and self.error_single > 0:
                noise_gate = cirq.depolarize(self.error_single).on_each(*op_qubits)
                step_circuit.append(noise_gate)
            elif len(op_qubits) > 1 and self.error_multi is not None and self.error_multi > 0:
                noise_gate = cirq.depolarize(self.error_multi).on_each(*op_qubits)
                step_circuit.append(noise_gate)

        # 极速演化：在上一刻的态 (self.rho) 上演化这一个带噪声的短电路
        res = self.simulator.simulate(step_circuit, initial_state=self.rho, qubit_order=self.qubits)

        raw_rho = res.final_density_matrix

        # 清洗并检查物理态
        cleaned_rho = self._clean_density_matrix(raw_rho)

        # 🌟 【绝对物理严谨的防线】：吸收态截断
        if cleaned_rho is None:
            # 物理模拟已经因为 Agent 的极端动作组合而引发数值崩溃。
            # 不要伪造态！直接判定为死路，结束本回合！
            #terminal = True

            # 给一个比普通失败（-0.01）更重的惩罚，比如 -0.1
            # 告诉网络：“这个动作组合连物理模拟器都跑不下去，绝对是个烂操作”
            reward = -0.1
            info = {'msg': 'Numerical Simulation Collapsed', 'goal_achieved': False}

            # 因为序列提前终止，为了保证 obs 的连续性，可以返回上一步的旧 rho，或者一个全0向量
            # 这不会影响物理计算，因为 done=True，网络只会把它当做截断点
            return self._get_obs(), reward, False, True, info

        # 如果态是健康的，正常赋值
        self.rho = cleaned_rho

        #self.rho = self._clean_density_matrix(res.final_density_matrix)

        # 记录门历史 (仅用作日志记录，不再用于重复模拟！)
        self.circuit_gates.append(action_gate)

        # --- 3. 处理门操作：核心加固 (检测纠缠门) ---
        target_gates = (cirq.CNotPowGate, cirq.CCXPowGate, cirq.CZPowGate)
        if isinstance(action_gate, cirq.Circuit):
            for op in action_gate.all_operations():
                if hasattr(op, 'gate') and isinstance(op.gate, target_gates):
                    self.is_entangling = True
                    self.used_cnot = True
                    break
        elif hasattr(action_gate, 'gate') and isinstance(action_gate.gate, target_gates):
            self.is_entangling = True
            self.used_cnot = True

        # =========================================================
        # 🚀 4. 目标导向：自动截断机制
        # =========================================================
        reward = 0.0
        info = {}

        # 只要用了 CNOT，就偷偷检查是否已经达到了目标！
        if self.used_cnot:
            eval_reward, eval_info = self._evaluate_purification_protocol()

            # 如果你的评估函数判定已经成功 (保真度提升 >= 0.01 且符合纠缠度要求)
            if eval_info.get('goal_achieved', False):
                terminal = True
                # 还可以选择是否达到真正的纯化阈值 (比如 0.95)
                # if eval_info['final_f'] >= self.fidelity_threshold: ...

                info = eval_info
                info['msg'] = 'Auto-Success! Target Reached!'
                return self._get_obs(), eval_reward, terminal, truncated, info

        # --- 5. 时间步判定 ---
        if self.step_counter >= self.max_timesteps:
            if not self.used_cnot:
                truncated = True  # 你原本逻辑写了 truncated=True
                return self._get_obs(), -0.1, False, truncated, {'msg': 'Failed: No CNOT used'}
            else:
                #terminal = True
                truncated = True
                reward, info = self._evaluate_purification_protocol()

        return self._get_obs(), reward, terminal, truncated, info

    def step_0423_can_work(self, action):
        self.step_counter += 1
        action_gate = self.action_gates[action]  # 这里的 action 是 0-1535 的索引

        terminal = False
        truncated = False

        # --- 1. 处理终止动作 (保持原逻辑) ---
        # --- 1. 处理终止动作 ---
        # 只要是字符串类型的动作，就说明 Agent 要收网了
        if isinstance(action_gate, str) and action_gate.startswith("FINISH_ACTION"):
            terminal = True

            # 防呆机制：不能第一步就结束，也不能不加门就结束
            if self.step_counter <= 1:
                return self._get_obs(), -0.2, True, False, {'msg': 'Forbidden immediate finish'}
            if not self.used_cnot:
                return self._get_obs(), -0.2, True, False, {'msg': 'Failed: No CNOT used'}

            # 🚀 核心：解析 Agent 选的是 P1 还是 P2
            selected_mode = 'P1' if action_gate == "FINISH_ACTION_P1" else 'P2'

            # 🚀 核心：把 Agent 选的模式作为参数传给评估函数！
            reward, info = self._evaluate_purification_protocol(resource_mode=selected_mode)

            # 记录在 info 里，方便你 debug 和看曲线
            info['finish_mode'] = selected_mode

            return self._get_obs(), reward, terminal, truncated, info

        # =========================================================
        # 🚀 2. 复合动作极速演化 (1536 体系的核心)
        # =========================================================
        # 直接使用 action_gate (即生成的复合电路)，不要拆解成单门
        # 注意：这里我们给整个复合电路挂载噪声模型
        noisy_circuit = cirq.Circuit()
        if isinstance(action_gate, cirq.Circuit):
            # 将复合电路包装并注入噪声
            for op in action_gate.all_operations():
                noisy_circuit.append(op)
                # 注入去极化噪声
                qbs = op.qubits
                error = self.error_single if len(qbs) == 1 else self.error_multi
                if error and error > 0:
                    noisy_circuit.append(cirq.depolarize(error).on_each(*qbs))
        else:
            # 兼容单比特门
            noisy_circuit.append(action_gate)
            if self.error_single and self.error_single > 0:
                noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*action_gate.qubits))

        # 演化 9-qubit 总态
        res = self.simulator.simulate(noisy_circuit, initial_state=self.rho, qubit_order=self.qubits)
        raw_rho = res.final_density_matrix

        # =========================================================
        # 🚀 3. 即时探测与辅助比特重置 (重要物理环节)
        # =========================================================
        # 模拟“探测-测量-重置”循环。如果不重置，Agent 下一跳就会观测到脏数据

        # A. 提取辅助比特 q6, q7, q8 的测量概率/结果 (Syndrome)
        # 假设我们通过计算投影后的期望值来模拟测量结果
        # syndromes, rho_after_measurement = self._perform_immediate_measurement(raw_rho)
        #
        # B. 物理清洗
        cleaned_rho = self._clean_density_matrix(raw_rho)#(rho_after_measurement)
        if cleaned_rho is None:
            return self._get_obs(), -0.2, False, True, {'msg': 'Simulation Collapsed'}

        self.rho = cleaned_rho
        #self.last_syndromes = syndromes  # 存入实例变量，供 _get_obs 使用

        # 记录历史
        self.circuit_gates.append(action_gate)

        # --- 检测纠缠门记录 ---
        if not self.used_cnot:
            target_gates = (cirq.CNotPowGate, cirq.CCXPowGate, cirq.CZPowGate)
            ops_to_check = action_gate.all_operations() if isinstance(action_gate, cirq.Circuit) else [action_gate]
            for op in ops_to_check:
                if hasattr(op, 'gate') and isinstance(op.gate, target_gates):
                    self.used_cnot = True
                    break

        # --- 4. 自动截断与评价 ---
        reward = 0.0
        if self.step_counter >= self.max_timesteps:
            truncated = True  # 标记为超时截断

            # 给一个比普通单步 (-0.01) 更重的超时惩罚，告诉它“拖延症是不好的”
            reward = -0.2

            # 直接返回初始保真度，表示这次提纯不仅没成功，连结算都没完成
            info = {
                'msg': 'Timeout: Agent failed to choose a FINISH action.',
                'goal_achieved': False,
                'final_f': self.F_in,
                'p_succ': 0.0,
                'rounds': 0
            }
            return self._get_obs(), reward, False, truncated, info

            # 如果还没到最大步数，正常返回 Step in progress
        reward = 0#-0.01  # 给予微小的探索惩罚，鼓励寻找最短路径
        info = {'msg': 'Step in progress'}

        return self._get_obs(), reward, False, False, info

    def step(self, action):
        self.step_counter += 1
        action_gate = self.action_gates[action]

        if isinstance(action_gate, str) and action_gate in self.topology_action_map:
            if self.topology_selected:
                return self._get_obs(), -0.2, True, False, {
                    'msg': 'Forbidden immediate finish',
                    'invalid_reason': 'Topology already selected.',
                    'topology': self.current_topology,
                }

            self._select_topology(action_gate)
            self.step_counter -= 1
            return self._get_obs(), 0.0, False, False, {
                'msg': 'Topology selected.',
                'topology': self.current_topology,
            }

        if not self.topology_selected:
            return self._get_obs(), -0.2, True, False, {
                'msg': 'Forbidden immediate finish',
                'invalid_reason': 'Select topology before protocol actions.',
            }

        # --- 1. 处理终止动作 (解析模式与掩码) ---
        if isinstance(action_gate, str) and action_gate.startswith("FINISH"):
            if self.step_counter <= 1:
                return self._get_obs(), -0.2, True, False, {
                    'msg': 'Forbidden immediate finish',
                    'invalid_reason': 'Finish immediately after topology selection.',
                    'topology': self.current_topology,
                }

            if not self.used_cnot:
                return self._get_obs(), -0.2, True, False, {
                    'msg': 'Failed: No two-qubit gate used',
                    'invalid_reason': 'Finish without any two-qubit/topology-local entangling action.',
                    'topology': self.current_topology,
                }

            if action_gate not in self.finish_action_map:
                raise ValueError(f"Unknown finish action: {action_gate}")

            mode, mask_policy = self.finish_action_map[action_gate]

            if self.ancilla_count == 2:
                reward, info = self._evaluate_dynamic_post_selection_10bit(mode, mask_policy)
            else:
                reward, info = self._evaluate_dynamic_post_selection(mode, mask_policy)

            info["finish_mode"] = "ANC_ONLY" if mode is None else mode
            info["finish_mask"] = mask_policy

            return self._get_obs(), reward, True, False, info

        if not self._is_action_allowed_by_topology(action_gate):
            return self._get_obs(), -0.2, True, False, {
                'msg': 'Forbidden immediate finish',
                'invalid_reason': 'Invalid nonlocal action for selected topology.',
                'topology': self.current_topology,
            }

        # =========================================================
        # 🚀 2. 复合动作极速演化 (注入硬件噪声，保持全相干)
        # =========================================================
        noisy_circuit = cirq.Circuit()
        if isinstance(action_gate, cirq.Circuit):
            for op in action_gate.all_operations():
                noisy_circuit.append(op)
                qbs = op.qubits
                error = self.error_single if len(qbs) == 1 else self.error_multi
                if error and error > 0:
                    noisy_circuit.append(cirq.depolarize(error).on_each(*qbs))
        else:
            noisy_circuit.append(action_gate)
            if self.error_single and self.error_single > 0:
                noisy_circuit.append(cirq.depolarize(self.error_single).on_each(*action_gate.qubits))

        # 演化 9-qubit 总态，不塌缩
        res = self.simulator.simulate(noisy_circuit, initial_state=self.rho, qubit_order=self.qubits)
        raw_rho = res.final_density_matrix

        # =========================================================
        # 🚀 3. 纯数值清理 (警告：仅限除浮点误差，绝对禁止偏迹/重置)
        # =========================================================
        cleaned_rho = self._clean_density_matrix(raw_rho)
        if cleaned_rho is None:
            return self._get_obs(), -0.2, False, True, {'msg': 'Simulation Collapsed'}

        self.rho = cleaned_rho
        self.circuit_gates.append(action_gate)

        # 检测纠缠门记录
        if not self.used_cnot:
            target_gates = (cirq.CNotPowGate, cirq.CCXPowGate, cirq.CZPowGate, cirq.SwapPowGate)
            ops_to_check = action_gate.all_operations() if isinstance(action_gate, cirq.Circuit) else [action_gate]
            for op in ops_to_check:
                if hasattr(op, 'gate') and isinstance(op.gate, target_gates):
                    self.used_cnot = True
                    break

        # =========================================================
        # 🚀 4. 自动截断与探索惩罚
        # =========================================================
        if self.step_counter >= self.max_timesteps:
            truncated = True
            reward = -0.2  # 严厉的拖延惩罚
            info = {
                'msg': 'Timeout: Failed to FINISH.',
                'goal_achieved': False,
                'final_f': self.F_in,
                'p_succ': 0.0,
            }
            return self._get_obs(), reward, False, truncated, info

        # 微小的步骤惩罚，逼迫 Agent 压缩量子线路深度
        reward = 0#-0.005
        info = {'msg': 'Step in progress'}

        return self._get_obs(), reward, False, False, info

    def _perform_immediate_measurement(self, rho_9q):
        """
        物理逻辑：对 q6, q7, q8 进行计算基测量，获取 syndrome，并将它们重置为 |0>。
        """
        # 1. 计算三个辅助比特分别为 1 的概率，模拟 Syndrome
        # 这里简化处理：Agent 观察到的是辅助比特处于 |1> 的“可能性”
        syndromes = []
        for anc_idx in [6, 7, 8]:
            # 构造 Z 算符矩阵
            z_op = cirq.Z(self.qubits[anc_idx]).matrix(qubits=self.qubits)
            # 期望值 <Z> = P(0) - P(1)
            exp_z = np.real(np.trace(rho_9q @ z_op))
            # 转换为 P(1) = (1 - <Z>)/2
            p_one = (1.0 - exp_z) / 2.0
            syndromes.append(float(p_one))

        # 2. 物理重置：对辅助比特求偏迹 (Trace out)
        # 抛弃辅助比特，保留 6 个主比特 (q0-q5)
        rho_6q = self.manual_partial_trace(rho_9q, keep_indices=list(range(6)), dims=[2] * 9)

        # 3. 重新扩容：为了保持 self.rho 始终是 9-qubit (512x512)
        # 补上干净的 |000><000|
        rho_anc_clean = np.zeros((8, 8), dtype=complex);
        rho_anc_clean[0, 0] = 1.0
        rho_9q_new = np.kron(rho_6q, rho_anc_clean)

        return syndromes, rho_9q_new

    def _calculate_purification_metrics(self, rho_4q):
        """
        辅助方法：计算 4-qubit 密度矩阵在测量 q2, q3 后的后选择保真度
        qubit_order: [q0, q1, q2, q3] -> Alice(0,2), Bob(1,3)
        """
        # 1. 定义测量 q2, q3 结果一致的投影算子: P = |00><00| + |11><11| (作用在 q2, q3 上)
        # 对应的 4-qubit 投影矩阵
        dim = 2 ** len(self.qubits)
        p_succ_mask = np.zeros((dim, dim), dtype=complex)

        for i in range(dim):
            # 提取 q2, q3 的状态 (第 2 和 第 3 位，假设 little-endian)
            # 根据你 reset 时的 qubit_order 调整位运算
            q2_val = (i >> 2) & 1
            q3_val = (i >> 3) & 1

            if q2_val == q3_val:  # 测量结果一致的分支
                p_succ_mask[i, i] = 1.0

        # 2. 应用投影
        rho_projected = p_succ_mask @ rho_4q @ p_succ_mask
        p_succ = np.trace(rho_projected).real

        if p_succ < 1e-6:
            return 0.0, 0.0

        # 3. 归一化投影后的态
        rho_normalized = rho_projected / p_succ

        # 4. 对 q2, q3 求偏迹，得到剩下的 q0, q1 的密度矩阵
        # 使用你原本代码中的 manual_partial_trace
        # 追踪掉 index 2 和 3
        rho_01 = self.manual_partial_trace(rho_normalized, [2, 3], [2, 2, 2, 2])

        # 5. 计算 rho_01 与理想贝尔态 |Φ+> 的保真度
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        fidelity = np.real(np.vdot(phi_plus, rho_01 @ phi_plus))

        return fidelity, p_succ

    def calc_history_fidelity(self):
        """计算历史状态之间的保真度，并返回需要惩罚的 transition 索引"""
        penalty_transitions = []  # 用于存储需要惩罚的transition, 每个元素是 (start_index, end_index)
        if self.state_history is None or len(self.state_history) < 2:
            return penalty_transitions  # 返回空列表
        his_state = self.state_history  # 不需要copy，直接使用
        for i in range(len(his_state)):
            for j in range(i + 1, len(his_state)):
                inner = np.inner(np.conj(his_state[i]), his_state[j])  # 计算内积
                F_h = np.abs(inner) ** 2  # 保真度计算
                if np.isclose(F_h, 1, atol=1e-4):
                    for k in range(i + 1, j + 1):
                        penalty_transitions.append((k - 1, k))  # 添加所有的transition
        return penalty_transitions

    def get_action_matrix(self, action_gate):
        """
        获取 Cirq 量子门的矩阵表示
        """
        if isinstance(action_gate, cirq.Operation):
            return cirq.unitary(action_gate.gate)
        elif isinstance(action_gate, cirq.Circuit):
            return cirq.unitary(action_gate)
        else:
            return cirq.unitary(action_gate)

    def is_near_identity(self, matrix, tol=1e-5):
        """判断矩阵是否接近单位矩阵"""
        identity = np.eye(matrix.shape[0])
        return np.allclose(matrix, identity, atol=tol)

    def entangle(self, entangle_list):
        entangle_list.append(self.current_concurrence())
        return entangle_list

    def fidelity_getlist(self, fidelity_list):
        fidelity_list.append(self.current_concurrence())
        return fidelity_list

    def _get_current_state(self):
        circuit = self._get_cirq(maybe_add_noise=True)
        state = self.simulator.simulate(circuit).final_state_vector
        return state

    def calculate_repetition_penalty(self):
        """计算重复操作的惩罚"""
        penalty = 0
        if len(self.operation_history) < self.window_size:
            return penalty

        # 1. 检测作用在相同量子比特上的非连续重复自逆门
        for i in range(len(self.operation_history)):
            for j in range(i + 1, len(self.operation_history)):
                if self.are_same_self_inverse_gates(self.operation_history[i],
                                                    self.operation_history[j]):
                    gate1 = self.action_gates[self.operation_history[i]]
                    gate2 = self.action_gates[self.operation_history[j]]
                    if isinstance(gate1, cirq.Operation) and isinstance(gate2, cirq.Operation) and len(
                            gate1.qubits) > 0 and len(gate2.qubits) > 0 and gate1.qubits == gate2.qubits:  # 作用在同一个量子比特
                        penalty -= self.penalty_weight

        # 2. 检测作用在相同量子比特上非连续的周期性旋转门
        for i in range(len(self.operation_history)):
            for j in range(i + 1, len(self.operation_history)):
                action1 = self.operation_history[i]
                action2 = self.operation_history[j]
                action_gate1 = self.action_gates[action1]
                action_gate2 = self.action_gates[action2]
                if isinstance(action_gate1, cirq.Operation) and action_gate1.gate in [cirq.rz, cirq.ry] and isinstance(
                        action_gate2, cirq.Operation) and action_gate2.gate in [cirq.rz, cirq.ry]:  # 修改
                    if action_gate1.qubits == action_gate2.qubits:  # 作用在同一个量子比特
                        angle1 = self.get_rotation_angle(action_gate1.gate)
                        angle2 = self.get_rotation_angle(action_gate2.gate)
                        if self.is_close_to_multiple_of_2pi(angle1 + angle2):  # 判断角度之和是否构成周期
                            penalty -= self.penalty_weight  # 使用所有参数计算角度

        # 3. 检测重复的门序列
        for length in range(2, len(self.operation_history) // 2 + 1):
            for i in range(len(self.operation_history) - 2 * length + 1):
                if all(self.operation_history[i + j] == self.operation_history[i + length + j] for j in
                       range(length)):
                    penalty -= self.penalty_weight

        return penalty

    def is_close_to_multiple_of_2pi(self, angle, tol=1e-4):
        """判断角度是否是 2pi 的整数倍"""
        return np.isclose(angle % (2 * np.pi), 0, atol=tol)

    def are_same_self_inverse_gates(self, action1, action2):
        """判断两个门是否是相同的自逆门"""
        if not isinstance(action1, int) or not isinstance(action2, int):
            return False

        gate1 = self.action_gates[action1]
        gate2 = self.action_gates[action2]

        if isinstance(gate1, cirq.Operation) and isinstance(gate2, cirq.Operation):
            return gate1.gate in [cirq.X, cirq.Y, cirq.Z, cirq.H] and gate1.gate == gate2.gate
        elif isinstance(gate1, cirq.Operation) and isinstance(gate2, str):  # str 只能是 之前定义的 h, x, y, z, 避免重复定义
            return gate1.gate in [cirq.X, cirq.Y, cirq.Z, cirq.H] and gate1.gate.__name__ == gate2
        else:
            return False  # 不是自逆门

    def get_rotation_angle(self, gate):
        """获取旋转门的旋转角度"""
        if hasattr(gate, "angle"):
            return gate.angle.real
        elif hasattr(gate, "exponent"):
            return gate.exponent * np.pi
        else:
            return 0  # 如果不是旋转门，则返回0

    def _update_last_state(self, current_state):
        self.last_state = current_state

    def _update_last_last_state(self, last_state):
        self.last_last_state = last_state

    # FIDELITY FUNCTIONS
    # 0 -> 1 (1 is when both Unitaries are the same)
    def Fnorm(self, achieved_goal):
        desired_goal = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0]
        ])
        max_axis = achieved_goal.ndim
        frobenius_norm = np.linalg.norm(achieved_goal - desired_goal, ord='fro',
                                        axis=(max_axis - 2, max_axis - 1)) / 2 ** (3 + 1)
        # F norms gives us distance so Fidelity = 1 - distance
        return 1 - frobenius_norm

    def render(self, mode='human'):
        if self.render_mode is not None:
            if mode != self.render_mode:
                raise AttributeError(
                    f"render mode asked to render with {mode} but environment has been initialized to render with {self.render_mode}."
                )
        else:
            self.render_mode = mode

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n' + self._get_cirq(False).__str__() + '\n')

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
        return None

    def target_concurrence(self):
        d1 = (self.target[0] * self.target[0] * self.target[7] * self.target[7] + self.target[1] * self.target[1] *
              self.target[6] * self.target[6] + self.target[2] *
              self.target[2] * self.target[5] * self.target[5] + self.target[4] * self.target[4] * self.target[3] *
              self.target[3])
        d2 = (self.target[0] * self.target[7] * self.target[3] * self.target[4] + self.target[0] * self.target[7] *
              self.target[5] * self.target[2] + self.target[
                  0] * self.target[7] * self.target[6] * self.target[1]
              + self.target[3] * self.target[4] * self.target[5] * self.target[2] + self.target[3] * self.target[4] *
              self.target[6] * self.target[1]
              + self.target[5] * self.target[2] * self.target[6] * self.target[1])
        d3 = self.target[0] * self.target[6] * self.target[5] * self.target[3] + self.target[7] * self.target[1] * \
             self.target[2] * self.target[4]
        target_con = 4 * abs(d1 - 2 * d2 + 4 * d3)
        return target_con

    def current_concurrence(self):
        circuit = self._get_cirq(maybe_add_noise=True)
        pred = self.simulator.simulate(circuit).final_state_vector
        d1 = (pred[0] * pred[0] * pred[7] * pred[7] + pred[1] * pred[1] * pred[6] * pred[6] + pred[2] *
              pred[2] * pred[5] * pred[5] + pred[4] * pred[4] * pred[3] * pred[3])
        d2 = (pred[0] * pred[7] * pred[3] * pred[4] + pred[0] * pred[7] * pred[5] * pred[2] + pred[0] * pred[7] * pred[
            6] * pred[1]
              + pred[3] * pred[4] * pred[5] * pred[2] + pred[3] * pred[4] * pred[6] * pred[1]
              + pred[5] * pred[2] * pred[6] * pred[1])
        d3 = pred[0] * pred[6] * pred[5] * pred[3] + pred[7] * pred[1] * pred[2] * pred[4]
        current_con = 4 * abs(d1 - 2 * d2 + 4 * d3)
        return current_con

    def current_unitary(self):
        circuit = self._get_cirq(maybe_add_noise=True)
        current_unitary = cirq.unitary(circuit)
        return current_unitary

    def unitary_fidelity(self):
        """计算两个酉矩阵之间的平均门保真度."""
        # 定义目标 Toffoli 门矩阵 (使用与 Cirq 相同的基态排序 - little-endian)
        target_toffoli = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0]
        ])
        circuit = self._get_cirq(maybe_add_noise=True)
        circuit_unitary = cirq.unitary(circuit)
        d = target_toffoli.shape[0]
        overlap = np.abs(np.trace(circuit_unitary.conj().T @ target_toffoli))
        unitary_fidelity = (overlap / d)  # + 1) / (d + 1)
        return unitary_fidelity

    def concurrence_pure(self):
        """计算两量子比特纯态的 concurrence。

        公式: C(|ψ⟩) = |⟨ψ|σ_y ⊗ σ_y|ψ*⟩|

        其中:
            - |ψ*⟩ 是 |ψ⟩ 的复共轭态
            - σ_y 是 Pauli-Y 矩阵

        参数:
            psi: 一个长度为 4 的 NumPy 数组，表示两量子比特纯态的状态向量。

        返回:
            该状态的 concurrence，一个 0 到 1 之间的浮点数。
        """
        psi = self.target
        if psi.shape != (4,):
            raise ValueError("输入状态向量的形状必须是 (4,)。")

        #  σ_y 矩阵
        sigma_y = np.array([[0, -1j], [1j, 0]])
        # σ_y ⊗ σ_y (张量积)
        sigma_y_otimes_sigma_y = np.kron(sigma_y, sigma_y)

        # |ψ*⟩ (psi 的复共轭态)
        psi_star = np.conj(psi)

        # C(|ψ⟩) = |⟨ψ|σ_y ⊗ σ_y|ψ*⟩|
        concurrence_val = np.abs(np.dot(psi_star, np.dot(sigma_y_otimes_sigma_y, psi)))

        return concurrence_val

    def concurrence_pure_current(self):
        """计算两量子比特纯态的 concurrence。

        公式: C(|ψ⟩) = |⟨ψ|σ_y ⊗ σ_y|ψ*⟩|

        其中:
            - |ψ*⟩ 是 |ψ⟩ 的复共轭态
            - σ_y 是 Pauli-Y 矩阵

        参数:
            psi: 一个长度为 4 的 NumPy 数组，表示两量子比特纯态的状态向量。

        返回:
            该状态的 concurrence，一个 0 到 1 之间的浮点数。
        """
        circuit = self._get_cirq(maybe_add_noise=True)
        psi = self.simulator.simulate(circuit).final_state_vector
        if psi.shape != (4,):
            raise ValueError("输入状态向量的形状必须是 (4,)。")

        #  σ_y 矩阵
        sigma_y = np.array([[0, -1j], [1j, 0]])
        # σ_y ⊗ σ_y (张量积)
        sigma_y_otimes_sigma_y = np.kron(sigma_y, sigma_y)

        # |ψ*⟩ (psi 的复共轭态)
        psi_star = np.conj(psi)

        # C(|ψ⟩) = |⟨ψ|σ_y ⊗ σ_y|ψ*⟩|
        concurrence_val = np.abs(np.dot(psi_star, np.dot(sigma_y_otimes_sigma_y, psi)))

        return concurrence_val

    def fidelity_measure(self):
        """
        计算当前含噪声电路产生的混合态与预定义目标纯态 self.target 之间的保真度。
        """
        # 1. 构建并获取包含噪声的电路
        try:
            circuit_with_noise = self._get_cirq(maybe_add_noise=True)
            if not isinstance(circuit_with_noise, cirq.Circuit):
                print("Warning: _get_cirq did not return a cirq.Circuit object.")
                return 0.0  # 或者抛出错误
        except Exception as e:
            print(f"Error getting circuit with noise: {e}")
            return 0.0  # 返回 0 保真度表示失败

        # 检查目标状态向量是否有效
        if self.target is None or not isinstance(self.target, np.ndarray):
            print("Error: Target state vector (self.target) is not defined or not a NumPy array.")
            return 0.0

        # 2. 使用密度矩阵模拟器模拟含噪声电路
        try:
            simulator = cirq.DensityMatrixSimulator(seed=42)  # 可以考虑是否需要固定种子
            result = simulator.simulate(circuit_with_noise, qubit_order=self.qubits)  # 推荐指定 qubit_order
            density_matrix = result.final_density_matrix

            # 可选：强制厄米性以提高数值稳定性
            density_matrix = (density_matrix + density_matrix.conj().T) / 2
            density_matrix = density_matrix / np.trace(density_matrix)

        except Exception as e:
            print(f"Error during density matrix simulation: {e}")
            print(f"Circuit that caused error:\n{circuit_with_noise}")
            return 0.0  # 返回 0 保真度表示失败

        # 3. 计算最终密度矩阵与目标状态向量之间的保真度
        # try:
        # cirq.fidelity 可以直接处理密度矩阵和状态向量
        # fidelity_M = cirq.fidelity(density_matrix, self.target)
        # 保真度理论上应该在 [0, 1] 之间，但数值计算可能导致微小偏差
        # fidelity_M = np.clip(np.real(fidelity_M), 0.0, 1.0) # 取实部并限制范围
        # 3. 计算最终密度矩阵与目标状态向量之间的保真度
        try:
            # === 修改开始 ===
            # 方法：直接计算 <psi | rho | psi>
            # 这种方法比 cirq.fidelity 快，且不会因为微小的负特征值报错
            # 计算 rho * |psi>
            rho_times_target = density_matrix @ self.target

            # 计算 <psi| * (rho * |psi>)
            # np.vdot 会自动对第一个参数取共轭
            overlap = np.vdot(self.target, rho_times_target)

            # 理论结果必须是实数，取实部即可
            fidelity_M = np.real(overlap)

            # 限制在 [0, 1] 之间 (处理数值误差导致的 1.00000001 或 -0.00000001)
            fidelity_M = np.clip(fidelity_M, 0.0, 1.0)
            # === 修改结束 ===

        except ValueError as e:
            print(f"Error calculating fidelity: {e}")
            print(f"Density matrix shape: {density_matrix.shape}")
            print(f"Target vector shape: {self.target.shape}")
            # 检查维度是否匹配
            num_qubits = len(self.qubits)
            expected_dim = 2 ** num_qubits
            if density_matrix.shape != (expected_dim, expected_dim) or self.target.shape != (expected_dim,):
                print(f"Dimension mismatch: Expected ({expected_dim}, {expected_dim}) and ({expected_dim},)")
            return 0.0
        except Exception as e:
            print(f"Unexpected error calculating fidelity: {e}")
            return 0.0

        return fidelity_M

    def _mixed_state_ent_diff(self):
        circuit_with_noise = self._get_cirq(maybe_add_noise=True)
        simulator = cirq.DensityMatrixSimulator(seed=42)  # 可以考虑是否需要固定种子
        result = simulator.simulate(circuit_with_noise, qubit_order=self.qubits)  # 推荐指定 qubit_order
        density_matrix = result.final_density_matrix

        # 可选：强制厄米性以提高数值稳定性
        density_matrix = (density_matrix + density_matrix.conj().T) / 2
        # print("\n计算带噪声状态的对数负度:")
        ln_noisy_0_vs_12 = self.calculate_log_negativity(density_matrix, [0], len(self.qubits))
        # print(f"  划分 (0 | 12): {ln_noisy_0_vs_12:.4f}")
        ln_noisy_1_vs_02 = self.calculate_log_negativity(density_matrix, [1], len(self.qubits))
        # print(f"  划分 (1 | 02): {ln_noisy_1_vs_02:.4f}")
        ln_noisy_2_vs_01 = self.calculate_log_negativity(density_matrix, [2], len(self.qubits))
        # print(f"  划分 (2 | 01): {ln_noisy_2_vs_01:.4f}")

        target_matrix = cirq.density_matrix_from_state_vector(self.target)
        t_0_vs_12 = self.calculate_log_negativity(target_matrix, [0], len(self.qubits))
        t_1_vs_02 = self.calculate_log_negativity(target_matrix, [1], len(self.qubits))
        t_2_vs_01 = self.calculate_log_negativity(target_matrix, [2], len(self.qubits))

        diff_ln = abs(ln_noisy_0_vs_12 - t_0_vs_12) + abs(ln_noisy_1_vs_02 - t_1_vs_02) + abs(
            ln_noisy_2_vs_01 - t_2_vs_01)
        diff_ln = diff_ln / 3

        return diff_ln

    def partial_transpose(self, density_matrix: np.ndarray,
                          subsystem_A_indices: List[int],
                          subsystem_dims: List[int]) -> np.ndarray:
        """计算密度矩阵相对于子系统 A 的部分转置。"""
        num_subsystems = len(subsystem_dims)
        total_dim = np.prod(subsystem_dims)
        if density_matrix.shape != (total_dim, total_dim):
            raise ValueError(f"密度矩阵形状 {density_matrix.shape} 与 "
                             f"子系统维度 {subsystem_dims} (总维度 {total_dim}) 不符。")
        reshaped_dm = density_matrix.reshape(subsystem_dims + subsystem_dims)
        axes = list(range(2 * num_subsystems))
        for idx in subsystem_A_indices:
            axes[idx], axes[idx + num_subsystems] = axes[idx + num_subsystems], axes[idx]
        pt_tensor = np.transpose(reshaped_dm, axes=axes)
        pt_matrix = pt_tensor.reshape((total_dim, total_dim))
        # print(pt_matrix)
        return pt_matrix

    def calculate_log_negativity(self, density_matrix: np.ndarray,
                                 subsystem_A_indices: List[int],
                                 num_qubits: int,
                                 tol: float = 1e-9) -> float:
        """计算给定二分划下的对数负度。"""
        if density_matrix.shape != (2 ** num_qubits, 2 ** num_qubits):
            raise ValueError(f"密度矩阵形状应为 ({2 ** num_qubits}, {2 ** num_qubits})")
        subsystem_dims = [2] * num_qubits
        pt_rho = self.partial_transpose(density_matrix, subsystem_A_indices, subsystem_dims)
        singular_values = scipy.linalg.svdvals(pt_rho)
        trace_norm = np.sum(singular_values)
        if trace_norm <= 1.0 + tol:
            return 0.0
        else:
            log_negativity = np.log2(trace_norm)
            return max(0.0, log_negativity)

    def calculate_renyi_entropy(self, rho: np.ndarray, alpha: float, tol: float = 1e-12) -> float:
        """
        计算密度矩阵 rho 的 alpha 阶 Rényi 熵 S_alpha(rho)。
        Includes handling for alpha=0, 1, inf and numerical stability.
        """
        if alpha < 0:
            raise ValueError("Rényi 熵的阶数 alpha 必须是非负的。")
        rho = np.asarray(rho)
        if not np.allclose(rho, rho.T.conj(), atol=tol):  # Use tolerance for check
            warnings.warn("输入矩阵可能不是厄米矩阵（在容忍度内）。")
            # Force Hermiticity? Maybe just warn.
            # rho = 0.5 * (rho + rho.T.conj())

        # Handle alpha = 1 (Von Neumann)
        if np.isclose(alpha, 1.0):
            try:
                eigenvalues = np.linalg.eigvalsh(rho)
            except np.linalg.LinAlgError:
                warnings.warn("特征值计算失败。")
                return np.nan
            positive_eigenvalues = eigenvalues[eigenvalues > tol]
            if len(positive_eigenvalues) == 0: return 0.0
            try:
                log_eigs = np.log2(positive_eigenvalues)
                von_neumann_entropy = -np.sum(positive_eigenvalues * log_eigs)
            except (ValueError, FloatingPointError):
                warnings.warn("数值误差导致 log2 计算问题 for S_1。")
                return np.nan
            if np.isnan(von_neumann_entropy) or np.isinf(von_neumann_entropy):
                warnings.warn("冯诺依曼熵计算结果为 NaN 或 Inf。")
                # Return NaN instead of potential Inf/NaN from sum
                return np.nan
            return von_neumann_entropy

        # Handle alpha = 0 (Hartley/Max-entropy)
        if np.isclose(alpha, 0.0):
            try:
                eigenvalues = np.linalg.eigvalsh(rho)
            except np.linalg.LinAlgError:
                warnings.warn("特征值计算失败。")
                return np.nan
            rank = np.sum(eigenvalues > tol)
            if rank == 0: return 0.0
            return np.log2(rank)

        # Handle alpha = inf (Min-entropy)
        if alpha == np.inf:
            try:
                eigenvalues = np.linalg.eigvalsh(rho)
            except np.linalg.LinAlgError:
                warnings.warn("特征值计算失败。")
                return np.nan
            max_eigenvalue = np.max(eigenvalues)
            if max_eigenvalue <= tol:
                warnings.warn("最大特征值接近零或为负，最小熵未定义或无穷大。")
                return np.inf
            try:
                # Ensure argument to log2 is positive real
                min_entropy = -np.log2(max_eigenvalue.real)
            except (ValueError, FloatingPointError):
                warnings.warn("数值误差导致 log2 计算问题 for S_inf。")
                return np.nan
            if np.isnan(min_entropy) or np.isinf(min_entropy):
                warnings.warn("最小熵计算结果为 NaN 或 Inf。")
                return np.nan  # Return NaN for consistency
            return min_entropy

        # General alpha != 0, 1, inf
        try:
            eigenvalues = np.linalg.eigvalsh(rho)
        except np.linalg.LinAlgError:
            warnings.warn("特征值计算失败。")
            return np.nan

        positive_eigenvalues = eigenvalues[eigenvalues > tol].real  # Take real part
        positive_eigenvalues = positive_eigenvalues[positive_eigenvalues > tol]  # Filter again

        if len(positive_eigenvalues) == 0:
            warnings.warn("密度矩阵没有正特征值，Tr(rho^alpha) 为 0。")
            if alpha < 1:
                return -np.inf  # Or potentially 0 depending on convention
            else:
                return np.inf  # Or potentially 0

        try:
            # Check for potential overflow with large alpha?
            with np.errstate(over='raise'):
                powered_eigs = positive_eigenvalues ** alpha
            trace_rho_alpha = np.sum(powered_eigs)
        except FloatingPointError:
            warnings.warn(f"计算 Tr(rho^alpha) = sum(lambda^alpha) 时出错 (可能溢出)。")
            return np.nan  # Or Inf? Depends on alpha

        if trace_rho_alpha <= tol or not np.isfinite(trace_rho_alpha):
            warnings.warn(f"Tr(rho^alpha) 计算结果 {trace_rho_alpha} 非常接近于零、为负或非有限。")
            if alpha < 1:
                return -np.inf
            else:
                return np.inf

        try:
            log_trace = np.log2(trace_rho_alpha)
        except (ValueError, FloatingPointError):
            warnings.warn(f"计算 log2(Tr(rho^alpha)) 时出错。")
            return np.nan

        if np.isclose(alpha, 1.0):  # Should not happen here, but as safety
            warnings.warn("Alpha=1 case handled improperly in general calculation.")
            return np.nan
        renyi_entropy = (1.0 / (1.0 - alpha)) * log_trace

        if np.isnan(renyi_entropy):  # Don't check for Inf here, Inf is a possible valid output
            warnings.warn(f"Rényi entropy calculation resulted in NaN for alpha={alpha}.")
            pass  # Keep NaN

        return renyi_entropy

    # Manual Partial Trace Implementation (保持不变)
    def manual_partial_trace_over(self, rho: np.ndarray, trace_over_indices: List[int], dims: List[int]) -> np.ndarray:
        """
        Manually performs partial trace using numpy.einsum.
        """
        num_subsystems = len(dims)
        if not all(d >= 1 for d in dims):
            raise ValueError("所有子系统维度必须至少为 1。")
        if not rho.shape == (np.prod(dims), np.prod(dims)):
            raise ValueError("rho 维度与 dims 不匹配。")
        if not all(0 <= i < num_subsystems for i in trace_over_indices):
            raise ValueError("trace_over_indices 包含无效索引。")

        kept_indices = [i for i in range(num_subsystems) if i not in trace_over_indices]

        tensor_shape = tuple(dims + dims)
        try:
            tensor_rho = rho.reshape(tensor_shape)
        except ValueError as e:
            print(f"Reshaping rho (shape {rho.shape}) to {tensor_shape} failed: {e}")
            raise e

        all_indices_list = list(range(2 * num_subsystems))
        kept_axes_rows = kept_indices
        kept_axes_cols = [i + num_subsystems for i in kept_indices]
        output_indices_list = kept_axes_rows + kept_axes_cols

        alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if 2 * num_subsystems > len(alphabet):
            raise ValueError("子系统过多，无法使用默认字母表进行 einsum。")

        input_map = {axis: alphabet[i] for i, axis in enumerate(all_indices_list)}
        output_map = {axis: alphabet[i] for i, axis in
                      enumerate(output_indices_list)}  # Use different indices for output map

        input_str_list = [input_map[i] for i in all_indices_list]
        output_str_list = [output_map[i] for i in output_indices_list]  # Map to output letters

        # Handle trace indices: make row and column indices the same letter in input
        current_input_str_list = list(input_str_list)  # Copy to modify
        for i in trace_over_indices:
            row_axis_idx = i
            col_axis_idx = i + num_subsystems
            trace_letter = current_input_str_list[row_axis_idx]
            current_input_str_list[col_axis_idx] = trace_letter  # Assign same letter

        # Reconstruct strings
        input_str = "".join(current_input_str_list)
        # Output string uses letters corresponding to kept indices only
        final_output_str_list = []
        current_output_idx = 0
        for i in kept_axes_rows:
            final_output_str_list.append(alphabet[current_output_idx])
            current_output_idx += 1
        for i in kept_axes_cols:
            final_output_str_list.append(alphabet[current_output_idx])
            current_output_idx += 1
        output_str = "".join(final_output_str_list)

        # Ensure the output letters match the positions correctly
        # Rebuild output string mapping kept row/col letters to a compact alphabet a,b,c...
        final_output_str_map = {}
        out_idx = 0
        for i in kept_axes_rows:
            final_output_str_map[input_map[i]] = alphabet[out_idx]
            out_idx += 1
        for i in kept_axes_cols:
            final_output_str_map[input_map[i]] = alphabet[out_idx]
            out_idx += 1
        output_str = "".join(alphabet[k] for k in range(out_idx))

        # Rebuild input string with trace indices identified by repetition
        final_input_str_list = list(alphabet[:2 * num_subsystems])  # Start with unique letters
        for i in trace_over_indices:
            final_input_str_list[i + num_subsystems] = final_input_str_list[i]  # Make letters same
        input_str = "".join(final_input_str_list)

        # Now build the output string using only the letters corresponding to kept indices
        final_output_str_list_v2 = []
        for i in kept_axes_rows:
            final_output_str_list_v2.append(final_input_str_list[i])
        for i in kept_axes_cols:
            final_output_str_list_v2.append(final_input_str_list[i])
        output_str = "".join(final_output_str_list_v2)

        einsum_str = f"{input_str}->{output_str}"
        # print(f"Debug: Einsum string: {einsum_str}") # Uncomment for debugging

        try:
            traced_tensor = np.einsum(einsum_str, tensor_rho, optimize='greedy')  # Added optimize flag
        except ValueError as e:
            print(f"Einsum failed with string '{einsum_str}': {e}")
            raise e

        dim_kept = np.prod([dims[i] for i in kept_indices])
        if dim_kept == 0:  # Handle case where all subsystems are traced out
            if traced_tensor.ndim == 0:  # Should be a scalar
                return traced_tensor  # Return the scalar trace result
            else:
                raise ValueError("Trace result has unexpected dimension when all traced.")

        # Ensure the result dimension matches expectation
        expected_output_dim = dim_kept * dim_kept
        if traced_tensor.size != expected_output_dim:
            # If einsum produced a scalar (dim_kept=1), reshape needs size 1
            if dim_kept == 1 and traced_tensor.size == 1:
                rho_reduced = traced_tensor.reshape(1, 1)
            else:
                print(f"Einsum output size {traced_tensor.size} != expected {expected_output_dim}")
                print(f"Einsum string: {einsum_str}")
                print(f"Input shape: {tensor_rho.shape}")
                print(f"Output shape: {traced_tensor.shape}")
                print(f"Kept indices: {kept_indices}, Dims: {dims}, Dim_kept: {dim_kept}")

                # Fallback to np.trace if einsum logic is flawed
                # This is much less efficient for large numbers of traces
                # warnings.warn("Einsum result dimension mismatch, potential issue in string generation.")
                # This fallback is complex to write correctly on the fly. Sticking with einsum debug.
                raise ValueError("Einsum result dimension mismatch.")

        else:
            rho_reduced = traced_tensor.reshape(dim_kept, dim_kept)

        return rho_reduced

    def calculate_system_and_subsystem_renyi(self, rho: np.ndarray, num_qubits: int, alpha: float) -> Tuple[
        float, float, float]:
        """
        计算整个系统以及中间划分的两个子系统的 **归一化** Rényi 熵。
        归一化因子是各自子系统的最大可能熵 log2(d) = num_qubits_in_subsystem。
        """
        expected_dim = 2 ** num_qubits
        if rho.shape != (expected_dim, expected_dim):
            raise ValueError(f"密度矩阵维度 {rho.shape} 与 {num_qubits} 个 qubit "
                             f"(期望维度 {expected_dim}x{expected_dim}) 不符。")

        # --- 1. 计算原始（未归一化）熵 ---
        s_total_raw = self.calculate_renyi_entropy(rho, alpha)

        # --- 2. 定义划分和维度 ---
        num_qubits_A = num_qubits // 2
        num_qubits_B = num_qubits - num_qubits_A
        keep_indices_A = list(range(num_qubits_A))
        keep_indices_B = list(range(num_qubits_A, num_qubits))
        trace_over_indices_A = keep_indices_B  # Indices to trace out to get rho_A
        trace_over_indices_B = keep_indices_A  # Indices to trace out to get rho_B

        s_A_raw = np.nan  # Initialize
        s_B_raw = np.nan

        if num_qubits >= 2:  # Only calculate subsystem entropy if partitioning is possible
            all_qubit_dims = [2] * num_qubits

            # --- 3/4. 计算子系统 A 的熵 ---
            try:
                # 使用 manual_partial_trace (追踪掉 B)
                rho_A = self.manual_partial_trace(rho, trace_over_indices_A, all_qubit_dims)
                s_A_raw = self.calculate_renyi_entropy(rho_A, alpha)
            except Exception as e:
                warnings.warn(f"计算子系统 A (保留 {keep_indices_A}) 的熵时出错: {e}")
                s_A_raw = np.nan  # Mark as failed

            # --- 5/6. 计算子系统 B 的熵 ---
            try:
                # 使用 manual_partial_trace (追踪掉 A)
                rho_B = self.manual_partial_trace(rho, trace_over_indices_B, all_qubit_dims)
                s_B_raw = self.calculate_renyi_entropy(rho_B, alpha)
            except Exception as e:
                warnings.warn(f"计算子系统 B (保留 {keep_indices_B}) 的熵时出错: {e}")
                s_B_raw = np.nan  # Mark as failed

        elif num_qubits == 1:
            # For 1 qubit, S_A = S_total, S_B is not applicable (or trace of 0-dim)
            s_A_raw = s_total_raw
            s_B_raw = np.nan  # Subsystem B doesn't exist meaningfully
            warnings.warn("1-qubit system: S_B is NaN.")
        else:  # num_qubits == 0
            s_A_raw = s_total_raw  # Should be 0
            s_B_raw = s_total_raw  # Should be 0

        # --- 7. 计算归一化因子 (最大熵) ---
        # log2(d) = num_qubits
        max_s_total = float(num_qubits) if num_qubits > 0 else 0.0
        max_s_A = float(num_qubits_A) if num_qubits_A > 0 else 0.0
        max_s_B = float(num_qubits_B) if num_qubits_B > 0 else 0.0

        # --- 8. 进行归一化 ---
        # Handle division by zero and non-finite raw entropies
        norm_s_total = np.nan
        if np.isfinite(s_total_raw):
            if max_s_total > 1e-9:  # Use tolerance for float comparison
                norm_s_total = s_total_raw / max_s_total
            elif np.isclose(s_total_raw, 0.0):  # Handle 0/0 case -> 0
                norm_s_total = 0.0
            # else: raw entropy is non-zero but max is zero -> leave as NaN

        norm_s_A = np.nan
        if np.isfinite(s_A_raw):
            if max_s_A > 1e-9:
                norm_s_A = s_A_raw / max_s_A
            elif np.isclose(s_A_raw, 0.0):
                norm_s_A = 0.0

        norm_s_B = np.nan
        if np.isfinite(s_B_raw):
            if max_s_B > 1e-9:
                norm_s_B = s_B_raw / max_s_B
            elif np.isclose(s_B_raw, 0.0):
                norm_s_B = 0.0

        # Special handling for Inf results from raw entropy
        if not np.isfinite(s_total_raw): norm_s_total = s_total_raw  # Propagate Inf/-Inf
        if not np.isfinite(s_A_raw): norm_s_A = s_A_raw
        if not np.isfinite(s_B_raw): norm_s_B = s_B_raw

        return norm_s_total, norm_s_A, norm_s_B

    def _get_renyi0415(self,rho_input = None):
        # circuit = self._get_cirq(maybe_add_noise=True)
        # state_v = self.simulator.simulate(circuit).final_state_vector
        # rho = cirq.density_matrix_from_state_vector(state_v)
        # rho = (rho + rho.conj().T) / 2
        if rho_input is not None:
            density_matrix = rho_input
        else:
            circuit_with_noise = self._get_cirq(maybe_add_noise=True)
            # simulator = cirq.DensityMatrixSimulator(seed=42)  # 可以考虑是否需要固定种子
            result = self.simulator.simulate(
                circuit_with_noise,
                initial_state=self.rho_initial,
                qubit_order=self.qubits
            )
            # result = simulator.simulate(circuit_with_noise, qubit_order=self.qubits)  # 推荐指定 qubit_order
            density_matrix = result.final_density_matrix

        # 可选：强制厄米性以提高数值稳定性
        density_matrix = (density_matrix + density_matrix.conj().T) / 2
        density_matrix = density_matrix / np.trace(density_matrix)
        s_tot_norm, s_a_norm, s_b_norm = self.calculate_system_and_subsystem_renyi(density_matrix, len(self.qubits), 2)

        # s_tot_norm, s_a_norm, s_b_norm = self.calculate_system_and_subsystem_renyi(rho, len(self.qubits), 2)
        return s_tot_norm, s_a_norm, s_b_norm

    def _get_renyi(self, rho_input=None):
        if rho_input is not None:
            density_matrix = rho_input
            # 【关键修复】：根据传入矩阵的维度动态计算比特数
            # 维度 d = 2^n，所以 n = log2(d)
            num_qubits = int(np.log2(density_matrix.shape[0]))
        else:
            circuit_with_noise = self._get_cirq(maybe_add_noise=True)
            result = self.simulator.simulate(
                circuit_with_noise,
                initial_state=self.rho_initial,
                qubit_order=self.qubits
            )
            density_matrix = result.final_density_matrix
            # 如果没有传入局部矩阵，才使用全局的系统比特数
            num_qubits = len(self.qubits)

        # 可选：强制厄米性以提高数值稳定性
        #density_matrix = (density_matrix + density_matrix.conj().T) / 2
        #density_matrix = density_matrix / np.trace(density_matrix).real

            # ================= 修改后的安全代码 =================
        density_matrix = (density_matrix + density_matrix.conj().T) / 2

        # 【新增：零迹保护】
        trace_val = np.trace(density_matrix).real
        if trace_val < 1e-12:
            # 如果态已经坍缩或测量失败，直接返回 0 熵，防止 NaN 传染
            return 0.0, 0.0, 0.0

        density_matrix = density_matrix / trace_val

        # 将动态计算出的 num_qubits 传给底层函数
        s_tot_norm, s_a_norm, s_b_norm = self.calculate_system_and_subsystem_renyi(density_matrix, num_qubits, 2)

        return s_tot_norm, s_a_norm, s_b_norm

    def _get_renyi_target(self):
        rho_target = cirq.density_matrix_from_state_vector(self.target)
        rho_target = (rho_target + rho_target.conj().T) / 2
        s_tot_norm_tar, s_a_norm_tar, s_b_norm_tar = self.calculate_system_and_subsystem_renyi(rho_target,
                                                                                               len(self.qubits), 2)
        return s_tot_norm_tar, s_a_norm_tar, s_b_norm_tar

    def _get_renyi_target9bit(self):
        # 1. 将 9-qubit 目标态矢量转换为 512x512 的密度矩阵
        rho_target_9q = cirq.density_matrix_from_state_vector(self.target)
        rho_target_9q = (rho_target_9q + rho_target_9q.conj().T) / 2

        # 2. 【核心修改】偏迹掉不需要的比特，只保留 Target (q0, q1, q2)
        # dims 必须设为 [2]*9，表示原本是 9-qubit 的张量
        rho_target_3q = self.manual_partial_trace(rho_target_9q, keep_indices=[0, 1, 2], dims=[2] * 9)
        rho_target_3q = self._clean_density_matrix(rho_target_3q)  # 清理极小复数计算误差

        # 3. 【核心修改】计算保留的 3-qubit 系统的 Renyi 熵
        # 注意：这里的第二个参数必须写死为 3，而不能再用 len(self.qubits)
        s_tot_norm_tar, s_a_norm_tar, s_b_norm_tar = self.calculate_system_and_subsystem_renyi(
            rho_target_3q,
            3,  # <--- 因为矩阵已经被压缩成了 8x8 (3-qubit)
            2  # alpha = 2
        )

        return s_tot_norm_tar, s_a_norm_tar, s_b_norm_tar

    def _get_renyi_01bit(self, rho_01bit):
        """
        rho_01bit: 2-qubit density matrix (4x4),
                   already post-selected target state
        """

        # 数值稳定性
        rho = (rho_01bit + rho_01bit.conj().T) / 2
        rho = rho / np.trace(rho)

        # 只算这个子系统本身
        purity = np.trace(rho @ rho).real
        S2 = -np.log(purity)

        # 如果你非要“子系统的子系统”（单 qubit）
        rho_a = cirq.partial_trace(rho, keep_indices=[0], dims=[2, 2])
        rho_b = cirq.partial_trace(rho, keep_indices=[1], dims=[2, 2])

        Sa = -np.log(np.trace(rho_a @ rho_a).real)
        Sb = -np.log(np.trace(rho_b @ rho_b).real)

        return S2, Sa, Sb

    def _evaluate_purification_protocol_old_4(self):
        """
        这是最终用于训练的评估函数。
        它要求 Agent 的电路在 5 轮循环中表现出保真度单调上升。
        """
        # 1. 初始非对称态 (DEJMPS 搜索的关键)
        f_in = 0.73
        l11, l01, l10 = 0.15, 0.06, 0.06
        current_rho = self.get_bell_diagonal_state(f_in, l11, l01, l10)

        fidelity_history = [f_in]
        num_rounds = 5
        sim = cirq.DensityMatrixSimulator()
        circuit = self._get_cirq(maybe_add_noise=True)  # Agent 当前生成的电路

        completed_rounds = 0
        for r in range(num_rounds):
            # 构造两对当前的态作为输入
            combined_rho = np.kron(current_rho, current_rho)

            # 运行模拟
            result = sim.simulate(circuit, initial_state=combined_rho, qubit_order=self.qubits)

            # 使用“基底对齐/系数提取”逻辑计算保真度
            # 调用刚才验证脚本里的逻辑：计算四个贝尔基底投影，取最大值
            rho_out, f_out = self.post_selection_and_align(result.final_density_matrix)

            # --- 训练核心判定：单调性 ---
            # 要求每轮必须有实质性提升。Identity 会在这里被卡死
            if f_out <= fidelity_history[-1] + 0.001:
                break

            # 更新态进入下一轮（重置基底，保留纯度）
            # current_rho = self.get_werner_state(f_out)  # 将纯度重置为标准态 WERNER
            # 方案 1：不重置，直接传递上一轮产生的真实密度矩阵（推荐用于搜索 DEJMPS）
            current_rho = rho_out

            # 方案 2：如果非要重置（为了数值稳定），也要重置为“非对称”态
            # 保持相位错误和比特错误的比例
            # current_rho = self.reconstruct_non_symmetric_state(f_out)
            fidelity_history.append(f_out)
            completed_rounds += 1

        # --- 计算最终奖励 ---
        final_f = fidelity_history[-1]

        # 1. 最高的奖励：完美达成目标（跑满 num_rounds 轮且保真度达标）
        if completed_rounds == num_rounds and final_f > 0.85:
            # 给一个巨大的通关奖金
            reward = 20.0 + (final_f - 0.73) * 100.0 - (0.01 * len(self.circuit_gates))
            goal_achieved = True

        # 2. 阶段性奖励：没跑满 5 轮，但至少跑通了 1 轮（保真度有提升）
        elif completed_rounds >= 1:
            # 根据跑通的轮数和保真度提升给分，给 Agent 一个改进的信号
            # 这个分数比上面的小，但必须是正的
            reward = (final_f - 0.73) * 50.0 + (completed_rounds * 1.0) + 2
            goal_achieved = False

        # 3. 失败惩罚：一轮都没跑通（比如 Identity 作弊或乱加门导致保真度下降）
        else:
            # 给一个小惩罚，不要太重，-1.0 左右即可
            reward = -1.0
            goal_achieved = False

        return reward, {
            'final_f': final_f,
            'rounds': completed_rounds,
            'f_history': [round(f, 4) for f in fidelity_history],
            'goal_achieved': goal_achieved
        }

    def _evaluate_purification_protocol_old(self):
        # --- 1. 准备初始状态 ---
        f_noisy = 0.73
        f_cata = 0.96
        # 构造非对称贝尔对角态
        rho_target = self.get_bell_diagonal_state(f_noisy, 0.15, 0.06, 0.06)  # q0, q1
        rho_res = self.get_bell_diagonal_state(f_noisy, 0.15, 0.06, 0.06)  # q2, q3
        rho_cata = self.get_bell_diagonal_state(f_cata, 0.02, 0.01, 0.01)  # q4, q5 (催化剂)

        # 初始 6-qubit 状态
        current_rho = np.kron(np.kron(rho_target, rho_res), rho_cata)
        self.catalyst_ref = rho_cata  # 用于对比损耗的参考

        sim = cirq.DensityMatrixSimulator()
        circuit = self._get_cirq(maybe_add_noise=True)

        # --- 2. 模拟演化 ---
        # 注意：6-qubit 演化消耗内存较多，确保 simulation_seed 固定或不溢出
        result = sim.simulate(circuit, initial_state=current_rho, qubit_order=self.qubits)
        rho_final = result.final_density_matrix

        # --- 3. 后选择测量资源对 (q2, q3) ---
        # 测量 q2, q3 结果一致的投影
        rho_post, p_succ = self._post_select_6bit_ancilla(rho_final, q_indices=[2, 3])

        if p_succ < 1e-4:
            return -10.0, {'msg': 'Protocol Failed (P_succ=0)'}

        # --- 4. 状态提取与保真度计算 ---
        # 提取 Target (q0, q1)
        # 追踪掉 [2, 3, 4, 5]
        rho_01 = self.manual_partial_trace(rho_post, [2, 3, 4, 5], [2] * 6)
        f_target = self._get_max_bell_fidelity(rho_01)

        # 提取 Catalyst (q4, q5)
        # 追踪掉 [0, 1, 2, 3]
        rho_45 = self.manual_partial_trace(rho_post, [0, 1, 2, 3], [2] * 6)
        # 计算催化剂保真度损耗 (1.0 表示完美还原)
        f_cata_current = np.real(np.vdot(self.catalyst_ref, rho_45 @ self.catalyst_ref))
        cata_loss = 0#1.0 - f_cata_current

        # --- 5. 综合奖励结算 ---
        delta_f = f_target - f_noisy  # 目标保真度增量

        # 催化剂守恒约束：如果损耗 > 2%，视为失败或给予极大惩罚
        if cata_loss > 0.02:
            reward = -5.0 - (cata_loss * 100)  # 损耗越重罚得越狠
            goal_achieved = False
        elif delta_f > 0.001:
            # 成功的奖励 = (提升量 * 权重) - (极小的催化损耗 * 惩罚)
            reward = (delta_f * 200.0) + (p_succ * 10.0) - (cata_loss * 500.0)
            goal_achieved = (f_target > 0.85)  # 设定通关保真度
        else:
            reward = -2.0  # 没纯化效果
            goal_achieved = False

        return reward, {
            'target_f': f_target,
            'cata_loss': cata_loss,
            'p_succ': p_succ,
            'goal_achieved': goal_achieved
        }

    def _evaluate_purification_protocol_bell(self):
        # --- 1. 初始参数提取 ---
        f_target_init = 0.7#0.73 0.6
        # 每一轮新领取的原材料 (Resource Pair q23) - 保持重置
        #raw_resource_rho = self.get_bell_diagonal_state(f_target_init, 0.15, 0.06, 0.06)

        # 初始状态：Target 和 Catalyst 直接继承 reset 中定义的初始值
        current_target_rho = self.rho_coherent#self.get_bell_diagonal_state(f_target_init, 0.15, 0.06, 0.06)
        current_catalyst_rho = self.rho_catalyst_initial  # reset中定义的矩阵
        #current_catalyst_rho = self.get_bell_diagonal_state(f_target_init, 0.15, 0.06, 0.06)
        #current_target_rho = self.get_dejmps_limit_state()
        #rho_res = self.get_dejmps_limit_state()

        num_rounds = 3
        fidelity_history = [f_target_init]
        fidelity45_history = [f_target_init]
        sim = cirq.DensityMatrixSimulator()
        circuit = self._get_cirq(maybe_add_noise=True)

        completed_rounds = 0
        total_catalyst_loss = 0

        for r in range(num_rounds):
            # --- 2. 组合 6-qubit 输入 ---
            # 顺序：q01(Target) ⊗ q23(Resource) ⊗ q45(Catalyst)
            #input_rho = np.kron(np.kron(current_target_rho, raw_resource_rho), current_catalyst_rho)
            # input_rho = np.kron(np.kron(current_target_rho, raw_resource_rho), current_catalyst_rho)

            # 修改后（递归模式）：Target(当前) ⊗ Resource(与Target相同) ⊗ Catalyst(当前)
            input_rho = np.kron(np.kron(current_target_rho, current_target_rho), current_catalyst_rho)#current_catalyst_rho)

            # 【关键加固 1】在演化前清洗总矩阵
            input_rho = self._clean_density_matrix(input_rho)

            # --- 3. 运行模拟 ---
            result = sim.simulate(circuit, initial_state=input_rho, qubit_order=self.qubits)
            rho_6q = result.final_density_matrix

            # --- 4. 后选择测量 Resource 对 (q2, q3) ---
            rho_post, p_succ = self._post_select_6bit_ancilla(rho_6q, q_indices=[2, 3])
            if p_succ < 1e-4: break

            # --- 5. 提取输出（不重置，保留原始演化） ---
            # 提取目标对 (q0, q1) 并立即清洗
            rho_01_out = self.manual_partial_trace(rho_post, [2, 3, 4, 5], [2] * 6)
            rho_01_out = self._clean_density_matrix(rho_01_out)  # 【关键加固 2】

            # 提取催化剂 (q4, q5) 并立即清洗
            #rho_45_out = self.manual_partial_trace(rho_post, [0, 1, 2, 2], [2] * 6)  # 注意这里的索引修正
            # 修正：Trace 0,1,2,3 得到 4,5
            rho_45_out = self.manual_partial_trace(rho_post, [0, 1, 2, 3], [2] * 6)
            rho_45_out = self._clean_density_matrix(rho_45_out)  # 【关键加固 3】

            # --- 6. 性能评估 ---
            # 目标保真度：取最大贝尔分量
            f_target = self._get_max_bell_fidelity(rho_01_out)
            f_target_45 = self._get_max_bell_fidelity(rho_45_out)

            # 催化剂损耗：使用 <psi_init | rho_out | psi_init>
            #self.rho_catalyst_initial_vec 应当在 reset 中保存为向量
            #cata_f = np.real(np.vdot(self.rho_catalyst_initial_vec, rho_45_out @ self.rho_catalyst_initial_vec))
            current_loss = 0#(1.0 - cata_f)
            # 中断判定：目标保真度不涨，或催化剂损耗过大
            if f_target <= fidelity_history[-1] + 0.0005 or current_loss > 0.03:#or f_target_45 <= fidelity45_history[-1]:
                break

            # --- 7. 更新状态进入下一轮 ---
            # 核心：直接把原始演化出的 rho 传给下一轮，不转回标准态
            current_target_rho = rho_01_out
            current_catalyst_rho = rho_45_out

            fidelity_history.append(f_target)
            fidelity45_history.append(f_target_45)
            total_catalyst_loss += current_loss
            completed_rounds += 1
            self.prob_sum *= p_succ
            #ss_tot, ss_a, ss_b = self._get_renyi_01bit(rho_01_out)


        # --- 8. 奖励结算 ---
        final_f = fidelity_history[-1]
        final_f45 = fidelity45_history[-1]
        delta_f = final_f - f_target_init
        delta_f45 = 0#final_f45 - f_target_init


        if completed_rounds == num_rounds:
            # 如果成功跑完 3 轮，给予大幅奖励
            reward = 10.0 + ((delta_f*5 + delta_f45*2) * 40.0)*self.prob_sum*2 #- (total_catalyst_loss * 1000.0) #- ss_tot*50
            self.goal_achieved = True
            if final_f > 0.95:
                reward += 10
            #if final_f*final_f45 > 0.8:
            #    reward += 15
        elif completed_rounds >= 1:
            reward = 2.0 + ((delta_f*5 + delta_f45*2) * 10.0)*self.prob_sum*1 #- (total_catalyst_loss * 500.0) #- ss_tot*20
            self.goal_achieved = False
        else:
            # 一轮都没跑通，给小惩罚
            reward = -1.0
            self.goal_achieved = False

        return reward, {
            'final_f': final_f,
            'cata_loss': total_catalyst_loss,
            'rounds': completed_rounds,
            'goal_achieved': self.goal_achieved,
            'final_f45': final_f45
        }

    def _evaluate_purification_protocol_backup0401(self):
        # --- 1. 初始参数 ---
        #f_target_init = self.F_in#0.7

        # 初始 3-qubit 态 (Target 和 Resource 用同一个分布)
        # 注意：这里我们假设不做催化(因为6比特不够放催化剂了)，只做递归
        #current_target_rho = self.get_ghz_state(f_target_init)
        #current_target_rho = self.rho_3q  # 使用你注入的 custom_rho 要留
        #f_target_init = self._get_ghz_fidelity(current_target_rho)  # 动态获取初态保真度 要留
        current_target_rho = self.manual_partial_trace(self.rho_initial, keep_indices=[0, 1, 2], dims=[2] * 6)
        f_target_init = self.F_in
        f_target = f_target_init
        final_f1 = f_target

        num_rounds = 1
        fidelity_history = [f_target_init]

        sim = cirq.DensityMatrixSimulator()
        # 注意：RL 输出的 Circuit 必须是作用在 6 个 qubit 上的 (0-2是Sys, 3-5是Res)
        circuit = self._get_cirq(maybe_add_noise=True)

        completed_rounds = 0

        for r in range(num_rounds):
            input_rho = self.rho_initial

            # 【清洗】
            input_rho = self._clean_density_matrix(input_rho)

            # --- 3. 运行模拟 ---
            # 确保 circuit 是定义在 cirq.LineQubit.range(6) 上的
            result = sim.simulate(circuit, initial_state=input_rho, qubit_order=self.qubits[:6])
            rho_6q = result.final_density_matrix

            # --- 4. 后选择测量 Resource Copy (q3, q4, q5) ---
            # GHZ 纯化通常要求 Resource 测得结果为 |000> (Z基) 或校验位为偶
            # 这里我们需要一个新的 post_select 函数
            #rho_post, p_succ = self._post_select_ghz_ancilla(rho_6q, measure_indices=[3, 4, 5])
            rho_post, p_succ = self._post_select_ghz_ancilla_even(rho_6q,  mode='P1',measure_indices=[3, 4, 5])

            if p_succ < 1e-4: break

            # --- 5. 提取输出 ---

            # 提取目标 System (保留前3个: 0, 1, 2)
            # 输入是 6-qubit (Target=0,1,2; Resource=3,4,5)
            rho_sys_out = self.manual_partial_trace(rho_post, keep_indices=[0, 1, 2], dims=[2] * 6)
            rho_sys_out = self._clean_density_matrix(rho_sys_out)

            # (可选) 如果你想观察 Resource 变成了什么样（虽然纯化协议里通常是扔掉它）
            # 保留后3个: 3, 4, 5
            #rho_res_out = self.manual_partial_trace(rho_post, keep_indices=[3, 4, 5], dims=[2] * 6)

            # --- 6. 性能评估 ---
            # 计算 3-qubit GHZ 保真度
            f_target = self._get_ghz_fidelity(rho_sys_out)

            # --- 7. 更新 ---
            # 成功！更新 Target 为更纯的态
            #if f_target <= fidelity_history[-1] + 0.0005 :
                # 这种非单调性检查在 GHZ 中很微妙，建议先注释掉，让它跑完看 Reward
            #    break

            current_target_rho = rho_sys_out
            fidelity_history.append(f_target)
            completed_rounds += 1
            self.prob_sum *= p_succ

        final_f = fidelity_history[-1]
        #final_f45 = fidelity45_history[-1]
        delta_f = final_f - f_target_init
        delta_f45 = 0  #final_f45 - f_target_init

        #if completed_rounds == num_rounds:
        if final_f >= f_target_init + 0.01:#005:  # fidelity_history[-1] + 0.0005:
            rho_renyi = np.kron(current_target_rho, current_target_rho)
            s_tot, s_a, s_b = self._get_renyi(rho_renyi)
            con_difference = abs(self.s_a_norm_tar - s_a)
            #reward = delta_f * self.prob_sum * 100 - s_tot - 0.5#delta_f * self.prob_sum * 500 - s_a - 0.5
            self.goal_achieved = True
            #exploration_weight = 0.4 *(1-final_f)
            #if fidelity >= self.fidelity_threshold:  # 优先判断 fidelity > fidelity_threshold
            reward = ( delta_f * self.prob_sum * 200 + (
                    1 - con_difference)) + 1 - self.step_counter/self.max_timesteps  # -exploration_weight * con_difference注重保真度和门数
        else:
            self.goal_achieved = False
            reward = -0.01
            final_f = final_f1

        return reward, {
            'final_f': final_f,
            #'cata_loss': total_catalyst_loss,
            'rounds': completed_rounds,
            'goal_achieved': self.goal_achieved,
            'rho_out':current_target_rho,
            'p_succ': self.prob_sum,
            #'final_f45': final_f45
        }

    def _evaluate_purification_protocol_0418(self):
        """极速 6-qubit 评估：直接读取 self.rho，消除 O(N^2) 重复物理模拟"""
        f_target_init = self.F_in

        # 1. 直接获取当前 6-qubit 的最新物理态 (64x64)
        input_rho = self.rho

        # 2. 后选择测量 Resource (q3, q4, q5)
        #rho_post, p_succ = self._post_select_ghz_ancilla_even(input_rho, mode='P1', measure_indices=[3, 4, 5])
        #rho_post, p_succ = self._post_select_ghz_ancilla(input_rho, measure_indices=[3, 4, 5])
        rho_post, p_succ = self._post_select_ghz_and_ancillas(input_rho)

        # 3. 失败判定：如果概率极低，直接按失败处理
        if p_succ < 1e-4:
            self.goal_achieved = False
            return -0.1, {
                'final_f': f_target_init,
                'rounds': 1,
                'goal_achieved': False,
                'rho_out': None,
                'p_succ': 0.0
            }

        # 4. 提取输出 Target System (保留 q0, q1, q2)
        #rho_sys_out = self.manual_partial_trace(rho_post, keep_indices=[0, 1, 2], dims=[2] * 6)
        #rho_sys_out = self._clean_density_matrix(rho_sys_out)
        rho_sys_out = self.manual_partial_trace(rho_post, keep_indices=[0, 1, 2], dims=[2] * 9)
        rho_sys_out = self._clean_density_matrix(rho_sys_out)

        en_out = self._get_logarithmic_negativity(rho_sys_out)

        # 5. 性能评估
        final_f = self._get_ghz_fidelity(rho_sys_out)
        delta_f = final_f - f_target_init
        delta_en = en_out - self.en_in
        # 为了兼容你后面的计算，先更新内部变量
        self.prob_sum = p_succ  # 由于 num_rounds=1，直接赋值即可

        # 6. 核心 Reward 结算 (完美保留你的公式)
        if final_f >= f_target_init + 0.00015*100 :#and delta_en > 1e-5:
            #rho_renyi = np.kron(rho_sys_out, rho_sys_out)
            s_tot, s_a, s_b = self._get_renyi(rho_sys_out)
            con_difference = abs(self.s_tot_norm_tar - s_tot)
            con_difference_a = abs(self.s_a_norm_tar - s_a)

            self.goal_achieved = True

            # 你的自定义多目标奖励公式 (兼顾保真度、成功率、纠缠差、步数)
            reward = (delta_f * self.prob_sum * 200 + (1 - con_difference) + (1-con_difference_a)) + 1 - self.step_counter / self.max_timesteps
        else:
            self.goal_achieved = False
            # rho_renyi = np.kron(rho_sys_out, rho_sys_out)
            # s_tot, s_a, s_b = self._get_renyi(rho_renyi)
            # con_difference = abs(self.s_tot_norm_tar - s_tot)
            # con_difference_a = abs(self.s_a_norm_tar - s_a)
            # reward = -0.01 + (delta_f*(1-final_f)  - con_difference*final_f - con_difference_a*final_f)*0.028
            reward = -0.1
            #final_f = final_f  # 失败时退回初态保真度

        return reward, {
            'final_f': final_f,
            'rounds': 1,
            'goal_achieved': self.goal_achieved,
            'rho_out': rho_sys_out,
            'p_succ': self.prob_sum,
        }

    def _evaluate_purification_protocol_0420(self):
        """极速 9-qubit 评估：完美融合张量降维与前馈校正"""
        f_target_init = self.F_in

        # 1. 获取当前 9-qubit 的最新物理态 (512x512)
        input_rho = self.rho

        # 2. 核心：一步完成 后选择 + 偏迹降维 + 前馈校正
        # 调用我们新写的 9-qubit 混合后选择函数
        # 如果是递归纯化，强烈建议 resource_mode='P2'；如果是 Murao 第一轮，用 'P1'
        rho_sys_out, p_succ = self._post_select_hybrid_9q(input_rho, resource_mode='P1')

        # 3. 失败判定：如果概率极低，直接按失败处理
        if p_succ < 1e-4:
            self.goal_achieved = False
            return -0.1, {
                'final_f': f_target_init,
                'rounds': 1,
                'goal_achieved': False,
                'rho_out': None,
                'p_succ': 0.0
            }

        # 4. 提取输出 Target System (保留 q0, q1, q2)
        # 🚨 警告：rho_sys_out 已经是 8x8 矩阵了！直接清理即可，千万别再 partial_trace！
        rho_sys_out = self._clean_density_matrix(rho_sys_out)

        en_out = self._get_logarithmic_negativity(rho_sys_out)

        # 5. 性能评估
        final_f = self._get_ghz_fidelity(rho_sys_out)
        delta_f = final_f - f_target_init
        delta_en = en_out - self.en_in

        # 兼容你的内部变量
        self.prob_sum = p_succ

        # 6. 核心 Reward 结算 (完美保留你的公式)
        if final_f >= f_target_init + 0.00015 * 100:
            s_tot, s_a, s_b = self._get_renyi(rho_sys_out)
            con_difference = abs(self.s_tot_norm_tar - s_tot)
            con_difference_a = abs(self.s_a_norm_tar - s_a)

            self.goal_achieved = True

            # 你的自定义多目标奖励公式
            reward = 1 + (delta_f * self.prob_sum * 200 + (1 - con_difference) + (
                        1 - con_difference_a)) + 1 - self.step_counter / self.max_timesteps
        else:
            self.goal_achieved = False
            reward = -0.1

        return reward, {
            'final_f': final_f,
            'rounds': 1,
            'goal_achieved': self.goal_achieved,
            'rho_out': rho_sys_out,
            'p_succ': self.prob_sum,
        }

    def _evaluate_purification_protocol(self, resource_mode='P2'):
        f_target_init = self.F_in
        input_rho = self.rho

        # 🚀 为了让你完美的 [0, 3, 5, 6] 偶宇称掩码生效
        # 如果是 P1，必须先对 Resource 施加 H 门，将其从 X 基底转回 Z 基底供代码切片
        if resource_mode == 'P1':
            basis_change_circuit = cirq.Circuit(cirq.H.on_each(self.qubits[3], self.qubits[4], self.qubits[5]))
            res = self.simulator.simulate(basis_change_circuit, initial_state=input_rho, qubit_order=self.qubits)
            input_rho = res.final_density_matrix

        # 把态送入你的极速降维函数
        # 此时 P1 找 [0, 3, 5, 6]，P2 找 [0, 7]，物理逻辑完美闭环！
        rho_sys_out, p_succ = self._post_select_hybrid_9q(input_rho, resource_mode=resource_mode)

        # 3. 失败判定：如果概率极低，直接按失败处理
        if p_succ < 1e-4:
            self.goal_achieved = False
            return -0.2, {
                'final_f': f_target_init,
                'rounds': 1,
                'goal_achieved': False,
                'rho_out': None,
                'p_succ': 0.0
            }

        # 4. 提取输出 Target System (保留 q0, q1, q2)
        # 🚨 警告：rho_sys_out 已经是 8x8 矩阵了！直接清理即可，千万别再 partial_trace！
        rho_sys_out = self._clean_density_matrix(rho_sys_out)

        en_out = self._get_logarithmic_negativity(rho_sys_out)

        # 5. 性能评估
        final_f = self._get_ghz_fidelity(rho_sys_out)
        delta_f = final_f - f_target_init
        delta_en = en_out - self.en_in

        # 兼容你的内部变量
        self.prob_sum = p_succ
        step_penalty = self.step_counter * 0.005

        # 6. 核心 Reward 结算 (完美保留你的公式)
        if final_f >= f_target_init + 0.00015 * 100:
            # 1. 过滤掉字符串(如"FINISH_P1")，把所有有效门强行拼成一个完整的独立线路
            full_circuit = cirq.Circuit([a for a in self.circuit_gates if not isinstance(a, str)])

            # 2. 初始化计数器
            single_qubit_count = 0
            two_qubit_count = 0

            # 3. 摊平线路，遍历每一个物理门
            for op in full_circuit.all_operations():
                # op.qubits 返回的是这个门作用的比特元组，求长度就是比特数
                if len(op.qubits) == 1:
                    single_qubit_count += 1
                elif len(op.qubits) == 2:
                    two_qubit_count += 1

            total_gates_plenty = single_qubit_count*0.05 + two_qubit_count*0.1

            s_tot, s_a, s_b = self._get_renyi(rho_sys_out)
            con_difference = abs(self.s_tot_norm_tar - s_tot)
            con_difference_a = abs(self.s_a_norm_tar - s_a)

            self.goal_achieved = True

            # 你的自定义多目标奖励公式
            reward = 4 + (delta_f * self.prob_sum * 200 + (1 - con_difference) + (
                        1 - con_difference_a)) + 1 - self.step_counter / self.max_timesteps - total_gates_plenty
        else:
            self.goal_achieved = False
            reward = -0.15#-step_penalty
            #reward = max(reward, -0.4)

        return reward, {
            'final_f': final_f,
            'rounds': 1,
            'goal_achieved': self.goal_achieved,
            'rho_out': rho_sys_out,
            'p_succ': self.prob_sum,
        }

    def _post_select_6bit_ancilla(self, rho_6q, q_indices=[2, 3]):
        """对第 2, 3 位比特进行测量后选择"""
        dim = 64
        p_mask = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            # 提取指定比特位的值 (假设 little-endian)
            q2 = (i >> (5 - q_indices[0])) & 1
            q3 = (i >> (5 - q_indices[1])) & 1
            if q2 == q3:
                p_mask[i, i] = 1.0

        rho_post = p_mask @ rho_6q @ p_mask
        p_succ = np.trace(rho_post).real
        if p_succ > 0:
            rho_post /= p_succ
        return rho_post, p_succ

    def _get_max_bell_fidelity(self, rho_2q):
        """计算相对于四个贝尔态的最大保真度（基底对齐）"""
        phi_p = np.array([1, 0, 0, 1]) / np.sqrt(2)
        phi_m = np.array([1, 0, 0, -1]) / np.sqrt(2)
        psi_p = np.array([0, 1, 1, 0]) / np.sqrt(2)
        psi_m = np.array([0, 1, -1, 0]) / np.sqrt(2)
        fids = [np.real(np.vdot(b, rho_2q @ b)) for b in [phi_p, phi_m, psi_p, psi_m]]
        return max(fids)

    def get_bell_diagonal_state(self, l00, l11, l01, l10):
        """生成非对称贝尔对角态"""
        phi_p = np.array([1, 0, 0, 1]) / np.sqrt(2)
        phi_m = np.array([1, 0, 0, -1]) / np.sqrt(2)
        psi_p = np.array([0, 1, 1, 0]) / np.sqrt(2)
        psi_m = np.array([0, 1, -1, 0]) / np.sqrt(2)

        rho = (l00 * np.outer(phi_p, phi_p.conj()) +
               l11 * np.outer(phi_m, phi_m.conj()) +
               l01 * np.outer(psi_p, psi_p.conj()) +
               l10 * np.outer(psi_m, psi_m.conj()))
        return rho

    def get_werner_state(self, f):
        """生成标准 Werner 态"""
        return self.get_bell_diagonal_state(f, (1 - f) / 3, (1 - f) / 3, (1 - f) / 3)

    def post_selection_and_align(self, rho_4q):
        """进行后选择测量，并提取四个贝尔基底下的最大保真度"""
        # 1. 定义标准贝尔基底
        phi_p = np.array([1, 0, 0, 1]) / np.sqrt(2)
        phi_m = np.array([1, 0, 0, -1]) / np.sqrt(2)
        psi_p = np.array([0, 1, 1, 0]) / np.sqrt(2)
        psi_m = np.array([0, 1, -1, 0]) / np.sqrt(2)
        bell_basis = [phi_p, phi_m, psi_p, psi_m]

        # 2. 测量 q2, q3 并后选择一致结果 (00 或 11)
        dim = 16
        p_mask = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            # 假设比特序是 [q0, q1, q2, q3]，q2, q3 对应最后两个索引位
            q2 = (i >> 1) & 1
            q3 = (i >> 0) & 1
            if q2 == q3:
                p_mask[i, i] = 1.0

        rho_post = p_mask @ rho_4q @ p_mask
        p_succ = np.trace(rho_post).real

        if p_succ < 1e-6:
            return None, 0

        rho_post /= p_succ

        # 3. 追踪掉 q2, q3 得到 q0, q1 的密度矩阵
        # 使用你原本类中就有的 manual_partial_trace 方法
        # 追踪掉索引为 2 和 3 的比特，剩下索引为 0 和 1 的比特
        rho_01 = self.manual_partial_trace(rho_post, [2, 3], [2, 2, 2, 2])

        # 4. 计算四个贝尔基底上的投影概率
        lambdas = [np.real(np.vdot(b, rho_01 @ b)) for b in bell_basis]

        # 提取最大的 lambda 作为当前轮次的“对齐后保真度”
        # 这样处理是为了让 Agent 不需要手动做基底复位也能感知到纯度的提升
        f_max = max(lambdas)

        # 返回重置为标准基底的密度矩阵（为了下一轮循环）和保真度
        new_rho = self.get_bell_diagonal_state(f_max, (1 - f_max) / 3, (1 - f_max) / 3, (1 - f_max) / 3)
        return new_rho, f_max

    def _calculate_metrics_with_rho(self, rho_4q):
        """进行后选择测量，并返回纯化后的完整密度矩阵"""
        # 投影算子 P = |00><00| + |11><11| 作用在第 2, 3 位比特 (辅助对比特)
        dim = 16
        p_mask = np.zeros((dim, dim))
        for i in range(dim):
            # 假设 qubit_order 是 [q0, q1, q2, q3]
            q2 = (i >> 2) & 1
            q3 = (i >> 3) & 1
            if q2 == q3:
                p_mask[i, i] = 1.0

        rho_post = p_mask @ rho_4q @ p_mask
        p_succ = np.trace(rho_post).real

        if p_succ < 1e-6:
            return 0.0, 0.0, None

        rho_post /= p_succ
        # 追踪掉 q2, q3 得到 q0, q1 的 2-qubit 密度矩阵
        # 使用你现有的 manual_partial_trace 方法
        rho_01 = self.manual_partial_trace(rho_post, [2, 3], [2, 2, 2, 2])

        # 计算相对于 |Φ+> 的保真度
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        fidelity = np.real(np.vdot(phi_plus, rho_01 @ phi_plus))

        return fidelity, p_succ, rho_01

    def _calculate_purification_metrics(self, rho_4q):
        """提取测量后剩余第一对贝尔对的保真度"""
        # 投影算子 P = |00><00| + |11><11| 作用在 q2, q3
        # 这里的索引 2, 3 必须与 qubit_order 严格对应
        dim = 16
        p_mask = np.zeros((dim, dim))
        for i in range(dim):
            # 假设 little-endian: i = (q3 q2 q1 q0)_2
            q2 = (i >> 2) & 1
            q3 = (i >> 3) & 1
            if q2 == q3:
                p_mask[i, i] = 1.0

        rho_post = p_mask @ rho_4q @ p_mask
        p_succ = np.trace(rho_post).real

        if p_succ < 1e-6:
            return 0.0, 0.0

        rho_post /= p_succ
        # 追踪掉 q2, q3 得到 q0, q1 的减缩密度矩阵
        rho_01 = self.manual_partial_trace(rho_post, [2, 3], [2, 2, 2, 2])

        # 计算与 |Φ+> 的保真度
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        fidelity = np.real(np.vdot(phi_plus, rho_01 @ phi_plus))

        return fidelity, p_succ

    def get_werner_state(self, f):
        """生成一个保真度为 f 的 2-qubit Werner 态密度矩阵"""
        # 理想贝尔对 |Φ+> = (|00>+|11>)/sqrt(2)
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        pure_bell = np.outer(phi_plus, phi_plus.conj())
        # Werner 态公式: f*|Φ+><Φ+| + (1-f)/3 * (I - |Φ+><Φ+|)
        identity = np.eye(4)
        mixed = f * pure_bell + ((1.0 - f) / 3.0) * (identity - pure_bell)
        return mixed

    def reconstruct_rho_from_fidelity(self, f):
        """
        为了循环计算，我们将输出的保真度重新映射回一个标准 Werner 态。
        这是纠缠纯化理论中常用的简化处理（Pauli Twirling 假设）。
        """
        return self.get_werner_state(f)

    import numpy as np

    def insert_resource_in_middle(self,rho_tc, rho_r):
        """
        将资源态 rho_r (2-qubit) 插入到 rho_tc (4-qubit) 的中间。

        参数:
            rho_tc: 维度 16x16 (2^4). 逻辑顺序为 [Target_0, Target_1, Catalyst_0, Catalyst_1]
                    这是上一轮循环后保留下来的 Target 和 Catalyst 的联合状态。
            rho_r:  维度 4x4 (2^2). 逻辑顺序为 [Resource_0, Resource_1]
                    这是每一轮新生成的纯净（但含噪）资源态。

        返回:
            rho_total: 维度 64x64 (2^6). 物理顺序为 [Target, Resource, Catalyst]
                       即: [T0, T1, R0, R1, C0, C1]
        """
        # 1. 计算直积：先简单组合，此时逻辑顺序为 [T, C, R]
        # T(0,1) + C(2,3) + R(4,5) -> 总共 6 个逻辑位置
        rho_combined = np.kron(rho_tc, rho_r)

        # 2. 将矩阵 Reshape 为 12 阶张量 (6个输入脚 ket, 6个输出脚 bra)
        # 形状: (2, 2, 2, 2, 2, 2,   2, 2, 2, 2, 2, 2)
        tensor_shape = [2] * 12
        rho_tensor = rho_combined.reshape(tensor_shape)

        # 3. 定义轴的置换规则 (Permutation)
        # 当前张量的轴顺序 (indices):
        # Ket (行): 0(T0), 1(T1), 2(C0), 3(C1), 4(R0), 5(R1)
        # Bra (列): 6(T0), 7(T1), 8(C0), 9(C1), 10(R0), 11(R1)

        # 我们需要的目标物理顺序: [T0, T1, R0, R1, C0, C1]
        # 即要把 R (当前索引 4,5) 移到中间，把 C (当前索引 2,3) 移到最后

        # 新的 Ket 顺序: [0, 1, 4, 5, 2, 3]
        # 新的 Bra 顺序: [6, 7, 10, 11, 8, 9] (即 Ket索引 + 6)

        perm = [0, 1, 4, 5, 2, 3, 6, 7, 10, 11, 8, 9]

        # 4. 执行转置 (移动量子比特的线路，但不破坏信息)
        rho_permuted = np.transpose(rho_tensor, axes=perm)

        # 5. 展平回 64x64 矩阵
        rho_final = rho_permuted.reshape(64, 64)

        return rho_final

    def get_pure_catalyst(self, p):
        """生成非极大纠缠纯态: sqrt(p)|00> + sqrt(1-p)|11>"""
        psi = np.array([np.sqrt(p), 0, 0, np.sqrt(1 - p)])
        return np.outer(psi, psi.conj())

    def make_psd(self, rho, tol=1e-12):
        """强制使矩阵满足密度矩阵的物理条件：厄米、迹为1、半正定"""
        # 1. 确保厄米性
        rho = (rho + rho.conj().T) / 2
        # 2. 特征值分解并去除负值
        evals, evecs = np.linalg.eigh(rho)
        evals = np.maximum(evals, 0)
        rho = evecs @ np.diag(evals) @ evecs.conj().T
        # 3. 重新归一化迹
        trace = np.trace(rho)
        if trace > 0:
            rho /= trace
        return rho.astype(np.complex128)

    def get_non_maximal_pure_state(self, p):
        """
        生成 sqrt(p)|00> + sqrt(1-p)|11> 的密度矩阵
        这是最经典的催化剂形式。
        """
        # 构造状态向量
        psi = np.zeros(4, dtype=complex)
        psi[0] = np.sqrt(p)  # |00>
        psi[3] = np.sqrt(1 - p)  # |11>

        # 转为密度矩阵
        rho = np.outer(psi, psi.conj())
        return rho

    def get_catalyst_fidelity(self, rho_current):
        """计算当前催化剂与初始定义的 0.6/0.4 态的重合度"""
        # self.rho_catalyst_initial 是在 reset 中保存的那个 0.6/0.4 态
        fidelity = np.real(np.vdot(self.rho_catalyst_initial, rho_current @ self.rho_catalyst_initial))
        return fidelity

    def _clean_density_matrix_old(self, rho):
        """强制修复密度矩阵：确保厄米性、迹为1、特征值非负"""
        if rho is None: return None

        # 1. 强制厄米性 (rho = rho^H)
        rho = (rho + rho.conj().T) / 2.0

        # 2. 强制正定性：截断负特征值
        # eigh 专门用于厄米矩阵，比 eig 更稳定
        evals, evecs = np.linalg.eigh(rho)
        if np.any(evals < 0):
            evals = np.maximum(evals, 0)
            rho = evecs @ np.diag(evals) @ evecs.conj().T

        # 3. 强制迹为 1
        t = np.trace(rho)
        rho = rho / t

        return rho.astype(complex)

    def _clean_density_matrix0330(self, rho):
        """强制修复密度矩阵：确保厄米性、迹为1、特征值非负"""
        if rho is None: return None

        # ==========================================
        # 【新增防线 1】：查杀 NaN 和 Inf 病毒
        # ==========================================
        if not np.isfinite(rho).all():
            # 如果矩阵已经被污染为 NaN，直接返回全 0 矩阵 (代表物理态已崩溃)
            return np.zeros_like(rho)

        # 1. 强制厄米性 (rho = rho^H)
        rho = (rho + rho.conj().T) / 2.0

        # 2. 强制正定性：截断负特征值
        # eigh 专门用于厄米矩阵，比 eig 更稳定
        evals, evecs = np.linalg.eigh(rho)
        if np.any(evals < 0):
            evals = np.maximum(evals, 0)
            rho = evecs @ np.diag(evals) @ evecs.conj().T

        # 3. 强制迹为 1
        t = np.trace(rho).real

        # ==========================================
        # 【新增防线 2】：防止 0 除以 0 产生 NaN
        # ==========================================
        if t < 1e-12:
            # 迹为 0 说明态已经空了（比如后选择失败），直接返回全 0 矩阵
            return np.zeros_like(rho)

        rho = rho / t

        return rho.astype(complex)

    def _clean_density_matrix0417(self, rho):
        """强制修复密度矩阵：确保厄米性、迹为1、特征值非负"""
        if rho is None: return None

        # ==========================================
        # 【新增防线 1】：查杀 NaN 和 Inf 病毒
        # ==========================================
        if not np.isfinite(rho).all():
            # 如果矩阵已经被污染为 NaN，直接返回全 0 矩阵 (代表物理态已崩溃)
            return np.zeros_like(rho)

        # 1. 强制厄米性 (rho = rho^H)
        rho = (rho + rho.conj().T) / 2.0

        # ==========================================
        # 🚀 【新增终极防线 3】：查杀底层数学库不收敛 Bug
        # ==========================================
        try:
            # 2. 强制正定性：截断负特征值
            # eigh 专门用于厄米矩阵，比 eig 更稳定
            evals, evecs = np.linalg.eigh(rho)
            if np.any(evals < 0):
                evals = np.maximum(evals, 0)
                rho = evecs @ np.diag(evals) @ evecs.conj().T
        except np.linalg.LinAlgError:
            # 如果 Agent 探索出了极端病态线路导致特征值不收敛
            # 直接将该态判死刑，返回全 0 矩阵！Agent 会因为保真度极低而受罚
            return np.zeros_like(rho)

        # 3. 强制迹为 1
        t = np.trace(rho).real

        # ==========================================
        # 【新增防线 2】：防止 0 除以 0 产生 NaN
        # ==========================================
        if t < 1e-12:
            # 迹为 0 说明态已经空了（比如后选择失败），直接返回全 0 矩阵
            return np.zeros_like(rho)

        rho = rho / t

        return rho.astype(complex)

    def _clean_density_matrix(self, rho):
        """
        严谨修复密度矩阵：确保厄米性、半正定性、迹严格为1。
        如果发生不可逆的数值崩坏（NaN, Inf, 迹下溢出），则返回 None，交由外层执行环境截断。
        """
        if rho is None:
            return None

        # ==========================================
        # 防线 1：查杀 NaN 和 Inf
        # ==========================================
        # 只要存在一个非有限数，矩阵运算的物理意义就彻底丧失
        if not np.isfinite(rho).all():
            return None

        # ==========================================
        # 防线 2：强制厄米性
        # ==========================================
        # 消除多次矩阵乘法累积的微小虚部不对称
        rho = (rho + rho.conj().T) / 2.0

        # ==========================================
        # 防线 3：强制半正定性与特征值截断
        # ==========================================
        try:
            # 使用 np.linalg.eigh (专用于厄米矩阵，比 eig 更高效且保证特征值为实数)
            evals, evecs = np.linalg.eigh(rho)

            # 由于浮点误差，原本应为 0 的特征值可能会变成微小的负数 (如 -1e-17)
            # 如果存在负特征值，将其硬截断为 0，这在量子层析成像(QST)中是标准的 Maximum Likelihood 处理手法
            if np.any(evals < 0):
                evals = np.maximum(evals, 0.0)
                # 利用非负特征值重新构建合法的密度矩阵
                rho = evecs @ np.diag(evals) @ evecs.conj().T
        except np.linalg.LinAlgError:
            # 极小概率发生：如果矩阵由于退相干变得极度病态，导致对角化算法不收敛
            return None

        # ==========================================
        # 防线 4：强制迹归一化 (保迹操作)
        # ==========================================
        # 取实部即可，因为之前的厄米化已经保证了对角线元素必须为实数
        t = np.trace(rho).real

        # 绝对物理底线：在幺正演化中，迹只能是 1。
        # 如果 t < 1e-15，说明前面的截断或浮点下溢出已经把矩阵的有效信息全部抹除
        if t < 1e-15:
            return None

        # 重新归一化：由于前面可能截断了负特征值，此时的 t 可能变成了 1.000000000001
        # 除以 t 可以确保最终结果的迹在机器精度上严格等于 1.0
        rho = rho / t

        # 确保数据类型统一，防止后续模拟器报错
        return rho.astype(np.complex128)

    def get_dejmps_limit_state(self):
        """
        构造一个对于标准 DEJMPS 协议已经是“尽头”的态。
        通常是 Bell-Diagonal State 的特定参数区域。
        """
        # 这是一个经典的 DEJMPS 难以处理的区域示例 (Pancake shape)
        # F < 0.5 或者是某种特定的非对称混合
        # 这里我们构造一个 Rank-2 态，它通常需要催化才能突破

        # 比如：Fidelity 0.68，但是主要噪声集中在 Phase Error 上，
        # 且正好处于标准纯化协议的“盲区”

        F = 0.6
        # 使得 rho 对称性破缺，阻碍简单的 H/S 门校正
        p1 = (1 - F) * 0.9
        p2 = (1 - F) * 0.1
        p3 = 0.0

        # 使用你之前的 get_bell_diagonal_state 函数
        # 注意：确保这个态满足 Majorization 条件 (Check Majorization 返回 True)
        rho = self.get_bell_diagonal_state(F, p1, p2, p3)

        return rho

    def get_general_coherent_state(self, weights, coherences=None):
        """
        生成通用的二比特量子态（包含非对角项）。

        参数:
        - weights: 列表 [l00, l11, l01, l10], 对应四个贝尔态的概率分布 (对角项)。
        - coherences: 字典, 键为 (i, j) 索引, 值为复数，表示贝尔态 i 和 j 之间的相干性。
                     索引映射: 0:Phi+, 1:Phi-, 2:Psi+, 3:Psi-
        """
        # 1. 定义四个贝尔基底
        phi_p = np.array([1, 0, 0, 1]) / np.sqrt(2)
        phi_m = np.array([1, 0, 0, -1]) / np.sqrt(2)
        psi_p = np.array([0, 1, 1, 0]) / np.sqrt(2)
        psi_m = np.array([0, 1, -1, 0]) / np.sqrt(2)
        bell_basis = [phi_p, phi_m, psi_p, psi_m]

        # 2. 初始化密度矩阵为 0
        rho = np.zeros((4, 4), dtype=complex)

        # 3. 添加对角项 (Bell-diagonal 部分)
        for i in range(4):
            rho += weights[i] * np.outer(bell_basis[i], bell_basis[i].conj())

        # 4. 添加非对角项 (Coherences 部分)
        # 这种项会产生类似 |00><01| 的相干效果
        if coherences is not None:
            for (i, j), val in coherences.items():
                # 由于密度矩阵必须是厄米的，我们同时添加 (i,j) 和 (j,i)
                term = val * np.outer(bell_basis[i], bell_basis[j].conj())
                rho += term
                rho += term.T.conj()  # 保证厄米性

        # 5. 强制物理检查：迹为 1，并进行归一化
        trace = np.trace(rho)
        if not np.isclose(trace, 1.0):
            rho /= trace

        # 6. 数值稳定性检查：确保正定性 (可选但推荐)
        # 如果 RL 搜出的操作让特征值变负，可以通过之前提到的 _clean_density_matrix 修复
        return rho



    def _post_select_ghz_ancilla000(self, rho_6q, measure_indices=[3, 4, 5]):
        """
        对 Resource (q3, q4, q5) 进行 Z 基投影测量。
        严格保留结果为 |000> 的分量。
        """
        dim = rho_6q.shape[0]  # 64

        # 1. 构建掩码 (Mask)
        # Cirq/NumPy 的密度矩阵通常遵循 qubit_order 顺序
        # Index i 的二进制表示为 q0 q1 q2 q3 q4 q5
        # 我们要检查 q3, q4, q5 是否全为 0

        # 构造 bitmask:
        # 假设 q0 是最高位 (MSB), q5 是最低位 (LSB)
        # q3 对应 4 (100), q4 对应 2 (010), q5 对应 1 (001)
        # mask = 4 | 2 | 1 = 7 (二进制 000111)
        target_mask = 0b000111

        indices = np.arange(dim)

        # 筛选条件：(index & target_mask) 必须等于 0
        # 这意味着最后三位必须是 000
        valid_indices_mask = (indices & target_mask) == 0

        # 2. 计算成功概率 (Trace of projected rho)
        # 只取对角线上满足条件的元素之和
        diags = np.diagonal(rho_6q)
        p_succ = np.sum(diags[valid_indices_mask]).real

        if p_succ < 1e-10:
            # 概率过低，返回全零或原始矩阵(取决于你的处理逻辑，建议返回全零并由外部break)
            return np.zeros_like(rho_6q), 0.0

        # 3. 实施投影 (P * rho * P)
        # 利用广播机制：保留 valid 行和 valid 列的交叉项，其余置零
        # mask_vec 形状 (64, 1)
        mask_vec = valid_indices_mask.astype(float).reshape(-1, 1)

        # rho_post[i, j] = rho[i, j] * mask[i] * mask[j]
        rho_post = rho_6q * (mask_vec @ mask_vec.T)

        # 4. 归一化
        rho_post /= p_succ

        return rho_post, p_succ

    def _post_select_ghz_ancilla_old(self, rho_6q, measure_indices=[3, 4, 5]):
        dim = rho_6q.shape[0]  # 64
        indices = np.arange(dim)

        # --- 修改掩码逻辑 ---
        target_mask = 0b000111  # 对应 q3, q4, q5

        # 条件 1: q3, q4, q5 全为 0
        mask_000 = (indices & target_mask) == 0

        # 条件 2: q3, q4, q5 全为 1 (二进制最后三位是 111，即 7)
        mask_111 = (indices & target_mask) == target_mask

        # 合并掩码：只要满足全 0 或全 1 即可
        valid_indices_mask = mask_000 | mask_111

        # --- 后续逻辑保持不变 ---
        diags = np.diagonal(rho_6q)
        p_succ = np.sum(diags[valid_indices_mask]).real

        if p_succ < 1e-10:
            return np.zeros_like(rho_6q), 0.0

        mask_vec = valid_indices_mask.astype(float).reshape(-1, 1)
        rho_post = rho_6q * (mask_vec @ mask_vec.T)
        rho_post /= p_succ

        return rho_post, p_succ

    def _post_select_ghz_ancillahhh(self, rho_6q, basis='Z', measure_indices=[3, 4, 5]):
        dim = rho_6q.shape[0]  # 64

        # ---------------- 新增：基矢旋转模块 ----------------
        if basis == 'X':
            # 定义单比特 H 门
            H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            # 前三个系统比特 (0, 1, 2) 保持不变，使用 8x8 单位阵
            I_sys = np.eye(8)
            # 后三个资源比特 (3, 4, 5) 施加 H 门
            H_res = np.kron(np.kron(H, H), H)
            # 生成全局 6-qubit 的基矢旋转矩阵
            U_rot = np.kron(I_sys, H_res)

            # 将资源比特从 X 基旋转到 Z 基
            rho_6q = U_rot @ rho_6q @ U_rot.conj().T
        # ----------------------------------------------------

        indices = np.arange(dim)
        target_mask = 0b000111

        mask_000 = (indices & target_mask) == 0
        mask_111 = (indices & target_mask) == target_mask

        # 物理逻辑分支：
        # 如果是 Z 基测量 (捕获 X 错误)，000 和 111 都是合法的 GHZ 关联
        if basis == 'Z':
            valid_indices_mask = mask_000 | mask_111
        # 如果是 X 基测量 (捕获 Z 错误)，完美的无相位错误对应于 |+++>
        # 旋转后即为 |000>。此时通常只保留 000 分支 (具体视你的校验逻辑而定)
        elif basis == 'X':
            valid_indices_mask = mask_000
            # 如果你的协议允许 |---> 且后续加 Z 门修正，则可以像 Z 基一样保留 mask_111

        diags = np.diagonal(rho_6q)
        p_succ = np.sum(diags[valid_indices_mask]).real

        if p_succ < 1e-10:
            return np.zeros_like(rho_6q), 0.0

        # 生成对应的掩码矩阵
        vec_000 = mask_000.astype(float).reshape(-1, 1)

        if basis == 'Z':
            vec_111 = mask_111.astype(float).reshape(-1, 1)
            rho_post = (rho_6q * (vec_000 @ vec_000.T)) + (rho_6q * (vec_111 @ vec_111.T))
        elif basis == 'X':
            # 如果 X 基下只接受 |+++> (即旋转后的 000)
            rho_post = (rho_6q * (vec_000 @ vec_000.T))

        rho_post /= p_succ

        # ---------------- 新增：状态恢复模块 ----------------
        # 测完之后，System 比特的状态已经被投影了。
        # 资源比特已经被摧毁/丢弃。但在数学模拟中，为了保持 64x64 维度一致，
        # 或者如果你后续还需要利用残余的密度矩阵，理论上不再需要转回去了。
        # 但如果为了物理表达的严谨性，确保未被测量的比特基矢对齐，
        # 通常不需要做逆旋转，因为测完 3,4,5 号位后，通常会通过 Partial Trace 把它们约化掉。
        # ----------------------------------------------------

        return rho_post, p_succ

    def _post_select_ghz_ancilla(self, rho_6q, measure_indices=[3, 4, 5]):
        dim = rho_6q.shape[0]  # 64
        indices = np.arange(dim)
        target_mask = 0b000111

        mask_000 = (indices & target_mask) == 0
        mask_111 = (indices & target_mask) == target_mask

        valid_indices_mask = mask_000 | mask_111
        diags = np.diagonal(rho_6q)
        p_succ = np.sum(diags[valid_indices_mask]).real

        if p_succ < 1e-10:
            return np.zeros_like(rho_6q), 0.0

        # 分别生成 000 和 111 的掩码矩阵，消除 000 和 111 之间的非物理相干性
        vec_000 = mask_000.astype(float).reshape(-1, 1)
        vec_111 = mask_111.astype(float).reshape(-1, 1)

        # 混合两个测量分支的密度矩阵
        rho_post = (rho_6q * (vec_000 @ vec_000.T)) + (rho_6q * (vec_111 @ vec_111.T))
        rho_post /= p_succ

        return rho_post, p_succ

    def _post_select_ghz_ancilla_even(self, rho_6q, mode='P2', measure_indices=[3, 4, 5]):
        """
        mode='P2': 严格一致掩码 (000, 111)，用于 Z 基底过滤 (捕获 X 错误)
        mode='P1': 偶宇称掩码 (000, 011, 101, 110)，用于 X 基底过滤 (捕获 Z 错误)
        """
        dim = rho_6q.shape[0]  # 64
        indices = np.arange(dim)
        target_mask = 0b000111  # 对应最后 3 个 bits (q3, q4, q5)

        # 获取每个索引在最后3位的值 (0~7)
        ancilla_vals = indices & target_mask

        # --- 1. 根据模式选择允许的测量结果 ---
        if mode == 'P1':
            # 偶宇称：二进制中 1 的个数为偶数
            # 000(0), 011(3), 101(5), 110(6)
            allowed_vals = [0, 3, 5, 6]
        else:
            # 默认 P2 模式：完全一致
            # 000(0), 111(7)
            allowed_vals = [0, 7]

        # --- 2. 计算总成功率 p_succ ---
        # 只要 ancilla_vals 在 allowed_vals 中，就是合法的索引
        valid_indices_mask = np.isin(ancilla_vals, allowed_vals)

        diags = np.diagonal(rho_6q)
        p_succ = np.sum(diags[valid_indices_mask]).real

        if p_succ < 1e-10:
            return np.zeros_like(rho_6q), 0.0

        # --- 3. 动态消除相干性，生成经典混合态 ---
        rho_post = np.zeros_like(rho_6q)

        # 遍历每一个被允许的经典测量结果，分别投影并相加
        # 这确保了不仅 000 和 111 之间没有相干性，011 和 101 之间也不会有！
        for val in allowed_vals:
            # 提取当前特定结果的掩码
            mask_val = (ancilla_vals == val)
            vec_val = mask_val.astype(float).reshape(-1, 1)

            # 投影出该分支的态，并累加到混合态中
            rho_post += rho_6q * (vec_val @ vec_val.T)

        # 归一化
        rho_post /= p_succ

        return rho_post, p_succ

    def manual_partial_trace(self, rho, keep_indices, dims=None):
        """
        计算部分迹 (Partial Trace) 的通用函数。

        参数:
        - rho: 密度矩阵 (N*N ndarray)
        - keep_indices: 你想【保留】的量子比特索引列表 (例如 [0, 2])
        - dims: 每个量子比特的维度列表。如果为 None，默认全是 2。
                对于 6 qubits，dims=[2]*6

        返回:
        - rho_reduced: 约化密度矩阵
        """
        # 1. 自动推导维度
        if dims is None:
            n_qubits = int(np.log2(rho.shape[0]))
            dims = [2] * n_qubits
        else:
            n_qubits = len(dims)

        # 校验输入
        if rho.shape[0] != np.prod(dims):
            raise ValueError(f"Rho dimension {rho.shape} does not match qubits {dims}")

        # 2. 识别要 Trace 掉的索引
        total_indices = set(range(n_qubits))
        keep_indices = list(keep_indices)  # 确保是列表
        trace_indices = list(total_indices - set(keep_indices))

        # 如果没有要 Trace 的，直接返回原矩阵
        if len(trace_indices) == 0:
            return rho.copy()
        # 如果 Trace 掉了所有，返回 1.0 (全迹)
        if len(keep_indices) == 0:
            return np.trace(rho)

        # 3. 核心数学操作：Reshape -> Transpose -> Trace

        # 3.1 Reshape: 将矩阵变成 2N 个轴的张量
        # 形状变为 [d1, d2, ..., dn, d1, d2, ..., dn]
        # 前 n 个是 Row 索引，后 n 个是 Column 索引
        rho_tensor = rho.reshape(dims + dims)

        # 3.2 Transpose: 重新排列轴
        # 我们需要把【保留的行】、【保留的列】移到前面
        # 把【Trace的行】、【Trace的列】移到后面
        # 目标顺序: [keep_row..., keep_col..., trace_row..., trace_col...]

        # 构建排列顺序
        perm_order = keep_indices + [x + n_qubits for x in keep_indices] + \
                     trace_indices + [x + n_qubits for x in trace_indices]

        rho_permuted = rho_tensor.transpose(perm_order)

        # 3.3 Flatten: 将保留部分和 Trace 部分分别合并
        # 现在的形状是 [Keep_Row, Keep_Col, Trace_Row, Trace_Col]
        dim_keep = int(np.prod([dims[i] for i in keep_indices]))
        dim_trace = int(np.prod([dims[i] for i in trace_indices]))

        rho_blocked = rho_permuted.reshape((dim_keep, dim_keep, dim_trace, dim_trace))

        # 3.4 Trace: 对最后两个维度求迹 (Sum of diagonal elements)
        # 也就是对 Trace 部分的 Row 和 Col 索引相等的元素求和
        rho_reduced = np.trace(rho_blocked, axis1=2, axis2=3)

        return rho_reduced

    def get_ghz_state(self, F):
        """生成 3-qubit GHZ 态: (|000> + |111>)/sqrt(2) 混合白噪声"""
        # 纯态
        psi = np.zeros(8)
        psi[0] = 1 / np.sqrt(2)
        psi[7] = 1 / np.sqrt(2)
        rho_pure = np.outer(psi, psi.conj())

        # 混合白噪声 (Werner-like)
        dim = 8
        rho_noise = np.eye(dim) / dim
        return F * rho_pure + (1 - F) * rho_noise

    def get_cluster_state_3q(self, F):
        """生成 3-qubit 线性簇态: CZ_12 CZ_23 |+++>"""
        # 1. 初始 |+++>
        p = np.array([1, 1]) / np.sqrt(2)
        psi = np.kron(np.kron(p, p), p)  # 长度8的向量

        # 2. 应用 CZ 门 (这里手动写或者用矩阵乘法)
        # 简单的做法是利用 projector
        # 这里假设你有生成 Cluster 态矩阵的函数
        # ... (省略矩阵生成细节，逻辑同上)
        rho_pure = np.outer(psi, psi.conj())

        # 混合噪声
        dim = 8
        return F * rho_pure + (1 - F) * np.eye(dim) / dim

    def _get_ghz_fidelity(self, rho):
        """
        计算 N-qubit GHZ 态保真度。
        Target: (|00...0> + |11...1>) / sqrt(2)
        """

        trace_val = np.trace(rho).real
        if trace_val < 1e-12:
            return 0.0  # 态不存在，保真度为 0
        rho = rho / trace_val

        dim = rho.shape[0]
        # 1. 构造 GHZ 态矢量 (不需要生成全矩阵)
        # 只有第 0 个分量 (|0...0>) 和最后第 -1 个分量 (|1...1>) 是 1/sqrt(2)
        target_state = np.zeros(dim, dtype=complex)
        target_state[0] = 1.0 / np.sqrt(2)
        target_state[-1] = 1.0 / np.sqrt(2)

        # 2. 计算 F = <psi| rho |psi>
        # vdot 是复数共轭点积: a^H . b
        # 先算 temp = rho @ target_state (矩阵 x 向量)
        # 再算 target_state^H . temp
        temp = rho @ target_state
        fidelity = np.vdot(target_state, temp).real

        return fidelity

    def _get_cluster_fidelity(self, rho):
        """
        计算 3-qubit 线性簇态 (Linear Cluster State) 保真度。
        Target: CZ_01 * CZ_12 * |+++>
        """
        dim = rho.shape[0]
        n_qubits = int(np.log2(dim))

        # 1. 获取目标态矢量 (建议在 reset 时缓存这个 vector，不要每次都算)
        target_state = self._generate_linear_cluster_vector(n_qubits)

        # 2. 计算 F = <psi| rho |psi>
        temp = rho @ target_state
        fidelity = np.vdot(target_state, temp).real

        return fidelity

    def _generate_linear_cluster_vector(self, n_qubits):
        """
        生成 N-qubit 线性簇态的态矢量。
        定义: |G> = (Product of CZ_{i, i+1}) |+>^{\otimes N}
        """
        # 1. 初始态 |+>^{\otimes N}
        # 也就是均匀叠加态，每个分量都是 1/sqrt(2^N)
        dim = 2 ** n_qubits
        psi = np.ones(dim, dtype=complex) / np.sqrt(dim)

        # 2. 应用 CZ 门相位
        # 只有当相邻两个 qubit 都是 1 时，相位翻转 (-1)
        # 我们遍历所有基底索引 i (0 到 dim-1)
        for i in range(dim):
            phase = 1.0
            # 检查每一对相邻 qubit (j, j+1)
            for j in range(n_qubits - 1):
                # 提取第 j 位和第 j+1 位的值
                # 注意：这里假设 Big-Endian (q0 是高位) 还是 Little-Endian?
                # 为了物理上的对称性，线性链怎么数都行，只要统一即可。
                # 这里使用 (indices >> shift) & 1 提取

                # 假设 q0 是最高位 (与大多数模拟器一致)
                bit_current = (i >> (n_qubits - 1 - j)) & 1
                bit_next = (i >> (n_qubits - 1 - (j + 1))) & 1

                if bit_current == 1 and bit_next == 1:
                    phase *= -1.0

            psi[i] *= phase

        return psi

    def _generate_ghz_state(self, F, coherence_strength=0.0):
        """
        生成含噪 GHZ 态: F * |GHZ><GHZ| + (1-F)*I/8
        可选: 添加相干噪声 (Coherence) 来增加难度
        """
        dim = 8
        # 1. 理想 GHZ 态 |000> + |111>
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1.0 / np.sqrt(2)
        psi[-1] = 1.0 / np.sqrt(2)
        rho_pure = np.outer(psi, psi.conj())

        # 2. 混合白噪声 (Werner Component)
        rho_noise = np.eye(dim) / dim
        rho = F * rho_pure + (1 - F) * rho_noise

        # 3. (可选) 添加相干噪声/基底旋转
        # 这会模拟例如 |000> 和 |001> 之间的相位干扰，逼迫 RL 学会校准
        if coherence_strength > 0:
            # 例如在 index 0 (|000>) 和 1 (|001>) 之间添加非对角项
            # 保持厄米性
            rho[0, 1] += coherence_strength
            rho[1, 0] += coherence_strength

            # 重新归一化并保证半正定
            rho = self._clean_density_matrix(rho)

        return rho

    def _peek_potential_fidelity(self):
        """
        虚拟结算：窥探如果当前立即 FINISH，能达到的保真度是多少。
        不会改变环境的真实状态。
        """
        # 1. 拿到当前走到这一步的物理电路
        circuit = self._get_cirq(maybe_add_noise=True)
        sim = cirq.DensityMatrixSimulator()

        # 2. 模拟演化到当前步
        result = sim.simulate(circuit, initial_state=self.rho_initial, qubit_order=self.qubits[:6])
        rho_6q = result.final_density_matrix

        # 3. 强行进行虚拟后选择测量
        # (这里用你环境里的 P1 或 P2 测量逻辑，根据你的设定)
        rho_post, p_succ = self._post_select_ghz_ancilla_even(rho_6q, mode='P1', measure_indices=[3, 4, 5])

        # 如果测量成功率极低，说明当前状态如果 FINISH 会彻底失败
        if p_succ < 1e-4:
            return 0.0, p_succ

        # 4. 提取 Target 的虚拟状态并计算保真度
        rho_sys_out = self.manual_partial_trace(rho_post, keep_indices=[0, 1, 2], dims=[2] * 6)
        rho_sys_out = self._clean_density_matrix(rho_sys_out)
        f_potential = self._get_ghz_fidelity(rho_sys_out)

        return f_potential, p_succ


    def _get_logarithmic_negativity(self, rho_3q):
        """
        【究极物理版】计算 3-qubit 密度矩阵关于三种二分划分的对数负性，
        并取最小值，严格衡量多方真实纠缠 (GME)。
        """
        # 1. 展开为 6 阶张量 (i0, i1, i2, j0, j1, j2)
        trace_val = np.trace(rho_3q).real
        if trace_val < 1e-12:
            return 0.0
        rho_3q = rho_3q / trace_val

        tensor = rho_3q.reshape((2, 2, 2, 2, 2, 2))

        def calc_bipartite_en(swap_ax1, swap_ax2):
            """计算指定划分的部分转置对数负性"""
            pt_tensor = np.swapaxes(tensor, swap_ax1, swap_ax2)
            # 压平为 8x8 矩阵 (Numpy的C-order会完美匹配新的行列索引)
            rho_pt = pt_tensor.reshape((8, 8))
            # 必须使用 eigvalsh (因为部分转置矩阵保证是厄米阵)
            eigvals = np.linalg.eigvalsh(rho_pt)
            trace_norm = np.sum(np.abs(eigvals))
            # 过滤极其微小的浮点噪音，确保最小为 0 (log2(1)=0)
            return np.log2(max(trace_norm, 1.0))

        # 2. 分别计算三个物理切面的纠缠度
        # 划分 q0 | q1, q2 (交换 i0 和 j0)
        en_0 = calc_bipartite_en(0, 3)
        # 划分 q1 | q0, q2 (交换 i1 和 j1)
        en_1 = calc_bipartite_en(1, 4)
        # 划分 q2 | q0, q1 (交换 i2 和 j2)
        en_2 = calc_bipartite_en(2, 5)

        # 3. 严格的木桶原理：只要有一个切面断开(为0)，整体多方纠缠即为 0
        return min(en_0, en_1, en_2)

    def _post_select_ghz_and_ancillas(self, rho_9q):
        """
        对 Resource (q3, q4, q5) 和 Ancilla (q6, q7, q8) 进行 Z 基投影。
        Resource 保留 |000> 或 |111>
        Ancilla 严格保留 |000>
        """
        dim = rho_9q.shape[0]  # 应该是 512
        indices = np.arange(dim)

        # 假设 cirq 的端序是 MSB (q0) -> LSB (q8)
        # Resource: q3, q4, q5 对应的掩码是 0b000 111 000 = 56
        res_mask = 0b000111000
        # Ancilla: q6, q7, q8 对应的掩码是 0b000 000 111 = 7
        anc_mask = 0b000000111

        # 1. 辅助比特严格为 000
        anc_valid = (indices & anc_mask) == 0

        # 2. 资源比特为 000 或 111
        res_000 = (indices & res_mask) == 0
        res_111 = (indices & res_mask) == res_mask

        # 取交集
        mask_000 = anc_valid & res_000
        mask_111 = anc_valid & res_111

        valid_indices_mask = mask_000 | mask_111
        diags = np.diagonal(rho_9q)
        p_succ = np.sum(diags[valid_indices_mask]).real

        if p_succ < 1e-10:
            return np.zeros_like(rho_9q), 0.0

        # 消除非物理相干性
        vec_000 = mask_000.astype(float).reshape(-1, 1)
        vec_111 = mask_111.astype(float).reshape(-1, 1)
        rho_post = (rho_9q * (vec_000 @ vec_000.T)) + (rho_9q * (vec_111 @ vec_111.T))

        rho_post /= p_succ
        return rho_post, p_succ


    def _post_select_hybrid_9q(self, rho_9q, resource_mode='P2'):
        """
        9-Qubit 极速后选择与张量降维：
        - 目标系统 Target (q0, q1, q2) -> 最终提取为 8x8 密度矩阵
        - 资源系统 Resource (q3, q4, q5) -> 根据协议选择 P1 (偶宇称) 或 P2 (全0/全1)
        - 辅助系统 Ancilla (q6, q7, q8) -> 严格要求全 0

        参数:
        rho_9q: 输入的 512x512 全局密度矩阵
        resource_mode: 'P2' (默认, 适合递归纯化) 或 'P1' (适合 Murao 第一轮)
        """
        dim = rho_9q.shape[0]  # 512
        indices = np.arange(dim)

        # --- 1. 位运算掩码与测量结果提取 ---
        res_mask = 0b000111000  # 对应 56
        anc_mask = 0b000000111  # 对应 7

        res_vals = (indices & res_mask) >> 3  # 右移 3 位，提取为 0~7
        anc_vals = indices & anc_mask  # 提取为 0~7

        # Ancilla 必须严格全 0
        anc_valid_mask = (anc_vals == 0)

        # Resource 允许的测量结果
        if resource_mode == 'P1':
            allowed_res = [0, 3, 5, 6]  # 偶宇称: 000, 011, 101, 110
        else:
            allowed_res = [0, 7]  # 严格一致: 000, 111

        res_valid_mask = np.isin(res_vals, allowed_res)

        # 物理上存活下来的完整态空间掩码
        valid_mask = anc_valid_mask & res_valid_mask

        # --- 2. 成功率计算 (提前终止以节约算力) ---
        diags = np.diagonal(rho_9q)
        p_succ = np.sum(diags[valid_mask]).real

        # 如果概率极低，直接返回 0 矩阵，避免除零报错
        if p_succ < 1e-10:
            return np.zeros((8, 8), dtype=np.complex128), 0.0

        # --- 3. 张量极速降维与经典概率混合 ---
        # 将 512x512 变形成 (8, 8, 8) 的输入输出张量
        # 维度含义: [Target_out, Resource_out, Ancilla_out, Target_in, Resource_in, Ancilla_in]
        tensor_rho = rho_9q.reshape((8, 8, 8, 8, 8, 8))

        rho_sys_out = np.zeros((8, 8), dtype=np.complex128)

        # 遍历允许的 Resource 经典测量结果
        for res_val in allowed_res:
            # 直接通过张量切片抠出目标分支！
            # ancilla 锁定为 0, resource 锁定为当前遍历的合法值
            # 这等价于 Tr_{res,anc} (P_{res,anc} * rho * P_{res,anc})
            rho_branch = tensor_rho[:, res_val, 0, :, res_val, 0]

            # 【物理核心】: 如我们确认的，无论是 111 还是 011，
            # 这里都不需要 Pauli 前馈校正，直接作为经典系综概率相加！
            rho_sys_out += rho_branch

        # --- 4. 归一化与保真保护 ---
        rho_sys_out /= p_succ

        # 强制确保厄米性 (Hermiticity)，防止后续保真度/熵计算因微小浮点误差报出虚数错误
        rho_sys_out = (rho_sys_out + rho_sys_out.conj().T) / 2.0

        return rho_sys_out.astype(np.complex128), p_succ

    def _get_ghz4_fidelity(self, rho_4q):
        """
        计算 4-qubit 态相对于理想 GHZ 态的保真度。
        """
        import numpy as np

        # 构造完美的测量基底
        psi_ideal = np.zeros(16, dtype=complex)
        psi_ideal[0] = 1.0 / np.sqrt(2)
        psi_ideal[15] = 1.0 / np.sqrt(2)
        rho_ideal = np.outer(psi_ideal, psi_ideal.conj())

        # 计算重叠度 Tr(rho_ideal * rho_4q)
        fidelity = np.trace(rho_ideal @ rho_4q).real

        # 滤除浮点数误差，强行截断在 [0, 1] 之间
        return max(0.0, min(1.0, float(fidelity)))

    def _generate_ghz4_state(self, F: float):
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

    def _get_logarithmic_negativity_4q_old(self, rho_4q):
        """
        计算 4-qubit 态的对数负度 (Logarithmic Negativity)。
        基于二分划分: Alice+Bob (q0, q1) vs Charlie (q2, q3)
        """
        import numpy as np
        # 1. 变成 8 个维度的张量: 前4个是行索引(ket)，后4个是列索引(bra)
        tensor = rho_4q.reshape((2, 2, 2, 2, 2, 2, 2, 2))

        # 2. 对前两个比特 (q0, q1) 进行部分转置
        # 原索引: 0,1,2,3 (行) | 4,5,6,7 (列)
        # 将 q0(0和4互换), q1(1和5互换)
        transposed_tensor = tensor.transpose((4, 5, 2, 3, 0, 1, 6, 7))

        # 3. 变回 16x16 矩阵
        rho_pt = transposed_tensor.reshape((16, 16))

        # 4. 求特征值 (部分转置后矩阵依然是厄米的，所以用 eigvalsh 更快更稳)
        eigvals = np.linalg.eigvalsh(rho_pt)

        # 5. LN 公式: log2( sum( |lambda_i| ) )
        ln = np.log2(np.sum(np.abs(eigvals)))

        return max(0.0, float(ln.real))

    def _get_logarithmic_negativity_4q(self, rho_4q):
        """
        计算 4-qubit 态所有可能二分划下的对数负度，并取最小值。
        这提供了一个真正的多体纠缠下界 (GME lower bound)。
        """
        import numpy as np

        # 变成 8 个维度的张量: 前4个是行(ket)，后4个是列(bra)
        tensor = rho_4q.reshape((2, 2, 2, 2, 2, 2, 2, 2))

        # 定义 4-qubit 的 7 种非平庸二分划 (只对子系统 A 进行部分转置)
        # 格式: (子系统 A 的 qubit 索引列表)
        partitions_A = [
            [0], [1], [2], [3],  # 1 vs 3 cuts
            [0, 1], [0, 2], [0, 3]  # 2 vs 2 cuts (其余 3 种对称，无需重复算)
        ]

        min_ln = float('inf')

        for part_A in partitions_A:
            # 初始化默认的轴顺序
            axes = [0, 1, 2, 3, 4, 5, 6, 7]

            # 对子系统 A 中的每个 qubit，交换其 ket 和 bra 的轴
            for q_idx in part_A:
                # 对应的 ket 轴是 q_idx, bra 轴是 q_idx + 4
                ket_axis = q_idx
                bra_axis = q_idx + 4
                # 交换
                axes[ket_axis], axes[bra_axis] = axes[bra_axis], axes[ket_axis]

            # 执行部分转置
            transposed_tensor = tensor.transpose(axes)
            rho_pt = transposed_tensor.reshape((16, 16))

            # 强制厄米化以消除计算机浮点数带来的微小虚部影响
            rho_pt = (rho_pt + rho_pt.conj().T) / 2.0

            # 求解特征值并计算对数负度
            eigvals = np.linalg.eigvalsh(rho_pt)
            ln_val = np.log2(np.sum(np.abs(eigvals)))

            # 更新最小值
            if ln_val < min_ln:
                min_ln = ln_val

        return max(0.0, float(min_ln.real))

    def _evaluate_dynamic_post_selection_10bit(self, mode, mask_policy):
        rho_10q = self.rho.copy()

        if mode is None and mask_policy != "ANC":
            raise ValueError("mode=None is only allowed for ANC-only finish action.")

        if mode == "P1" and mask_policy in ["RES", "BOTH"]:
            # Resource 是 q4, q5, q6, q7
            basis_change_circuit = cirq.Circuit(cirq.H.on_each(self.qubits[4], self.qubits[5], self.qubits[6], self.qubits[7]))
            res = self.simulator.simulate(basis_change_circuit, initial_state=rho_10q, qubit_order=self.qubits)
            rho_10q = res.final_density_matrix

        diag_elements = np.diagonal(rho_10q).real
        # 10-qubit: q0(bit9) q1(bit8) q2(bit7) q3(bit6) q4(bit5) q5(bit4) q6(bit3) q7(bit2) q8(bit1) q9(bit0)
        final_valid_mask = np.ones(1024, dtype=bool)

        # --- 2. 双辅助比特掩码 (ANC: q8 + q9) ---
        # q8 位于 bit 1 (Z-check), q9 位于 bit 0 (X-check)
        if mask_policy in ["ANC", "BOTH"]:
            anc_valid_mask = np.array([(i & 0b11) == 0 for i in range(1024)])
            final_valid_mask &= anc_valid_mask

        # --- 3. 资源比特掩码 (RES: q4, q5, q6, q7) ---
        # Resource 占据 bits 5,4,3,2 (右移 2 位跳过两个 ancilla)
        if mask_policy in ["RES", "BOTH"]:
            if mode == "P1":
                # X 基底下的 4-qubit 偶宇称态
                allowed_res = [0, 3, 5, 6, 9, 10, 12, 15]
            else:
                # Z 基底下的 4-qubit 全同态
                allowed_res = [0, 15]

            res_vals = np.array([((i >> 2) & 0b1111) for i in range(1024)])
            res_valid_mask = np.isin(res_vals, allowed_res)
            final_valid_mask &= res_valid_mask

        # 概率计算与归一化
        p_succ = np.sum(diag_elements[final_valid_mask])
        if p_succ < 1e-10:
            return -0.2, {'msg': 'P_succ zero', 'final_f': 0.0, 'p_succ': 0.0}

        # Project each retained measurement branch independently. This avoids
        # retaining coherence between distinct classical postselection results.
        if mask_policy == "ANC":
            measurement_bits_mask = 0b11
        elif mask_policy == "RES":
            measurement_bits_mask = 0b111100
        elif mask_policy == "BOTH":
            measurement_bits_mask = 0b111111
        else:
            raise ValueError(f"Unknown mask policy: {mask_policy}")

        projected_rho = np.zeros_like(rho_10q)
        valid_indices = np.where(final_valid_mask)[0]
        for result in set(index & measurement_bits_mask for index in valid_indices):
            branch_mask = np.array([
                (index & measurement_bits_mask) == result
                for index in range(1024)
            ])
            branch_indices = np.where(branch_mask)[0]
            if np.any(final_valid_mask[branch_indices]):
                vector = branch_mask.astype(float).reshape(-1, 1)
                projected_rho += rho_10q * (vector @ vector.T)
        projected_rho /= p_succ

        # 提取 4-qubit Target，dim=[2]*10（偏迹掉 q4-q9）
        target_rho = self.manual_partial_trace(projected_rho, keep_indices=[0, 1, 2, 3], dims=[2]*10)
        target_rho = self._apply_final_conversion_noise(target_rho)
        if target_rho is None:
            return -0.2, {
                'msg': 'Invalid target rho after topology conversion noise',
                'final_f': 0.0,
                'p_succ': float(p_succ),
            }

        final_f = self._get_ghz4_fidelity(target_rho)
        en_out = self._get_logarithmic_negativity_4q(target_rho)

        # if final_f <= self.F_in:
        #     reward = -0.1
        # else:
        #     # 核心：利用 p_succ * 提升度，迫使 Agent 寻找高产量、高质量的平衡点
        #     reward = p_succ * (final_f - self.F_in) * 10.0
        if final_f >= self.F_in + 0.0002 :#or delta_en > 1e-5:  # 0.005
            # 1. 过滤掉字符串(如"FINISH_P1")，把所有有效门强行拼成一个完整的独立线路
            full_circuit = cirq.Circuit([a for a in self.circuit_gates if not isinstance(a, str)])

            # 2. 初始化计数器
            single_qubit_count = 0
            two_qubit_count = 0

            # 3. 摊平线路，遍历每一个物理门
            for op in full_circuit.all_operations():
                # op.qubits 返回的是这个门作用的比特元组，求长度就是比特数
                if len(op.qubits) == 1:
                    single_qubit_count += 1
                elif len(op.qubits) == 2:
                    two_qubit_count += 1

            total_gates_plenty = single_qubit_count * 0.04 + two_qubit_count * 0.06

            s_tot, s_a, s_b = self._get_renyi(target_rho)
            con_difference = abs(self.s_tot_norm_tar - s_tot)
            con_difference_a = abs(self.s_a_norm_tar - s_a)

            self.goal_achieved = True

            # 你的自定义多目标奖励公式
            delta_f = final_f - self.F_in
            reward = (20 + (delta_f * p_succ * 200 + (1 - con_difference) + (
                    1 - con_difference_a)) + 1 - self.step_counter / self.max_timesteps - total_gates_plenty * 5)
            reward = max(0, reward)
        else:
            self.goal_achieved = False
            reward = -0.15  # -step_penalty
            # reward = max(reward, -0.4)

        return reward, {
            'final_f': final_f,
            'final_en': en_out,
            'p_succ': p_succ,
            'f_improvement': final_f - self.F_in,
            'en_improvement': en_out - self.en_in,
            'topology': self.current_topology,
            'topology_noise': self.topology_noise_report,
        }

    def _evaluate_dynamic_post_selection_3bit(self, mode, mask_policy):
        """
        基于 P1 (X基底/偶宇称) 和 P2 (Z基底/全同) 的动态后选择结算。
        """
        rho_9q = self.rho.copy()

        # --- 1. 物理旋转：如果是 P1 (相位纯化)，必须先将 Resource 旋转到 X 基底 ---
        if mode == "P1":
            # 对应你确认的逻辑：P1 给 H 门，准备测 0356 偶宇称
            basis_change_circuit = cirq.Circuit(cirq.H.on_each(self.qubits[3], self.qubits[4], self.qubits[5]))  # q3, q4, q5
            res = self.simulator.simulate(basis_change_circuit, initial_state=rho_9q, qubit_order=self.qubits)
            rho_9q = res.final_density_matrix

        diag_elements = np.diagonal(rho_9q).real
        final_valid_mask = np.ones(512, dtype=bool)

        # --- 2. 辅助比特掩码 (ANC): 严格全 0 ---
        if mask_policy in ["ANC", "BOTH"]:
            anc_valid_mask = np.array([(i & 0b000000111) == 0 for i in range(512)])
            final_valid_mask &= anc_valid_mask

        # --- 3. 资源比特掩码 (RES): 区分模式匹配合法状态 ---
        if mask_policy in ["RES", "BOTH"]:
            if mode == "P1":
                # X 基底下合法的 GHZ 稳定子状态：偶宇称
                allowed_res = [0, 3, 5, 6]  # 000, 011, 101, 110
            else:
                # Z 基底下合法的 GHZ 状态：全同
                allowed_res = [0, 7]  # 000, 111

            res_vals = np.array([((i >> 3) & 0b111) for i in range(512)])
            res_valid_mask = np.isin(res_vals, allowed_res)
            final_valid_mask &= res_valid_mask

        # --- 4. 概率计算与归一化 ---
        p_succ = np.sum(diag_elements[final_valid_mask])
        if p_succ < 1e-10:
            return -0.2, {'msg': 'P_succ zero', 'final_f': 0.0, 'p_succ': 0.0}

        # 投影与消除非物理相干性 (NEW)
        projected_rho = np.zeros_like(rho_9q)

        # 测量结果由辅助比特(q6,q7,q8)和资源比特(q3,q4,q5)共同决定，
        # 在 9-qubit (512维) 空间中，由于 q0-q2 占据高位，
        # 测量结果严格对应矩阵索引的后 6 位，即 i % 64
        valid_indices = np.where(final_valid_mask)[0]
        valid_measurement_results = set(valid_indices % 64)

        for val in valid_measurement_results:
            # 单独提取属于该确切测量结果的掩码
            branch_mask = np.array([(i % 64) == val for i in range(512)])

            # 只有当该测量分支符合您的后选择策略时，才进行投影
            if final_valid_mask[np.where(branch_mask)[0][0]]:
                vec = branch_mask.astype(float).reshape(-1, 1)
                # 独立投影并相加，彻底抹除不同分支间的量子相干性
                projected_rho += rho_9q * (vec @ vec.T)

        projected_rho /= p_succ

        # 使用你提供的 manual_partial_trace 提取 Target (q0, q1, q2)
        target_rho = self.manual_partial_trace(projected_rho, keep_indices=[0, 1, 2])

        # --- 5. 奖励结算 ---
        final_f = self._get_ghz_fidelity(target_rho)

        en_out = self._get_logarithmic_negativity(target_rho)
        delta_en = en_out - self.en_in

        # if final_f <= self.F_in:
        #     reward = -0.1
        # else:
        #     # 核心：利用 p_succ * 提升度，迫使 Agent 寻找高产量、高质量的平衡点
        #     reward = p_succ * (final_f - self.F_in) * 10.0
        if final_f >= self.F_in + 0.0001 or delta_en >1e-5:#0.005
            # 1. 过滤掉字符串(如"FINISH_P1")，把所有有效门强行拼成一个完整的独立线路
            full_circuit = cirq.Circuit([a for a in self.circuit_gates if not isinstance(a, str)])

            # 2. 初始化计数器
            single_qubit_count = 0
            two_qubit_count = 0

            # 3. 摊平线路，遍历每一个物理门
            for op in full_circuit.all_operations():
                # op.qubits 返回的是这个门作用的比特元组，求长度就是比特数
                if len(op.qubits) == 1:
                    single_qubit_count += 1
                elif len(op.qubits) == 2:
                    two_qubit_count += 1

            total_gates_plenty = single_qubit_count*0.04 + two_qubit_count*0.06

            s_tot, s_a, s_b = self._get_renyi(target_rho)
            con_difference = abs(self.s_tot_norm_tar - s_tot)
            con_difference_a = abs(self.s_a_norm_tar - s_a)

            self.goal_achieved = True

            # 你的自定义多目标奖励公式
            delta_f = final_f - self.F_in
            reward = (20 + (delta_f * p_succ * 200 + (1 - con_difference) + (
                        1 - con_difference_a)) + 1 - self.step_counter / self.max_timesteps - total_gates_plenty*25)
            reward = max(0.0, reward)
        else:
            self.goal_achieved = False
            reward = -0.15#-step_penalty
            #reward = max(reward, -0.4)

        return reward, {
            'final_f': final_f,
            'p_succ': p_succ,
            'f_improvement': final_f - self.F_in
        }

    def _evaluate_dynamic_post_selection(self, mode, mask_policy):
        rho_9q = self.rho.copy()

        # mode=None 只用于 FINISH_MASK_ANC
        if mode is None and mask_policy != "ANC":
            raise ValueError("mode=None is only allowed for ANC-only finish action.")

        # P1 只有在 resource 参与后选择时才需要切换到 X 基底
        if mode == "P1" and mask_policy in ["RES", "BOTH"]:
            basis_change_circuit = cirq.Circuit(
                cirq.H.on_each(
                    self.qubits[4],
                    self.qubits[5],
                    self.qubits[6],
                    self.qubits[7]
                )
            )
            res = self.simulator.simulate(
                basis_change_circuit,
                initial_state=rho_9q,
                qubit_order=self.qubits[:9]
            )
            rho_9q = res.final_density_matrix

        diag_elements = np.diagonal(rho_9q).real

        # 9-qubit 排序:
        # q0(bit8) q1(bit7) q2(bit6) q3(bit5)
        # q4(bit4) q5(bit3) q6(bit2) q7(bit1) q8(bit0)

        final_valid_mask = np.ones(512, dtype=bool)

        # --- ANC mask: q8 = 0 ---
        if mask_policy in ["ANC", "BOTH"]:
            anc_valid_mask = np.array([
                (i & 0b1) == 0 for i in range(512)
            ])
            final_valid_mask &= anc_valid_mask

        # --- RES mask: q4-q7 ---
        if mask_policy in ["RES", "BOTH"]:
            if mode == "P1":
                # X 基底下 4-qubit 偶宇称
                allowed_res = [0, 3, 5, 6, 9, 10, 12, 15]
            elif mode == "P2":
                # Z 基底下 4-qubit 全同
                allowed_res = [0, 15]
            else:
                raise ValueError("RES/BOTH mask requires mode to be P1 or P2.")

            res_vals = np.array([
                (i >> 1) & 0b1111 for i in range(512)
            ])
            res_valid_mask = np.isin(res_vals, allowed_res)
            final_valid_mask &= res_valid_mask

        p_succ = np.sum(diag_elements[final_valid_mask])

        if p_succ < 1e-10:
            return -0.2, {
                "msg": "P_succ zero",
                "final_f": 0.0,
                "p_succ": 0.0
            }

        # --- projection/decoherence ---
        if mask_policy == "ANC":
            measurement_bits_mask = 0b00001
        elif mask_policy == "RES":
            measurement_bits_mask = 0b11110
        elif mask_policy == "BOTH":
            measurement_bits_mask = 0b11111
        else:
            raise ValueError(f"Unknown mask_policy: {mask_policy}")

        projected_rho = np.zeros_like(rho_9q)

        valid_indices = np.where(final_valid_mask)[0]
        valid_measurement_results = set(
            i & measurement_bits_mask for i in valid_indices
        )

        for val in valid_measurement_results:
            branch_mask = np.array([
                (i & measurement_bits_mask) == val for i in range(512)
            ])

            branch_indices = np.where(branch_mask)[0]

            if np.any(final_valid_mask[branch_indices]):
                vec = branch_mask.astype(float).reshape(-1, 1)
                projected_rho += rho_9q * (vec @ vec.T)

        projected_rho /= p_succ

        target_rho = self.manual_partial_trace(
            projected_rho,
            keep_indices=[0, 1, 2, 3],
            dims=[2] * 9
        )
        target_rho = self._apply_final_conversion_noise(target_rho)
        if target_rho is None:
            return -0.2, {
                "msg": "Invalid target rho after topology conversion noise",
                "final_f": 0.0,
                "p_succ": float(p_succ),
                "topology": self.current_topology,
                "topology_noise": self.topology_noise_report,
            }

        final_f = self._get_ghz4_fidelity(target_rho)
        en_out = self._get_logarithmic_negativity_4q(target_rho)
        delta_en = en_out - self.en_in

        if final_f >= self.F_in + 0.0001: #or delta_en>0.0001 :#or self.F_in > 0.95:
            # 1. 过滤掉字符串(如"FINISH_P1")，把所有有效门强行拼成一个完整的独立线路
            full_circuit = cirq.Circuit([a for a in self.circuit_gates if not isinstance(a, str)])

            # 2. 初始化计数器
            single_qubit_count = 0
            two_qubit_count = 0

            # 3. 摊平线路，遍历每一个物理门
            for op in full_circuit.all_operations():
                # op.qubits 返回的是这个门作用的比特元组，求长度就是比特数
                if len(op.qubits) == 1:
                    single_qubit_count += 1
                elif len(op.qubits) == 2:
                    two_qubit_count += 1

            total_gates_plenty = single_qubit_count * 0.04 + two_qubit_count * 0.06
            #total_gates_plenty = 0

            s_tot, s_a, s_b = self._get_renyi(target_rho)
            con_difference = abs(self.s_tot_norm_tar - s_tot)
            con_difference_a = abs(self.s_a_norm_tar - s_a)

            self.goal_achieved = True

            # 你的自定义多目标奖励公式
            delta_f = final_f - self.F_in
            reward = (20 + (delta_f * p_succ * 200 + (1 - con_difference) + (
                    1 - con_difference_a)) + 1 - self.step_counter / self.max_timesteps - total_gates_plenty*5) + (delta_en) * p_succ * 20# *5+delta_en*p_succ*30+delta_f * p_succ*30
            if reward <=0:
                reward = max(0, (delta_en*0.5+delta_f*1.5) * p_succ * 20 - total_gates_plenty*2.5)#*2.5
                #reward = max(0, delta_en*p_succ*20 - total_gates_plenty * 5,delta_f * p_succ*20 - total_gates_plenty * 5)
        else:
            self.goal_achieved = False
            reward = -0.15  # -step_penalty

        topology_penalty = self._topology_conversion_penalty()
        reward -= topology_penalty

        return reward, {
            'final_f': final_f,
            'final_en': en_out,
            'p_succ': p_succ,
            'f_improvement': final_f - self.F_in,
            'en_improvement': en_out - self.en_in,
            'topology': self.current_topology,
            'topology_conversion_penalty': topology_penalty,
            'topology_noise': self.topology_noise_report,
        }
