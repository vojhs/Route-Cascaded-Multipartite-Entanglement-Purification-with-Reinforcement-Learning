from typing import Dict, List, Optional, Union

import cirq
import numpy as np
import itertools
from typing import Any
n = 3  # Example value for n
k1 = 1
k2 = 2

def get_default_gates2(qubits: List[cirq.LineQubit]) -> (List, int):
    gates = []
    num_fixed_actions = None
    for idx, qubit in enumerate(qubits):
        next_qubit = qubits[(idx + 1) % len(qubits)]
        fixed_gates = [
            cirq.rz(np.pi / 4.)(qubit),
            #cirq.ry(np.pi / -2.)(qubit),
            cirq.X(qubit),
            cirq.Y(qubit),
            cirq.Z(qubit),
            cirq.H(qubit),
            cirq.CNOT(qubit, next_qubit),
            cirq.CZ(qubit, next_qubit),
            #cirq.ry(np.pi / 2.)(qubit),
            #FGate(n, k1)(qubit, next_qubit),
            #FGate(n, k2)(qubit, next_qubit)
        ]
        if num_fixed_actions is None:
            num_fixed_actions = len(fixed_gates)  # 比如 11 个
        gates += fixed_gates
        # 添加参数化旋转门：返回一个函数用于后续根据连续参数构造门操作
        def param_ry(angle, qubit=qubit):
            return cirq.ry(angle)(qubit)
        gates.append(param_ry)
    return gates, num_fixed_actions

def get_default_gates3(qubits: List[cirq.LineQubit]) -> List[dict]:
    gates = []
    for idx, qubit in enumerate(qubits):
        next_qubit = qubits[(idx + 1) % len(qubits)]
        # 单量子位参数化门（RY/RZ）
        gates.append({
            "gate_constructor": lambda angle, q=qubit: cirq.ry(angle).on(q),
            "qubits": [qubit],
            "needs_param": True
        })
        gates.append({
            "gate_constructor": lambda angle, q=qubit: cirq.rz(angle).on(q),
            "qubits": [qubit],
            "needs_param": True
        })
        # 非参数化门（X, CNOT等）
        gates.append({
            "gate_constructor": lambda q=qubit: cirq.X(q),
            "qubits": [qubit],
            "needs_param": False
        })
        gates.append({
            "gate_constructor": lambda q=qubit: cirq.Y(q),
            "qubits": [qubit],
            "needs_param": False
        })
        gates.append({
            "gate_constructor": lambda q=qubit: cirq.Z(q),
            "qubits": [qubit],
            "needs_param": False
        })
        gates.append({
            "gate_constructor": lambda q=qubit: cirq.H(q),
            "qubits": [qubit],
            "needs_param": False
        })

        gates.append({
            "gate_constructor": lambda q1=qubit, q2=next_qubit: cirq.CNOT(q1, q2),
            "qubits": [qubit, next_qubit],
            "needs_param": False
        })
        gates.append({
            "gate_constructor": lambda q1=qubit, q2=next_qubit: cirq.CZ(q1, q2),
            "qubits": [qubit, next_qubit],
            "needs_param": False
        })
        gates.append({
            "gate_constructor": lambda q1=qubit, q2=next_qubit: FGate(n, k1)(q1, q2),
            "qubits": [qubit, next_qubit],
            "needs_param": False
        })
        gates.append({
            "gate_constructor": lambda q1=qubit, q2=next_qubit: FGate(n, k2)(q1, q2),
            "qubits": [qubit, next_qubit],
            "needs_param": False
        })
        # 其他门...
    return gates

def get_default_gates_qas(
        qubits: List[cirq.LineQubit]) -> List[cirq.Operation]: # 返回 Operation 更标准

    gates: List[cirq.Operation] = []
    num_qubits = len(qubits)

    if num_qubits == 0:
        return []

    # --- 定义 FGate 参数 (根据需要调整) ---
    # 注意：避免使用 'n' 作为变量名，因为它可能与 FGate 内部的 'n' 混淆
    n = num_qubits # 示例值
    k1 = 1 # 示例值
    k2 = 2 # 示例值

    # --- 1. 添加所有单比特门 ---
    # 使用原始循环结构添加单比特门比较自然
    for qubit in qubits: # 直接遍历 qubit 即可
        gates += [
            cirq.rz(np.pi / 4.)(qubit),
            cirq.X(qubit),
            cirq.Y(qubit),
            cirq.Z(qubit),
            cirq.H(qubit),
            cirq.ry(np.pi / 2.)(qubit),
            # --- 添加其他需要的单比特门 (从你的注释中选择) ---
            #cirq.T(qubit),
            cirq.S(qubit),
            #cirq.rz(np.pi/8.)(qubit), # T 门的两倍？
            #cirq.rz(np.pi/16.)(qubit),
            #cirq.rz(np.pi / 32.)(qubit),
            # ... 添加你需要的其他单比特门 ...
        ]

    # --- 2. 添加所有可能的双比特门 (独立于外层循环) ---
    if num_qubits >= 2:
        # 定义你想包含的所有双比特门类型
        two_qubit_gate_generators = [
            lambda q1, q2: cirq.CNOT(q1, q2),
            lambda q1, q2: cirq.CZ(q1, q2),
        ]

        # 遍历所有可能的有序量子比特对 (i, j) where i != j
        for i, j in itertools.permutations(range(num_qubits), 2):
            q_i = qubits[i]
            q_j = qubits[j]
            for gate_gen in two_qubit_gate_generators:
                gates.append(gate_gen(q_i, q_j))

    # --- 3. 添加所有可能的三比特门 (独立于外层循环) ---
    if num_qubits >= 3:
        # 定义你想包含的所有三比特门类型
        three_qubit_gate_generators = [
             #lambda q1, q2, q3: cirq.TOFFOLI(q1, q2, q3), # 如果需要
             # lambda q1, q2, q3: cirq.FREDKIN(q1, q2, q3), # 如果需要
        ]

        # 遍历所有可能的有序量子比特三元组 (i, j, k) where i, j, k distinct
        for i, j, k in itertools.permutations(range(num_qubits), 3):
            q_i = qubits[i]
            q_j = qubits[j]
            q_k = qubits[k]
            for gate_gen in three_qubit_gate_generators:
                 # 只有在三比特门列表不为空时才添加
                 if three_qubit_gate_generators:
                     gates.append(gate_gen(q_i, q_j, q_k))


    # --- 清理和信息输出 (可选) ---
    # 如果需要，可以去重，但理论上这种生成方式不会重复
    # print(f"Generated {len(gates)} total gate operations (fully connected).")

    return gates


def get_default_observables2(
        qubits: List[cirq.LineQubit]) -> List[cirq.GateOperation]:
    observables = []
    for qubit in qubits:
        observables += [
            cirq.X(qubit),
            cirq.Y(qubit),
            cirq.Z(qubit),
        ]
    return observables


def get_default_observables_4bit(qubits: List[cirq.LineQubit]) -> List[cirq.GateOperation]:
    # 假设比特映射为: q0, q2 归 Alice; q1, q3 归 Bob
    q0, q1, q2, q3 = qubits[0], qubits[1], qubits[2], qubits[3]

    # 定义关联观测算符 (Pauli String)
    # 纠缠纯化的核心就是将这些关联值从低变高
    ops = [
        # 1. 跨 Alice-Bob 的关联 (检测贝尔对的质量)
        cirq.PauliString(cirq.Z(q0), cirq.Z(q1)),
        cirq.PauliString(cirq.X(q0), cirq.X(q1)),
        cirq.PauliString(cirq.Z(q2), cirq.Z(q3)),
        cirq.PauliString(cirq.X(q2), cirq.X(q3)),

        # 2. Alice 本地的关联 (检测两对比特之间的协同)
        cirq.PauliString(cirq.Z(q0), cirq.Z(q2)),
        cirq.PauliString(cirq.X(q0), cirq.X(q2)),

        # 3. Bob 本地的关联
        cirq.PauliString(cirq.Z(q1), cirq.Z(q3)),
        cirq.PauliString(cirq.X(q1), cirq.X(q3)),
    ]
    return ops


def get_default_observables_bell(qubits: List[cirq.LineQubit]) -> List[cirq.GateOperation]:
    # 架构分配:
    # Alice: q0 (T), q2 (R), q4 (C)
    # Bob:   q1 (T), q3 (R), q5 (C)
    q0, q1, q2, q3, q4, q5 = qubits[0], qubits[1], qubits[2], qubits[3], qubits[4], qubits[5]

    ops = [
        # --- 1. 跨 Alice-Bob 的关联 (检测三对贝尔对各自的纠缠质量) ---
        # 目标对 (Target)
        cirq.PauliString(cirq.Z(q0), cirq.Z(q1)),
        cirq.PauliString(cirq.X(q0), cirq.X(q1)),

        # 资源对 (Resource/Ancilla)
        cirq.PauliString(cirq.Z(q2), cirq.Z(q3)),
        cirq.PauliString(cirq.X(q2), cirq.X(q3)),

        # 催化对 (Catalyst) - 必须监控，Agent 需要看到它是否保持稳定
        cirq.PauliString(cirq.Z(q4), cirq.Z(q5)),
        cirq.PauliString(cirq.X(q4), cirq.X(q5)),

        # --- 2. Alice 本地的关联 (检测本地三个比特之间的交互) ---
        # 目标与资源的交互 (q0-q2)
        cirq.PauliString(cirq.Z(q0), cirq.Z(q2)),
        # 资源与催化剂的交互 (q2-q4)
        cirq.PauliString(cirq.Z(q2), cirq.Z(q4)),
        # 目标与催化剂的交互 (q0-q4)
        cirq.PauliString(cirq.Z(q0), cirq.Z(q4)),

        # --- 3. Bob 本地的关联 (对称监控) ---
        cirq.PauliString(cirq.Z(q1), cirq.Z(q3)),
        cirq.PauliString(cirq.Z(q3), cirq.Z(q5)),
        cirq.PauliString(cirq.Z(q1), cirq.Z(q5)),

        # --- 4. (可选) 全局宇称或相位敏感算符 ---
        # 比如为了检测非对称噪声，可以增加 Y 基底的关联
        cirq.PauliString(cirq.Y(q0), cirq.Y(q1)),
    ]
    return ops


def get_default_observables6bit(qubits: List[cirq.LineQubit]) -> List[cirq.GateOperation]:
    # --- 1. 架构重新分配 (GHZ 3-Party) ---
    # Target (系统态): q0(Alice), q1(Bob), q2(Charlie)
    # Resource (资源态): q3(Alice), q4(Bob), q5(Charlie)
    # 物理分布:
    #   Alice: [q0, q3]
    #   Bob:   [q1, q4]
    #   Charlie:[q2, q5]

    # 解包 qubits (假设输入是标准的 0-5 顺序)
    t0, t1, t2 = qubits[0], qubits[1], qubits[2]  # Target Triplet
    r0, r1, r2 = qubits[3], qubits[4], qubits[5]  # Resource Triplet

    ops = [
        # =========================================================
        # A. 目标系统 (Target System) 的健康度检测
        # =========================================================
        # GHZ 态的 "Stabilizers" (稳定子)，Agent 通过这些判断 GHZ 纯不纯

        # 1. Z-基关联 (检测 Bit-flip 错误)
        # 只要 Z0*Z1=1 且 Z1*Z2=1，就说明是 |000> 或 |111>
        cirq.PauliString(cirq.Z(t0), cirq.Z(t1)),  # Alice-Bob 关联
        cirq.PauliString(cirq.Z(t1), cirq.Z(t2)),  # Bob-Charlie 关联

        cirq.PauliString(cirq.X(t0), cirq.X(t1)),  # <--- 必须加
        cirq.PauliString(cirq.X(t1), cirq.X(t2)),  # <--- 必须加

        cirq.PauliString(cirq.Y(t0), cirq.Y(t1)),  # 监控相位的两体退化
        cirq.PauliString(cirq.Y(t1), cirq.Y(t2)),
        # 2. X-基全局关联 (检测 Phase-flip 错误 / 相干性)
        # GHZ 的核心特征是 X*X*X = 1。如果这一项低，说明变成了混合态目标: XXX 趋近于 1; XYY, YXY, YYX 趋近于 -1
        cirq.PauliString(cirq.X(t0), cirq.X(t1), cirq.X(t2)),
        cirq.PauliString(cirq.X(t0), cirq.Y(t1), cirq.Y(t2)),
        cirq.PauliString(cirq.Y(t0), cirq.X(t1), cirq.Y(t2)),
        cirq.PauliString(cirq.Y(t0), cirq.Y(t1), cirq.X(t2)),

        # =========================================================
        # B. 资源系统 (Resource System) 的健康度检测
        # =========================================================
        # 同样的逻辑，监控资源态是否“干净”
        cirq.PauliString(cirq.Z(r0), cirq.Z(r1)),
        cirq.PauliString(cirq.Z(r1), cirq.Z(r2)),

        cirq.PauliString(cirq.X(r0), cirq.X(r1)),  # <--- 必须加
        cirq.PauliString(cirq.X(r1), cirq.X(r2)),  # <--- 必须加

        cirq.PauliString(cirq.Y(r0), cirq.Y(r1)),
        cirq.PauliString(cirq.Y(r1), cirq.Y(r2)),

        cirq.PauliString(cirq.X(r0), cirq.X(r1), cirq.X(r2)),
        cirq.PauliString(cirq.X(r0), cirq.Y(r1), cirq.Y(r2)),
        cirq.PauliString(cirq.Y(r0), cirq.X(r1), cirq.Y(r2)),
        cirq.PauliString(cirq.Y(r0), cirq.Y(r1), cirq.X(r2)),

        # =========================================================
        # C. 本地操作关联 (Local Correlations)
        # =========================================================
        # 这一部分非常重要！Agent 需要看到 CNOT 操作的效果。
        # 当 Alice 执行 CNOT(Target->Resource) 时，Z 关联会传播。

        # Alice 实验室内部 (Target <-> Resource)
        cirq.PauliString(cirq.Z(t0), cirq.Z(r0)),
        cirq.PauliString(cirq.X(t0), cirq.X(r0)),
        cirq.PauliString(cirq.Y(t0), cirq.Y(r0)),

        # Bob 实验室内部
        cirq.PauliString(cirq.Z(t1), cirq.Z(r1)),
        cirq.PauliString(cirq.X(t1), cirq.X(r1)),
        cirq.PauliString(cirq.Y(t1), cirq.Y(r1)),

        # Charlie 实验室内部
        cirq.PauliString(cirq.Z(t2), cirq.Z(r2)),
        cirq.PauliString(cirq.X(t2), cirq.X(r2)),
        cirq.PauliString(cirq.Y(t2), cirq.Y(r2)),
    ]

    return ops


def get_default_observables_3bit_0521(qubits: List[cirq.LineQubit]) -> List[cirq.GateOperation]:
    # =========================================================
    # 架构分配 (9-Qubit Bipartite Asymmetric)
    # =========================================================
    # Alice 实验室 (6 qubits): [q0, q1] (Target), [q3, q4] (Resource), [q6, q7] (Ancilla)
    # Bob   实验室 (3 qubits): [q2]     (Target), [q5]     (Resource), [q8]     (Ancilla)

    t0, t1, t2 = qubits[0], qubits[1], qubits[2]  # Target
    r0, r1, r2 = qubits[3], qubits[4], qubits[5]  # Resource
    a0, a1, a2 = qubits[6], qubits[7], qubits[8]  # Ancilla

    ops = [
        # =========================================================
        # A. 目标与资源的全局健康度 (跨越 A-B 的纠缠度)
        # =========================================================
        # Target GHZ
        cirq.PauliString(cirq.Z(t0), cirq.Z(t1)),
        cirq.PauliString(cirq.Z(t1), cirq.Z(t2)),
        cirq.PauliString(cirq.X(t0), cirq.X(t1)),
        cirq.PauliString(cirq.X(t1), cirq.X(t2)),
        cirq.PauliString(cirq.X(t0), cirq.X(t1), cirq.X(t2)),
        cirq.PauliString(cirq.X(t0), cirq.Y(t1), cirq.Y(t2)),
        cirq.PauliString(cirq.Y(t0), cirq.X(t1), cirq.Y(t2)),
        cirq.PauliString(cirq.Y(t0), cirq.Y(t1), cirq.X(t2)),

        # Resource GHZ (可选，如果你算力吃紧可以删减部分，但保留 ZZ 和 XX 有助于 RL 感知资源态)
        cirq.PauliString(cirq.Z(r0), cirq.Z(r1)),
        cirq.PauliString(cirq.Z(r1), cirq.Z(r2)),
        cirq.PauliString(cirq.X(r0), cirq.X(r1)),
        cirq.PauliString(cirq.X(r1), cirq.X(r2)),

        # =========================================================
        # B. 辅助比特 (Ancilla) 的单体状态
        # =========================================================
        # 监控 3 个便签纸当前吸收了什么错误
        cirq.Z(a0), cirq.X(a0), cirq.Y(a0),  # Alice Ancilla 1
        cirq.Z(a1), cirq.X(a1), cirq.Y(a1),  # Alice Ancilla 2
        cirq.Z(a2), cirq.X(a2), cirq.Y(a2),  # Bob Ancilla

        # =========================================================
        # C. 实验室本地交叉关联 (Local Correlations) - 核心
        # =========================================================
        # --- Alice 实验室 (q0, q1, q3, q4, q6, q7) ---
        # 1. Target <-> Resource 互查 (Alice 内部的资源泵浦)
        cirq.PauliString(cirq.Z(t0), cirq.Z(r0)),
        cirq.PauliString(cirq.X(t0), cirq.X(r0)),
        cirq.PauliString(cirq.Z(t1), cirq.Z(r1)),
        cirq.PauliString(cirq.X(t1), cirq.X(r1)),
        cirq.PauliString(cirq.Z(t0), cirq.Z(r1)),  # 交叉校验
        cirq.PauliString(cirq.Z(t1), cirq.Z(r0)),  # 交叉校验

        # 2. Target <-> Ancilla (错误提取)
        cirq.PauliString(cirq.Z(t0), cirq.Z(a0)),
        cirq.PauliString(cirq.X(t0), cirq.X(a0)),
        cirq.PauliString(cirq.Z(t1), cirq.Z(a1)),
        cirq.PauliString(cirq.X(t1), cirq.X(a1)),

        # 3. Resource <-> Ancilla (资源态清洗)
        cirq.PauliString(cirq.Z(r0), cirq.Z(a0)),
        cirq.PauliString(cirq.Z(r1), cirq.Z(a1)),

        # 4. Target 内部关联 (Alice 独有的本地优势，监控目标比特对的退相干)
        cirq.PauliString(cirq.Z(t0), cirq.Z(t1)),
        cirq.PauliString(cirq.X(t0), cirq.X(t1)),

        # --- Bob 实验室 (q2, q5, q8) ---
        # Bob 只有 3 个比特，关联相对简单
        cirq.PauliString(cirq.Z(t2), cirq.Z(r2)),
        cirq.PauliString(cirq.X(t2), cirq.X(r2)),
        cirq.PauliString(cirq.Z(t2), cirq.Z(a2)),
        cirq.PauliString(cirq.X(t2), cirq.X(a2)),
        cirq.PauliString(cirq.Z(r2), cirq.Z(a2)),
    ]

    return ops

def get_default_observables_051810bit(qubits: List[cirq.LineQubit]) -> List[cirq.PauliString]:
    # =========================================================
    # 架构分配 (10-Qubit Asymmetric Star/Chain Topology: 0+1+23)
    # =========================================================
    # Alice   (2 qubits): [q0] (Target), [q4] (Resource)
    # Bob     (2 qubits): [q1] (Target), [q5] (Resource)
    # Charlie (6 qubits): [q2, q3] (Target), [q6, q7] (Resource), [q8, q9] (Ancilla)

    t_A, t_B, t_C1, t_C2 = qubits[0], qubits[1], qubits[2], qubits[3]  # Targets
    r_A, r_B, r_C1, r_C2 = qubits[4], qubits[5], qubits[6], qubits[7]  # Resources
    a_Z = qubits[8]  # Charlie Z-check ancilla (CNOT 模式抓 X 错)
    a_X = qubits[9]  # Charlie X-check ancilla (H-CZ-H 模式抓 Z 错)

    ops = [
        # =========================================================
        # A. 4-Qubit 全局健康度 (多体纠缠稳定子)
        # =========================================================
        # Target 4-qubit GHZ 稳定子
        cirq.PauliString(cirq.Z(t_A), cirq.Z(t_B)),
        cirq.PauliString(cirq.Z(t_B), cirq.Z(t_C1)),
        cirq.PauliString(cirq.Z(t_C1), cirq.Z(t_C2)),
        cirq.PauliString(cirq.X(t_A), cirq.X(t_B), cirq.X(t_C1), cirq.X(t_C2)), # 全局宇称

        # Resource 4-qubit GHZ 稳定子 (帮助 Agent 感知备份资源的质量)
        cirq.PauliString(cirq.Z(r_A), cirq.Z(r_B)),
        cirq.PauliString(cirq.Z(r_B), cirq.Z(r_C1)),
        cirq.PauliString(cirq.Z(r_C1), cirq.Z(r_C2)),
        cirq.PauliString(cirq.X(r_A), cirq.X(r_B), cirq.X(r_C1), cirq.X(r_C2)),

        # =========================================================
        # B. 辅助比特单体状态 (Charlie 拥有 q8, q9)
        # =========================================================
        cirq.Z(a_Z), cirq.X(a_Z), cirq.Y(a_Z),  # Z-check ancilla
        cirq.Z(a_X), cirq.X(a_X), cirq.Y(a_X),  # X-check ancilla

        # =========================================================
        # C. 实验室本地交叉关联 (Local Correlations) - RL 决策核心
        # =========================================================
        # --- Alice 本地 ---
        cirq.PauliString(cirq.Z(t_A), cirq.Z(r_A)),
        cirq.PauliString(cirq.X(t_A), cirq.X(r_A)),

        # --- Bob 本地 ---
        cirq.PauliString(cirq.Z(t_B), cirq.Z(r_B)),
        cirq.PauliString(cirq.X(t_B), cirq.X(r_B)),

        # --- Charlie 本地 (极其丰富，因为他拥有 6 个比特) ---
        # 1. Target <-> Resource (本地泵浦潜力)
        cirq.PauliString(cirq.Z(t_C1), cirq.Z(r_C1)),
        cirq.PauliString(cirq.X(t_C1), cirq.X(r_C1)),
        cirq.PauliString(cirq.Z(t_C2), cirq.Z(r_C2)),
        cirq.PauliString(cirq.X(t_C2), cirq.X(r_C2)),

        # 2. Target 内部 & Resource 内部关联
        cirq.PauliString(cirq.Z(t_C1), cirq.Z(t_C2)),
        cirq.PauliString(cirq.X(t_C1), cirq.X(t_C2)),
        cirq.PauliString(cirq.Z(r_C1), cirq.Z(r_C2)),
        cirq.PauliString(cirq.X(r_C1), cirq.X(r_C2)),

        # 3. Z-check ancilla (q8) 探针关联 (监控 X 错误提取)
        cirq.PauliString(cirq.Z(t_C1), cirq.Z(a_Z)),
        cirq.PauliString(cirq.X(t_C1), cirq.X(a_Z)),
        cirq.PauliString(cirq.Z(t_C2), cirq.Z(a_Z)),
        cirq.PauliString(cirq.X(t_C2), cirq.X(a_Z)),

        # 4. X-check ancilla (q9) 探针关联 (监控 Z 错误提取)
        cirq.PauliString(cirq.Z(t_C1), cirq.Z(a_X)),
        cirq.PauliString(cirq.X(t_C1), cirq.X(a_X)),
        cirq.PauliString(cirq.Z(t_C2), cirq.Z(a_X)),
        cirq.PauliString(cirq.X(t_C2), cirq.X(a_X)),

        # 5. 双 ancilla 交叉关联 (监控两个 ancilla 之间的信息流动)
        cirq.PauliString(cirq.Z(a_Z), cirq.Z(a_X)),
        cirq.PauliString(cirq.X(a_Z), cirq.X(a_X)),
    ]

    return ops

def get_default_observables(qubits: List[cirq.LineQubit]) -> List[cirq.PauliString]:
    # =========================================================
    # 架构分配 (9-Qubit Asymmetric Star/Chain Topology: 0+1+23)
    # =========================================================
    # Alice   (2 qubits): [q0] (Target), [q4] (Resource)
    # Bob     (2 qubits): [q1] (Target), [q5] (Resource)
    # Charlie (5 qubits): [q2, q3] (Target), [q6, q7] (Resource), [q8] (Ancilla)

    t_A, t_B, t_C1, t_C2 = qubits[0], qubits[1], qubits[2], qubits[3]  # Targets
    r_A, r_B, r_C1, r_C2 = qubits[4], qubits[5], qubits[6], qubits[7]  # Resources
    ancilla = qubits[8]  # Charlie 唯一的辅助比特 (轮流承担 Z-check 和 X-check)

    ops = [
        # =========================================================
        # A. 4-Qubit 全局健康度 (多体纠缠稳定子)
        # =========================================================
        # Target 4-qubit GHZ 稳定子
        cirq.PauliString(cirq.Z(t_A), cirq.Z(t_B)),
        cirq.PauliString(cirq.Z(t_B), cirq.Z(t_C1)),
        cirq.PauliString(cirq.Z(t_C1), cirq.Z(t_C2)),
        cirq.PauliString(cirq.X(t_A), cirq.X(t_B), cirq.X(t_C1), cirq.X(t_C2)), # 全局宇称

        # Resource 4-qubit GHZ 稳定子 (帮助 Agent 感知备份资源的质量)
        cirq.PauliString(cirq.Z(r_A), cirq.Z(r_B)),
        cirq.PauliString(cirq.Z(r_B), cirq.Z(r_C1)),
        cirq.PauliString(cirq.Z(r_C1), cirq.Z(r_C2)),
        cirq.PauliString(cirq.X(r_A), cirq.X(r_B), cirq.X(r_C1), cirq.X(r_C2)),

        # =========================================================
        # B. 辅助比特单体状态 (Charlie 拥有 q8)
        # =========================================================
        cirq.Z(ancilla), cirq.X(ancilla), cirq.Y(ancilla),

        # =========================================================
        # C. 实验室本地交叉关联 (Local Correlations) - RL 决策核心
        # =========================================================
        # --- Alice 本地 ---
        cirq.PauliString(cirq.Z(t_A), cirq.Z(r_A)),
        cirq.PauliString(cirq.X(t_A), cirq.X(r_A)),

        # --- Bob 本地 ---
        cirq.PauliString(cirq.Z(t_B), cirq.Z(r_B)),
        cirq.PauliString(cirq.X(t_B), cirq.X(r_B)),

        # --- Charlie 本地 (5 个比特的丰富关联) ---
        # 1. Target <-> Resource (本地泵浦潜力)
        cirq.PauliString(cirq.Z(t_C1), cirq.Z(r_C1)),
        cirq.PauliString(cirq.X(t_C1), cirq.X(r_C1)),
        cirq.PauliString(cirq.Z(t_C2), cirq.Z(r_C2)),
        cirq.PauliString(cirq.X(t_C2), cirq.X(r_C2)),

        # 2. Target 内部 & Resource 内部关联
        cirq.PauliString(cirq.Z(t_C1), cirq.Z(t_C2)),
        cirq.PauliString(cirq.X(t_C1), cirq.X(t_C2)),
        cirq.PauliString(cirq.Z(r_C1), cirq.Z(r_C2)),
        cirq.PauliString(cirq.X(r_C1), cirq.X(r_C2)),

        # 3. 辅助比特探针关联 (监控错误提取状态)
        cirq.PauliString(cirq.Z(t_C1), cirq.Z(ancilla)),
        cirq.PauliString(cirq.X(t_C1), cirq.X(ancilla)),
        cirq.PauliString(cirq.Z(t_C2), cirq.Z(ancilla)),
        cirq.PauliString(cirq.X(t_C2), cirq.X(ancilla)),
    ]

    return ops

def get_default_observables2(qubits: List[cirq.LineQubit]) -> List[cirq.GateOperation]:
    q0, q1, q2, q3, q4, q5 = qubits
    return [
        cirq.Z(q0)*cirq.Z(q1), cirq.X(q0)*cirq.X(q1), # Target 质量
        cirq.Z(q4)*cirq.Z(q5), cirq.X(q4)*cirq.X(q5), # Catalyst 质量
        cirq.Z(q0)*cirq.Z(q4), # Target-Catalyst 关联
        cirq.Z(q2)*cirq.Z(q4), # Resource-Catalyst 关联
    ]


def get_bell_state() -> np.ndarray:
    target = np.zeros(2**2, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    return target


def get_ghz_state(n_qubits: int = 3) -> np.ndarray:
    target = np.zeros(2**n_qubits, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    return target


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


def get_default_gates_purify_old(qubits):
    # qubits[0], qubits[2] 归 Alice; qubits[1], qubits[3] 归 Bob
    alice_q = [qubits[0], qubits[2]]
    bob_q = [qubits[1], qubits[3]]
    gates = []

    for q_set in [alice_q, bob_q]:
        # 单比特门
        for q in q_set:
            gates += [cirq.H(q), cirq.X(q), cirq.Y(q), cirq.Z(q),
                      cirq.rx(np.pi / 2)(q), cirq.ry(np.pi / 2)(q)]
        # 本地双比特门 (Alice 内部或 Bob 内部)
        gates.append(cirq.CNOT(q_set[0], q_set[1]))
        gates.append(cirq.CNOT(q_set[1], q_set[0]))
        gates.append(cirq.Circuit(cirq.CNOT(qubits[0], qubits[2]), cirq.CNOT(qubits[1], qubits[3])))
        gates.append(cirq.Circuit(cirq.CNOT(qubits[2], qubits[0]), cirq.CNOT(qubits[3], qubits[1])))

    gates.append("FINISH_ACTION")
    return gates


def get_default_gates2(qubits):
    # Alice(A): q0, q2, q4 | Bob(B): q1, q3, q5
    # Target: q0,q1 | Resource: q2,q3 | Catalyst: q4,q5
    A = [qubits[0], qubits[2], qubits[4]]
    B = [qubits[1], qubits[3], qubits[5]]
    gates = []

    # --- 1. 双边宏动作 (保持对称性，快速搜索基础序列) ---
    for gate_func in [cirq.H, cirq.rx(np.pi / 2), cirq.T]:
        for i in range(3):  # 对三对比特施加同步旋转
            gates.append(cirq.Circuit(gate_func(A[i]), gate_func(B[i])))

    # 双边 CNOT (Target-Resource, Resource-Catalyst, Target-Catalyst)
    pairs = [(0, 1, 2, 3), (2, 3, 4, 5), (0, 1, 4, 5)]
    for a_ctrl, b_ctrl, a_tar, b_tar in pairs:
        gates.append(cirq.Circuit(cirq.CNOT(qubits[a_ctrl], qubits[a_tar]),
                                  cirq.CNOT(qubits[b_ctrl], qubits[b_tar])))

    # --- 2. 增加单边/非对称独立操作 (处理非对称噪声或相位校正) ---
    for q in A + B:
        gates += [cirq.H(q), cirq.rx(np.pi / 2)(q), cirq.rz(np.pi / 4)(q)]

    # --- 3. 允许催化剂作为控制比特 (反馈/修正逻辑) ---
    # Alice 侧内部 CNOT: 催化剂(4) 控制 目标(0) 或 资源(2)
    gates.append(cirq.CNOT(qubits[4], qubits[0]))
    gates.append(cirq.CNOT(qubits[4], qubits[2]))
    # Bob 侧内部 CNOT: 催化剂(5) 控制 目标(1) 或 资源(3)
    gates.append(cirq.CNOT(qubits[5], qubits[1]))
    gates.append(cirq.CNOT(qubits[5], qubits[3]))

    # --- 4. 加入 SWAP 门 (模拟载体穿梭或状态置换) ---
    # Alice 侧本地交换
    #gates.append(cirq.SWAP(qubits[0], qubits[4]))  # 目标与催化剂交换
    #gates.append(cirq.SWAP(qubits[2], qubits[4]))  # 资源与催化剂交换
    # Bob 侧同步交换
    #gates.append(cirq.SWAP(qubits[1], qubits[5]))
    #gates.append(cirq.SWAP(qubits[3], qubits[5]))

    gates.append("FINISH_ACTION")
    return gates


def get_default_gates_p2(qubits):
    q0, q1, q2, q3, q4, q5 = qubits
    gates = []




    for q_pair in [(q0, q1,q2,), (q3,q4,q5),]:#(q4, q5)]:#(q0, q1), (q2, q3),
        for gate_func in [cirq.H, cirq.S,]: # cirq.rx(np.pi / 2), cirq.T,cirq.Z,cirq.X,cirq.Y]:  # 增加 S 门，对相位纠错很重要
            gates.append(cirq.Circuit(gate_func(q_pair[0]), gate_func(q_pair[1]), gate_func(q_pair[2])))

    # --- C. 双边相互作用 (CNOT) ---
    # 维持跨区的同步 CNOT，这是纯化所必须的
    gates.append(cirq.Circuit(cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5)))  # T-R 交互
    gates.append(cirq.Circuit(cirq.H.on_each(qubits),cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5), cirq.H(q0), cirq.H(q1), cirq.H(q2)))

    # --- D. 催化剂内部本地操作 (可选) ---
    # 允许 Alice 侧的目标和催化剂本地交互
    #gates.append(cirq.CNOT(q0, q3))
    #gates.append(cirq.CNOT(q1, q4))
    #gates.append(cirq.CNOT(q2, q5))
    # 3. 【新增建议】极致宏动作：全员双边基底转换 (P2 协议的神器)
    # 物理意义：一步进入 X 基底进行相位纯化
    gates.append(cirq.Circuit(cirq.H.on_each(qubits)))
    # 或者全员 S 门（针对某些复杂的相位错误）
    gates.append(cirq.Circuit(cirq.S.on_each(qubits)))

    gates.append("FINISH_ACTION")
    return gates

def get_default_gates_6bit(qubits):
    q0, q1, q2, q3, q4, q5 = qubits
    gates = []




    for q_cata in [q0,q1, q2,q3,q4, q5]:
        for gate_func in [cirq.H, cirq.T,cirq.Z,cirq.X,cirq.Y, cirq.S]:
            gates.append(gate_func(q_cata))
    #for q_cata in [q0,q1, q2,q3]:
    #    for gate_func in [cirq.H, cirq.rx(np.pi / 2), cirq.rz(np.pi / 4), cirq.T,cirq.Z,cirq.X,cirq.Y, cirq.S,cirq.ry(np.pi / 4),cirq.S**-1]:
    #        gates.append(gate_func(q_cata))

    # --- C. 双边相互作用 (CNOT) ---
    # 维持跨区的同步 CNOT，这是纯化所必须的
   # gates.append(cirq.Circuit(cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5)))  # T-R 交互
    #gates.append(cirq.Circuit(cirq.H.on_each(qubits),cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5), cirq.H(q0), cirq.H(q1), cirq.H(q2)))
    #gates.append(cirq.Circuit(cirq.CNOT(q0, q3), cirq.CNOT(q1, q3)))
    #gates.append(cirq.Circuit(cirq.CNOT(q2, q4), cirq.CNOT(q2, q5)))

    #gates.append(cirq.Circuit(cirq.CNOT(q3, q0), cirq.CNOT(q3, q1)))
    #gates.append(cirq.Circuit(cirq.CNOT(q4, q2), cirq.CNOT(q5, q2)))

    #gates.append(cirq.Circuit(cirq.CZ(q0, q3), cirq.CZ(q1, q3)))
    #gates.append(cirq.Circuit(cirq.CZ(q2, q4), cirq.CZ(q2, q5)))


    # --- D. 催化剂内部本地操作 (可选) ---
    # 允许 Alice 侧的目标和催化剂本地交互
    gates.append(cirq.CNOT(q0, q1))
    gates.append(cirq.CNOT(q0, q3))
    gates.append(cirq.CNOT(q1, q3))

    gates.append(cirq.CNOT(q2, q5))
    #gates.append(cirq.CNOT(q4, q5))
    #gates.append(cirq.CNOT(q2, q4))#liu
    #feifa
    gates.append(cirq.CNOT(q0, q4))
    gates.append(cirq.CNOT(q1, q4))
    gates.append(cirq.CNOT(q3, q4))

    #gates.append(cirq.CNOT(q0, q2))
    #gates.append(cirq.CNOT(q1, q2))
    #gates.append(cirq.CNOT(q3, q2))
    #gates.append(cirq.CNOT(q4, q2))

    #gates.append(cirq.CNOT(q2, q0))
    #gates.append(cirq.CNOT(q2, q1))
    #gates.append(cirq.CNOT(q2, q3))
    #gates.append(cirq.CNOT(q4, q5))#liu

    #gates.append(cirq.CNOT(q4, q5))
    #gates.append(cirq.CNOT(q2, q4))
    #gates.append(cirq.CNOT(q2, q4))
    #gates.append(cirq.CNOT(q4, q5))
    #gates.append(cirq.CNOT(q3, q4))

    gates.append(cirq.CNOT(q1, q0))
    gates.append(cirq.CNOT(q3, q0))
    gates.append(cirq.CNOT(q3, q1))

    #gates.append(cirq.CNOT(q4, q2))
    gates.append(cirq.CNOT(q5, q2))
    #gates.append(cirq.CNOT(q5, q4))#liu

    gates.append(cirq.CNOT(q4, q0))
    gates.append(cirq.CNOT(q4, q1))
    gates.append(cirq.CNOT(q4, q3))

    gates.append(cirq.Circuit(cirq.H.on_each(qubits)))
    gates.append(cirq.Circuit(cirq.H(q0), cirq.H(q1),cirq.H(q2)))
    # 或者全员 S 门（针对某些复杂的相位错误）
    gates.append(cirq.Circuit(cirq.S.on_each(qubits)))
    gates.append(cirq.Circuit(cirq.H(q3), cirq.H(q4), cirq.H(q5)))

    #gates.append(cirq.TOFFOLI(q0, q1, q3))
    #gates.append(cirq.TOFFOLI(q3, q1, q0))
    #gates.append(cirq.TOFFOLI(q3, q0, q1))
    #gates.append(cirq.TOFFOLI(q0, q1, q4))
    #gates.append(cirq.TOFFOLI(q4, q1, q0))
    #gates.append(cirq.TOFFOLI(q4, q0, q1))

    #gates.append(cirq.TOFFOLI(q2, q4, q5))
    #gates.append(cirq.TOFFOLI(q5, q4, q2))
    #gates.append(cirq.TOFFOLI(q2, q5, q4))

    #gates.append(cirq.CNOT(q1, q0))
    #gates.append(cirq.CNOT(q3, q0))
    #gates.append(cirq.CNOT(q3, q1))
    #gates.append(cirq.CNOT(q5, q2))
    #gates.append(cirq.CNOT(q4, q3))

    #gates.append(cirq.CNOT(q5, q4))

    gates.append("FINISH_ACTION")
    return gates

def get_default_gates_0420(qubits):
    q0, q1, q2, q3, q4, q5, q6, q7, q8 = qubits

    alice_qubits = [q0, q1, q3, q4, q6, q7]
    bob_qubits = [q2, q5, q8]

    gates = []




    for q_cata in [q0,q1, q2,q3,q4, q5,q6,q7,q8]:
        for gate_func in [cirq.H,cirq.Z,cirq.X, cirq.S]:
            gates.append(gate_func(q_cata))
    #for q_cata in [q0,q1, q2,q3]:
    #    for gate_func in [cirq.H, cirq.rx(np.pi / 2), cirq.rz(np.pi / 4), cirq.T,cirq.Z,cirq.X,cirq.Y, cirq.S,cirq.ry(np.pi / 4),cirq.S**-1]:
    #        gates.append(gate_func(q_cata))

    # --- C. 双边相互作用 (CNOT) ---
    # 维持跨区的同步 CNOT，这是纯化所必须的
   # gates.append(cirq.Circuit(cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5)))  # T-R 交互
    #gates.append(cirq.Circuit(cirq.H.on_each(qubits),cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5), cirq.H(q0), cirq.H(q1), cirq.H(q2)))
    #gates.append(cirq.Circuit(cirq.CNOT(q0, q3), cirq.CNOT(q1, q3)))
    #gates.append(cirq.Circuit(cirq.CNOT(q2, q4), cirq.CNOT(q2, q5)))

    #gates.append(cirq.Circuit(cirq.CNOT(q3, q0), cirq.CNOT(q3, q1)))
    #gates.append(cirq.Circuit(cirq.CNOT(q4, q2), cirq.CNOT(q5, q2)))

    #gates.append(cirq.Circuit(cirq.CZ(q0, q3), cirq.CZ(q1, q3)))
    #gates.append(cirq.Circuit(cirq.CZ(q2, q4), cirq.CZ(q2, q5)))



    # 生成 Alice 内部所有的 CNOT (排列组合，自动区分控制位和目标位)
    for control, target in itertools.permutations(alice_qubits, 2):
        gates.append(cirq.CNOT(control, target))

    # 生成 Bob 内部所有的 CNOT
    for control, target in itertools.permutations(bob_qubits, 2):
        gates.append(cirq.CNOT(control, target))

    # 3. 【新增建议】极致宏动作：全员双边基底转换 (P2 协议的神器)
    # 物理意义：一步进入 X 基底进行相位纯化
    gates.append(cirq.Circuit(cirq.H.on_each(qubits)))
    gates.append(cirq.Circuit(cirq.H(q0), cirq.H(q1),cirq.H(q2),cirq.H(q3), cirq.H(q4),cirq.H(q5)))
    # 或者全员 S 门（针对某些复杂的相位错误）
    gates.append(cirq.Circuit(cirq.S.on_each(qubits)))
    gates.append(cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.H(q2)))

    #gates.append(cirq.TOFFOLI(q0, q1, q3))
    #gates.append(cirq.TOFFOLI(q3, q1, q0))
    #gates.append(cirq.TOFFOLI(q3, q0, q1))
    #gates.append(cirq.TOFFOLI(q0, q1, q4))
    #gates.append(cirq.TOFFOLI(q4, q1, q0))
    #gates.append(cirq.TOFFOLI(q4, q0, q1))

    #gates.append(cirq.TOFFOLI(q2, q4, q5))
    #gates.append(cirq.TOFFOLI(q5, q4, q2))
    #gates.append(cirq.TOFFOLI(q2, q5, q4))

    #gates.append(cirq.CNOT(q1, q0))
    #gates.append(cirq.CNOT(q3, q0))
    #gates.append(cirq.CNOT(q3, q1))
    #gates.append(cirq.CNOT(q5, q2))
    #gates.append(cirq.CNOT(q4, q3))

    #gates.append(cirq.CNOT(q5, q4))

    #gates.append("FINISH_ACTION")
    return gates

def get_default_gates_false(qubits):
    """
    基于群分解理论 (H群 * B群 * S_Alice群 * S_Bob群) 的 1536 种保真复合动作空间。
    完美适配 9-qubit 非对称拓扑:
    - Target: q0, q1, q2
    - Resource: q3, q4, q5
    - Ancilla: q6, q7 (Alice), q8 (Bob)
    """
    q0, q1, q2, q3, q4, q5, q6, q7, q8 = qubits

    # 物理节点定义：每个节点持有一份 Target 和一份对应的 Resource
    # Alice 拥有节点 0(q0, q3) 和 1(q1, q4)；Bob 拥有节点 2(q2, q5)
    nodes = [(q0, q3), (q1, q4), (q2, q5)]

    gates = []

    # ==========================================
    # 1. 定义四个维度的基础物理群 (Templates)
    # ==========================================

    # --- A. H群 (全局同质群) - 6 种 ---
    # 负责 Target 和 Resource 之间的纠缠交换与 X 错误转移
    h_templates = [
        lambda t, r: [],  # Identity (无操作)
        lambda t, r: [cirq.SWAP(t, r)],
        lambda t, r: [cirq.CNOT(t, r)],  # Target 控 Resource
        lambda t, r: [cirq.CNOT(r, t)],  # Resource 控 Target
        lambda t, r: [cirq.CNOT(r, t), cirq.CNOT(t, r)],  # DCX 1
        lambda t, r: [cirq.CNOT(t, r), cirq.CNOT(r, t)]  # DCX 2
    ]

    # --- B. B群 (跨区双局域关联) - 8 种组合 ---
    # 负责建立 Alice (q0) 和 Bob (q2) 之间的 Z 基底纠错关联
    # 组合开关: (has_CZ_02, has_S_0, has_S_2)
    b_combos = list(itertools.product([0, 1], repeat=3))

    # --- C. Alice 本地 S群 (超级节点特权) - 8 种组合 ---
    # 利用非对称特权：q6 专职查 X 错误，q7 专职查 Z 错误，物理完全隔离防干涉
    # 组合开关: (用q6做X探测, 用q7做Z探测, 对q6加S门改变基底)
    sa_combos = list(itertools.product([0, 1], repeat=3))

    # --- D. Bob 本地 S群 (普通节点校验) - 4 种互斥组合 ---
    # Bob 只有一个辅助比特 q8，因此探测必须互斥
    # 0: 无操作, 1: 仅X探测, 2: 仅Z探测, 3: 仅对自己加S门调整相位
    sb_options = [0, 1, 2, 3]

    # ==========================================
    # 2. 笛卡尔积组装 1536 个宏动作 (Macro-Actions)
    # ==========================================
    # 总计: 6 (H) * 8 (B) * 8 (S_A) * 4 (S_B) = 1536
    for h_func in h_templates:
        for b_bits in b_combos:
            for sa_bits in sa_combos:
                for sb_val in sb_options:

                    # 每一组循环生成一个完整的复合量子电路
                    macro_circuit = cirq.Circuit()

                    # 【装载 H群】：全局同步施加，保证纠缠不断裂
                    for t, r in nodes:
                        sub_gates = h_func(t, r)
                        if sub_gates:
                            macro_circuit.append(sub_gates)

                    # 【装载 B群】：跨节点相位操作
                    if b_bits[0]: macro_circuit.append(cirq.CZ(q0, q2))
                    if b_bits[1]: macro_circuit.append(cirq.S(q0))
                    if b_bits[2]: macro_circuit.append(cirq.S(q2))

                    # 【装载 Alice S群】：双辅助比特并行探测
                    if sa_bits[0]:
                        # X-Check (位翻转探测)，绑定在 q6
                        macro_circuit.append([cirq.CNOT(q0, q6), cirq.CNOT(q1, q6)])
                    if sa_bits[1]:
                        # Z-Check (相位翻转探测)，绑定在 q7
                        macro_circuit.append([cirq.H(q7), cirq.CZ(q0, q7), cirq.CZ(q1, q7), cirq.H(q7)])
                    if sa_bits[2]:
                        # S门相位补偿
                        macro_circuit.append(cirq.S(q6))

                    # 【装载 Bob S群】：单辅助比特互斥探测
                    if sb_val == 1:
                        # X-Check
                        macro_circuit.append(cirq.CNOT(q2, q8))
                    elif sb_val == 2:
                        # Z-Check
                        macro_circuit.append([cirq.H(q8), cirq.CZ(q2, q8), cirq.H(q8)])
                    elif sb_val == 3:
                        # S门相位补偿
                        macro_circuit.append(cirq.S(q8))

                    # 将组装好的宏动作加入动作库
                    gates.append(macro_circuit)

    # ==========================================
    # 3. 收尾动作 (资源解耦与结算)
    # ==========================================
    # 将基底选择权交给 Agent，由 AI 自主决定最佳的解耦方式

    # 动作 1536: 以 P1 模式 (偶宇称) 结算，通常用于过滤相位噪声
    gates.append("FINISH_ACTION_P1")

    # 动作 1537: 以 P2 模式 (严格 000/111) 结算，通常用于过滤位翻转噪声
    gates.append("FINISH_ACTION_P2")

    return gates


def get_default_gates_0421_new(qubits):
    """
    生成 4628 维物理、代数、工程三重完备的 RL 动作空间。

    Qubits 物理映射 (9 qubits):
    - Node 1 (Alice): Target q0, Resource q3
    - Node 2 (Alice): Target q1, Resource q4
    - Node 3 (Bob):   Target q2, Resource q5
    - Ancilla (Alice): q6 (Target X-Check), q7 (Target Z-Check)
    - Ancilla Hack:    q8 (Resource Check) - 逻辑上由 Alice 接管，用于探测 q3, q4
    """
    q0, q1, q2 = qubits[0], qubits[1], qubits[2]
    q3, q4, q5 = qubits[3], qubits[4], qubits[5]
    q6, q7, q8 = qubits[6], qubits[7], qubits[8]

    gates = []

    # =====================================================================
    # 1. H 群 (Homogeneous Group) - 6 种
    # =====================================================================
    h_templates = [
        lambda: [],  # 1. Identity
        lambda: [cirq.SWAP(q0, q3), cirq.SWAP(q1, q4), cirq.SWAP(q2, q5)],  # 2. SWAP
        lambda: [cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5)],  # 3. CNOT_TR
        lambda: [cirq.CNOT(q3, q0), cirq.CNOT(q4, q1), cirq.CNOT(q5, q2)],  # 4. CNOT_RT
        lambda: [cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5),
                 cirq.CNOT(q3, q0), cirq.CNOT(q4, q1), cirq.CNOT(q5, q2)],  # 5. DCX_TR
        lambda: [cirq.CNOT(q3, q0), cirq.CNOT(q4, q1), cirq.CNOT(q5, q2),
                 cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5)]  # 6. DCX_RT
    ]

    # =====================================================================
    # 2. B12 群 (Node 1 & Node 2 双局域相位关联) - 8 种
    # =====================================================================
    b12_combos = list(itertools.product([0, 1], repeat=3))

    # =====================================================================
    # 3. B23 群 (Node 2 & Node 3 双局域相位关联) - 8 种
    # =====================================================================
    b23_combos = list(itertools.product([0, 1], repeat=3))

    # =====================================================================
    # 4. SA 群 (Alice 端 Target 探测) - 4 种
    # =====================================================================
    sa_combos = list(itertools.product([0, 1], repeat=2))

    # =====================================================================
    # 5. SB 群 (Hack: Alice 端 Resource 探测) - 3 种
    # 用 q8 对 Resource(q3, q4) 进行合法奇偶校验，彻底替换原有的 Bob 坍缩探测
    # =====================================================================
    sb_options = [0, 1, 2]  # 0: 无操作, 1: X-Check, 2: Z-Check

    # =====================================================================
    # 组装宏动作空间：6 * 8 * 8 * 4 * 3 = 4608 维
    # =====================================================================
    for h_func in h_templates:
        for b12 in b12_combos:
            for b23 in b23_combos:
                for sa_bits in sa_combos:
                    for sb_val in sb_options:
                        macro_circuit = cirq.Circuit()

                        # 挂载 H 群
                        macro_circuit.append(h_func())

                        # 挂载 B12 群
                        if b12[0]: macro_circuit.append([cirq.CZ(q0, q3), cirq.CZ(q1, q4)])
                        if b12[1]: macro_circuit.append([cirq.S(q0), cirq.S(q1)])
                        if b12[2]: macro_circuit.append([cirq.S(q3), cirq.S(q4)])

                        # 挂载 B23 群
                        if b23[0]: macro_circuit.append([cirq.CZ(q1, q4), cirq.CZ(q2, q5)])
                        if b23[1]: macro_circuit.append([cirq.S(q1), cirq.S(q2)])
                        if b23[2]: macro_circuit.append([cirq.S(q4), cirq.S(q5)])

                        # 挂载 SA: 探测 Target
                        if sa_bits[0]:  # X-Check
                            macro_circuit.append([cirq.CNOT(q0, q6), cirq.CNOT(q1, q6)])
                        if sa_bits[1]:  # Z-Check
                            macro_circuit.append([cirq.H(q7), cirq.CZ(q0, q7), cirq.CZ(q1, q7), cirq.H(q7)])

                        # 挂载 SB: 探测 Resource (由 Alice 异地调用 q8)
                        if sb_val == 1:  # Resource X-Check
                            macro_circuit.append([cirq.CNOT(q3, q8), cirq.CNOT(q4, q8)])
                        elif sb_val == 2:  # Resource Z-Check
                            macro_circuit.append([cirq.H(q8), cirq.CZ(q3, q8), cirq.CZ(q4, q8), cirq.H(q8)])

                        gates.append(macro_circuit)

    # =====================================================================
    # 6. 追加 Pauli 修正手术刀 (18 维)
    # 为全部 6 个主比特独立挂载 X, Y, Z
    # =====================================================================
    for q in [q0, q1, q2, q3, q4, q5]:
        gates.append(cirq.Circuit(cirq.X(q)))
        gates.append(cirq.Circuit(cirq.Y(q)))
        gates.append(cirq.Circuit(cirq.Z(q)))

    # =====================================================================
    # 7. 追加收网动作 (2 维)
    # =====================================================================
    gates.append("FINISH_ACTION_P1")
    gates.append("FINISH_ACTION_P2")

    return gates


def get_default_gatesHB(qubits):
    """
    生成 420 维极致解耦的 RL 动作空间 (适配 DensityMatrixSimulator)
    包含 Wang(2025) 对称群 + Ancilla 探测 + Alice 局域不对称特权 + Pauli 手术刀
    """
    q0, q1, q2 = qubits[0], qubits[1], qubits[2]
    q3, q4, q5 = qubits[3], qubits[4], qubits[5]
    q6, q7, q8 = qubits[6], qubits[7], qubits[8]

    gates = []

    # =====================================================================
    # 基础生成元定义
    # =====================================================================
    h_templates = [
        lambda: [],
        lambda: [cirq.SWAP(q0, q3), cirq.SWAP(q1, q4), cirq.SWAP(q2, q5)],
        lambda: [cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5)],
        lambda: [cirq.CNOT(q3, q0), cirq.CNOT(q4, q1), cirq.CNOT(q5, q2)],
        lambda: [cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5),
                 cirq.CNOT(q3, q0), cirq.CNOT(q4, q1), cirq.CNOT(q5, q2)],
        lambda: [cirq.CNOT(q3, q0), cirq.CNOT(q4, q1), cirq.CNOT(q5, q2),
                 cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5)]
    ]

    b12_combos = list(itertools.product([0, 1], repeat=3))
    b23_combos = list(itertools.product([0, 1], repeat=3))
    sa_combos = list(itertools.product([0, 1], repeat=2))
    sb_options = [0, 1, 2]

    # =====================================================================
    # 动作区 A: 纯宏观演化 (Action 0 ~ 383) -> 共 384 维
    # =====================================================================
    for h_func in h_templates:
        for b12 in b12_combos:
            for b23 in b23_combos:
                macro_circuit = cirq.Circuit()
                macro_circuit.append(h_func())

                if b12[0]: macro_circuit.append([cirq.CZ(q0, q3), cirq.CZ(q1, q4)])
                if b12[1]: macro_circuit.append([cirq.S(q0), cirq.S(q1)])
                if b12[2]: macro_circuit.append([cirq.S(q3), cirq.S(q4)])

                if b23[0]: macro_circuit.append([cirq.CZ(q1, q4), cirq.CZ(q2, q5)])
                if b23[1]: macro_circuit.append([cirq.S(q1), cirq.S(q2)])
                if b23[2]: macro_circuit.append([cirq.S(q4), cirq.S(q5)])

                gates.append(macro_circuit)

    # =====================================================================
    # 动作区 B: 标准 Ancilla 探测 (Action 384 ~ 395) -> 共 12 维
    # =====================================================================
    for sa_bits in sa_combos:
        for sb_val in sb_options:
            probe_circuit = cirq.Circuit()
            if sa_bits[0]: probe_circuit.append([cirq.CNOT(q0, q6), cirq.CNOT(q1, q6)])
            if sa_bits[1]: probe_circuit.append([cirq.H(q7), cirq.CZ(q0, q7), cirq.CZ(q1, q7), cirq.H(q7)])
            if sb_val == 1:
                probe_circuit.append([cirq.CNOT(q3, q8), cirq.CNOT(q4, q8)])
            elif sb_val == 2:
                probe_circuit.append([cirq.H(q8), cirq.CZ(q3, q8), cirq.CZ(q4, q8), cirq.H(q8)])
            gates.append(probe_circuit)

    # =====================================================================
    # ✨ 动作区 B+: Alice 局域高抗噪特权 (Action 396 ~ 399) -> 共 4 维 ✨
    # 将 Target (q0,q1) 的错误以极低代价硬挤压进 Resource (q3,q4)
    # =====================================================================
    alice_privilege_gates = [
        [cirq.CNOT(q0, q3), cirq.CNOT(q1, q3)],  # 特权1: X 错误提取到 q3
        [cirq.CNOT(q0, q4), cirq.CNOT(q1, q4)],  # 特权2: X 错误提取到 q4
        [cirq.H(q3), cirq.CZ(q0, q3), cirq.CZ(q1, q3), cirq.H(q3)],  # 特权3: Z 错误提取到 q3
        [cirq.H(q4), cirq.CZ(q0, q4), cirq.CZ(q1, q4), cirq.H(q4)]  # 特权4: Z 错误提取到 q4
    ]
    for priv_gate in alice_privilege_gates:
        gates.append(cirq.Circuit(priv_gate))

        # =====================================================================
        # 🚀 动作区 B++: Alice 全局联合校验 (Weight-4 Stabilizer Check) -> 共 6 维
        # 作用: 用极少的信息暴露度完成 4 比特体检。
        # 修正: Alice 拥有 q6, q7, q8 三张便签纸，全部加入调度池供 Agent 自由选择！
        # =====================================================================
        global_checks = []

        # 遍历 Alice 拥有的 3 个辅助比特
        for anc in [q6, q7, q8]:
            # 1. 全局 X 错误 (位翻转) 校验
            # 探测 q0, q1, q3, q4 中的奇数个 X 错误
            global_checks.append(
                [cirq.CNOT(q0, anc), cirq.CNOT(q1, anc), cirq.CNOT(q3, anc), cirq.CNOT(q4, anc)]
            )

            # 2. 全局 Z 错误 (相位翻转) 校验
            # 借助 H 门，探测 q0, q1, q3, q4 中的奇数个 Z 错误
            global_checks.append(
                [cirq.H(anc), cirq.CZ(q0, anc), cirq.CZ(q1, anc), cirq.CZ(q3, anc), cirq.CZ(q4, anc), cirq.H(anc)]
            )

        for g_check in global_checks:
            gates.append(cirq.Circuit(g_check))
    # =====================================================================
    # 动作区 C: Pauli 修正手术刀 (Action 400 ~ 417) -> 共 18 维
    # =====================================================================
    for q in [q0, q1, q2, q3, q4, q5]:
        gates.append(cirq.Circuit(cirq.X(q)))
        gates.append(cirq.Circuit(cirq.Y(q)))
        gates.append(cirq.Circuit(cirq.Z(q)))

    # =====================================================================
    # 动作区 D: 收网动作 (Action 418 ~ 419) -> 共 2 维
    # =====================================================================
    # gates.append("FINISH_ACTION_P1")
    # gates.append("FINISH_ACTION_P2")
    #gates.append("FINISH_P1_MASK_NONE")
    gates.append("FINISH_P1_MASK_ANC")
    gates.append("FINISH_P1_MASK_RES")
    gates.append("FINISH_P1_MASK_BOTH")
    #gates.append("FINISH_P2_MASK_NONE")
    gates.append("FINISH_P2_MASK_ANC")
    gates.append("FINISH_P2_MASK_RES")
    gates.append("FINISH_P2_MASK_BOTH")

    return gates


def get_default_gates_basic(qubits: List[cirq.LineQubit]) -> List[Any]:
    """
    终极原生基础量子门集合 (Primitive Gate Set)
    适用于 utils.py 中的独立函数调用。
    """
    gates = []

    # =====================================================================
    # 1. 单比特门 (剔除 cirq.I，只保留改变态的算符)
    # =====================================================================
    single_ops = [
        cirq.X, cirq.Y, cirq.Z,
        cirq.H,
        cirq.S,
        lambda q: cirq.S(q) ** -1  # S_dag
    ]

    # 直接遍历传入的 qubits 列表 (9 * 6 = 54 个动作)
    for q in qubits:
        for op in single_ops:
            gates.append(cirq.Circuit(op(q)))

    # =====================================================================
    # 2. 双比特纠缠门 - 严格的 LOCC 物理隔离墙
    # =====================================================================
    # Alice 实验室: 4 个数据位 + 2 个辅助位 (共 6 个比特)
    alice_qubits = [qubits[i] for i in [0, 1, 3, 4, 6, 7]]

    # Bob 实验室: 2 个数据位 + 1 个辅助位 (共 3 个比特)
    bob_qubits = [qubits[i] for i in [2, 5, 8]]

    # --- Alice 实验室内部操作 ---
    # CNOT 门 (排列: 6 取 2 = 30 个动作)
    for control, target in itertools.permutations(alice_qubits, 2):
        gates.append(cirq.Circuit(cirq.CNOT(control, target)))

    # CZ 门 (组合: 6 取 2 = 15 个动作)
    for q_a, q_b in itertools.combinations(alice_qubits, 2):
        gates.append(cirq.Circuit(cirq.CZ(q_a, q_b)))

    # --- Bob 实验室内部操作 ---
    # CNOT 门 (排列: 3 取 2 = 6 个动作)
    for control, target in itertools.permutations(bob_qubits, 2):
        gates.append(cirq.Circuit(cirq.CNOT(control, target)))

    # CZ 门 (组合: 3 取 2 = 3 个动作)
    for q_a, q_b in itertools.combinations(bob_qubits, 2):
        gates.append(cirq.Circuit(cirq.CZ(q_a, q_b)))

    # =====================================================================
    # 3. 收网结算动作 (FINISH)
    # =====================================================================
    finish_actions = [
        "FINISH_P1_MASK_ANC",
        "FINISH_P1_MASK_RES",
        "FINISH_P1_MASK_BOTH",
        "FINISH_P2_MASK_ANC",
        "FINISH_P2_MASK_RES",
        "FINISH_P2_MASK_BOTH"
    ]
    gates.extend(finish_actions)

    #print(f"🚀 Native Gate Set Built! Total Action Space Dimension: {len(gates)}")
    # 最终维度: 54 + 30 + 15 + 6 + 3 + 6 = 114 维
    return gates


def get_default_gates_0428(qubits: List[cirq.LineQubit]) -> List[Any]:
    """
    终极量子架构搜索 (Ultimate QAS) 动作集合 82 维版。
    已修复所有缩进嵌套错误，精确对齐最终物理版图。
    """
    q0, q1, q2 = qubits[0], qubits[1], qubits[2]  # Target -> Alice: 0,1 | Bob: 2
    q3, q4, q5 = qubits[3], qubits[4], qubits[5]  # Resource -> Alice: 3,4 | Bob: 5
    q6, q7, q8 = qubits[6], qubits[7], qubits[8]  # Ancilla -> Alice: 6,7 | Bob: 8

    gates = []

    # =====================================================================
    # 🚀 动作区 1: Wang 的 H 群 (双边纠缠操作) -> 共 7 维
    # =====================================================================
    h_group = [
        [],  # Wait/Identity
        [cirq.SWAP(q0, q3), cirq.SWAP(q1, q4), cirq.SWAP(q2, q5)],
        [cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5)],
        [cirq.CNOT(q3, q0), cirq.CNOT(q4, q1), cirq.CNOT(q5, q2)],
        [cirq.CZ(q0, q3), cirq.CZ(q1, q4), cirq.CZ(q2, q5)],
        [cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5),
         cirq.CNOT(q3, q0), cirq.CNOT(q4, q1), cirq.CNOT(q5, q2)],
        [cirq.CNOT(q3, q0), cirq.CNOT(q4, q1), cirq.CNOT(q5, q2),
         cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5)]
    ]
    for ops in h_group:
        if not ops:
            gates.append(cirq.Circuit(cirq.I(q0)))
        else:
            gates.append(cirq.Circuit(ops))

    # =====================================================================
    # 🚀 动作区 2: 宏观保对称操作 (Macro Symmetry Ops) -> 共 15 维
    # =====================================================================
    macro_symmetry_ops = [
        # --- 局域保对称 S 门 ---
        [cirq.S(q0), cirq.S(q1)],
        [cirq.S(q1), cirq.S(q2)],
        [cirq.S(q0), cirq.S(q2)],
        [cirq.S(q3), cirq.S(q4)],
        [cirq.S(q4), cirq.S(q5)],
        [cirq.S(q3), cirq.S(q5)],
        # --- 局域保对称 CZ 门 ---
        [cirq.CZ(q0, q3), cirq.CZ(q1, q4)],
        [cirq.CZ(q1, q4), cirq.CZ(q2, q5)],
        [cirq.CZ(q0, q3), cirq.CZ(q2, q5)],
        # --- 全局基底转换 ---
        [cirq.H(q0), cirq.H(q1), cirq.H(q2)],
        [cirq.H(q3), cirq.H(q4), cirq.H(q5)],
        # # --- 全局奇偶翻转 ---
        # [cirq.X(q0), cirq.X(q1), cirq.X(q2)],
        # [cirq.Z(q0), cirq.Z(q1), cirq.Z(q2)],
        # [cirq.X(q3), cirq.X(q4), cirq.X(q5)],
        # [cirq.Z(q3), cirq.Z(q4), cirq.Z(q5)]
    ]
    for ops in macro_symmetry_ops:
        gates.append(cirq.Circuit(ops))

    # =====================================================================
    # 🚀 动作区 3: 辅助比特探测池 (探针解耦) -> 共 12 维
    # =====================================================================
    probe_gates = []

    probe_gates.append([cirq.CNOT(q0, q6), cirq.CNOT(q1, q6)])
    probe_gates.append([cirq.H(q7), cirq.CZ(q0, q7), cirq.CZ(q1, q7), cirq.H(q7)])
    probe_gates.append([cirq.CNOT(q2, q8), cirq.CNOT(q5, q8)])
    probe_gates.append([cirq.H(q8), cirq.CZ(q2, q8), cirq.CZ(q5, q8), cirq.H(q8)])

    # alice_privilege = [
    #     [cirq.CNOT(q0, q3), cirq.CNOT(q1, q3)],
    #     [cirq.CNOT(q0, q4), cirq.CNOT(q1, q4)],
    #     [cirq.H(q3), cirq.CZ(q0, q3), cirq.CZ(q1, q3), cirq.H(q3)],
    #     [cirq.H(q4), cirq.CZ(q0, q4), cirq.CZ(q1, q4), cirq.H(q4)]
    # ]
    # probe_gates.extend(alice_privilege)

    for anc in [q6, q7]:
        probe_gates.append([cirq.CNOT(q0, anc), cirq.CNOT(q1, anc), cirq.CNOT(q3, anc), cirq.CNOT(q4, anc)])
        probe_gates.append(
            [cirq.H(anc), cirq.CZ(q0, anc), cirq.CZ(q1, anc), cirq.CZ(q3, anc), cirq.CZ(q4, anc), cirq.H(anc)])

    for ops in probe_gates:
        gates.append(cirq.Circuit(ops))

    # =====================================================================
    # 🚀 动作区 4: 独立的单比特门微操 -> 共 36 维cirq.Y
    # =====================================================================
    single_ops = [
        cirq.X,  cirq.Z,
        cirq.H,
        #cirq.S,
        #lambda q: cirq.S(q) ** -1
    ]

    for q in [q0, q1, q2, q3, q4, q5]:
        for op in single_ops:
            gates.append(cirq.Circuit(op(q)))


    # =====================================================================
    # 🚀 动作区 7: 容错纠缠探针 (Fault-Tolerant Bell Probes) ✨新增✨
    # 作用: 使用 Shor/Steane 容错范式，用 Bell 态联合提取奇偶性，防止交叉感染。
    # 流程: 制备 Bell 态 -> 交互 -> 解纠缠 -> 测量
    # =====================================================================
    ft_bell_probes = []

    # --- 1. Alice 容错探测 Target (q0, q1) 的 X 错 (位翻转) ---
    ft_bell_probes.append([
        # Step 1: 制备 q6, q7 的 Bell 态 |Φ+>
        cirq.H(q6), cirq.CNOT(q6, q7),
        # Step 2: 容错交互 (q6摸q0, q7摸q1，横向 CNOT)
        cirq.CNOT(q0, q6), cirq.CNOT(q1, q7),
        # Step 3: 解纠缠 (还原回计算基底，供 MASK_ANC 测量)
        cirq.CNOT(q6, q7), cirq.H(q6)
    ])

    # --- 2. Alice 容错探测 Target (q0, q1) 的 Z 错 (相位翻转) ---
    ft_bell_probes.append([
        # Step 1: 制备 Bell 态
        cirq.H(q6), cirq.CNOT(q6, q7),
        # Step 2: 容错交互 (横向 CZ 门)
        cirq.CZ(q0, q6), cirq.CZ(q1, q7),
        # Step 3: 解纠缠
        cirq.CNOT(q6, q7), cirq.H(q6)
    ])

    # --- 3. Alice 容错探测 Resource (q3, q4) 的 X 错 ---
    ft_bell_probes.append([
        cirq.H(q6), cirq.CNOT(q6, q7),
        cirq.CNOT(q3, q6), cirq.CNOT(q4, q7),
        cirq.CNOT(q6, q7), cirq.H(q6)
    ])

    # --- 4. Alice 容错探测 Resource (q3, q4) 的 Z 错 ---
    ft_bell_probes.append([
        cirq.H(q6), cirq.CNOT(q6, q7),
        cirq.CZ(q3, q6), cirq.CZ(q4, q7),
        cirq.CNOT(q6, q7), cirq.H(q6)
    ])

    for ops in ft_bell_probes:
        gates.append(cirq.Circuit(ops))
    # =====================================================================
    # 🚀 动作区 6: 动态收网结算动作 -> 共 6 维
    # =====================================================================
    finish_actions = [
        "FINISH_P1_MASK_ANC",
        "FINISH_P1_MASK_RES",
        "FINISH_P1_MASK_BOTH",
        "FINISH_P2_MASK_ANC",
        "FINISH_P2_MASK_RES",
        "FINISH_P2_MASK_BOTH"
    ]
    gates.extend(finish_actions)

    print(f"✅ Ultimate QAS Gates Finalized! Total Dimension: {len(gates)}")
    # 7 + 15 + 12 + 36 + 6 + 6 = 82 维
    return gates


def get_default_gates_3bit_0521(qubits: List[cirq.LineQubit]) -> List[Any]:
    """
    终极量子架构搜索 (Ultimate QAS) 宏动作合并版 - 120 维。
    采用 Wang 理论中的笛卡尔积 (H群 × B群)，大幅缩短 RL 决策步长。
    """
    q0, q1, q2 = qubits[0], qubits[1], qubits[2]  # Target -> Alice: 0,1 | Bob: 2
    q3, q4, q5 = qubits[3], qubits[4], qubits[5]  # Resource -> Alice: 3,4 | Bob: 5
    q6, q7, q8 = qubits[6], qubits[7], qubits[8]  # Ancilla -> Alice: 6,7 | Bob: 8

    gates = []

    # =====================================================================
    # 🚀 动作区 1+2: 宏观动作合并区 (H-Group × B-Group Cartesian Product)
    # 维度: 7 (H) × 12 (B + 空操作) = 84 维
    # =====================================================================
    h_base = [
        [],  # 0. H 组空操作
        [cirq.SWAP(q0, q3), cirq.SWAP(q1, q4), cirq.SWAP(q2, q5)],
        [cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5)],
        [cirq.CNOT(q3, q0), cirq.CNOT(q4, q1), cirq.CNOT(q5, q2)],
        #[cirq.CZ(q0, q3), cirq.CZ(q1, q4), cirq.CZ(q2, q5)],
        [cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5),
         cirq.CNOT(q3, q0), cirq.CNOT(q4, q1), cirq.CNOT(q5, q2)],
        [cirq.CNOT(q3, q0), cirq.CNOT(q4, q1), cirq.CNOT(q5, q2),
         cirq.CNOT(q0, q3), cirq.CNOT(q1, q4), cirq.CNOT(q2, q5)]
    ]

    b_base = [
        [],  # 0. B 组空操作
        # --- S 门保对称 ---
        [cirq.S(q0), cirq.S(q1)],
        [cirq.S(q1), cirq.S(q2)],
        [cirq.S(q0), cirq.S(q2)],
        [cirq.S(q3), cirq.S(q4)],
        [cirq.S(q4), cirq.S(q5)],
        [cirq.S(q3), cirq.S(q5)],
        # --- CZ 门保对称 ---
        [cirq.CZ(q0, q3), cirq.CZ(q1, q4)],
        [cirq.CZ(q1, q4), cirq.CZ(q2, q5)],
        [cirq.CZ(q0, q3), cirq.CZ(q2, q5)],
        # --- 全局基底切换 ---
        #[cirq.H(q0), cirq.H(q1), cirq.H(q2)],
        #[cirq.H(q3), cirq.H(q4), cirq.H(q5)]
    ]

    # 生成笛卡尔积
    for h_ops in h_base:
        for b_ops in b_base:
            macro_circuit = cirq.Circuit()
            combined_ops = h_ops + b_ops

            if not combined_ops:
                # 只有当 H 和 B 都为空时，才添加一个占位的 I 门
                macro_circuit.append(cirq.I(q0))
            else:
                macro_circuit.append(combined_ops)
            gates.append(macro_circuit)

    # =====================================================================
    # 🚀 动作区 3: 辅助比特探测池 (标准探针 + 全局体检) -> 共 9 维
    # =====================================================================
    probe_gates = []

    # 标准探测
    probe_gates.append([cirq.CNOT(q0, q6), cirq.CNOT(q1, q6)])
    probe_gates.append([cirq.H(q7), cirq.CZ(q0, q7), cirq.CZ(q1, q7), cirq.H(q7)])
    probe_gates.append([cirq.CNOT(q2, q8), cirq.CNOT(q5, q8)])
    probe_gates.append([cirq.H(q8), cirq.CZ(q2, q8), cirq.CZ(q5, q8), cirq.H(q8)])

    # Alice 全局体检
    for anc in [q6, q7]:
        probe_gates.append([cirq.CNOT(q0, anc), cirq.CNOT(q1, anc), cirq.CNOT(q3, anc), cirq.CNOT(q4, anc)])
        probe_gates.append(
            [cirq.H(anc), cirq.CZ(q0, anc), cirq.CZ(q1, anc), cirq.CZ(q3, anc), cirq.CZ(q4, anc), cirq.H(anc)])

    # 双 ancilla 并行 X+Z 探测 (q6 抓 X 错 + q7 抓 Z 错, Target + Resource)
    probe_gates.append([
        cirq.CNOT(q0, q6), cirq.CNOT(q1, q6), cirq.CNOT(q3, q6), cirq.CNOT(q4, q6),
        cirq.H(q7), cirq.CZ(q0, q7), cirq.CZ(q1, q7), cirq.CZ(q3, q7), cirq.CZ(q4, q7), cirq.H(q7),
    ])

    for ops in probe_gates:
        gates.append(cirq.Circuit(ops))


    # =====================================================================
    # 🚀 动作区 5: 独立的单比特门微操 (仅保留必要生成元) -> 共 18 维
    # =====================================================================
    # single_ops = [cirq.X, cirq.Z, ]#cirq.H]
    # for q in [q0, q1, q2, q3, q4, q5]:
    #     for op in single_ops:
    #         gates.append(cirq.Circuit(op(q)))

    # =====================================================================
    # 🚀 动作区 6: 动态收网结算动作 -> 共 6 维
    # =====================================================================
    finish_actions = [
        "FINISH_P1_MASK_ANC",
        "FINISH_P1_MASK_RES",
        "FINISH_P1_MASK_BOTH",
        "FINISH_P2_MASK_ANC",
        "FINISH_P2_MASK_RES",
        "FINISH_P2_MASK_BOTH"
    ]
    gates.extend(finish_actions)

    print(f"🔥 Cartesian QAS Gates Built! Total Dimension: {len(gates)}")
    # 验证维度: 84(宏组合) + 8(标准探针) + 4(容错探针) + 18(单比特) + 6(收网) = 120 维
    return gates


def get_default_gates_0518_10bit(qubits: List[cirq.LineQubit]) -> List[Any]:
    """
    4-qubit 拓扑的终极 QAS 宏动作合并版 (10-qubit 物理空间)。
    严格匹配 0+1+23 架构，Charlie 拥有 2 个辅助比特: q8 (Z-check) 和 q9 (X-check)。
    """
    # =========================================================
    # 🚀 1. 严格适配 10-qubit 的解包逻辑
    # =========================================================
    q0, q1, q2, q3 = qubits[0:4]  # Target (A:0, B:1, C:2,3)
    q4, q5, q6, q7 = qubits[4:8]  # Resource (A:4, B:5, C:6,7)
    q8 = qubits[8]  # Ancilla Z-check (Charlie, CNOT 模式抓 X 错)
    q9 = qubits[9]  # Ancilla X-check (Charlie, H-CZ-H 模式抓 Z 错)

    gates = []

    # =====================================================================
    # 🚀 动作区 1+2: H群 × B群 笛卡尔积
    # =====================================================================
    h_base = [
        [],  # 0. 空操作
        [cirq.SWAP(q0, q4), cirq.SWAP(q1, q5), cirq.SWAP(q2, q6), cirq.SWAP(q3, q7)],
        [cirq.CNOT(q0, q4), cirq.CNOT(q1, q5), cirq.CNOT(q2, q6), cirq.CNOT(q3, q7)],
        [cirq.CNOT(q4, q0), cirq.CNOT(q5, q1), cirq.CNOT(q6, q2), cirq.CNOT(q7, q3)],
        [cirq.CNOT(q0, q4), cirq.CNOT(q1, q5), cirq.CNOT(q2, q6), cirq.CNOT(q3, q7),
         cirq.CNOT(q4, q0), cirq.CNOT(q5, q1), cirq.CNOT(q6, q2), cirq.CNOT(q7, q3)],
        [cirq.CNOT(q4, q0), cirq.CNOT(q5, q1), cirq.CNOT(q6, q2), cirq.CNOT(q7, q3),
         cirq.CNOT(q0, q4), cirq.CNOT(q1, q5), cirq.CNOT(q2, q6), cirq.CNOT(q3, q7)]
    ]

    b_base = [
        [],  # 0. 空操作
        # --- 局域保对称 S 门 (抵消相位) ---
        [cirq.S(q0), cirq.S(q1)], [cirq.S(q0), cirq.S(q2)], [cirq.S(q0), cirq.S(q3)],
        [cirq.S(q1), cirq.S(q2)], [cirq.S(q1), cirq.S(q3)], [cirq.S(q2), cirq.S(q3)],
        [cirq.S(q4), cirq.S(q5)], [cirq.S(q4), cirq.S(q6)], [cirq.S(q4), cirq.S(q7)],
        [cirq.S(q5), cirq.S(q6)], [cirq.S(q5), cirq.S(q7)], [cirq.S(q6), cirq.S(q7)],
        # --- 跨边保对称 CZ 门 ---
        [cirq.CZ(q0, q4), cirq.CZ(q1, q5)],
        [cirq.CZ(q2, q6), cirq.CZ(q3, q7)],
        [cirq.CZ(q0, q4), cirq.CZ(q3, q7)]
    ]

    for h_ops in h_base:
        for b_ops in b_base:
            macro_circuit = cirq.Circuit()
            combined_ops = h_ops + b_ops
            if not combined_ops:
                macro_circuit.append(cirq.I(q0))
            else:
                macro_circuit.append(combined_ops)
            gates.append(macro_circuit)

    # =====================================================================
    # 🚀 动作区 3: 双 ancilla 局域探针 (q8=Z-check, q9=X-check)
    # =====================================================================
    probe_gates = []

    # --- q8 单 ancilla Z-check (抓 X 错 / 比特翻转) ---
    probe_gates.append([cirq.CNOT(q2, q8), cirq.CNOT(q3, q8)])  # Target 校验
    probe_gates.append([cirq.CNOT(q2, q8), cirq.CNOT(q3, q8), cirq.CNOT(q6, q8), cirq.CNOT(q7, q8)])  # 全局校验

    # --- q9 单 ancilla X-check (抓 Z 错 / 相位翻转) ---
    probe_gates.append([cirq.H(q9), cirq.CZ(q2, q9), cirq.CZ(q3, q9), cirq.H(q9)])  # Target 校验
    probe_gates.append([cirq.H(q9), cirq.CZ(q2, q9), cirq.CZ(q3, q9), cirq.CZ(q6, q9), cirq.CZ(q7, q9), cirq.H(q9)])  # 全局校验

    # --- 双 ancilla 并行探针 (同时抓 X+Z 错，组合成一个宏动作) ---
    probe_gates.append([
        cirq.CNOT(q2, q8), cirq.CNOT(q3, q8),  # q8 Z-check Target
        cirq.H(q9), cirq.CZ(q2, q9), cirq.CZ(q3, q9), cirq.H(q9)  # q9 X-check Target
    ])
    probe_gates.append([
        cirq.CNOT(q2, q8), cirq.CNOT(q3, q8), cirq.CNOT(q6, q8), cirq.CNOT(q7, q8),  # q8 Z-check Global
        cirq.H(q9), cirq.CZ(q2, q9), cirq.CZ(q3, q9), cirq.CZ(q6, q9), cirq.CZ(q7, q9), cirq.H(q9)  # q9 X-check Global
    ])

    for ops in probe_gates:
        gates.append(cirq.Circuit(ops))

    # =====================================================================
    # 🚀 动作区 4: 单比特门微操 (作用于 8 个数据比特 + 2 个 ancilla)
    # =====================================================================
    single_ops = [cirq.X, cirq.Z, cirq.H]
    for q in [q0, q1, q2, q3, q4, q5, q6, q7]:
        for op in single_ops:
            gates.append(cirq.Circuit(op(q)))

    # =====================================================================
    # 🚀 动作区 5: 动态收网结算动作
    # =====================================================================
    finish_actions = [
        "FINISH_P1_MASK_ANC",
        "FINISH_P1_MASK_RES",
        "FINISH_P1_MASK_BOTH",
        "FINISH_P2_MASK_ANC",
        "FINISH_P2_MASK_RES",
        "FINISH_P2_MASK_BOTH"
    ]
    gates.extend(finish_actions)

    return gates

def get_default_gates_13_two_ancillas(qubits: List[cirq.LineQubit]) -> List[Any]:
    """Action space for a fixed 1|3 GHZ4 topology with two remote ancillas.

    Qubits q8 and q9 are co-located with logical qubits 1, 2, and 3.  The
    probe actions measure the two independent even-parity stabilizers Z1Z2
    and Z2Z3, either separately or in one parallel macro action.
    """
    if len(qubits) != 10:
        raise ValueError("The 1|3 two-ancilla action space requires 10 qubits.")

    q0, q1, q2, q3 = qubits[0:4]
    q4, q5, q6, q7 = qubits[4:8]
    q8, q9 = qubits[8:10]

    gates = [
        "SELECT_TOPO_13",
        "SELECT_TOPO_112",
        "SELECT_TOPO_1111",
        "SELECT_TOPO_22",
    ]

    h_base = [
        [],
        [cirq.SWAP(q0, q4), cirq.SWAP(q1, q5), cirq.SWAP(q2, q6), cirq.SWAP(q3, q7)],
        [cirq.CNOT(q0, q4), cirq.CNOT(q1, q5), cirq.CNOT(q2, q6), cirq.CNOT(q3, q7)],
        [cirq.CNOT(q4, q0), cirq.CNOT(q5, q1), cirq.CNOT(q6, q2), cirq.CNOT(q7, q3)],
        [
            cirq.CNOT(q0, q4), cirq.CNOT(q1, q5), cirq.CNOT(q2, q6), cirq.CNOT(q3, q7),
            cirq.CNOT(q4, q0), cirq.CNOT(q5, q1), cirq.CNOT(q6, q2), cirq.CNOT(q7, q3),
        ],
        [
            cirq.CNOT(q4, q0), cirq.CNOT(q5, q1), cirq.CNOT(q6, q2), cirq.CNOT(q7, q3),
            cirq.CNOT(q0, q4), cirq.CNOT(q1, q5), cirq.CNOT(q2, q6), cirq.CNOT(q3, q7),
        ],
    ]
    b_base = [
        [],
        [cirq.S(q0), cirq.S(q1)], [cirq.S(q0), cirq.S(q2)], [cirq.S(q0), cirq.S(q3)],
        [cirq.S(q1), cirq.S(q2)], [cirq.S(q1), cirq.S(q3)], [cirq.S(q2), cirq.S(q3)],
        [cirq.S(q4), cirq.S(q5)], [cirq.S(q4), cirq.S(q6)], [cirq.S(q4), cirq.S(q7)],
        [cirq.S(q5), cirq.S(q6)], [cirq.S(q5), cirq.S(q7)], [cirq.S(q6), cirq.S(q7)],
        [cirq.CZ(q0, q4), cirq.CZ(q1, q5)],
        [cirq.CZ(q2, q6), cirq.CZ(q3, q7)],
        [cirq.CZ(q0, q4), cirq.CZ(q3, q7)],
    ]
    for h_ops in h_base:
        for b_ops in b_base:
            gates.append(cirq.Circuit(h_ops + b_ops or [cirq.I(q0)]))

    target_qubits = [q0, q1, q2, q3]
    resource_qubits = [q4, q5, q6, q7]
    for i, j in itertools.combinations(range(4), 2):
        ti, tj = target_qubits[i], target_qubits[j]
        ri, rj = resource_qubits[i], resource_qubits[j]
        gates.append(cirq.Circuit(cirq.CZ(ti, tj), cirq.CZ(ri, rj)))
        gates.append(cirq.Circuit(cirq.CNOT(ti, tj), cirq.CNOT(ri, rj)))
        gates.append(cirq.Circuit(cirq.CNOT(tj, ti), cirq.CNOT(rj, ri)))

    # Separate syndrome checks requested for the 1|3 remote three-qubit node.
    gates.append(cirq.Circuit(cirq.CNOT(q1, q8), cirq.CNOT(q2, q8)))
    gates.append(cirq.Circuit(cirq.CNOT(q2, q9), cirq.CNOT(q3, q9)))

    # Simultaneous measurement of Z1Z2 with q8 and Z2Z3 with q9.
    gates.append(cirq.Circuit(
        cirq.CNOT(q1, q8), cirq.CNOT(q2, q8),
        cirq.CNOT(q2, q9), cirq.CNOT(q3, q9),
    ))

    for q in [q0, q1, q2, q3, q4, q5, q6, q7]:
        for op in [cirq.X, cirq.Z, cirq.H]:
            gates.append(cirq.Circuit(op(q)))

    gates.extend([
        "FINISH_MASK_ANC",
        "FINISH_P1_MASK_RES",
        "FINISH_P1_MASK_BOTH",
        "FINISH_P2_MASK_RES",
        "FINISH_P2_MASK_BOTH",
    ])
    return gates


def get_default_gates(qubits: List[cirq.LineQubit]) -> List[Any]:
    """
    4-qubit 拓扑的终极 QAS 宏动作合并版 (9-qubit 物理空间)。
    第一版动态拓扑使用固定动作空间：episode 第一步先选择分布方式，
    之后环境根据所选分布惩罚不合法的非局域动作。
    """
    # =========================================================
    # 🚀 1. 严格适配 9-qubit 的解包逻辑
    # =========================================================
    q0, q1, q2, q3 = qubits[0:4]  # Target (A:0, B:1, C:2,3)
    q4, q5, q6, q7 = qubits[4:8]  # Resource (A:4, B:5, C:6,7)
    q8 = qubits[8]  # 唯一的 Ancilla (Charlie)

    gates = [
        "SELECT_TOPO_13",
        "SELECT_TOPO_112",
        "SELECT_TOPO_1111",
        "SELECT_TOPO_22",
    ]

    # =====================================================================
    # 🚀 动作区 1+2: H群 × B群 笛卡尔积 (操作数据比特，保持不变)
    # =====================================================================
    h_base = [
        [],  # 0. 空操作
        [cirq.SWAP(q0, q4), cirq.SWAP(q1, q5), cirq.SWAP(q2, q6), cirq.SWAP(q3, q7)],
        [cirq.CNOT(q0, q4), cirq.CNOT(q1, q5), cirq.CNOT(q2, q6), cirq.CNOT(q3, q7)],
        [cirq.CNOT(q4, q0), cirq.CNOT(q5, q1), cirq.CNOT(q6, q2), cirq.CNOT(q7, q3)],
        [cirq.CNOT(q0, q4), cirq.CNOT(q1, q5), cirq.CNOT(q2, q6), cirq.CNOT(q3, q7),
         cirq.CNOT(q4, q0), cirq.CNOT(q5, q1), cirq.CNOT(q6, q2), cirq.CNOT(q7, q3)],
        [cirq.CNOT(q4, q0), cirq.CNOT(q5, q1), cirq.CNOT(q6, q2), cirq.CNOT(q7, q3),
         cirq.CNOT(q0, q4), cirq.CNOT(q1, q5), cirq.CNOT(q2, q6), cirq.CNOT(q3, q7)]
    ]

    b_base = [
        [],  # 0. 空操作
        # --- 局域保对称 S 门 (抵消相位) ---
        [cirq.S(q0), cirq.S(q1)], [cirq.S(q0), cirq.S(q2)], [cirq.S(q0), cirq.S(q3)],
        [cirq.S(q1), cirq.S(q2)], [cirq.S(q1), cirq.S(q3)], [cirq.S(q2), cirq.S(q3)],
        [cirq.S(q4), cirq.S(q5)], [cirq.S(q4), cirq.S(q6)], [cirq.S(q4), cirq.S(q7)],
        [cirq.S(q5), cirq.S(q6)], [cirq.S(q5), cirq.S(q7)], [cirq.S(q6), cirq.S(q7)],
        # --- 跨边保对称 CZ 门 ---
        [cirq.CZ(q0, q4), cirq.CZ(q1, q5)],
        [cirq.CZ(q2, q6), cirq.CZ(q3, q7)],
        [cirq.CZ(q0, q4), cirq.CZ(q3, q7)]
    ]

    for h_ops in h_base:
        for b_ops in b_base:
            macro_circuit = cirq.Circuit()
            combined_ops = h_ops + b_ops
            if not combined_ops:
                macro_circuit.append(cirq.I(q0))
            else:
                macro_circuit.append(combined_ops)
            gates.append(macro_circuit)

    # Candidate intra-lab pair couplings. The environment decides which
    # logical pairs are legal after the topology action is selected.
    target_qubits = [q0, q1, q2, q3]
    resource_qubits = [q4, q5, q6, q7]
    for i, j in itertools.combinations(range(4), 2):
        ti, tj = target_qubits[i], target_qubits[j]
        ri, rj = resource_qubits[i], resource_qubits[j]
        gates.append(cirq.Circuit(cirq.CZ(ti, tj), cirq.CZ(ri, rj)))
        gates.append(cirq.Circuit(cirq.CNOT(ti, tj), cirq.CNOT(ri, rj)))
        gates.append(cirq.Circuit(cirq.CNOT(tj, ti), cirq.CNOT(rj, ri)))

    # =====================================================================
    # 🚀 动作区 3: 单 ancilla 局域探针 (q8 轮流担任 Z-check 和 X-check)
    # =====================================================================
    probe_gates = []

    # --- Z-check 模式 (抓 X 错 / 比特翻转) ---
    probe_gates.append([cirq.CNOT(q2, q8), cirq.CNOT(q3, q8)])  # Target 校验
    probe_gates.append([cirq.CNOT(q2, q8), cirq.CNOT(q3, q8), cirq.CNOT(q6, q8), cirq.CNOT(q7, q8)])  # 全局校验

    # --- X-check 模式 (抓 Z 错 / 相位翻转，全部改由 q8 承担) ---
    probe_gates.append([cirq.H(q8), cirq.CZ(q2, q8), cirq.CZ(q3, q8), cirq.H(q8)])  # Target 校验
    probe_gates.append([cirq.H(q8), cirq.CZ(q2, q8), cirq.CZ(q3, q8), cirq.CZ(q6, q8), cirq.CZ(q7, q8), cirq.H(q8)])  # 全局校验

    # 注意：删除了原有的“双 ancilla 并行探针”宏动作，因为现在只有一个物理辅助比特。
    # Agent 必须学会串行调度，例如：在这个 step 用 q8 做 Z-check，下个 step 重新利用 q8 做 X-check。

    for ops in probe_gates:
        gates.append(cirq.Circuit(ops))

    # =====================================================================
    # 🚀 动作区 4: 单比特门微操 (作用于 8 个数据比特)
    # =====================================================================
    single_ops = [cirq.X, cirq.Z, cirq.H]
    for q in [q0, q1, q2, q3, q4, q5, q6, q7]:
        for op in single_ops:
            gates.append(cirq.Circuit(op(q)))

    # =====================================================================
    # 🚀 动作区 5: 动态收网结算动作
    # =====================================================================
    finish_actions = [
        "FINISH_MASK_ANC",
        "FINISH_P1_MASK_RES",
        "FINISH_P1_MASK_BOTH",
        "FINISH_P2_MASK_RES",
        "FINISH_P2_MASK_BOTH"
    ]
    gates.extend(finish_actions)

    return gates

def get_default_gates22(qubits):
    q0, q1, q2, q3, q4, q5 = qubits
    gates = []

    # 动作 0: 双边 H
    gates.append(cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.H(q2), cirq.H(q3)))
    # 动作 1: 双边 S (修正相干的关键)
    gates.append(cirq.Circuit(cirq.S(q0), cirq.S(q1), cirq.S(q2), cirq.S(q3)))
    # 动作 2: 双边 T (更精细的修正)
    gates.append(cirq.Circuit(cirq.T(q0), cirq.T(q1), cirq.T(q2), cirq.T(q3)))
    # 动作 3: 经典 CNOT
    gates.append(cirq.Circuit(cirq.CNOT(q0, q2), cirq.CNOT(q1, q3)))
    # 动作 4: 终结
    gates.append("FINISH_ACTION")

    return gates
def get_default_gates2(qubits):
    q0, q1, q2, q3, q4, q5 = qubits
    gates = []

    # ==================================================
    # 策略：强制双边对称 (Force Bilateral Symmetry)
    # 这将大幅缩小搜索空间，并防止 Agent 破坏纠缠结构
    # ==================================================

    # 定义基础单比特门集合
    # 建议先去掉 T 门和 Rz，先用简单的 H, Rx, Ry, Z 跑通，再加复杂的
    base_single_gates = [
        cirq.H,
        cirq.rx(np.pi / 2),
        #cirq.Z,
        #cirq.X,
        #cirq.T, # 初期可以先注释掉 T 门，它会显著增加搜索难度
        # cirq.rz(np.pi / 4)
    ]

    # --- A. 所有对的双边单比特操作 ---
    # 包括：Target(0,1), Resource(2,3), Catalyst(4,5)
    # 强制 Alice 和 Bob 做一样的旋转
    pairs = [(q0, q1), (q2, q3), ]#(q4, q5)]

    for gate_func in base_single_gates:
        for qa, qb in pairs:
            # 这是一个动作：Alice 和 Bob 同时对各自的比特做 gate_func
            gates.append(cirq.Circuit(gate_func(qa), gate_func(qb)))

    # --- B. 跨系统的双边 CNOT (关键交互) ---
    # 逻辑：Alice 和 Bob 必须同步执行 CNOT

    # 1. Standard DEJMPS (Target <-> Resource)
    # Target 控 Resource
    gates.append(cirq.Circuit(cirq.CNOT(q0, q2), cirq.CNOT(q1, q3)))
    # Resource 控 Target (有时需要反向)
    #gates.append(cirq.Circuit(cirq.CNOT(q2, q0), cirq.CNOT(q3, q1)))

    # 2. Catalysis Interaction (Target <-> Catalyst)
    # [修复点]：必须包含双向交互

    # Catalyst 控 Target (把 Catalyst 的信息传给 Target)
    gates.append(cirq.Circuit(cirq.CNOT(q4, q0), cirq.CNOT(q5, q1)))

    # [新增] Target 控 Catalyst (利用相位反冲或反向传播)
    gates.append(cirq.Circuit(cirq.CNOT(q0, q4), cirq.CNOT(q1, q5)))

    gates.append(cirq.Circuit(cirq.CZ(q4, q0), cirq.CZ(q5, q1)))
    gates.append(cirq.Circuit(cirq.CZ(q0, q4), cirq.CZ(q1, q5)))

    # 3. (可选) Resource <-> Catalyst
    # 如果允许 Catalyst 也可以和 Resource 交互（有些协议允许）
    #gates.append(cirq.Circuit(cirq.CNOT(q4, q2), cirq.CNOT(q5, q3)))
    #催化剂单边操作
    for q_cata in [q4, q5]:
        for gate_func in [cirq.H, cirq.rx(np.pi / 2), cirq.rz(np.pi / 16), cirq.T,cirq.Z,cirq.X,cirq.Y,]:
            gates.append(gate_func(q_cata))

    # --- C. 结束动作 ---
    gates.append("FINISH_ACTION")

    #print(f"Action Space Size: {len(gates)}")  # 打印出来看看，最好控制在 20 以内
    return gates

class FGate(cirq.Gate):
    """
    Cirq custom gate for the F_gate.
    """

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.theta = np.arccos(np.sqrt(1 / (n - k + 1)))

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        q0, q1 = qubits
        yield cirq.ry(-self.theta)(q1)
        yield cirq.CZ(q0, q1)
        yield cirq.ry(self.theta)(q1)

    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo(
            wire_symbols=(f"F*CTR(k={self.k})", f"F⊕TAR(k={self.k})")
        )

    def __str__(self):
        return f"FGate(n={self.n}, k={self.k})"


class CXRVGate(cirq.Gate):
    """
    Cirq custom gate for the cxrv gate (reverse CNOT).
    """

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        q0, q1 = qubits
        yield cirq.H(q0)
        yield cirq.H(q1)
        yield cirq.CNOT(q1, q0)
        yield cirq.H(q0)
        yield cirq.H(q1)

    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo(
            wire_symbols=("CXRV_TAR⊕", "CXRV_CTR*")  # 指定 q0 为目标比特，q1 为控制比特
        )

    def __str__(self):
        return "CXRVGate"
