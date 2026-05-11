"""
QHDALabs-qmnet : Sieć relacji kwantowych z formalizmem Page-Woottersa.
Symuluje scramblery, mosty CZ i relacyjny czas w Qiskit.

Autor: Krzysztof Banasiewicz (krzyshtof.com)
Licencja: krzyshtof.com/license
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict
from collections import Counter

import numpy as np
from scipy.linalg import eig, expm
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import CZGate, CCZGate, CCXGate
from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler import CouplingMap

# Optional: AER for execution tests
try:
    from qiskit_aer import AerSimulator  # type: ignore
    _HAS_AER = True
except Exception:
    AerSimulator = None  # type: ignore
    Sampler = None
    _HAS_AER = False


# ------------------------------------------------------------
# 0) Ustawienia: topologia (linia 5 kubitów) i ulgi transpilerowe
# ------------------------------------------------------------
LINE_COUPLING = CouplingMap([(0, 1), (1, 2), (2, 3), (3, 4)])
BASIS_GATES = ["rz", "sx", "x", "cx", "cz"]


def transpile_line(qc_meas: QuantumCircuit, backend=None, optimization_level: int = 2) -> QuantumCircuit:
    return transpile(
        qc_meas,
        backend=backend,
        coupling_map=LINE_COUPLING if backend is None else None,
        basis_gates=BASIS_GATES if backend is None else None,
        optimization_level=optimization_level,
    )


# ------------------------------------------------------------
# 1) Przygotowanie stanów |+> oraz graf-state (krawędzie CZ)
# ------------------------------------------------------------
def prep_plus(qc: QuantumCircuit, qubits: Iterable[int]) -> None:
    for qubit in qubits:
        qc.h(qubit)


def graph_state_circuit(n_qubits: int, edges: List[Tuple[int, int]]) -> QuantumCircuit:
    """Stan grafowy: |+>^{⊗n} + CZ na każdej krawędzi."""
    q = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(q, name="graph_state")
    prep_plus(qc, range(n_qubits))
    for (u, v) in edges:
        qc.cz(q[u], q[v])
    return qc


def stabilizers_for_graph(n_qubits: int, edges: List[Tuple[int, int]]) -> List[SparsePauliOp]:
    """Zwraca stabilizatory graf-state: K_v = X_v ⊗_{u∈N(v)} Z_u."""
    neighbors: Dict[int, set] = {i: set() for i in range(n_qubits)}
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    ops: List[SparsePauliOp] = []
    for v in range(n_qubits):
        pauli = ["I"] * n_qubits
        pauli[v] = "X"
        for u in neighbors[v]:
            if pauli[u] == "I":
                pauli[u] = "Z"
        # Qiskit: qubit 0 to najmłodszy bit => odwracamy etykietę
        label = "".join(reversed(pauli))
        ops.append(SparsePauliOp.from_list([(label, 1.0)]))
    return ops


# ------------------------------------------------------------
# 2) „Most” CZ i brickwork po linii
# ------------------------------------------------------------
def cz_bridge(qc: QuantumCircuit, a, b) -> None:
    qc.append(CZGate(), [a, b])


def brickwork_edges_line(n_qubits: int, parity: int) -> List[Tuple[int, int]]:
    """
    Brickwork na linii:
      parity=0 -> (0,1),(2,3)...
      parity=1 -> (1,2),(3,4)...
    """
    edges = []
    start = 0 if parity == 0 else 1
    for i in range(start, n_qubits - 1, 2):
        edges.append((i, i + 1))
    return edges


# ------------------------------------------------------------
# 3) Scrambler U(T) oparty o CZ + H_all
# ------------------------------------------------------------
def apply_scrambler_layer(qc: QuantumCircuit, n_qubits: int, edges: List[Tuple[int, int]]) -> None:
    """Jedna warstwa scramblera: H na wszystkich + CZ na brickwork krawędziach."""
    for i in range(n_qubits):
        qc.h(i)
    for u, v in edges:
        qc.cz(u, v)


def build_scrambler_U(n_qubits: int, T: int, topology: str = "line") -> QuantumCircuit:
    """Buduje unitarny scrambler U(T) jako T warstw (H_all + CZ_brickwork)."""
    q = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(q, name=f"U_CZ_H_T{T}")
    for step in range(T):
        if topology != "line":
            raise ValueError("Wspieram tylko topology='line'.")
        edges = brickwork_edges_line(n_qubits, parity=step % 2)
        apply_scrambler_layer(qc, n_qubits, edges)
    return qc


def build_scrambler_U_dagger(n_qubits: int, T: int, topology: str = "line") -> QuantumCircuit:
    """U† = warstwy w odwrotnej kolejności (H i CZ są samoodwracalne)."""
    q = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(q, name=f"Udagger_CZ_H_T{T}")
    for step in reversed(range(T)):
        if topology != "line":
            raise ValueError("Wspieram tylko topology='line'.")
        edges = brickwork_edges_line(n_qubits, parity=step % 2)
        # inverse of layer: CZ then H_all
        for u, v in reversed(edges):
            qc.cz(u, v)
        for i in range(n_qubits):
            qc.h(i)
    return qc


# ------------------------------------------------------------
# 4) Echo eksperyment (Loschmidt echo) — metryka stabilizatorowa
# ------------------------------------------------------------
@dataclass
class EchoResult:
    T: int
    Kvals: List[float]


def build_echo_circuit(
    n_qubits: int,
    graph_edges: List[Tuple[int, int]],
    T: int,
    W_qubit: int = 0,
    W_op: str = "z",
    topology: str = "line",
) -> QuantumCircuit:
    """
    |G> -> U(T) -> W -> U†(T)
    Bez pomiarów (Estimator-friendly). Mierzymy później stabilizatory.
    """
    base = graph_state_circuit(n_qubits, graph_edges)
    U = build_scrambler_U(n_qubits, T, topology=topology)
    Udg = build_scrambler_U_dagger(n_qubits, T, topology=topology)

    qc = QuantumCircuit(n_qubits, name=f"echo_T{T}")
    qc.compose(base, inplace=True)
    qc.compose(U, inplace=True)

    op = W_op.lower()
    if op == "z":
        qc.z(W_qubit)
    elif op == "x":
        qc.x(W_qubit)
    elif op == "y":
        qc.y(W_qubit)
    else:
        raise ValueError("W_op must be one of: 'x','y','z'")

    qc.compose(Udg, inplace=True)
    return qc


def estimate_expectations_ideal(circuit: QuantumCircuit, ops: List[SparsePauliOp]) -> List[float]:
    """Idealny Estimator (bez shotów) — szybka walidacja."""
    est = StatevectorEstimator()
    pub = (circuit, ops)
    job = est.run([pub])
    result = job.result()[0]
    return [float(v) for v in result.data.evs]


def run_echo_scan_ideal(
    n_qubits: int,
    graph_edges: List[Tuple[int, int]],
    T_list: List[int],
    W_qubit: int = 0,
    W_op: str = "z",
    topology: str = "line",
) -> List[EchoResult]:
    Ks = stabilizers_for_graph(n_qubits, graph_edges)
    out: List[EchoResult] = []
    for T in T_list:
        qc = build_echo_circuit(
            n_qubits=n_qubits,
            graph_edges=graph_edges,
            T=T,
            W_qubit=W_qubit,
            W_op=W_op,
            topology=topology,
        )
        kvals = estimate_expectations_ideal(qc, Ks)
        out.append(EchoResult(T=T, Kvals=kvals))
    return out


# ------------------------------------------------------------
# 5) Pomiar jako “paliwo”: most warunkowy (dynamic circuits)
# ------------------------------------------------------------
def conditional_cz_bridge_dynamic(n_qubits: int, q_a: int, q_i: int, q_j: int) -> QuantumCircuit:
    """
    Most CZ sterowany mid-circuit measurement (wymaga backendu z dynamic circuits):
      - mierz q_a -> bit ca
      - jeśli ca == 1, wykonaj CZ(q_i, q_j)
    """
    q = QuantumRegister(n_qubits, "q")
    ca = ClassicalRegister(1, "ca")
    qc = QuantumCircuit(q, ca, name="conditional_bridge_dynamic")

    qc.h(q[q_a])
    qc.measure(q[q_a], ca[0])

    with qc.if_test((ca[0], 1)):
        qc.cz(q[q_i], q[q_j])

    return qc


# ------------------------------------------------------------
# 5b) Pomiar jako paliwo: most warunkowy (c_if) — działa na zwykłych backendach
# ------------------------------------------------------------
def conditional_bridge_controlled_demo(n_qubits: int = 5, a: int = 1, i: int = 0, j: int = 4) -> QuantumCircuit:
    """
    Twardy test „most działa” bez mid-circuit measurement i bez if_test/c_if.
    Robimy kontrolowany most: CCZ(a,i,j).
    Intuicja: jeśli a=1, to na parze (i,j) działa fazowy „most” jak CZ.
    """
    q = QuantumRegister(n_qubits, "q")
    c = ClassicalRegister(3, "c")   # c[0]=a, c[1]=i, c[2]=j
    qc = QuantumCircuit(q, c, name="bridge_controlled_CCZ")

    # a w |+> -> superpozycja gałęzi "most OFF/ON"
    qc.h(q[a])

    # para (i,j) w |++>
    qc.h(q[i])
    qc.h(q[j])

    # kontrolowany most
    qc.append(CCZGate(), [q[a], q[i], q[j]])

    # Mierzymy:
    # - a w Z (wybór gałęzi)
    qc.measure(q[a], c[0])

    # - i, j w bazie X (H przed pomiarem)
    qc.h(q[i])
    qc.h(q[j])
    qc.measure(q[i], c[1])
    qc.measure(q[j], c[2])

    return qc


def run_bridge_demo_on_aer(shots: int = 8192) -> None:
    if not _HAS_AER:
        print("\n[Bridge demo] Brak qiskit-aer. Zainstaluj: pip install qiskit-aer")
        return

    qc = conditional_bridge_controlled_demo()
    sim = AerSimulator()
    # POPRAWKA: Przekazujemy qc, bo qc_meas tu nie występuje
    tqc = transpile_line(qc, backend=sim, optimization_level=1)

    res = sim.run(tqc, shots=shots).result()
    counts = res.get_counts()

    # Helper: policz statystyki dla danej mapy bitów out (order = "12" lub "21")
    def compute_stats(order: str):
        stats = {0: {"00": 0, "01": 0, "10": 0, "11": 0, "N": 0},
                 1: {"00": 0, "01": 0, "10": 0, "11": 0, "N": 0}}
        for bitstr, cnt in counts.items():
            s = bitstr.replace(" ", "")
            # s ma 3 bity dla rejestru c[3], typowo MSB..LSB: c2 c1 c0
            c2, c1, c0 = s[0], s[1], s[2]

            a_bit = int(c0)  # a jest mierzone do c[0] => LSB

            if order == "12":
                out = c1 + c2   # c[1] then c[2]
            else:
                out = c2 + c1   # c[2] then c[1]

            stats[a_bit][out] += cnt
            stats[a_bit]["N"] += cnt
        return stats

    def x_parity_corr(d):
        N = d["N"]
        if N == 0:
            return None
        same = d["00"] + d["11"]
        diff = d["01"] + d["10"]
        return (same - diff) / N

    stats12 = compute_stats("12")
    stats21 = compute_stats("21")

    c0_12, c1_12 = x_parity_corr(stats12[0]), x_parity_corr(stats12[1])
    c0_21, c1_21 = x_parity_corr(stats21[0]), x_parity_corr(stats21[1])

    def score(c0, c1):
        if c0 is None or c1 is None:
            return -1e9
        return (-abs(c0)) + (c1)

    best = "12" if score(c0_12, c1_12) > score(c0_21, c1_21) else "21"
    stats = stats12 if best == "12" else stats21
    c0 = x_parity_corr(stats[0])
    c1 = x_parity_corr(stats[1])

    print("\n[Bridge demo on Aer] Controlled bridge via CCZ (robust)")
    print("Chosen out-bit order:", best, "(means out = c1c2 if '12' else c2c1)")
    print("Counts conditioned on a=0:", {k: stats[0][k] for k in ["00","01","10","11"]}, "N=", stats[0]["N"])
    print("Counts conditioned on a=1:", {k: stats[1][k] for k in ["00","01","10","11"]}, "N=", stats[1]["N"])
    print("X-parity correlation E[(-1)^(x_i xor x_j)]")
    print("  a=0 ->", c0, "(should be ~0: bridge OFF)")
    print("  a=1 ->", c1, "(should be >0: bridge ON)")


# ------------------------------------------------------------
# 6) Entanglement swapping (dynamic circuits) — raw only unless backend supports it
# ------------------------------------------------------------
def entanglement_swapping_circuits():
    q = QuantumRegister(4, "q")
    c_bell = ClassicalRegister(2, "m_bell")

    c_measZ = ClassicalRegister(2, "m_Z")
    qcZ = QuantumCircuit(q, c_bell, c_measZ, name="swap_ZZ")

    for start in [0, 2]:
        qcZ.h(q[start])
        qcZ.cx(q[start], q[start + 1])

    qcZ.cx(q[1], q[2])
    qcZ.h(q[1])
    qcZ.measure(q[1], c_bell[0])
    qcZ.measure(q[2], c_bell[1])

    with qcZ.if_test((c_bell[0], 1)):
        qcZ.z(q[3])
    with qcZ.if_test((c_bell[1], 1)):
        qcZ.x(q[3])

    qcZ.measure(q[0], c_measZ[0])
    qcZ.measure(q[3], c_measZ[1])

    c_measX = ClassicalRegister(2, "m_X")
    qcX = QuantumCircuit(q, c_bell, c_measX, name="swap_XX")

    for start in [0, 2]:
        qcX.h(q[start])
        qcX.cx(q[start], q[start + 1])

    qcX.cx(q[1], q[2])
    qcX.h(q[1])
    qcX.measure(q[1], c_bell[0])
    qcX.measure(q[2], c_bell[1])

    with qcX.if_test((c_bell[0], 1)):
        qcX.z(q[3])
    with qcX.if_test((c_bell[1], 1)):
        qcX.x(q[3])

    qcX.h(q[0])
    qcX.h(q[3])
    qcX.measure(q[0], c_measX[0])
    qcX.measure(q[3], c_measX[1])

    return qcZ, qcX


def bridge_bell_gate_demo(n_qubits: int = 5, a: int = 1, i: int = 0, j: int = 4) -> QuantumCircuit:
    """
    'Bridge ON' creates entanglement (EPR-like correlation) between i and j conditioned on a=1,
    using a coherent control (Toffoli / CCX) — no dynamic circuits, no mid-circuit measurement.
    """
    q = QuantumRegister(n_qubits, "q")
    c = ClassicalRegister(3, "c")
    qc = QuantumCircuit(q, c, name="bridge_bell_gate_demo")

    qc.h(q[a])
    qc.h(q[i])
    qc.append(CCXGate(), [q[a], q[i], q[j]])

    qc.measure(q[a], c[0])
    qc.measure(q[i], c[1])
    qc.measure(q[j], c[2])
    return qc


def run_bridge_bell_demo_on_aer(shots: int = 8192) -> None:
    if not _HAS_AER:
        print("\n[Bridge demo] Brak qiskit-aer. Zainstaluj: pip install qiskit-aer")
        return

    qc = bridge_bell_gate_demo()
    sim = AerSimulator()
    tqc = transpile_line(qc, backend=sim, optimization_level=1)

    res = sim.run(tqc, shots=shots, memory=True).result()
    mem = res.get_memory()

    stats = {0: {"00": 0, "01": 0, "10": 0, "11": 0, "N": 0},
             1: {"00": 0, "01": 0, "10": 0, "11": 0, "N": 0}}

    for s in mem:
        s = s.replace(" ", "")
        j_bit, i_bit, a_bit = s[0], s[1], s[2]  # "j i a"
        a_val = int(a_bit)
        out = i_bit + j_bit
        stats[a_val][out] += 1
        stats[a_val]["N"] += 1

    def zz_corr(d):
        N = d["N"]
        if N == 0:
            return None
        same = d["00"] + d["11"]
        diff = d["01"] + d["10"]
        return (same - diff) / N

    c0 = zz_corr(stats[0])
    c1 = zz_corr(stats[1])

    print("\n[Bridge demo on Aer] Coherent controlled entanglement via CCX (robust)")
    print("Counts conditioned on a=0:", {k: stats[0][k] for k in ["00","01","10","11"]}, "N=", stats[0]["N"])
    print("Counts conditioned on a=1:", {k: stats[1][k] for k in ["00","01","10","11"]}, "N=", stats[1]["N"])
    print("ZZ correlation E[(-1)^(z_i xor z_j)]")
    print("  a=0 ->", c0, "(expected ~0: no entangling action on j)")
    print("  a=1 ->", c1, "(expected >0: j follows i -> strong correlation)")


def build_echo_measurement_fueled(
    n_qubits: int,
    graph_edges: List[Tuple[int, int]],
    T: int,
    rule_qubit: int = 1,
    bridge_pair: Tuple[int, int] = (0, 4),
    W_qubit: int = 0,
    W_op: str = "z",
    topology: str = "line",
) -> QuantumCircuit:
    i, j = bridge_pair

    qc = QuantumCircuit(n_qubits, 1, name=f"echo_meas_T{T}")
    base = graph_state_circuit(n_qubits, graph_edges)
    U = build_scrambler_U(n_qubits, T, topology=topology)
    Udg = build_scrambler_U_dagger(n_qubits, T, topology=topology)

    qc.compose(base, inplace=True)
    qc.compose(U, inplace=True)

    op = W_op.lower()
    if op == "z":
        qc.z(W_qubit)
    elif op == "x":
        qc.x(W_qubit)
    elif op == "y":
        qc.y(W_qubit)

    qc.h(rule_qubit)
    qc.measure(rule_qubit, 0)

    with qc.if_test((qc.cregs[0][0], 1)):
        qc.cz(i, j)

    qc.compose(Udg, inplace=True)
    return qc


def add_final_measure_all(qc: QuantumCircuit) -> QuantumCircuit:
    n = qc.num_qubits
    c_out = ClassicalRegister(n, "out")
    qc2 = qc.copy()
    qc2.add_register(c_out)
    qc2.measure(range(n), c_out)
    return qc2


# POPRAWKA: Bezpieczniejsze parsowanie na wypadek dodatkowych spacji u Qiskita.
def prob_all_zero_from_counts(counts: Dict[str, int], n_qubits: int, out_reg_first: bool = True) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0

    p0 = 0
    for key, c in counts.items():
        parts = key.split()
        if not parts:
            continue
        # Pierwszy człon na lewo to zawsze 'c_out' dodane pod koniec (Qiskit wyświetla odwrotnie)
        out = parts[0] if out_reg_first else parts[-1]
        if out == "0" * n_qubits:
            p0 += c
    return p0 / total


def run_time_engine_echo_on_aer(
    n_qubits: int,
    graph_edges: List[Tuple[int, int]],
    T_list: List[int],
    shots: int = 4096,
    rule_qubit: int = 1,
    bridge_pair: Tuple[int, int] = (0, 4),
    W_qubit: int = 0,
    W_op: str = "z",
    topology: str = "line",
    postselect_m: int | None = None,
) -> None:
    if not _HAS_AER:
        print("\n[Time-engine echo] Brak qiskit-aer. Zainstaluj: pip install qiskit-aer")
        return

    sim = AerSimulator()

    print("\n[TIME-ENGINE ECHO on Aer] (measurement-fueled)")
    print(f"Rule qubit={rule_qubit}, bridge={bridge_pair}, W={W_op}@{W_qubit}, shots={shots}")
    if postselect_m is not None:
        print(f"Post-selection: keep only shots with m={postselect_m}")

    for T in T_list:
        qc = build_echo_measurement_fueled(
            n_qubits=n_qubits,
            graph_edges=graph_edges,
            T=T,
            rule_qubit=rule_qubit,
            bridge_pair=bridge_pair,
            W_qubit=W_qubit,
            W_op=W_op,
            topology=topology,
        )
        qc_meas = add_final_measure_all(qc)

        # POPRAWKA: Transpilujemy i puszczamy `qc_meas`, NIE `qc` !
        tqc = transpile_line(qc_meas, backend=sim, optimization_level=1)
        res = sim.run(tqc, shots=shots).result()
        counts = res.get_counts()

        m_counts = {0: 0, 1: 0}
        filtered = Counter()

        for key, c in counts.items():
            # POPRAWKA: Rozbicie na spacjach od Qiskita: "c_out m"
            parts = key.split()
            if len(parts) != 2:
                continue

            out_str = parts[0]
            mbit = int(parts[1])
            m_counts[mbit] += c

            if postselect_m is None or mbit == postselect_m:
                filtered[key] += c

        p0 = prob_all_zero_from_counts(filtered, n_qubits=n_qubits)
        total_f = sum(filtered.values())
        frac = total_f / shots if shots else 0

        print(f"  T={T:2d} | P(out=0..0)={p0:.4f} | m0={m_counts[0]} m1={m_counts[1]} | kept={frac:.3f}")


def _extract_m_and_out_bits_from_key(key: str, n_qubits: int) -> tuple[int, str]:
    """
    POPRAWKA:
    Wyciąga oddzielnie 'm' (1 bit po prawej) oraz 'out' (n_qubits po lewej) uciekając od indeksowania
    za pomocą -n_qubits w zwartych stringach.
    """
    parts = key.split()
    if len(parts) == 2:
        return int(parts[1]), parts[0]

    # Fallback awaryjny gdyby jakimś cudem spacji nie było
    s = key.replace(" ", "")
    out = s[:n_qubits]
    mbit = int(s[n_qubits:])
    return mbit, out


def _z_from_bit(bit_char: str) -> int:
    return +1 if bit_char == "0" else -1


def exp_Zk_from_counts_postselected(
    counts: dict[str, int],
    n_qubits: int,
    k: int,
    postselect_m: int,
) -> tuple[float | None, int]:
    num = 0
    den = 0
    for key, c in counts.items():
        mbit, out = _extract_m_and_out_bits_from_key(key, n_qubits)
        if mbit != postselect_m:
            continue
        z = _z_from_bit(out[-(k + 1)])
        num += c * z
        den += c
    if den == 0:
        return None, 0
    return num / den, den


def exp_ZkZl_from_counts_postselected(
    counts: dict[str, int],
    n_qubits: int,
    k: int,
    l: int,
    postselect_m: int,
) -> tuple[float | None, int]:
    num = 0
    den = 0
    for key, c in counts.items():
        mbit, out = _extract_m_and_out_bits_from_key(key, n_qubits)
        if mbit != postselect_m:
            continue
        zk = _z_from_bit(out[-(k + 1)])
        zl = _z_from_bit(out[-(l + 1)])
        num += c * (zk * zl)
        den += c
    if den == 0:
        return None, 0
    return num / den, den


def run_Z_observables_vs_T_time_engine(
    n_qubits: int,
    graph_edges: list[tuple[int, int]],
    T_list: list[int],
    shots: int = 4096,
    rule_qubit: int = 1,
    bridge_pair: tuple[int, int] = (0, 4),
    W_qubit: int = 0,
    W_op: str = "z",
    topology: str = "line",
    k: int = 0,
    l: int = 4,
) -> None:
    if not _HAS_AER:
        print("\n[Z observables] Brak qiskit-aer. Zainstaluj: pip install qiskit-aer")
        return

    sim = AerSimulator()

    print("\n[Z OBSERVABLES vs T] (measurement-fueled, postselected)")
    print(f"Observables: <Z{k}>, <Z{k}Z{l}>  | rule={rule_qubit}, bridge={bridge_pair}, W={W_op}@{W_qubit}, shots={shots}")
    print("Columns: T | kept(m0) <Zk>0 <ZkZl>0 || kept(m1) <Zk>1 <ZkZl>1")

    for T in T_list:
        qc = build_echo_measurement_fueled(
            n_qubits=n_qubits,
            graph_edges=graph_edges,
            T=T,
            rule_qubit=rule_qubit,
            bridge_pair=bridge_pair,
            W_qubit=W_qubit,
            W_op=W_op,
            topology=topology,
        )
        qc_meas = add_final_measure_all(qc)

        tqc = transpile_line(qc_meas, backend=sim, optimization_level=1)
        res = sim.run(tqc, shots=shots).result()
        counts = res.get_counts()

        z0_m0, n0 = exp_Zk_from_counts_postselected(counts, n_qubits, k=k, postselect_m=0)
        zz_m0, _ = exp_ZkZl_from_counts_postselected(counts, n_qubits, k=k, l=l, postselect_m=0)

        z0_m1, n1 = exp_Zk_from_counts_postselected(counts, n_qubits, k=k, postselect_m=1)
        zz_m1, _ = exp_ZkZl_from_counts_postselected(counts, n_qubits, k=k, l=l, postselect_m=1)

        def fmt(x):
            return "None" if x is None else f"{x:+.4f}"

        print(f"  T={T:2d} | {n0:4d} {fmt(z0_m0):>7} {fmt(zz_m0):>8} || {n1:4d} {fmt(z0_m1):>7} {fmt(zz_m1):>8}")


# ============================================================================
# ROZSZERZENIE: FORMALIZM PAGE–WOOTTERSA (relacyjny czas)
# ============================================================================

def build_star_hamiltonian(N_sys: int, J: float = 1.0) -> SparsePauliOp:
    pauli_list = []
    coeffs = []
    for i in range(1, N_sys):
        z0z_str = ''.join(['Z' if idx == 0 or idx == i else 'I' for idx in range(N_sys)])
        pauli_list.append(z0z_str)
        coeffs.append(J)
    return SparsePauliOp(pauli_list, coeffs)

def build_history_state(N_clock: int,
                        H_s: SparsePauliOp,
                        psi0: Statevector) -> Statevector:
    T = 2 ** N_clock
    dim_c = T
    dim_s = 2 ** int(np.log2(len(psi0.data)))

    Hs_mat = H_s.to_matrix()
    history = np.zeros(dim_c * dim_s, dtype=complex)

    for t in range(T):
        U_t = expm(-1j * Hs_mat * t)
        psi_t = U_t @ psi0.data

        clock_t = np.zeros(dim_c, dtype=complex)
        clock_t[t] = 1.0

        history += np.kron(clock_t, psi_t)

    history /= np.sqrt(T)
    return Statevector(history)

def page_wootters_demo(N_clock: int = 3, N_sys: int = 2, omega: float = 0.5, J: float = 1.0):
    print("\n" + "="*60)
    print(f"  DEMO: Formalizm Page–Woottersa (relacyjny czas), N_clock={N_clock}")
    print("="*60)

    pauli_c = []
    coeff_c = []
    for i in range(N_clock):
        z_term = ['I'] * N_clock
        z_term[i] = 'Z'
        pauli_c.append(''.join(z_term))
        coeff_c.append(-omega * (2**i) / 2)
    H_c = SparsePauliOp(pauli_c, coeff_c)

    H_s = build_star_hamiltonian(N_sys, J=J)
    X0_op = SparsePauliOp.from_list([('X' + 'I'*(N_sys-1), 1.0)])
    psi0_s = Statevector.from_label("+0")
    psi_phys = build_history_state(N_clock, H_s, psi0_s)

    print("\nZbudowano stan historii Page–Woottersa (bez szukania eigenów H_total).")

    def clock_time_evolution(t: int) -> QuantumCircuit:
        qc = QuantumCircuit(N_clock)
        for i in range(N_clock):
            phase = omega * (2**i) * t
            qc.rz(2 * phase, i)
        return qc

    def conditional_expectation(psi: Statevector,
                                t: int,
                                observable_sys: SparsePauliOp):
        """
        POPRAWKA: W formalizmie Page-Woottersa chcąc wyciągnąć stan z history state,
        rzutujesz na wektor bazowy |t>. Aplikowanie przed tym Ewolucji Zegara niszczy korelację!
        Dlatego odrzucono `qc.compose(...)`.
        """
        qc = QuantumCircuit(N_clock + N_sys)
        qc.initialize(psi, range(N_clock + N_sys))

        backend = AerSimulator(method='statevector')
        qc.save_statevector()
        job = backend.run(qc)
        psi_t = job.result().get_statevector()

        dim_c = 2**N_clock
        dim_s = 2**N_sys
        vec = psi_t.data.reshape(dim_c, dim_s)

        t_bin = format(t, f"0{N_clock}b")
        idx_t = int(t_bin, 2)
        vec_t = vec[idx_t, :]
        norm = np.linalg.norm(vec_t)
        if norm < 1e-12:
            return 0.0
        vec_t = vec_t / norm
        psi_s = Statevector(vec_t)
        return psi_s.expectation_value(observable_sys)

    times = list(range(2**N_clock))
    cond_vals = []
    for t in times:
        val = conditional_expectation(psi_phys, t, X0_op)
        cond_vals.append(val)

    H_s_mat = H_s.to_matrix()
    U_s = expm(-1j * H_s_mat)
    psi_s0 = psi0_s

    orig_vals = []
    state = psi_s0
    for _ in times:
        orig_vals.append(state.expectation_value(X0_op))
        state = Statevector(U_s @ state.data)

    print("\nt (zegar) | <X0> warunkowe | <X0> ewolucja H_s | różnica")
    for t, cv, ov in zip(times, cond_vals, orig_vals):
        diff = cv - ov
        print(f"{t:3d} | {cv:10.4f} | {ov:10.4f} | {diff:10.4f}")
    mse = np.mean([(cv-ov)**2 for cv, ov in zip(cond_vals, orig_vals)])
    print(f"Średni błąd kwadratowy: {mse:.6f}")


if __name__ == "__main__":
    n = 5

    edges_star = [(0, 1), (0, 2), (0, 3), (0, 4)]
    gs = graph_state_circuit(n, edges_star)
    print("Graph-state...", len(transpile_line(gs).data))

    Ks = stabilizers_for_graph(n, edges_star)
    kvals0 = estimate_expectations_ideal(gs, Ks)
    print("⟨K_v⟩ dla graf-state (gwiazda):", [float(f"{v:.3f}") for v in kvals0])

    T_list = [0, 1, 2, 3, 4, 6, 8]
    results = run_echo_scan_ideal(
        n_qubits=n,
        graph_edges=edges_star,
        T_list=T_list,
        W_qubit=0,
        W_op="z",
        topology="line",
    )

    print("\nECHO (ideal) — stabilizatory po U(T) Z U†(T):")
    for r in results:
        meanK = sum(r.Kvals) / len(r.Kvals)
        print(f"  T={r.T:2d} | mean ⟨K⟩ = {meanK:+.4f} | K = {[float(f'{v:+.3f}') for v in r.Kvals]}")

    cond_dc = conditional_cz_bridge_dynamic(5, q_a=1, q_i=0, q_j=4)
    print("\nConditional CZ (dynamic if_test) — depth (raw):", cond_dc.depth(), "| ops:", len(cond_dc.data))

    swapZZ, swapXX = entanglement_swapping_circuits()
    print("Entanglement swapping ZZ — depth (raw):", swapZZ.depth(), "| ops:", len(swapZZ.data))
    print("Entanglement swapping XX — depth (raw):", swapXX.depth(), "| ops:", len(swapXX.data))

    print("\nUwaga / Note:")
    print("  Obwody z if_test wymagają backendu wspierającego dynamic circuits.")
    print("  Dlatego nie wołamy transpile() na nich bez podania takiego backendu.")

    run_bridge_bell_demo_on_aer(shots=8192)

    T_list2 = [0, 1, 2, 3, 4, 6, 8]
    run_time_engine_echo_on_aer(
        n_qubits=n,
        graph_edges=edges_star,
        T_list=T_list2,
        shots=4096,
        rule_qubit=1,
        bridge_pair=(0, 4),
        W_qubit=0,
        W_op="z",
        topology="line",
        postselect_m=None,
    )

    run_time_engine_echo_on_aer(
        n_qubits=n,
        graph_edges=edges_star,
        T_list=T_list2,
        shots=4096,
        rule_qubit=1,
        bridge_pair=(0, 4),
        W_qubit=0,
        W_op="z",
        topology="line",
        postselect_m=0,
    )
    run_time_engine_echo_on_aer(
        n_qubits=n,
        graph_edges=edges_star,
        T_list=T_list2,
        shots=4096,
        rule_qubit=1,
        bridge_pair=(0, 4),
        W_qubit=0,
        W_op="z",
        topology="line",
        postselect_m=1,
    )

    run_Z_observables_vs_T_time_engine(
        n_qubits=n,
        graph_edges=edges_star,
        T_list=[0, 1, 2, 3, 4, 6, 8],
        shots=4096,
        rule_qubit=1,
        bridge_pair=(0, 4),
        W_qubit=0,
        W_op="z",
        topology="line",
        k=0,
        l=4,
    )
    page_wootters_demo(N_clock=3)
