"""
===============================================================================
QHDALabs-qmnet  |  Quantum Relational Network
Measurement-Fueled Bridge Experiment — clean single-file version

Author  : Krzysztof Banasiewicz
Contact : https://krzyshtof.com

Core question:
  Does a mid-circuit measurement, used as "fuel" to conditionally add
  a CZ edge (bridge) between two otherwise-disconnected qubits, affect
  the system's ability to return to its initial state after unscrambling?

Experiment flow:
  |G⟩  →  U(T)  →  W  →  [measure rule_qubit → m]
       →  (if m==1: CZ bridge on (i,j))
       →  U†(T)  →  final measurement

Observables:
  - P(return)   : P(out = 0...0) as echo return proxy
  - ⟨Z_k⟩       : single-qubit Z expectation, postselected on m
  - ⟨Z_k Z_l⟩   : two-qubit ZZ correlator, postselected on m

All results are reported separately for m=0 and m=1 branches.
===============================================================================
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import CCXGate, CCZGate, CZGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import CouplingMap

try:
    from qiskit_aer import AerSimulator
    _HAS_AER = True
except ImportError:
    AerSimulator = None
    _HAS_AER = False


# ============================================================
# CONFIGURATION
# ============================================================

# Default 5-qubit line topology for transpilation
LINE_COUPLING = CouplingMap([(0, 1), (1, 2), (2, 3), (3, 4)])
BASIS_GATES = ["rz", "sx", "x", "cx", "cz"]


def transpile_for_sim(qc: QuantumCircuit, backend, optimization_level: int = 1) -> QuantumCircuit:
    """Transpile circuit for AerSimulator with line topology constraints."""
    return transpile(
        qc,
        backend=backend,
        coupling_map=LINE_COUPLING,
        basis_gates=BASIS_GATES,
        optimization_level=optimization_level,
    )


# ============================================================
# SECTION 1: GRAPH STATE
# ============================================================

def graph_state_circuit(n_qubits: int, edges: List[Tuple[int, int]]) -> QuantumCircuit:
    """
    Build a graph state: |+>^{⊗n} with CZ on each edge.

    The graph defines which qubits are "connected" in the initial network.
    Edges here are the natural connections — the bridge will later add
    a conditional connection outside this base topology.
    """
    q = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(q, name="graph_state")
    for i in range(n_qubits):
        qc.h(q[i])
    for u, v in edges:
        qc.cz(q[u], q[v])
    return qc


def stabilizers_for_graph(n_qubits: int, edges: List[Tuple[int, int]]) -> List[SparsePauliOp]:
    """
    Build stabilizer operators K_v = X_v ⊗ Z_{neighbors(v)} for a graph state.

    Used for ideal (statevector) echo quality measurement:
    ⟨K_v⟩ = +1 means the qubit v is back to its stabilized state.
    """
    neighbors: Dict[int, set] = {i: set() for i in range(n_qubits)}
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    ops = []
    for v in range(n_qubits):
        pauli = ["I"] * n_qubits
        pauli[v] = "X"
        for u in neighbors[v]:
            pauli[u] = "Z"
        label = "".join(reversed(pauli))   # Qiskit: qubit 0 = rightmost bit
        ops.append(SparsePauliOp.from_list([(label, 1.0)]))
    return ops


# ============================================================
# SECTION 2: SCRAMBLER U(T) AND ITS INVERSE
# ============================================================

def _brickwork_edges(n_qubits: int, parity: int) -> List[Tuple[int, int]]:
    """
    Brickwork CZ pairs on a line for one scrambler layer.
      parity=0 → (0,1), (2,3), ...
      parity=1 → (1,2), (3,4), ...
    """
    start = parity % 2
    return [(i, i + 1) for i in range(start, n_qubits - 1, 2)]


def build_scrambler(n_qubits: int, T: int) -> QuantumCircuit:
    """
    Build U(T): T layers of (H on all qubits + CZ brickwork).

    This is a unitary scrambling circuit that spreads local information
    across the full network. More layers = more scrambling.
    """
    qc = QuantumCircuit(n_qubits, name=f"U_T{T}")
    for step in range(T):
        for i in range(n_qubits):
            qc.h(i)
        for u, v in _brickwork_edges(n_qubits, step):
            qc.cz(u, v)
    return qc


def build_scrambler_inverse(n_qubits: int, T: int) -> QuantumCircuit:
    """
    Build U†(T): exact inverse of build_scrambler(n_qubits, T).

    Since H and CZ are self-inverse, reversing the layer order suffices.
    If no bridge were applied, U†(T) · U(T) = I (perfect echo return).
    The bridge breaks this symmetry — that's the experiment.
    """
    qc = QuantumCircuit(n_qubits, name=f"Udg_T{T}")
    for step in reversed(range(T)):
        for u, v in reversed(_brickwork_edges(n_qubits, step)):
            qc.cz(u, v)
        for i in range(n_qubits):
            qc.h(i)
    return qc


# ============================================================
# SECTION 3: ECHO BASELINE (ideal, no bridge)
# ============================================================

@dataclass
class EchoResult:
    T: int
    stabilizer_vals: List[float]

    @property
    def mean_stabilizer(self) -> float:
        return float(np.mean(self.stabilizer_vals))


def build_echo_circuit_ideal(
    n_qubits: int,
    graph_edges: List[Tuple[int, int]],
    T: int,
    W_qubit: int = 0,
    W_op: str = "z",
) -> QuantumCircuit:
    """
    Build the baseline echo circuit (no measurement, no bridge):
      |G⟩ → U(T) → W → U†(T)

    If W is identity (or U†U = I), all stabilizers return to +1.
    The decay of ⟨K_v⟩ with T shows how hard the echo is to close.
    """
    qc = QuantumCircuit(n_qubits, name=f"echo_baseline_T{T}")
    qc.compose(graph_state_circuit(n_qubits, graph_edges), inplace=True)
    qc.compose(build_scrambler(n_qubits, T), inplace=True)

    op = W_op.lower()
    if op == "z":
        qc.z(W_qubit)
    elif op == "x":
        qc.x(W_qubit)
    elif op == "y":
        qc.y(W_qubit)
    else:
        raise ValueError(f"W_op must be 'x', 'y', or 'z', got: {W_op!r}")

    qc.compose(build_scrambler_inverse(n_qubits, T), inplace=True)
    return qc


def run_echo_baseline_ideal(
    n_qubits: int,
    graph_edges: List[Tuple[int, int]],
    T_list: List[int],
    W_qubit: int = 0,
    W_op: str = "z",
) -> List[EchoResult]:
    """
    Run the baseline echo for each T using ideal statevector simulation.
    Returns stabilizer expectation values — no shots, no noise.
    """
    stabilizers = stabilizers_for_graph(n_qubits, graph_edges)
    estimator = StatevectorEstimator()
    results = []

    for T in T_list:
        qc = build_echo_circuit_ideal(n_qubits, graph_edges, T, W_qubit, W_op)
        job = estimator.run([(qc, stabilizers)])
        kvals = [float(v) for v in job.result()[0].data.evs]
        results.append(EchoResult(T=T, stabilizer_vals=kvals))

    return results


# ============================================================
# SECTION 4: MEASUREMENT-FUELED BRIDGE EXPERIMENT
# ============================================================

def build_bridge_experiment_circuit(
    n_qubits: int,
    graph_edges: List[Tuple[int, int]],
    T: int,
    rule_qubit: int,
    bridge_pair: Tuple[int, int],
    W_qubit: int = 0,
    W_op: str = "z",
) -> QuantumCircuit:
    """
    Build the measurement-fueled bridge experiment circuit:

      |G⟩ → U(T) → W → H(rule) → measure(rule) → m
           → if m==1: CZ(bridge_pair)
           → U†(T)

    The mid-circuit measurement of rule_qubit produces a classical bit m.
    If m=1, a CZ gate is applied between bridge_pair qubits.
    This CZ is NOT part of the original graph or scrambler —
    it is an extra edge, conditionally added by the measurement outcome.

    The key question: does this conditional bridge affect echo return?
    Compare P(return | m=0) vs P(return | m=1).

    Args:
        n_qubits    : total number of qubits
        graph_edges : base graph topology
        T           : scrambler depth
        rule_qubit  : qubit measured mid-circuit to fuel the bridge
        bridge_pair : (i, j) — qubits connected if m=1
        W_qubit     : qubit where perturbation W is applied
        W_op        : perturbation operator ('x', 'y', 'z')

    Returns:
        Circuit without final measurement (add separately for flexibility).
        Classical register 0 holds the bridge measurement bit m.
    """
    i, j = bridge_pair

    # 1 classical bit for the bridge measurement (m)
    qc = QuantumCircuit(n_qubits, 1, name=f"bridge_exp_T{T}")

    # Step 1: Prepare graph state
    qc.compose(graph_state_circuit(n_qubits, graph_edges), inplace=True)

    # Step 2: Scramble
    qc.compose(build_scrambler(n_qubits, T), inplace=True)

    # Step 3: Perturbation W
    op = W_op.lower()
    if op == "z":
        qc.z(W_qubit)
    elif op == "x":
        qc.x(W_qubit)
    elif op == "y":
        qc.y(W_qubit)
    else:
        raise ValueError(f"W_op must be 'x', 'y', or 'z', got: {W_op!r}")

    # Step 4: Measurement as fuel
    # Put rule_qubit in superposition, then measure → m
    qc.h(rule_qubit)
    qc.measure(rule_qubit, 0)

    # Step 5: Conditional bridge
    # If m==1: add CZ between bridge_pair (extra edge, not in base graph)
    with qc.if_test((qc.cregs[0][0], 1)):
        qc.cz(i, j)

    # Step 6: Attempt to un-scramble
    # Without bridge: U†U = I, system returns perfectly.
    # With bridge (m=1): extra CZ breaks the symmetry → partial or no return.
    qc.compose(build_scrambler_inverse(n_qubits, T), inplace=True)

    return qc


def _add_final_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """Add a separate classical register for final qubit readout."""
    n = qc.num_qubits
    c_out = ClassicalRegister(n, "out")
    qc2 = qc.copy()
    qc2.add_register(c_out)
    qc2.measure(range(n), c_out)
    return qc2


# ============================================================
# SECTION 5: RESULTS PARSING
# ============================================================

def _parse_key(key: str, n_qubits: int) -> Tuple[int, str]:
    """
    Parse a Qiskit counts key into (m_bit, out_string).

    Qiskit prints registers separated by spaces, last-added register
    appears leftmost. Since we add 'out' after the bridge register,
    the format is: "out_bits m_bit"
    Example for n=5: "01010 1"
    """
    parts = key.split()
    if len(parts) == 2:
        return int(parts[1]), parts[0]
    # Fallback: no spaces (shouldn't happen with two registers)
    s = key.replace(" ", "")
    return int(s[-1]), s[:n_qubits]


def _z_eigenvalue(bit: str) -> int:
    """Map measurement bit to Z eigenvalue: '0' → +1, '1' → −1."""
    return +1 if bit == "0" else -1


def compute_return_probability(
    counts: Dict[str, int],
    n_qubits: int,
    postselect_m: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Compute P(out = 0...0) as echo return proxy.

    A perfect echo (no bridge, no perturbation) would return P=1.
    Perturbation W reduces this. The bridge (m=1) may reduce it further.

    Returns (probability, n_shots_kept).
    """
    total = 0
    match = 0
    zero_string = "0" * n_qubits

    for key, cnt in counts.items():
        m, out = _parse_key(key, n_qubits)
        if postselect_m is not None and m != postselect_m:
            continue
        total += cnt
        if out == zero_string:
            match += cnt

    if total == 0:
        return 0.0, 0
    return match / total, total


def compute_Zk_expectation(
    counts: Dict[str, int],
    n_qubits: int,
    k: int,
    postselect_m: int,
) -> Tuple[Optional[float], int]:
    """
    Compute ⟨Z_k⟩ postselected on m = postselect_m.

    Z_k = +1 if qubit k measured 0, −1 if measured 1.
    Qiskit bit ordering: out[-(k+1)] corresponds to qubit k.
    """
    num = 0
    den = 0
    for key, cnt in counts.items():
        m, out = _parse_key(key, n_qubits)
        if m != postselect_m:
            continue
        z = _z_eigenvalue(out[-(k + 1)])
        num += cnt * z
        den += cnt
    if den == 0:
        return None, 0
    return num / den, den


def compute_ZkZl_expectation(
    counts: Dict[str, int],
    n_qubits: int,
    k: int,
    l: int,
    postselect_m: int,
) -> Tuple[Optional[float], int]:
    """
    Compute ⟨Z_k Z_l⟩ postselected on m = postselect_m.

    Measures whether qubits k and l are correlated after the echo.
    The bridge connects bridge_pair — if k,l = bridge_pair, this
    directly measures the effect of the conditional CZ.
    """
    num = 0
    den = 0
    for key, cnt in counts.items():
        m, out = _parse_key(key, n_qubits)
        if m != postselect_m:
            continue
        zk = _z_eigenvalue(out[-(k + 1)])
        zl = _z_eigenvalue(out[-(l + 1)])
        num += cnt * (zk * zl)
        den += cnt
    if den == 0:
        return None, 0
    return num / den, den


# ============================================================
# SECTION 6: FULL EXPERIMENT RUNNER
# ============================================================

def run_bridge_experiment(
    n_qubits: int = 5,
    graph_edges: Optional[List[Tuple[int, int]]] = None,
    T_list: Optional[List[int]] = None,
    rule_qubit: int = 1,
    bridge_pair: Tuple[int, int] = (0, 4),
    W_qubit: int = 0,
    W_op: str = "z",
    shots: int = 4096,
    run_baseline: bool = True,
) -> None:
    """
    Run the full measurement-fueled bridge experiment.

    Prints:
      1. Baseline echo stabilizer values (ideal) for each T
      2. For each T and each m branch (0 and 1):
         - number of shots kept
         - P(return)
         - ⟨Z_k⟩ for k = bridge_pair[0]
         - ⟨Z_k Z_l⟩ for k,l = bridge_pair

    Args:
        n_qubits    : number of qubits (default 5, line topology)
        graph_edges : base graph; default is star (qubit 0 as hub)
        T_list      : scrambler depths to test
        rule_qubit  : qubit measured to fuel the bridge
        bridge_pair : (i, j) — extra CZ edge when m=1
        W_qubit     : where perturbation W is applied
        W_op        : 'x', 'y', or 'z'
        shots       : number of shots for Aer simulation
        run_baseline: whether to run ideal baseline echo first
    """
    if not _HAS_AER:
        print("ERROR: qiskit-aer not found. Install with: pip install qiskit-aer")
        return

    if graph_edges is None:
        graph_edges = [(0, 1), (0, 2), (0, 3), (0, 4)]  # star topology

    if T_list is None:
        T_list = [0, 1, 2, 3, 4, 6, 8]

    k, l = bridge_pair
    sim = AerSimulator()

    # ----------------------------------------------------------
    # PART 1: Baseline echo (ideal, no bridge, no shots)
    # ----------------------------------------------------------
    if run_baseline:
        print("\n" + "=" * 70)
        print("BASELINE ECHO (ideal statevector, no bridge)")
        print(f"Graph edges: {graph_edges}")
        print(f"Perturbation: {W_op.upper()} on qubit {W_qubit}")
        print("=" * 70)
        print(f"{'T':>4} | {'mean ⟨K⟩':>10} | stabilizer values")
        print("-" * 70)

        baseline_results = run_echo_baseline_ideal(
            n_qubits, graph_edges, T_list, W_qubit, W_op
        )
        for r in baseline_results:
            kstr = "  ".join(f"{v:+.3f}" for v in r.stabilizer_vals)
            print(f"{r.T:>4} | {r.mean_stabilizer:>+10.4f} | {kstr}")

    # ----------------------------------------------------------
    # PART 2: Measurement-fueled bridge experiment
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("MEASUREMENT-FUELED BRIDGE EXPERIMENT")
    print(f"Rule qubit : {rule_qubit}  (measured mid-circuit → m)")
    print(f"Bridge pair: {bridge_pair}  (CZ added when m=1)")
    print(f"Perturbation: {W_op.upper()} on qubit {W_qubit}")
    print(f"Shots: {shots}")
    print("=" * 70)
    print(
        f"{'T':>4} | "
        f"{'n(m=0)':>7} {'P_ret|m0':>9} {'<Zk>|m0':>9} {'<ZkZl>|m0':>11} | "
        f"{'n(m=1)':>7} {'P_ret|m1':>9} {'<Zk>|m1':>9} {'<ZkZl>|m1':>11}"
    )
    print("-" * 70)

    for T in T_list:
        qc = build_bridge_experiment_circuit(
            n_qubits=n_qubits,
            graph_edges=graph_edges,
            T=T,
            rule_qubit=rule_qubit,
            bridge_pair=bridge_pair,
            W_qubit=W_qubit,
            W_op=W_op,
        )
        qc_full = _add_final_measurements(qc)
        tqc = transpile_for_sim(qc_full, sim)
        counts = sim.run(tqc, shots=shots).result().get_counts()

        def fmt(x: Optional[float]) -> str:
            return "  None" if x is None else f"{x:+.4f}"

        p0, n0 = compute_return_probability(counts, n_qubits, postselect_m=0)
        p1, n1 = compute_return_probability(counts, n_qubits, postselect_m=1)

        zk0, _ = compute_Zk_expectation(counts, n_qubits, k=k, postselect_m=0)
        zk1, _ = compute_Zk_expectation(counts, n_qubits, k=k, postselect_m=1)

        zkzl0, _ = compute_ZkZl_expectation(counts, n_qubits, k=k, l=l, postselect_m=0)
        zkzl1, _ = compute_ZkZl_expectation(counts, n_qubits, k=k, l=l, postselect_m=1)

        print(
            f"{T:>4} | "
            f"{n0:>7} {p0:>9.4f} {fmt(zk0):>9} {fmt(zkzl0):>11} | "
            f"{n1:>7} {p1:>9.4f} {fmt(zk1):>9} {fmt(zkzl1):>11}"
        )

    print("\nHow to read the table:")
    print(f"  P_ret   : P(all qubits return to |0⟩) — higher = better echo")
    print(f"  <Zk>    : ⟨Z_{k}⟩ expectation — 0 means maximally mixed on qubit {k}")
    print(f"  <ZkZl>  : ⟨Z_{k} Z_{l}⟩ correlator — nonzero = bridge created correlation")
    print(f"  m=0 branch: bridge was NOT fired (rule measured 0)")
    print(f"  m=1 branch: bridge WAS fired (CZ added between {bridge_pair})")
    print(f"\nKey signal: if P_ret differs between m=0 and m=1 at same T,")
    print(f"the conditional CZ bridge is affecting the echo dynamics.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_bridge_experiment(
        n_qubits=5,
        graph_edges=[(0, 1), (0, 2), (0, 3), (0, 4)],  # star: qubit 0 as hub
        T_list=[0, 1, 2, 3, 4, 6, 8],
        rule_qubit=1,
        bridge_pair=(0, 4),   # bridge connects hub to leaf — maximally disruptive
        W_qubit=0,
        W_op="z",
        shots=4096,
        run_baseline=True,
    )
