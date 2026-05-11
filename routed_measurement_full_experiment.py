# routed_measurement_full_experiment.py
# ============================================================
# QHDALabs
# Routed Measurement Experiment v1
#
# PL:
# Pełny eksperyment w jednym pliku:
# - porównanie silnego pomiaru bezpośredniego
# - routingu informacji przez ancillę
# - weak-like routingu przez częściowe sprzężenie
#
# EN:
# Full single-file experiment:
# - direct strong measurement
# - ancilla-routed information extraction
# - weak-like routed extraction via partial coupling
# ============================================================

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Statevector,
    DensityMatrix,
    partial_trace,
    state_fidelity,
    entropy,
    Operator,
)
from qiskit_aer import AerSimulator


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ExperimentConfig:
    shots: int = 4096
    theta_values: Tuple[float, ...] = (
        0.0,
        np.pi / 32,
        np.pi / 16,
        np.pi / 8,
        np.pi / 6,
        np.pi / 4,
        np.pi / 3,
        np.pi / 2,
        2 * np.pi / 3,
        3 * np.pi / 4,
        np.pi,
    )
    run_counts: bool = True
    make_plots: bool = True
    save_plots: bool = False
    plot_prefix: str = "routed_measurement"
    print_density_matrix: bool = False


# ============================================================
# BASIC HELPERS
# ============================================================

def purity(rho: DensityMatrix) -> float:
    """Compute purity Tr(rho^2)."""
    mat = np.array(rho.data, dtype=complex)
    return float(np.real(np.trace(mat @ mat)))


def rho_to_numpy(rho: DensityMatrix) -> np.ndarray:
    """Convert density matrix to numpy array."""
    return np.array(rho.data, dtype=complex)


def reduced_system_density(dm_2q: DensityMatrix) -> DensityMatrix:
    """
    Trace out ancilla q1 and keep system q0.
    Qubit ordering follows Qiskit convention.
    """
    return partial_trace(dm_2q, [1])


def bloch_components(rho: DensityMatrix) -> Dict[str, float]:
    """
    Compute Bloch vector components for a single-qubit density matrix:
    x = Tr(rho X), y = Tr(rho Y), z = Tr(rho Z)
    """
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    mat = rho_to_numpy(rho)
    x = float(np.real(np.trace(mat @ X)))
    y = float(np.real(np.trace(mat @ Y)))
    z = float(np.real(np.trace(mat @ Z)))

    return {"x": x, "y": y, "z": z}


def pretty_complex_matrix(mat: np.ndarray, precision: int = 5) -> str:
    """Nicely format a complex matrix for printing."""
    rows = []
    for row in mat:
        row_str = []
        for val in row:
            re = np.real(val)
            im = np.imag(val)
            if abs(im) < 10 ** (-precision):
                row_str.append(f"{re:.{precision}f}")
            elif abs(re) < 10 ** (-precision):
                row_str.append(f"{im:.{precision}f}j")
            else:
                sign = "+" if im >= 0 else "-"
                row_str.append(f"{re:.{precision}f}{sign}{abs(im):.{precision}f}j")
        rows.append("[ " + ", ".join(row_str) + " ]")
    return "\n".join(rows)


# ============================================================
# QUANTUM CHANNELS / MEASUREMENT MODELS
# ============================================================

def apply_projective_measurement_channel_z_on_q0(rho_2q: DensityMatrix) -> DensityMatrix:
    """
    Apply a non-selective projective measurement channel in Z basis on q0.

    PL:
    To nie jest pojedynczy wynik 0/1, tylko kanał pomiarowy bez wyboru gałęzi.
    Niszczy koherencję względem bazy Z na q0.

    EN:
    This is a non-selective projective measurement channel, not a single-shot outcome.
    It destroys coherence in the Z basis on q0.
    """
    mat = rho_to_numpy(rho_2q)

    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    M0 = np.kron(P0, I)
    M1 = np.kron(P1, I)

    measured = M0 @ mat @ M0.conj().T + M1 @ mat @ M1.conj().T
    return DensityMatrix(measured)


# ============================================================
# STATE PREPARATION
# ============================================================

def prepare_two_qubit_base_state() -> QuantumCircuit:
    """
    q0 = system
    q1 = ancilla

    Prepare:
    q0 -> |+>
    q1 -> |0>
    """
    qc = QuantumCircuit(2)
    qc.h(0)
    return qc


def get_reference_plus_state() -> Statevector:
    """Reference single-qubit |+> state."""
    return Statevector.from_label("+")


# ============================================================
# EXPERIMENT VARIANTS
# ============================================================

def channel_strong_direct_z() -> Dict[str, Any]:
    """
    Strong direct measurement channel on system qubit q0.
    Implemented as a projective Z measurement channel (non-selective).
    """
    qc = prepare_two_qubit_base_state()
    sv = Statevector.from_instruction(qc)
    rho = DensityMatrix(sv)

    rho_after = apply_projective_measurement_channel_z_on_q0(rho)
    rho_sys = reduced_system_density(rho_after)

    return {
        "label": "strong_direct_z",
        "qc": qc,
        "rho_2q": rho_after,
        "rho_sys": rho_sys,
    }


def channel_ancilla_cx() -> Dict[str, Any]:
    """
    Strong ancilla-routed readout:
    q0 controls q1 through CX, then ancilla is treated as record/environment.
    The system state is obtained by tracing out the ancilla.
    """
    qc = prepare_two_qubit_base_state()
    qc.cx(0, 1)

    sv = Statevector.from_instruction(qc)
    rho = DensityMatrix(sv)
    rho_sys = reduced_system_density(rho)

    return {
        "label": "ancilla_cx",
        "qc": qc,
        "rho_2q": rho,
        "rho_sys": rho_sys,
    }


def channel_ancilla_weak_ry(theta: float) -> Dict[str, Any]:
    """
    Weak-like routed interaction:
    q0 -> ancilla through controlled RY(theta).
    Then ancilla is treated as the information carrier/environment.
    """
    qc = prepare_two_qubit_base_state()
    qc.cry(theta, 0, 1)

    sv = Statevector.from_instruction(qc)
    rho = DensityMatrix(sv)
    rho_sys = reduced_system_density(rho)

    return {
        "label": f"ancilla_weak_ry_theta_{theta:.6f}",
        "qc": qc,
        "rho_2q": rho,
        "rho_sys": rho_sys,
        "theta": theta,
    }


# ============================================================
# COUNTS / SAMPLING
# ============================================================

def sample_counts(qc: QuantumCircuit, shots: int = 4096) -> Dict[str, int]:
    """
    Run sampling on the full 2-qubit circuit with final measurement.
    This is informational only; the density-matrix analysis above is primary.
    """
    sim = AerSimulator()
    qc_m = qc.copy()
    qc_m.measure_all()
    result = sim.run(qc_m, shots=shots).result()
    return result.get_counts()


def sample_direct_z_counts(shots: int = 4096) -> Dict[str, int]:
    """
    Sampling version for direct Z readout of q0 plus optional readout of q1.
    q1 remains untouched in |0>.
    """
    sim = AerSimulator()
    qc = prepare_two_qubit_base_state()
    qc.measure_all()
    result = sim.run(qc, shots=shots).result()
    return result.get_counts()


# ============================================================
# METRICS
# ============================================================

def compute_system_metrics(rho_sys: DensityMatrix) -> Dict[str, float]:
    ref_plus = get_reference_plus_state()
    return {
        "purity": purity(rho_sys),
        "entropy": float(entropy(rho_sys, base=2)),
        "fidelity_to_plus": float(state_fidelity(rho_sys, ref_plus)),
        **bloch_components(rho_sys),
    }


# ============================================================
# ANALYSIS WRAPPERS
# ============================================================

def analyze_strong_direct_z(cfg: ExperimentConfig) -> Dict[str, Any]:
    res = channel_strong_direct_z()
    metrics = compute_system_metrics(res["rho_sys"])
    counts = sample_direct_z_counts(cfg.shots) if cfg.run_counts else None

    return {
        "mode": "strong_direct_z",
        "theta": None,
        "metrics": metrics,
        "counts": counts,
        "rho_sys": rho_to_numpy(res["rho_sys"]),
        "rho_2q": rho_to_numpy(res["rho_2q"]),
        "circuit": res["qc"],
    }


def analyze_ancilla_cx(cfg: ExperimentConfig) -> Dict[str, Any]:
    res = channel_ancilla_cx()
    metrics = compute_system_metrics(res["rho_sys"])
    counts = sample_counts(res["qc"], cfg.shots) if cfg.run_counts else None

    return {
        "mode": "ancilla_cx",
        "theta": None,
        "metrics": metrics,
        "counts": counts,
        "rho_sys": rho_to_numpy(res["rho_sys"]),
        "rho_2q": rho_to_numpy(res["rho_2q"]),
        "circuit": res["qc"],
    }


def analyze_ancilla_weak_ry(theta: float, cfg: ExperimentConfig) -> Dict[str, Any]:
    res = channel_ancilla_weak_ry(theta)
    metrics = compute_system_metrics(res["rho_sys"])
    counts = sample_counts(res["qc"], cfg.shots) if cfg.run_counts else None

    return {
        "mode": "ancilla_weak_ry",
        "theta": theta,
        "metrics": metrics,
        "counts": counts,
        "rho_sys": rho_to_numpy(res["rho_sys"]),
        "rho_2q": rho_to_numpy(res["rho_2q"]),
        "circuit": res["qc"],
    }


# ============================================================
# REPORTING
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_metrics_block(label: str, metrics: Dict[str, float]) -> None:
    print(f"\n[{label}]")
    for k, v in metrics.items():
        print(f"  {k:18s}: {v:+.6f}")


def print_counts_block(counts: Dict[str, int] | None) -> None:
    if counts is None:
        return
    print("  counts:")
    for k in sorted(counts.keys()):
        print(f"    {k}: {counts[k]}")


def print_result(result: Dict[str, Any], cfg: ExperimentConfig) -> None:
    label = result["mode"]
    theta = result["theta"]
    metrics = result["metrics"]

    if theta is None:
        print_header(f"RESULT: {label}")
    else:
        print_header(f"RESULT: {label} | theta = {theta:.6f} rad")

    print_metrics_block("system_metrics", metrics)
    print_counts_block(result["counts"])

    if cfg.print_density_matrix:
        print("\n  rho_sys:")
        print(pretty_complex_matrix(result["rho_sys"]))


def summarize_weak_results(results: List[Dict[str, Any]]) -> None:
    print_header("WEAK ROUTING SUMMARY TABLE")
    print(
        f"{'theta(rad)':>12} | {'purity':>10} | {'entropy':>10} | {'fidelity':>10} | {'bloch_x':>10} | {'bloch_z':>10}"
    )
    print("-" * 80)

    for r in results:
        m = r["metrics"]
        print(
            f"{r['theta']:12.6f} | "
            f"{m['purity']:10.6f} | "
            f"{m['entropy']:10.6f} | "
            f"{m['fidelity_to_plus']:10.6f} | "
            f"{m['x']:10.6f} | "
            f"{m['z']:10.6f}"
        )


# ============================================================
# PLOTTING
# ============================================================

def plot_weak_sweep(results: List[Dict[str, Any]], cfg: ExperimentConfig) -> None:
    thetas = [r["theta"] for r in results]
    purity_vals = [r["metrics"]["purity"] for r in results]
    entropy_vals = [r["metrics"]["entropy"] for r in results]
    fidelity_vals = [r["metrics"]["fidelity_to_plus"] for r in results]
    bloch_x_vals = [r["metrics"]["x"] for r in results]
    bloch_z_vals = [r["metrics"]["z"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(thetas, purity_vals, marker="o", label="purity")
    plt.plot(thetas, entropy_vals, marker="s", label="entropy")
    plt.plot(thetas, fidelity_vals, marker="^", label="fidelity_to_plus")
    plt.xlabel("theta [rad]")
    plt.ylabel("metric value")
    plt.title("Weak routed measurement sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if cfg.save_plots:
        plt.savefig(f"{cfg.plot_prefix}_metrics.png", dpi=200, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(thetas, bloch_x_vals, marker="o", label="bloch_x")
    plt.plot(thetas, bloch_z_vals, marker="s", label="bloch_z")
    plt.xlabel("theta [rad]")
    plt.ylabel("Bloch component")
    plt.title("System Bloch components vs weak routing angle")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if cfg.save_plots:
        plt.savefig(f"{cfg.plot_prefix}_bloch.png", dpi=200, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    print_header("CONFIG")
    print(asdict(cfg))

    strong_direct = analyze_strong_direct_z(cfg)
    ancilla_cx = analyze_ancilla_cx(cfg)
    weak_results = [analyze_ancilla_weak_ry(theta, cfg) for theta in cfg.theta_values]

    print_result(strong_direct, cfg)
    print_result(ancilla_cx, cfg)

    for wr in weak_results:
        print_result(wr, cfg)

    summarize_weak_results(weak_results)

    if cfg.make_plots:
        plot_weak_sweep(weak_results, cfg)

    return {
        "strong_direct_z": strong_direct,
        "ancilla_cx": ancilla_cx,
        "ancilla_weak_ry_sweep": weak_results,
    }


# ============================================================
# INTERPRETATION NOTES
# ============================================================

def print_interpretation_notes() -> None:
    print_header("INTERPRETATION NOTES")
    print(
        """
1. strong_direct_z
   - reprezentuje silny kanał pomiarowy w bazie Z na qubicie systemowym
   - niszczy koherencję |+> względem osi X
   - oczekiwany efekt:
     purity ~ 1.0
     entropy ~ 1.0
     fidelity_to_plus ~ 0.5
     bloch_x ~ 0.0
     bloch_z ~ 0.0

2. ancilla_cx
   - informacja o q0 jest przenoszona do ancilli przez CX
   - po śledzeniu ancilli system traci koherencję podobnie jak przy silnym odczycie
   - to jest routed strong readout

3. ancilla_weak_ry(theta)
   - częściowe sprzężenie z ancillą
   - dla małego theta:
       mały wyciek informacji
       mniejsza utrata koherencji
       fidelity do |+> pozostaje wyższa
   - dla większego theta:
       zachowanie zbliża się do silniejszego kanału

4. sens eksperymentu
   - nie testujemy tylko "czy jest pomiar"
   - testujemy:
       ile informacji wycieka
       jaką drogą wycieka
       jak siła sprzężenia wpływa na degradację stanu
"""
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    cfg = ExperimentConfig(
        shots=4096,
        theta_values=(
            0.0,
            np.pi / 32,
            np.pi / 16,
            np.pi / 8,
            np.pi / 6,
            np.pi / 4,
            np.pi / 3,
            np.pi / 2,
            2 * np.pi / 3,
            3 * np.pi / 4,
            np.pi,
        ),
        run_counts=True,
        make_plots=True,
        save_plots=False,
        plot_prefix="routed_measurement",
        print_density_matrix=False,
    )

    print_interpretation_notes()
    run_experiment(cfg)
