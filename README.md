# QHDALabs / qmnet

**Quantum Measurement Routing & Relational Network Research**

*Krzysztof Banasiewicz — independent researcher*

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Qiskit%201.0%2B-purple)](https://qiskit.org)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/QHDALabs/qmnet/pulls)

---

## What this is

This repository explores a practical question in near-term quantum computing:

> *Can we treat mid-circuit measurements not as passive readout, but as active resources — using measurement outcomes to steer the topology of subsequent quantum operations?*

The work has evolved through two directions that share the same foundation:

**Direction 1 — Decoherence routing**: treating information leakage as a dial rather than a switch, using ancilla coupling angle to tune how much information escapes and through which path.

**Direction 2 — Measurement-fueled bridges**: using a mid-circuit measurement result to conditionally add a CZ edge (bridge) between otherwise-disconnected qubits, and observing how this affects a Loschmidt echo experiment.

A technical note documenting the bridge experiment results is available in this repository.

---

## Published note

**[Measurement-Fueled Conditional Bridges in Graph-State Echo Experiments](qhdalabs-Bridge-note-draft.pdf)**

Key findings from the bridge experiment:

- Conditional CZ bridge applied in the m=1 measurement branch suppresses echo return probability to **P_ret = 0.000** at scrambler depths T=2 and T=4, while the m=0 branch retains P_ret ≈ 0.12–0.13
- At T=0 and T=8, ⟨Z_0⟩ is sharply polarized: **+1.000 in m=0, −1.000 in m=1** — a deterministic Z-flip conditioned on the bridge measurement
- A topology sweep across five bridge pairs shows that the pair **(1,3)** — whose qubits share no brickwork layer with the scrambler — is approximately half as disruptive as other pairs, suggesting scrambler-edge overlap drives disruption strength

These are preliminary results from ideal (noise-free) simulation on 5 qubits. Physical interpretation and scalability remain open questions.

---

## Repository structure

```
qmnet/
├── qmnet_v3.py                            # Bridge experiment — clean version (use this)
├── qmnet_v4.py                            # Bridge experiment + topology sweep
├── qmnet.py                               # Earlier exploration: scrambling, PW engine
├── routed_measurement_full_experiment.py  # Decoherence routing experiment
├── qhdalabs-Bridge-note-draft.pdf         # Technical note (draft 1.0)
├── LICENSE
└── README.md
```

**Where to start:** `qmnet_v3.py` for the bridge experiment, `routed_measurement_full_experiment.py` for the decoherence routing work. `qmnet.py` is an earlier monolithic file kept for reference.

---

## Experiments

### `routed_measurement_full_experiment.py`
**Decoherence as a dial**

Compares three information-extraction regimes on a single system qubit:

| Mode | What happens |
|---|---|
| `strong_direct_z` | Projective Z channel — full decoherence |
| `ancilla_cx` | Information routed via CX to ancilla — equivalent decoherence, different path |
| `ancilla_weak_ry(θ)` | Partial coupling via controlled RY — tunable leak |

The weak routing sweep across θ ∈ [0, π] produces a smooth trade-off: purity drops, entropy rises, fidelity to |+⟩ degrades continuously. Bloch vector components track this at each θ.

---

### `qmnet_v3.py` / `qmnet_v4.py`
**Measurement-fueled bridge experiment**

Full circuit sequence:

```
|G⟩ → U(T) → Z(q0) → H(q1) → measure(q1) → m
     → if m==1: CZ(bridge_pair)
     → U†(T) → final measurement
```

`qmnet_v3.py` runs bridge pair `(0,4)` across scrambler depths T ∈ {0,1,2,3,4,6,8} with baseline stabilizer echo for comparison.

`qmnet_v4.py` adds a topology sweep across five bridge pairs to test whether scrambler-edge overlap determines disruption strength.

---

### `qmnet.py`
**Earlier exploration (reference)**

Includes: graph state generation, Loschmidt echo scan with stabilizer estimator, CCZ-based bridge demo, entanglement swapping circuits, and a Page–Wootters history state engine. Kept for reference; `qmnet_v3.py` is the cleaner successor for the bridge work.

---

## Current status

| Result | Status |
|---|---|
| Stabilizer values track scrambling depth | ✅ confirmed (ideal sim) |
| Bridge suppresses P_ret to 0 at T=2,4 in m=1 branch | ✅ confirmed (ideal sim) |
| Z-flip ⟨Z_0⟩ = ±1 conditioned on bridge at T=0,8 | ✅ confirmed (ideal sim) |
| Topology sweep: (1,3) least disruptive | ✅ confirmed (ideal sim) |
| Noise model validation | ⬜ not yet |
| Scaling beyond 5 qubits | ⬜ not yet |
| Physical QPU run | ⬜ not yet — IBM Heron is target |

---

## Dependencies

```
qiskit >= 1.0
qiskit-aer >= 0.13
numpy
scipy
matplotlib
qiskit-ibm-runtime  # optional, for QPU access
```

```bash
pip install qiskit qiskit-aer scipy matplotlib
```

---

## Running

```bash
# Decoherence routing sweep + plots (~1 min)
python routed_measurement_full_experiment.py

# Bridge experiment, single pair, full T sweep (~2 min)
python qmnet_v3.py

# Bridge experiment + topology sweep (~5 min)
python qmnet_v4.py
```

---

## Open questions — where collaboration is welcome

- **Noise model**: does the bridge suppression effect survive under realistic depolarizing + T1/T2 noise?
- **Scaling**: does scrambler-edge overlap remain the dominant factor at 7–10 qubits?
- **Formal channel analysis**: express the bridge as a quantum channel acting on the echo and derive its effect analytically — needs QEC/density matrix background
- **QPU access**: IBM Heron or equivalent for hardware validation of the T=2 case (circuit is shallow enough)
- **Endorsement for arXiv**: if you have quant-ph endorser status and find this work interesting, please get in touch

---

## Contact

**Krzysztof Banasiewicz**
qhdalabs.contact@gmail.com | [LinkedIn](https://www.linkedin.com/in/krzyshtoof)

If you find something wrong, say so directly. If you find something interesting, let's talk. If you want to use this in your own work — go ahead, just link back.

---

## License

MIT — see `LICENSE`.

---

*Independent research. No institutional affiliation. No grants. Just curiosity, Qiskit, and too many late nights.*
