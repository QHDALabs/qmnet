# QHDALabs / qmnet

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A51.0-6929C4?style=flat-square&logo=qiskit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT%20(pending)-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20%2F%20Experimental-orange?style=flat-square&logo=flask&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Qiskit%20Aer%20Simulator-blueviolet?style=flat-square&logo=ibm&logoColor=white)
![Target QPU](https://img.shields.io/badge/Target%20QPU-IBM%20Heron-005F9E?style=flat-square&logo=ibm&logoColor=white)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=flat-square&logo=github&logoColor=white)

**Quantum Measurement Routing & Relational Network Research**

*Krzysztof Banasiewicz — independent researcher*

---

## What this is

This repository contains two experimental modules exploring a practical question:
> *How much control do we have over where quantum information goes — and when?*

Most near-term quantum hardware research focuses on gate fidelity and error correction.
This work takes a different angle: **routing** — treating the flow of information from a
quantum system into its environment not as an unavoidable problem, but as a controllable
parameter.

The two modules grew from the same question but attack it at different levels.

---

## Modules

### `routed_measurement_full_experiment.py`

**Decoherence as a dial, not a switch**

A direct comparison of three information-extraction regimes:

| Mode | What happens |
| --- | --- |
| `strong_direct_z` | Projective Z measurement channel — full decoherence |
| `ancilla_cx` | Information routed through ancilla via CX — equivalent decoherence, different path |
| `ancilla_weak_ry(θ)` | Partial coupling via controlled RY — tunable information leak |

The weak routing sweep across θ ∈ [0, π] produces a clean trade-off curve: as coupling
angle increases, purity drops, entropy rises, and fidelity to the original state degrades.
The Bloch vector components track this continuously.

**Why it matters for hardware:** Near-term devices (NISQ era) cannot eliminate decoherence. But if we can *route* it —
choosing which ancilla qubit absorbs which information, and how much — that is a
primitive for building smarter error mitigation layers above the hardware abstraction.

---

### `qmnet.py`

**Dynamic topology and relational time**

A more experimental module. Three ideas developed together:

**1. Graph states as a connectivity substrate**

5-qubit star and line graph states with full stabilizer verification. The Loschmidt echo
scan (`run_echo_scan_ideal`) measures how a local perturbation W propagates and scrambles
across the network as a function of circuit depth T. Stabilizer expectation values decay
as scrambling increases — a quantitative handle on information spreading.

**2. Measurement-fueled bridges**

The `conditional_cz_bridge_dynamic` and `bridge_bell_gate_demo` circuits implement a
topology switch: a mid-circuit measurement result determines whether a CZ gate fires
between two otherwise-disconnected qubits. The CCZ-based variant avoids dynamic circuit
requirements while preserving the conditional entanglement effect.

The key observable: ZZ correlation conditioned on the ancilla measurement outcome.
When the bridge fires (a=1), strong ZZ correlation appears between qubits 0 and 4 —
qubits that share no direct gate in the base circuit.

**3. Page–Wootters relational time**

Implementation of the Page–Wootters mechanism: a "history state" entangling a clock
register (N_clock qubits) with a system register. Conditional expectation values
extracted by projecting onto clock basis states reproduce standard Schrödinger evolution
to within numerical precision (MSE < 1e-5 for N_clock=3).

This is a verification implementation, not a new result — but it is a working,
self-contained PW engine that can be extended.

---

## Current status

Both modules produce numerically consistent results on Qiskit Aer simulator.
No physical QPU runs yet — IBM Quantum Heron is the target for next validation.

**What is confirmed:**

- ✅ Stabilizer values track scrambling depth as expected
- ✅ ZZ bridge correlation: ~0.0 for a=0, ~0.8–1.0 for a=1 (8192 shots)
- ✅ Weak measurement trade-off curve is smooth and physically interpretable
- ✅ PW history state matches Schrödinger evolution for N_clock ∈ {2, 3}

**What is not claimed:**

- ❌ Novel physics results
- ❌ Hardware benchmarks
- ❌ Scalability beyond ~5–7 qubits without error correction

---

## Dependencies

![qiskit](https://img.shields.io/badge/qiskit-%E2%89%A51.0-6929C4?style=flat-square&logo=qiskit&logoColor=white)
![qiskit-aer](https://img.shields.io/badge/qiskit--aer-%E2%89%A50.13-6929C4?style=flat-square&logo=qiskit&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-latest-013243?style=flat-square&logo=numpy&logoColor=white)
![scipy](https://img.shields.io/badge/scipy-latest-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-latest-11557C?style=flat-square&logo=python&logoColor=white)

```
qiskit >= 1.0
qiskit-aer >= 0.13
qiskit-ibm-runtime (optional, for QPU access)
numpy
scipy
matplotlib
```

Install:

```bash
pip install qiskit qiskit-aer scipy matplotlib
```

---

## Running

```bash
# Weak measurement sweep + plots
python routed_measurement_full_experiment.py

# Full qmnet demo: graph states, echo, bridges, Page-Wootters
python qmnet.py
```

⏱️ Expected runtime on a standard laptop: **2–5 minutes** for qmnet full run.

---

## Repository structure

```
.
├── LICENSE
├── README.md
├── qmnet.py    # Network, scrambling, PW engine
├── qmnet_v3.py # Bridge experiment
├── qmnet_v4.py # Bridge experiment
└── routed_measurement_full_experiment.py  # Decoherence routing experiment
```

---

## Open questions (where help is welcome)

This project is at the boundary of what one person can develop alone.
Contributions, critique, and collaboration are genuinely welcome on:

- 🔬 **Scaling the bridge mechanism** — does conditional topology switching remain coherent
beyond 7–10 qubits, and how does it interact with realistic noise models?
- 📐 **Formalization** — the routing layer concept needs proper density matrix channel
formalism. Looking for a collaborator with QEC background.
- 🖥️ **QPU validation** — access to IBM Heron or equivalent for hardware runs.
- 🕰️ **The PW engine** — extending to continuous-variable clocks and larger system registers.

---

## Contact & collaboration

[![Email](https://img.shields.io/badge/Email-qhdalabs.contact%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:qhdalabs.contact@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-QHDALabs-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/QHDALabs)

**Krzysztof Banasiewicz** <qhdalabs.contact@gmail.com>

If you find something wrong, say so directly. If you find something interesting,
let's talk. If you want to use this in your own work — go ahead, just link back.

---

## License

[![License](https://img.shields.io/badge/License-MIT%20(pending%20patent%20review)-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](LICENSE)

See `LICENSE` — currently all rights reserved pending patent review.
The code is shared for research visibility and collaboration purposes.
Open-source release under MIT is planned once the patent process concludes.

---

*🔭 This is independent research. No institutional affiliation. No grants.
Just curiosity, Qiskit, and too many late nights.*
