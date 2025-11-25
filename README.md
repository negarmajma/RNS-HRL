# RNS-HRL: Runtime Validation of a Responsive Neural Stimulation Device

This repository contains the code for a runtime validation framework for a Responsive Neural Stimulation (RNS) device used in epilepsy management.  
The model integrates three main components:

- **Timed Coloured Petri Nets (TCPN)** for modelling the execution logic of the device,
- **Fuzzy logic** for handling uncertainty in brain and physiological signals,
- **Reinforcement learning (RL)** – including a hierarchical manager – for adaptive decision making at runtime.

The code reproduces the experiments described in the manuscript, including:
- baseline learning and validation on 10 expert-defined scenarios, and  
- hierarchical RL training on an augmented dataset of 1530 scenarios with noisy and interpolated physiological profiles.

---

## Repository structure (main files)

- `main.py` – Baseline (non-hierarchical) RL training and evaluation on the 10 expert-defined scenarios using the TCPN + fuzzy model.
- `train.py` – Training script for the **hierarchical manager** on the augmented scenario dataset; saves the trained manager and the training reward curve.
- `Diagram.py` – Post-processing and plotting script; reads the scenario-wise evaluation file and generates the figure with mean returns and action probabilities.

Supporting modules:

- `data_loader.py` – Loads the base brain and physiological sensor values and expert labels.
- `tcpn.py` – Implementation of the timed coloured Petri net model of the RNS device.
- `fuzzy_system.py` – Fuzzy logic system for mapping TCPN states and sensor values to fuzzy risk levels (S1–S4).
- `hierarchical_env.py` – Hierarchical RL environment that couples the TCPN–fuzzy model with the high-level manager and low-level worker.

---

## Dependencies

The project uses Python and common scientific libraries. A typical environment includes:

- Python 3.x  
- `numpy`  
- `pandas`  
- `matplotlib`  
- `torch` (PyTorch)  
- `tqdm`

You can install the dependencies (example):

```bash
pip install numpy pandas matplotlib torch tqdm

For Run:
python generate_scenarios.py
python train.py
python eval_hrl_by_scenario.py
