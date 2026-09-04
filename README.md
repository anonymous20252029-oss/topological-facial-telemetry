# Coordinate-Free Topological Telemetry for Clinical Facial Palsy and rPPG

This repository contains the interactive validation platform for **Paper 1** developed at the **Signal Lab, Department of Cybernetics and Biomedical Engineering, VŠB - Technical University of Ostrava**.

## Core Mathematical Contributions
- **Multicritical Bifiltrations:** Decoupling non-rigid facial deformations from rigid out-of-plane head yaw rotations ($SE(3)$ invariance).
- **1-Lipschitz Stability:** Proven bounds against Hausdorff perturbations.
- **Adaptive Denoising:** Fast Transversal Filter (FTF) leveraging topological derivatives for optical pulse recovery.

## Quickstart (Local)
```bash
git clone [https://github.com/](https://github.com/)<your-username>/topological-facial-telemetry.git
cd topological-facial-telemetry
pip install -r requirements.txt
streamlit run app.py
