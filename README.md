# Simulating the Spring Dynamics of an EpiPen Autoinjector

**Group 1 – Sagentia 1**  
*Sam Bethell, Chenghe Tang, Charles Ying, Joseph Mace, Emeka Anagu*

---

## 📘 Overview

This repository contains the code and numerical models developed for the project:

> **Simulating the spring dynamics of an EpiPen autoinjector for reliable drug delivery**

The repository includes all scripts, simulation files, and data used in the modeling and analysis of the EpiPen mechanism as part of the MDM Sagentia project.

---

## ⚙️ Code Structure

| File | Description |
|------|--------------|
| `FinalModel.py` | NOT FINAL | 
| `Finalimpliciteuler.py` | Implicitly solving |
| `KelvinVoigtModel.py` | First draft  |
| `KelvinVoigtExplicit.py` | Using explicit euler |
| `KVexplicitlywithoutairgap.py` | Modelling spring dashpot model without an airgap |
| `Findingparameters.py` | Shows how fluid injected varies with parameters |
| `Force vs Damping.py` | Plotting force on needle against resistance |
| `Stroke vs Pressure.py` | Plotting stroke vs Pressure  |
| `fluid2.py` | An iteration of a differential equation describing fluid flow through needle |
| `fluid3.py` | An iteration of a differential equation describing fluid flow through needle |
| `0211.py` |  |
| `0211settingchanges.py` |  |
| `PDE` | Approximating the continuous model expressed as a partial differential equation |
| `README.md` | Project overview and documentation. |

---

## 🧩 Model Overview

The models in this repository simulate the dynamic behavior of the EpiPen’s internal spring and plunger system through multiple stages of deployment and injection.  
Each script corresponds to a different modeling approach or parameter study.

---

## 🧠 Technical Highlights

- Includes both **explicit** and **implicit** time integration methods  
- Supports **Kelvin–Voigt viscoelastic modeling**  
- Implements **parameter sweep scripts** for optimization  
- Produces time histories, force–displacement plots, and injection dynamics visualizations  

---

## 📁 Requirements

Python ≥ 3.9  

Required libraries:
```bash
numpy
matplotlib
scipy
