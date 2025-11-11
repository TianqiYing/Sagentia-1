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
| `FinalModel.py` |  |
| `Finalimpliciteuler.py` |  |
| `KelvinVoigtModel.py` |  |
| `KelvinVoigtExplicit.py` |  |
| `KVexplicitlywithoutairgap.py` |  |
| `Findingparameters.py` |  |
| `Force vs Damping.py` |  |
| `Stroke vs Pressure.py` |  |
| `fluid2.py` |  |
| `fluid3.py` |  |
| `0211.py` |  |
| `0211settingchanges.py` |  |
| `PDE` |  |
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
