Methanol Synthesis Reactor Digital Twin

This repository presents a digital twin framework for methanol synthesis reactors, combining process modeling, reaction kinetics, and system-level simulation. The project aims to bridge fundamental reaction engineering with real-time process analysis and optimization.

Overview

The model integrates:

Reaction kinetics (CO/CO₂ hydrogenation to methanol)
Energy balances and thermal behavior
Plug flow reactor (PFR) modelling
Digital twin concepts for process monitoring and prediction

It is designed as a research-oriented simulation platform for studying reactor performance under varying operating conditions.

Key Features
Python-based reactor simulation framework
Coupled mass and energy balances
Modular structure for easy extension
Visualization via HTML interface
Foundation for real-time digital twin applications

Repository Structure
Methanol Synthesis Platform.py → Core simulation model
methanol_digital_twin_2.py → Enhanced digital twin version
Methanol_Platform_V.2.html → Visualization interface
.gitignore → Environment configuration

Modelling Approach

The reactor is modeled using:

(Graaf, Bussche & Froment, Park) kinetics
Plug flow assumptions
Heat effects of reaction
Parametric sensitivity analysis

This enables evaluation of:

Conversion efficiency
Temperature profiles
Process stability

Requirements
Python 3.x
NumPy
Matplotlib

Future Improvements
Integration with real-time data streams
Advanced reactor models (multi-phase, CFD coupling)
Process optimization & control strategies
Techno-economic analysis (TEA) layer

Purpose

This project demonstrates how digital twins can be applied to chemical reactors, supporting:
Process understanding
Optimization
Scalable industrial applications

Author

Neel Augustine 
Process Engineer | Hydrogen & Syngas Technologies
