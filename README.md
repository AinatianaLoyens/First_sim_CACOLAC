#Description
The codes in this repository are simulations of semi-discrete population models (prey-predator models).\\
The models are ODE systems that have a discrete part.\\
The aim is to plot prey and predators population according to their functions in the models.

#Dependancies
numpy as np
odeint from scipy.integrate
matplotlib.pyplot as plt

# Main module: `double_exo` — Modeling Pest Control with Continuous and Impulsive Exogenous Mortality

This module provides tools to simulate and analyze a population dynamics model with both **continuous** and **impulsive** exogenous mortality. It allows flexible simulation of one- or two-species interactions (typically pest and parasitoid), with customizable functional forms for growth, predation, and mortality.

---

## Model Overview

The model implemented in this module is based on the following system:

dx/dt = g(x) * x - f(x, y) * y

dy/dt = γ * f(x, y) * y - m(x, y) * y

x(nT⁺) = x(nT) - Ẽₓ * hₓ(x(nT))

y(nT⁺) = y(nT) - Ẽᵧ * hᵧ(y(nT))


- `x` is the pest population  
- `y` is the parasitoid population  
- `g` is the pest growth rate  
- `f` is the predation function  
- `m` is the intrinsic mortality of the parasitoid  
- `hₓ`, `hᵧ` are impulsive exogenous mortality functions  
- `Ẽₓ`, `Ẽᵧ` are the strengths of impulsive control  
- `T` is the period between pulses  

### Notes:

- The mortality function `m(x, y)` may depend on both species.  
- `hₓ` and `hᵧ` can either be `0` (no impulsive mortality) or equal to the identity (`x` or `y`, respectively), to apply full impulsive mortality.  
- To model continuous exogenous mortality:
  - Replace `g(x)` by `ĝ(x) - Eₓ`  
  - Replace `m(x, y)` by `m̂(x, y) + Eᵧ`  

The model can simulate:
- One or two species (set `y0 = 0` for single species)  
- Any combination of continuous or impulsive mortality  
- Complete absence of exogenous mortality

- ## Module Structure

The core module is **`double_exo.py`**. It includes:

### 1. Predefined Functional Forms

Pre-implemented versions of the model’s functions:
- Growth (`g`), mortality (`m`), interaction (`f`), and impulsive mortality (`hₓ`, `hᵧ`)
- Variants depending on `x`, `y`, or both

### 2. Solvers

Functions to solve the model over time:
- Includes impulsive effects at multiples of the period `T`
- Option to delay the first pulse beyond one period
- Returns `x(t)`, `y(t)`, and the cumulative pest load `I = ∫ x(t) dt`

### 3. Comparison Tools

- Plotting functions for comparing continuous and impulsive models  
- Heatmaps and indicators to visualize relative efficiency  

### 4. Periodic Solution Simulations

- Tools to simulate and analyze non-trivial stable periodic solutions in 2D
