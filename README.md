# From Particles to Perils: SVGD-Based Hazardous Scenario Generation for Autonomous Driving Systems Testing

## Abstract
Simulation-based testing of autonomous driving systems (ADS) must expose realistic and diverse failures that emerge from dense traffic and complex interactions among heterogeneous dynamic objects (vehicles, cyclists, and pedestrians). The effectiveness of ADS testing is highly sensitive to the choice of initial conditions (seeds). However, existing search-based seeding approaches (e.g., genetic algorithms) struggle in the high-dimensional spaces induced by dense, heterogeneous traffic, often collapsing into a limited set of modes and leaving many realistic failure scenarios undiscovered.

We present **PtoP**, a novel framework for ADS testing. At its core, PtoP couples adaptive random seed generation—which produces seeds that cover diverse initial failure modes—with **Stein Variational Gradient Descent (SVGD)** to explore a wide range of failure-inducing initial conditions. Each particle represents the initial state of a dynamic object. SVGD jointly leverages gradient-driven attraction toward high-hazard regions and kernel-mediated repulsion to maintain diversity among particles, resulting in risk-seeking yet well-distributed seeds that span multiple distinct failure modes.

Beyond seed generation, PtoP functions as a plug-and-play framework that integrates seamlessly with online testing techniques (e.g., reinforcement learning–based testers), providing principled seeds that boost overall testing effectiveness.  

We evaluate PtoP in **CARLA**—a state-of-the-art simulator—on two ADS implementations: an industry-grade stack (**Apollo**) and an end-to-end ADS native to CARLA. When combined with state-of-the-art baselines, PtoP substantially improves performance: safety-violation rate (up to **+27.68%**), diversity (up to **+9.6%**), and map coverage (up to **+16.78%**). These results highlight PtoP as an efficient, principled approach to synthesizing realistic, hard-to-find hazardous scenarios for ADS validation.

[//]: # (<img src="https://github.com/asvonavnsnvononaon/AutoMT/blob/main/Images/ASE_overall.jpg" width="60%"/>)

## 📖 Table of Contents

- [Abstract](#abstract)  
- [Installation](#installation)  
  - [Install Packages](#install-packages)  
  - [Install CARLA](#1-install-carla)  
  - [Install Apollo](#2-install-apollo)  
  - [Install CARLA-Apollo Bridge](#3-install-carla-apollo-bridge)  
  - [File Setup](#4-file-setup)  
- [Running the System](#running-the-system)  
- [Baselines](#baselines)

---



## Installation

### Install Packages  
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 1. Install CARLA  
First, install **CARLA 0.9.13** from the official website: [CARLA](https://carla.org/).

### 2. Install Apollo  
Next, install **Apollo 8.0** from the official repository: [Apollo](https://github.com/ApolloAuto/apollo).

### 3. Install CARLA-Apollo Bridge  
Install the **CARLA-Apollo Bridge** from: [CARLA Apollo Bridge](https://github.com/MaisJamal/carla_apollo_bridge).

### 4. File Setup  
- Copy `manual_control.py` to `/carla_apollo_bridge/example`.  
- Copy `listener.py` to the Apollo root folder.  
- Copy the maps in the `map` folder to `Apollo/modules/map/data`.

## Running the System  

1. Start **CARLA**:  
   ```bash
   ./CarlaUE4.sh -RenderOffScreen
   ```

2. Run the CARLA configuration script in the **bridger docker**. Change the map name if you are using a different map:  
   ```bash
   python carla-python-0.9.13/util/config.py -m Town04 --host 172.17.0.1
   ```
3. Start the **manual control script**:  
   ```bash
   python examples/manual_control.py
   ```
4. Launch **Apollo** by following the instructions on the [CARLA Apollo Bridge repository](https://github.com/MaisJamal/carla_apollo_bridge).  
5Run the **CARLA Cyber Bridge** in another **bridge docker**:  
   ```bash
   python carla_cyber_bridge/run_bridge.py
   ```
5. Start the **listener script** in the Apollo docker:  
   ```bash
   python listener.py
   ```
   - If you encounter issues running `listener.py`, refer to the official Apollo documentation: [Apollo Cyber Python README](https://github.com/ApolloAuto/apollo/blob/master/cyber/python/README.md).

6. Run the **main script**:  
   ```bash
   python PtoP.py

## Baselines
Baseline implementations are provided in the baseline/ folder.