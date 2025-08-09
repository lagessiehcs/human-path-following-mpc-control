# Human Path Following MPC Control 
In this project, a differential drive mobile robot is controlled to follow a human path using Model Predictive Control (MPC). The optimization problem within the MPC framework is solved using [CasADi](https://web.casadi.org/), a symbolic framework for automatic differentiation and numerical optimization. This project contains two implementation versions, one in MATLAB and one in Python.

## Prerequisites
### MATLAB 
In the MATLAB implementation, the mobile robot is simulated using the [Mobile Robotics Simulation Toolbox](https://de.mathworks.com/matlabcentral/fileexchange/66586-mobile-robotics-simulation-toolbox) developed by [Mathworks](https://de.mathworks.com/).

Before running the code, ensure you have the following:
1. **[Mobile Robotics Simulation Toolbox](https://de.mathworks.com/matlabcentral/fileexchange/66586-mobile-robotics-simulation-toolbox)**: This toolbox is essential for the simulation. You can install it from the MATLAB Central File Exchange.

2. **[Phased Array System Toolbox](https://de.mathworks.com/products/phased-array.html )** for the rotz function.

3. **CasADi Package**: This project was tested with casadi-3.6.5, which can be downloaded [here](https://github.com/casadi/casadi/releases?page=3). Once downloaded, extract the package and place it in the MATLAB folder.

### Python

The following packages are required for the implementation in Python:
Windows:
````
python -m pip install -U pip 
pip install casadi
pip install numpy
python -m pip install matplotlib
python -m pip install scipy
````
Linux:
````
sudo apt update
sudo apt install python3-pip
pip install casadi
pip install numpy
pip install matplotlib
pip install scipy
````

## Run 
### MATLAB 
Open FollowControllerSimulator.m and run it.

### Python
Navigate to the Python folder and enter the following in the command line:
````
python3 follow_control_simulator.py
````
   
## Simulation Example
![Demo GIF](./Matlab/Animation/animated_plot.gif)

## Source
The MPC implementation in this project was inspired by the workshop [MPC and MHE implementation in Matlab using Casadi](https://www.youtube.com/watch?v=RrnkPrcpyEA).

