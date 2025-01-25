# photophysics_simulator
Tools for computing the expected fluorescence signal from fluorescent probes with complex photo-physics.  
The program solves analytically the kinetic rate equations of an arbitrarily complex network of interconnected electronic states.  
As an input for the simulation, the user should define the experimental pulse scheme, the kinetic matrix of the fluorophore and the characteristics of the detector.  
The program has been optimized to simulate the response of Reversibly Switchable Fluorescent Proteins (RSFPs, specifically rsEGFP2) to different pulse schemes.  

## Content
- `photophysics_simulator_engine.py`    
Kinetics solver given an arbitrary network of interconnected electronic states. This file contains all the methods necessary to run the simulation tool.

- `photophysics_simulator_model.py`  
Examples of fluorophore models that can be input in the program. The parameters declared in the model must be referenced in the fluorophore object defined in the engine.

- `experiments_type_example`  
This folder contains a complete description of how to simulate the three most common types of experiments in the characterization of RSFPs.
  - `off_switching_experiment.py`. Time-evolution of the fluorescence signal given an off-switching dose.  
  - `on_switching_experiment.py`. Fluorescence signal evolution given multiple on-switching doses.  
  - `fatigue_experiment.py`. Fluorescence signal evolution upon multiple photo-switching cycles.  
  
- `fitting_example`
This folder contains examples of how to perform `least_squares` fitting routines of two different simulated experiments. The folder includes two example experimental datasets.  
  - `fitting_on_switching.py`. Fit of the fluorescence signal response to an increasing 405 nm power density ramp. The goal is to find the optimum value for a given parameter by comparing the experimental curve with the simulated one. `on_switching_data.txt` includes the normalised fluorescence intensity as well as the experimental standard deviation.  
  - `fitting_fatigue.py`. Fit of the fluorescence evolution upon 2000 photo-switching cycles. The goal is to find the optimum value for a given parameter by performing a global fit of the simulated experiment to the data. `fatigue_data_592_timing.mat` includes the normalised fluorescence intensity as well as the experimental standard deviation.
 
## Installation and use
To use the code as standalone, download the program from the [releases page](https://github.com/marinaguilera/photophysics_simulator/releases) and open it in the development software of choice.      
The expected runtime for the scripts listed in `experiments_type_example` is < 1 min. The fitting routines in `fitting_example` are computationally heavier and the expected runtime depends on the parameter to fit.  
The code has been developed using PyCharm Community Edition 2021.3 and Python 3.9 as the Python interpreter. 
