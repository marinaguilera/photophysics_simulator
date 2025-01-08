# photophysics_simulator
Tools for computing the expected fluorescence signal from probes with complex photo-physics.  
The program solves analitically the kinetic rate equation of an arbitrarily complex network of interconnected electronic states.  
As an input for the simulation, the user should define the experimental pulse scheme, the kinetic matrix of the fluorophore and the characteristics of the detector.  
The program has been optimized to simulate the response of Reversibly Switchable Fluorescencent Proteins (RSFPs, specifically rsEGFP2) to different pulse schemes.  

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
This folder contains examples on how to perform `least_squares` fitting routines of two different simulated experiments. The folder inclues two example experimental datasets.  
  - `fitting_on_switching.py`. Fit of the fluorescence signal response to an increasing 405 nm power density ramp. The goal is to find the optimum value for a given parameter by comparing the experimental curve with the simulated one. `on_switching_data.txt` includes the normalised fluorescence intensity as well as the experimental standard deviation.  
  - `fitting_fatigue.py`. Fit of the fluorescence evolution upon 2000 photo-switching cycles. The goal is to find the optimum value for a given parameter by performing a global fit of the simulated experiment to the data. `fatigue_data_592_timing.mat` includes the normalised fluorescence intensity as well as the experimental standard deviation.
