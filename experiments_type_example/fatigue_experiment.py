import numpy as np
from matplotlib import pyplot as plt
import photophysics_simulator_engine as pse
import photophysics_simulator_model as model

########################################################################################################################
# Example of a photo-switching fatigue experiment.
# This script showcases how to model a photo-switching fatigue experiment. The detector and fluorophore modules are
# imported from the photophysics_simulator_model. Nonetheless, one can create a detector and/or fluorophore in this same
# file and execute the experiment with such custom modules.
# Specifically, in this example we simulate the response of rsEGFP2 to different 488 nm power densities. The goal is to
# visualize how the fatigue curves change with the excitation intensity.
########################################################################################################################
# Create the 488 nm power density array and initialize the fatigue_curve array that will store the signal
# evolution in the excitation window for each power density. In this example, we are also decreasing the exposure time
# of the 488 nm illumination as we increase the 488 nm power density to keep a constant energy dose. One can define the
# looping variable as the lead_vector at the beginning of the script.
power_density_488 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
exposure_times = np.array([[3, 5.3], [3, 4.2], [3, 3.9], [3, 3.7], [3, 3.6], [3, 3.5], [3, 3.5], [3, 3.4], [3, 3.4],
                           [3,3.4]])
lead_vector = power_density_488
Ncycles = 2000
fatigue_curve = np.zeros((Ncycles, lead_vector.shape[0]))

# Create a sub-routine that will perform the fatigue experiment recursively for each of the input 488 nm power
# densities. In this case, as we recursively change the 488 nm power density other illumination parameters should be
# modified. One can define these parameters from the input arrays (power_density_488 and exposure_times).
for i in range (lead_vector.shape[0]):
    power_density_488_exp = power_density_488[i]
    exposure_times_exp = exposure_times[i]
    pulseWidth_488_exp = (exposure_times_exp[1]-exposure_times_exp[0])
    tStart_488_exp = exposure_times_exp[0]

    fatigue_pulse = pse.ModulatedLasers(wavelengths=[405, 488, 592],
                                    powerDensities=[0.1, power_density_488_exp, 0],
                                    pulseWidths=[1, pulseWidth_488_exp, 0],
                                    tStart=[1, tStart_488_exp, 0],
                                    dwellTime= 10)

    camera = pse.Detector(exposureTime=exposure_times_exp, scalingAmplitude=1.6E7,
                          integrationTime=0.1)

    fatigue_experiment = pse.experiment(illumination=fatigue_pulse,
                                    fluorophore=model.rsEGFP2_full_model,
                                    detection=camera)

    # Reset the initial populations for each iteration of the lead_vector loop
    fatigue_experiment.fluorophore.initial_populations = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for x in range (Ncycles):
        fatigue_curve [x,i], populations, p0 = fatigue_experiment.detectorSignalIntegrated()
        fatigue_experiment.fluorophore.initial_populations = p0


# Plotting
pse.plottingFatigue(Ncycles, fatigue_curve)



