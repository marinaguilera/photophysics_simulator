import numpy as np
from matplotlib import pyplot as plt
import photophysics_simulator_engine as pse
import photophysics_simulator_model as model

########################################################################################################################
# Example of On-Switching experiment.
# This script showcases how to model an on-switching experiment. The detector and fluorophore modules are imported from
# the photophysics_simulator_model. Nonetheless, one can create a detector and/or fluorophore in this same file
# and execute the experiment with such custom modules.

# Specifically, in this example we simulate the response of rsEGFP2 to different 405 nm power densities. The goal is to
# visualize how the on-switching curves change with the activation intensity.
########################################################################################################################
# Create the 405 nm power density array and initialize the on_switching_curve array that will store the signal
# evolution in the excitation window for each power density.
power_density_405 = np.array([0.01, 0.1, 0.5, 1, 1.5, 5])
on_switching_curve = np.zeros(power_density_405.shape[0])

# Create a sub-routine that will perform the on-switching experiment recursively for each of the input 405 nm power
# densities. In this case, the only parameter that is actively modified in each loop is the 405 nm power density,
# however, other illumination parameters could be adapted in each iteration by passing them as array with the same size
# as power_density_405. The signal trace for all input power densities will be stored in on_switching_curve.
for i in range (power_density_405.shape[0]):
    power_density_405_exp = power_density_405[i]
    on_switching_pulse = pse.ModulatedLasers(wavelengths=[405, 488],
                                             powerDensities=[power_density_405_exp, 0.2],
                                             pulseWidths=[0.5,1],
                                             tStart=[1, 2],
                                             dwellTime=[4])
    # Define the camera exposure time. Typically the camera exposure time is the same as the fluorescence excitation
    # window.
    model.camera.exposureTime = [2,3]

    on_switching_experiment = pse.experiment(illumination=on_switching_pulse,
                                             fluorophore= model.rsEGFP2_simple_model,
                                             detection= model.camera)

    on_switching_curve[i], populations, p0 = on_switching_experiment.detectorSignalIntegrated()

# Plotting
pse.plottingOnSwitching(power_density_405, on_switching_curve)


