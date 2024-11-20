import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import photophysics_simulator_engine as pse
import photophysics_simulator_model as model

########################################################################################################################
# Example of Off-Switching experiment.
# This script showcases how to model an off-switching experiment. The detector and fluorophore modules are imported from
# the photophysics_simulator_model. Nonetheless, one can create a detector and/or fluorophore in this same file
# and execute the experiment with such custom modules.

# Specifically, in this example we simulate the response of rsEGFP2 to different 488 nm power densities. The goal is to
# visualize how the off-switching curves change with the excitation intensity and perform a simple mono-exponential
# fitting.
########################################################################################################################
# Define the camera exposure time. Typically the camera exposure time is the same as the fluorescence excitation window.
model.camera.exposureTime = [2.5, 52.5]
time = np.arange(model.camera.integrationTime,
                 (model.camera.exposureTime[1]-model.camera.exposureTime[0] + model.camera.integrationTime),
                 model.camera.integrationTime)

# Create the 488 nm power density array and initialize the off_switching_curve array that will store the signal
# evolution in the excitation window for each power density.
power_density_488 = np.array([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10])
off_switching_curve = np.zeros((time.shape[0], power_density_488.shape[0]))

# Create a sub-routine that will perform the off-switching experiment recursively for each of the input 488 nm power
# densities. In this case, the only parameter that is actively modified in each loop is the 488 nm power density,
# however, other illumination parameters could be adapted in each iteration by passing them as array with the same size
# as power_density_488. The signal trace for all input power densities will be stored in off_switching_curve.
for i in range (power_density_488.shape[0]):
    power_density_488_exp = power_density_488[i]
    off_switching_pulse = pse.ModulatedLasers(wavelengths=[405, 488],
                                          powerDensities=[0.1, power_density_488_exp],
                                          pulseWidths=[1, 50],
                                          tStart=[0.1, 2.5],
                                          dwellTime=60)

    off_switching_experiment = pse.experiment(illumination=off_switching_pulse,
                                          fluorophore= model.rsEGFP2_simple_model,
                                          detection=model.camera)

    off_switching_curve[:,i], populations = off_switching_experiment.detectorSignalTimeResolved()

# Perform a monoexponential fit from curve_fit to the off-switching decay data with some added gaussian noise.
def monoExpFit(x_data, y_data):
    # Define the fitting function
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    # Intialize varaibles
    tau = np.zeros((1,y_data.shape[1]))
    y_data_plus_noise = np.zeros(y_data.shape)
    for k in range(y_data.shape[1]):
        rng = np.random.default_rng()
        noise = 0.02 * rng.normal(size=time.size)
        norm_ydata = np.divide(y_data, np.amax(y_data, axis=0))
        y_data_plus_noise[:,k] = norm_ydata[:,k] + noise
        popt, pcov = curve_fit(func, x_data, y_data_plus_noise[:,k])
        tau[:,k] = popt[1]
    return tau

tau = monoExpFit(time, off_switching_curve)

# Plotting. Can use the default plotting function in the engine or adapt to the particular experiment.
pse.plottingOffSwitching(time, off_switching_curve)
plt.show()










