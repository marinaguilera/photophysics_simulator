import numpy as np
from numpy import linalg as linalg
from matplotlib import pyplot as plt
import photophysics_simulator_engine as pse
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from cycler import cycler
np.seterr(divide='ignore', invalid='ignore')

########################################################################################################################
# Example of a photo-switching fatigue experiment fitting.
# This script showcases how to model a photo-switching fatigue experiment. Specifically, in this example we simulate the
# response of rsEGFP2 to an additional 592 nm co-illumination pulse. The time delay between the on-set of 592 nm
# illumination and the 488 nm pulse is scanned between different values. The routine fits a free parameter by comparing
# the simulated experiment to an experimental dataset.

# The experimental parameters are listed below:
# 405 nm pulse: tStart = 1.0 ms, powerDensity = 240 W/cm2, pulseWidth = 1.0 ms
# 488 nm pulse: tStart = 3.0 ms, powerDensity = 300 W/cm2, pulseWidth = 0.9 ms
# 592 nm pulse: tStart = [3, 4, 5, 8.5, 2, -], powerDensity = 4.00 or 0.00 kW/cm2, pulseWidth = 1.0 ms
# camera exposure time = [3, 3.9]
########################################################################################################################
# Import experimental data from mat file
mat_fileName = (r'fatigue_data_592_timing.mat')
mat_contents = sio.loadmat(mat_fileName)
# Declare global variables
global system
global fatigue_data
# Extract the variable meanF from mat file
fatigue_data = mat_contents['fatigue_norm']
########################################################################################################################
# Methods
# Calculate standard deviation from jacobian matrix from least_squares
def standard_deviation(jac, rss):
    df = jac.shape[0] - jac.shape[1]
    hessian_matrix = np.matmul(np.transpose(jac), jac)
    cov_matrix = (rss/df)*linalg.inv(hessian_matrix)
    std = np.sqrt(np.diag(cov_matrix))
    return std

# Initialize empty experiment with empty illumination, fluorophore and detection modules
system = pse.experiment(illumination=[],
                        fluorophore=[],
                        detection=[])

# Fitting model. Includes the modelling of the experiment, in this case a global fit of fatigue curves at
# different experimental conditions.
def fitting_model(k, system=system):
    # Define model ingridients
    ingridients = model_ingridients(k)
    system.illumination = ingridients[0]
    system.detection = ingridients[2]
    system.fluorophore = ingridients[1]

    # Fatigue experiment
    Ncycles = 2000

    # Many variants of fatigue experiment. The loop is defined depending on the parameters that are being changed.
    # Define the lead_vector as the one to loop over.
    power_density_592 = np.array([4, 4, 4, 4, 4, 0])
    tStart_592 = np.array([3, 4, 5, 8.5, 2, 3])
    lead_vector = power_density_592
    fatigue_curve = np.zeros((Ncycles, lead_vector.shape[0]))

    for i in range (lead_vector.shape[0]):
        power_density_592_exp = power_density_592[i]
        tStart_592_exp = tStart_592[i]

        fatigue_experiment = pse.experiment(illumination=system.illumination,
                                            fluorophore=system.fluorophore,
                                            detection=system.detection)

        # For each iteration of the loop pass the corresponding illumination parameters. Reset the initial populations
        # for each iteration of the lead_vector loop
        fatigue_experiment.illumination.powerDensities[-1] = power_density_592_exp
        fatigue_experiment.illumination.tStartLambda[-1] = tStart_592_exp
        fatigue_experiment.fluorophore.initial_populations = [0,0,1,0,0,0,0,0,0,0,0,0,0]

        for x in range(Ncycles):
            fatigue_curve[x, i], populations, p0 = fatigue_experiment.detectorSignalIntegrated()
            fatigue_experiment.fluorophore.initial_populations = p0
    return fatigue_curve, Ncycles

# Define parameters of fitting model
def model_ingridients(k):
    # Illumination scheme
    fatigue_pulse = pse.ModulatedLasers(wavelengths=[405, 488, 592],
                                        powerDensities=[.24,.3, 4],
                                        pulseWidths=[1,0.9,1],
                                        tStart=[1, 3, 0],
                                        dwellTime=[10])
    # Fluorophore. k is an array for the free parameters to fit.
    rsEGFP2_triplet_bleaching = pse.NegativeSwitchers(extincion_coeff_on=[5260, 61560, 0],
                                           extincion_coeff_off=[22000, 60, 0],
                                           wavelength=[405, 488, 592],
                                           lifetime_on=1.6E-6,
                                           lifetime_off=20E-9,
                                           qy_cis_to_trans_anionic=1.73E-2,
                                           qy_trans_to_cis_neutral=0.33,
                                           qy_cis_to_trans_neutral=0.33,
                                           qy_trans_to_cis_anionic=1.73e-3,
                                           qy_fluorescence_on=0.35,
                                           initial_populations=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                           extincion_coeff_triplet=[2e3, 10e3, 7.5e3],
                                           lifetime_triplet=5,
                                           lifetime_triplet_excited=1E-9,
                                           inter_system_crossing=2.5e-3,
                                           reverse_inter_system_crossing=k[0],
                                           qy_bleaching_on=0,
                                           qy_bleaching_off= 1e-5,
                                           qy_bleaching_triplet=1.2e-3,
                                           qy_bleaching_triplet_exc= 5e-7,
                                           lifetime_deprot_cis=.825,
                                           lifetime_prot_trans=48e-3,
                                           pKa_cis=5.9,
                                           pH=7.5,
                                           qy_im_to_off= .12,
                                           qy_fluorescence_im= 0.35,
                                           lifetime_maturation= 5e-3,
                                           extincion_coeff_immature= [16e3, 28e3,0],
                                           nspecies=13)
    # Detection
    camera = pse.Detector(exposureTime=[3, 3.9], scalingAmplitude=1.6e7, integrationTime=0.1)
    return fatigue_pulse, rsEGFP2_triplet_bleaching, camera

# Compute residuals between theoretical curve and experimental one. This function will be fed into least_squares.
bias = 1
def residuals (k, system=system, fatigue_data=fatigue_data, bias=bias):
    # Model
    fatigue_theo, Ncycles = fitting_model(k,system=system)
    fatigue_theo = np.divide(fatigue_theo, fatigue_theo[0,:])
    # Experimental data
    fatigue_data_exp = fatigue_data[0:Ncycles,:]
    # Calculate residuals
    residual = (fatigue_data_exp.flatten('F') - fatigue_theo.flatten('F'))
    return residual
########################################################################################################################
# Plotting functions
def plot_fatigue(Ncycles, data_theo, data_exp):
    cycles = np.arange(1, Ncycles+1)
    custom_cycler = cycler(color=['b','g','r','c','m', 'y']) #,'m','y'])
    fig, ax = plt.subplots()
    ax.set_prop_cycle(custom_cycler)
    ax.plot(cycles, data_theo)
    ax.plot(cycles, data_exp, marker='.', linestyle = 'None')
    ax.set(title='Fatigue Curve', xlabel='# Cycles', ylabel='Normalized Signal')
    plt.ylim([0,1])
    plt.show()
########################################################################################################################
# Fitting routine. k0 stores the initial guess for the fitting parameters. The fitting routine can be fine-tuned by
# adapting the lower and upper bounds of the fitting. The optimization is carried out by minimizing the least_squares
# regression comparing the experimental data to a simulated experiment with the input parameter. From fit_output we
# obtain the optimal value for the fitted parameter as well as the jacobian matrix and the residue array at the
# convergence step. These last two parameters are used to calculate the uncertainty associated to the fit in the form of
# standard deviation and confidence intervals.
k0 = [2.5e-3]
lb = [1e-8]
ub = [5e-1]
fit_output = least_squares(residuals, k0, bounds=(lb,ub))
k = fit_output['x']
jac = fit_output['jac']
res = fit_output['cost']
std = standard_deviation(jac, res)
print(k, std, res)

#Run model with the optimized parameters
fatigue_curve, Ncycles = fitting_model(k0, system)
fatigue_curve = np.divide(fatigue_curve, fatigue_curve[0,:])

# Select experimental data according to Ncycles
fatigue_data_fit = fatigue_data[0:Ncycles,:]
fatigue_data_fit = np.divide(fatigue_data_fit, fatigue_data_fit[0,:])

# Plotting data
plot_fatigue(Ncycles, fatigue_curve, fatigue_data_fit)
custom_cycler = cycler(color=['b','g','r','c','m','y']) #,'m','y'])
fig, ax = plt.subplots()
ax.set_prop_cycle(custom_cycler)
ax.plot(fatigue_data_fit-fatigue_curve)
plt.show()