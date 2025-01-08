import numpy as np
from numpy import linalg as linalg
from matplotlib import pyplot as plt
import photophysics_simulator_engine as pse
import scipy.io as sio
from scipy.optimize import least_squares
from cycler import cycler
########################################################################################################################
# Example of a on-switching experiment fitting.
# This script showcases how to model an on-switching experiment. Specifically, in this example we simulate the
# response of rsEGFP2 to different 405 nm power densities and fits a free parameter by comparing it to an experimental
# dataset.
########################################################################################################################
# Declare global variables
global system
global on_switching_curve_exp
global on_switching_curve_exp_err
# Import experimental data txt
def import_data_txt(path):
    on_switching_curve = np.loadtxt(path, usecols=(4))
    on_switching_curve_exp_err = np.loadtxt(path,usecols=(5))
    return on_switching_curve, on_switching_curve_exp_err

# Import data from txt
data_path = (r'on_switching_data.txt')
on_switching_curve_exp, on_switching_curve_exp_err = import_data_txt(data_path)
# Import data from mat file
# mat_fileName = (r'')
# mat_contents = sio.loadmat(mat_fileName)
# Extract the variables of interest from the matfile
# on_switching_curve_exp = mat_contents['norm_norm_on'][:,0]
# on_switching_curve_exp_err = mat_contents['norm_err_on'][:,0]
########################################################################################################################
# Methods
# Calculate standard deviation using jacobian matrix and residues array at the last convergence step from least_squares
def standard_deviation(jac, rss):
    df = jac.shape[0] - jac.shape[1]
    hessian_matrix = np.matmul(np.transpose(jac), jac)
    cov_matrix = (rss/df)*linalg.inv(hessian_matrix)
    std = np.sqrt(np.diag(cov_matrix))
    return std

# Calculate the confidence interval from the fitted parameter and the calculated standard deviation
def confidenceInterval(k, std):
    ci_95 = np.zeros((k.shape[0],2))
    for i in range(k.shape[0]):
        ci_95[i,:] = [k[i]*np.exp(-1.96*(std[i]/k[i])), k[i]*np.exp(1.96*(std[i]/k[i]))]
    return ci_95

# Calculate correlation matrix using jacobian matrix and residues array at the last convergence step from least_squares
def correlation_matrix(jac,rss):
    df = jac.shape[0] - jac.shape[1]
    hessian_matrix = np.matmul(np.transpose(jac), jac)
    cov_matrix = (rss / df) * linalg.inv(hessian_matrix)
    corr_matrix = np.zeros((cov_matrix.shape[0], cov_matrix.shape[1]))
    for i in range (cov_matrix.shape[0]):
        for j in range (cov_matrix.shape[1]):
            corr_matrix[i,j] = np.divide(cov_matrix[i,j],np.sqrt(cov_matrix[i,i]*cov_matrix[j,j]))
    return corr_matrix

# Initialize empty experiment with empty illumination, fluorophore and detection modules
system = pse.experiment(illumination=[],
                        fluorophore=[],
                        detection=[])

# Fitting model. Includes the modelling of the experiment, in this case a global fit of on switching curve.
def fitting_model(k, system=system):
    # Define model ingridients
    ingridients = model_ingridients(k)
    system.illumination = ingridients[0]
    system.detection = ingridients[2]
    system.fluorophore = ingridients[1]
    # On-Switching Experiment
    power_density_405 = np.array([0,0,0,0.007, 0.014, 0.034, 0.068, 0.13, 0.27, 0.48, 0.73, 0.97])
    on_switching_curve = np.zeros(power_density_405.shape[0])

    on_switching_experiment = pse.experiment(illumination=system.illumination,
                                             fluorophore= system.fluorophore,
                                             detection= system.detection)

    for i in range (power_density_405.shape[0]):
        # For each iteration of the loop pass the corresponding illumination parameters.
        power_density_405_exp = power_density_405[i]
        on_switching_experiment.illumination.powerDensities[0] = power_density_405_exp
        on_switching_curve[i], populations, p0 = on_switching_experiment.detectorSignalIntegrated()
        on_switching_experiment.fluorophore.initial_populations = p0
    return on_switching_curve, power_density_405

# Define parameters of fitting model
def model_ingridients(k):
    # Illumination scheme
    on_switching_pulse = pse.ModulatedLasers(wavelengths=[405, 488],
                                             powerDensities=[0.2, 0.12],
                                             pulseWidths=[1, 2],
                                             tStart=[1, 3],
                                             dwellTime=[7])
    # Fluorophore
    rsEGFP2_triplet_bleaching = pse.NegativeSwitchers(extincion_coeff_on=[5260, 61560],
                                           extincion_coeff_off=[22000, 60],
                                           wavelength=[405, 488],
                                           lifetime_on=1.6E-6,
                                           lifetime_off=20E-9,
                                           qy_cis_to_trans_anionic=1.73E-2,
                                           qy_trans_to_cis_neutral=.33,
                                           qy_cis_to_trans_neutral=.33,
                                           qy_trans_to_cis_anionic=1.73e-3,
                                           qy_fluorescence_on=0.35,
                                           initial_populations=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                           extincion_coeff_triplet=[2e3, 10e3],
                                           lifetime_triplet=5,
                                           lifetime_triplet_excited=1E-9,
                                           inter_system_crossing=2.5e-3,
                                           reverse_inter_system_crossing=2.5e-3,
                                           qy_bleaching_on=0,
                                           qy_bleaching_off=0,
                                           qy_bleaching_triplet=0,
                                           lifetime_deprot_cis=825e-3,
                                           lifetime_prot_trans=48e-3,
                                           pKa_cis=5.9,
                                           pH=7.5,
                                           qy_im_to_off= k[0],
                                           qy_fluorescence_im= 0.35,
                                           lifetime_maturation= 5e-3,
                                           extincion_coeff_immature= [16e3, 28e3],
                                           nspecies=13)
    # Detection
    camera = pse.Detector(exposureTime=[3, 5], scalingAmplitude=1.6e7, integrationTime=0.1)
    return on_switching_pulse, rsEGFP2_triplet_bleaching, camera

# Compute residuals between theoretical curve and experimental one. This function will be fed into least_squares.
def residuals (k, system=system, on_switching_data=on_switching_curve_exp, on_switching_err = on_switching_curve_exp_err):
    # Model
    on_switching_curve_theo, power_density_405 = fitting_model(k, system=system)
    # Optionally, substract the background.
    #on_switching_curve_theo = np.subtract(on_switching_curve_theo, np.min(on_switching_curve_theo))
    on_switching_curve_theo = np.divide(on_switching_curve_theo, np.max(on_switching_curve_theo))
    # Experimental data
    on_switching_exp = on_switching_data
    # Calculate residuals. Can add the experimental standard deviation as regularization to calculate the residuals
    residual = np.divide((on_switching_exp.flatten('F') - on_switching_curve_theo.flatten('F')), on_switching_err.flatten('F'))
    #residual = on_switching_exp.flatten('F') - on_switching_curve_theo.flatten('F')
    return residual
########################################################################################################################
# Fitting routine. k0 stores the initial guess for the fitting parameters. The fitting routine can be fine-tuned by
# adapting the lower and upper bounds of the fitting. The optimization is carried out by minimizing the least_squares
# regression comparing the experimental data to a simulated experiment with the input parameter. From fit_output we
# obtain the optimal value for the fitted parameter as well as the jacobian matrix and the residue array at the
# convergence step. These last two parameters are used to calculate the uncertainty associated to the fit in the form of
# standard deviation and confidence intervals. The correlation matrix between free parameters (if there are more than
# one free parameters to be fitted) can also be calculated.
k0 = [.09]
lb = [.01]
ub = [.33]
fit_output = least_squares(residuals, k0, bounds=(lb,ub))
k = fit_output['x']
jac = fit_output['jac']
res = fit_output['cost']
std = standard_deviation(jac, res)
#ci_95 = confidenceInterval(k, std)
#corr_matrix = correlation_matrix(jac, res)

# Run the model with optimized parameters
on_switching_curve, power_density_405 = fitting_model(k,system=system)
# Optionally, substract the background.
#on_switching_curve = np.subtract(on_switching_curve, np.min(on_switching_curve))
on_switching_curve = np.divide(on_switching_curve, np.max(on_switching_curve))

# Plotting
fig, (ax0,ax1) = plt.subplots(2,1,gridspec_kw={'height_ratios': [5,1]})
# b selects the number of points to plotted. In this case, we skip the second step where the 405 nm power density is
# equal to 0 kW/cm2
b = [0,2,3,4,5,6,7,8,9,10,11]
ax0.plot(power_density_405[b], on_switching_curve[b],marker='o')
ax0.errorbar(power_density_405[b], on_switching_curve_exp[b], yerr=on_switching_curve_exp_err[b],fmt='-o')
ax0.set_ylim(-0.1, 1.1)
ax0.set(title = 'On-Switching curve', xlabel = '405 power density (kW/cm^2)', ylabel = 'Norm. Fluorescence')
ax0.legend(['Simulation', 'Experimental data'], loc='lower right')
ax1.plot(power_density_405[b], on_switching_curve_exp[b]-on_switching_curve[b])
fig.tight_layout()

