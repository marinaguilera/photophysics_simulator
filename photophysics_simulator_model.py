import numpy as np
import photophysics_simulator_engine as pse

########################################################################################################################
# Model to execute photo-physics simulations. In this file one can create the different modules of the simulation:
# detector, fluorophore model. The modules defined here can be imported to other files when running the simulated
# experiments. This file includes an example of different fluorophore models: negative switchers and non-switching
# probes.

########################################################################################################################

# Create detector
camera = pse.Detector(exposureTime=[], scalingAmplitude=1E5, integrationTime=0.1)
point_detector = pse.Detector(exposureTime=[], scalingAmplitude=1E5, integrationTime=0.001)

########################################################################################################################
# Create the fluorophore model

# Negative switchers
# Simple photo-switching model
rsEGFP2_simple_model = pse.NegativeSwitchers(extincion_coeff_on=[5260, 51560],
                                       extincion_coeff_off=[22000, 60],
                                       wavelength=[405,488],
                                       lifetime_on=1.6E-6,
                                       lifetime_off=20E-9,
                                       qy_cis_to_trans_anionic=1.65E-2,
                                       qy_trans_to_cis_neutral=0.33,
                                       qy_fluorescence_on=0.35,
                                       initial_populations=[0, 0, 1, 0])

# Simple photo-switching model with bleaching via the triplet state
rsEGFP2_triplet_bleaching = pse.NegativeSwitchers(extincion_coeff_on=[5260, 51560, 0],
                                                 extincion_coeff_off=[22000,60,0],
                                                 wavelength=[405,488,592],
                                                 lifetime_on=1.6E-6,
                                                 lifetime_off=20E-9,
                                                 qy_cis_to_trans_anionic=1.6E-2,
                                                 qy_trans_to_cis_neutral=0.33,
                                                 qy_cis_to_trans_neutral=0.33,
                                                 qy_trans_to_cis_anionic=2e-3,
                                                 qy_fluorescence_on=0.35,
                                                 initial_populations=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 extincion_coeff_triplet=[ 0,10e3,0],
                                                 lifetime_triplet= 5,
                                                 lifetime_triplet_excited=1E-9,
                                                 inter_system_crossing= 2.5e-3,
                                                 reverse_inter_system_crossing= 0,
                                                 qy_bleaching_on=0,
                                                 qy_bleaching_off=0,
                                                 qy_bleaching_triplet=1e-4,
                                                 lifetime_deprot_cis=825e-3,
                                                 lifetime_prot_trans=48e-3,
                                                 pKa_cis= 5.9,
                                                 pH = 7.5,
                                                  nspecies=11)
# Full photo-switching model with bleaching via the triplet state and on-switching mediated by a long-lived intermediate
rsEGFP2_full_model = pse.NegativeSwitchers(extincion_coeff_on=[5260, 61560, 0],
                                           extincion_coeff_off=[22000, 60, 0],
                                           wavelength=[405, 488, 592],
                                           lifetime_on=1.6E-6,
                                           lifetime_off=20E-9,
                                           qy_cis_to_trans_anionic=1.73E-2,
                                           qy_trans_to_cis_neutral=0.33,
                                           qy_cis_to_trans_neutral=0.33,
                                           qy_trans_to_cis_anionic=2e-2,
                                           qy_fluorescence_on=0.35,
                                           initial_populations=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                           extincion_coeff_triplet=[2e3, 10e3, 7.5e3],
                                           lifetime_triplet=5,
                                           lifetime_triplet_excited=1E-9,
                                           inter_system_crossing=2.5e-3,
                                           reverse_inter_system_crossing= 2.5e-3,
                                           qy_bleaching_on= 0,
                                           qy_bleaching_off= .8e-5,
                                           qy_bleaching_triplet= .8e-3,
                                           qy_bleaching_triplet_exc= 0,
                                           lifetime_deprot_cis= 825e-3,
                                           lifetime_prot_trans=48e-3,
                                           pKa_cis=5.9,
                                           pH=7.5,
                                           qy_im_to_off= .1,
                                           qy_fluorescence_im= 0.35,
                                           lifetime_maturation= 5e-3,
                                           extincion_coeff_immature= [16e3, 28e3,0],
                                           nspecies=13)

########################################################################################################################
# Non-switching probes

# Triplet probe with 1 dark state and bleaching
triplet_probe_simple_model = pse.TripletProbes(extincion_coeff_exc=[30000],
                                               wavelength=[488],
                                               lifetime_exc1=2.1e-6,
                                               lifetime_dark1=8e-3,
                                               quantum_yield_fluo=0.57,
                                               quantum_yield_isc=0.05,
                                               quantum_yield_bleach=0,
                                               starting_populations=[1,0,0,0])
# Triplet probe with 2 dark state and bleaching
triplet_probe_complete_model = pse.TripletProbes(extincion_coeff_exc=[30000],
                                                wavelength=[488],
                                                lifetime_exc1=2.1e-6,
                                                lifetime_dark1=20e-3,
                                                quantum_yield_fluo=0.5,
                                                quantum_yield_isc=0.01,
                                                quantum_yield_bleach=1e-4,
                                                starting_populations=[1,0,0,0,0,0],
                                                extincion_coeff_dark=[1000],
                                                lifetime_exc2=1e-9,
                                                lifetime_dark2=1e-9,
                                                quantum_yield_risc= 2.875e-12,
                                                nspecies=6)

########################################################################################################################