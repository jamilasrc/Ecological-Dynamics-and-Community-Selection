# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:59:50 2024

@author: jamil
"""

################## Packages #############

import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import pandas as pd

from community_dynamics_and_properties_v2 import *

########################################

interation_matrix_chaos = np.load('chaos_09_005.npy')
interation_matrix_oscillations = np.load('oscillations_09_005.npy')
interation_matrix_stable = np.load('stable_09_005.npy')

########################################

no_species = 50
lineages = np.arange(5)
interact_func = 'random'
interact_func_args = {'mu_a':0.9,'sigma_a':0.05}

community_dynamics_chaos = community(no_species,
                                "fixed", None,
                                interact_func, interact_func_args,
                                usersupplied_interactmat=interation_matrix_chaos)
community_dynamics_chaos.simulate_community(30000,'Default',lineages,init_cond_func_name="Mallmin")
community_dynamics_chaos.calculate_lyapunov_exponents(lineages,separation=1e-3,dt=7000)
community_dynamics_chaos.diversity
community_dynamics_chaos.lyapunov_exponents

plt.plot(community_dynamics_chaos.ODE_sols['lineage 4'].t,
         community_dynamics_chaos.ODE_sols['lineage 4'].y.T)

community_dynamics_oscillations = community(no_species,
                                "fixed", None,
                                interact_func, interact_func_args,
                                usersupplied_interactmat=interation_matrix_oscillations)
community_dynamics_oscillations.simulate_community(30000,"Default",lineages,init_cond_func_name="Mallmin")
community_dynamics_oscillations.calculate_lyapunov_exponents(lineages,separation=1e-3,dt=7000)
community_dynamics_oscillations.diversity
community_dynamics_oscillations.lyapunov_exponents

plt.plot(community_dynamics_oscillations.ODE_sols['lineage 3'].t,
         community_dynamics_oscillations.ODE_sols['lineage 3'].y.T)


community_dynamics_stable = community(no_species,
                                "fixed", None,
                                interact_func, interact_func_args,
                                usersupplied_interactmat=interation_matrix_stable)
community_dynamics_stable.simulate_community(30000,"Default",lineages,init_cond_func_name="Mallmin")
community_dynamics_stable.calculate_lyapunov_exponents(lineages,separation=1e-3,dt=7000)
community_dynamics_stable.lyapunov_exponents

plt.plot(community_dynamics_stable.ODE_sols['lineage 0'].t,
         community_dynamics_stable.ODE_sols['lineage 0'].y.T)

#####################################

three_species_community_chaos = pd.read_pickle('three_species_community_chaos.pkl')

plt.plot(three_species_community_chaos.ODE_sols['lineage 0'].t[1000:2000],
         three_species_community_chaos.ODE_sols['lineage 0'].y[:,1000:2000].T)

three_species_community_chaos.calculate_lyapunov_exponents(np.arange(5),
                                                           separation=1e-3,dt=7000)
three_species_community_chaos.lyapunov_exponents
three_species_community_chaos.mu_a
three_species_community_chaos.sigma_a

three_species_community = community(3,"fixed", None, interact_func, {'mu_a':0.9,'sigma_a':0.15})
three_species_community.simulate_community(30000,'Default',lineages,init_cond_func_name="Mallmin")
three_species_community.calculate_lyapunov_exponents(lineages,separation=1e-3,dt=7000)
three_species_community.lyapunov_exponents

plt.plot(three_species_community.ODE_sols['lineage 1'].t[1000:2000],
         three_species_community.ODE_sols['lineage 1'].y[:,1000:2000].T)
