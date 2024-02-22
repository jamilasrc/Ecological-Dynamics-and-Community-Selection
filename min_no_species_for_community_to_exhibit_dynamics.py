# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:39:21 2024

@author: Jamila
"""

################## Packages #############

import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import pandas as pd

from community_dynamics_and_properties_v2 import *
from community_selection_protocols_and_analysis import *

########################################

def min_no_species_for_dynamics(min_species,max_species,no_communities,no_lineages,t_end,
                                dynamics_function,**dynamics_function_args):
    
    print(dynamics_function_args['interact_func_args'])
    
    no_species_range = np.arange(min_species,max_species)
    
    lineages = np.arange(no_lineages)
    
    for no_species in no_species_range:
        
        no_species_with_dynamics = dynamics_function(lineages,no_species,no_communities,t_end,
                                                     **dynamics_function_args)
        
        if no_species_with_dynamics:
            
            print('Communities with ',str(no_species),' species can exhibit these dynamics.', end='\n')
            
            return [no_species,no_species_with_dynamics]
        
    print('No communities with ',str(min_species),'-',str(max_species),
          ' species exhibit these dynamics.', end='\n')
    

def find_chaos(lineages,no_species,no_communities,t_end,
               interact_func=None,interact_func_args=None,interaction_matrix=None):
    
    for comm in range(no_communities):
        
        community_dynamics = community(no_species,
                                        "fixed", None,
                                        interact_func, interact_func_args, usersupplied_interactmat=interaction_matrix)
        community_dynamics.simulate_community(t_end,"Default",lineages,init_cond_func_name="Mallmin")
        
        fluctuating_lineages = [lineage for lineage, fluctuations in community_dynamics.fluctuations.items() \
                               if fluctuations > 0.2]
            
        if fluctuating_lineages:
            
            fluctuating_lineage_dynamics = deepcopy(community_dynamics)
            
            fluctuating_lineage_dynamics.final_composition = \
                {lineage: community_dynamics.final_composition[lineage] for lineage in fluctuating_lineages}
                
            no_uniq_fluct_compositions, lineages_no = fluctuating_lineage_dynamics.unique_compositions()
            
            for lineage_no in lineages_no:
                
                
                lineage_period = find_period_ode_system(community_dynamics.ODE_sols['lineage ' + str(lineage_no)])
                
                if not np.isnan(lineage_period).all():
                
                    community_dynamics.calculate_lyapunov_exponents([lineage_no],
                                                          dt=np.nanmax(np.array(lineage_period)))
                    
                else:
                    
                    community_dynamics.calculate_lyapunov_exponents([lineage_no],
                                                          dt=10000)
                    
                le_lineage = list(deepcopy(community_dynamics.lyapunov_exponents).values())[0]
                
                if (-1 < le_lineage[0] < 15) & (le_lineage[1] >= 0.7):
                    
                    print('Chaos has been identified in lineage ',str(lineage_no),'.', end='\n')
                    print('This community exhibits chaos.', end='\n')
                
                    return community_dynamics
          
    print('No lineages are chaotic.', end='\n')

def find_multiple_regimes(lineages,no_species,
               interact_func=None,interact_func_args=None,interaction_matrix=None):
    
    community_dynamics = community(no_species,
                                    "fixed", None,
                                    interact_func, interact_func_args, usersupplied_interactmat=interaction_matrix)
    community_dynamics.simulate_community(t_end,"Default",lineages,init_cond_func_name="Mallmin")
    
    if community_dynamics.no_unique_compositions > 1:
        
        print('This community has multiple compositions.', end='\n')
        
        return community_dynamics
    
    else:
    
        print('This community has one composition.', end='\n')


def find_multistability(lineages,no_species,
               interact_func=None,interact_func_args=None,interaction_matrix=None):
    
    community_dynamics = community(no_species,
                                    "fixed", None,
                                    interact_func, interact_func_args, usersupplied_interactmat=interaction_matrix)
    community_dynamics.simulate_community(t_end,"Default",lineages,init_cond_func_name="Mallmin")
    
    stable_lineages = [lineage for lineage, fluctuations in community_dynamics.fluctuations.items() \
                           if fluctuations == False]
        
    stable_lineage_dynamics = deepcopy(community_dynamics)
    
    stable_lineage_dynamics.final_composition = \
        {lineage: community_dynamics.final_composition[lineage] for lineage in stable_lineages}
        
    no_uniq_stable_compositions = stable_lineage_dynamics.unique_compositions()[0]
    
    if no_uniq_stable_compositions > 1:
        
        print('This community may be multistable.', end='\n')
        
        return community_dynamics
    
    else:
        
        print('This community is not multistable.', end='\n')
        
    
################################# Main ################################        
   
interaction_distributions = generate_distribution([0.7,1.1], [0.05,0.2])

min_species = 3
max_species = 15

no_communities = 10
no_lineages = 5

t_end = 30000

min_species_for_chaos_per_dist = [[interact_dist,min_no_species_for_dynamics(min_species,
                                                     max_species, no_communities, no_lineages,
                                                     t_end, find_chaos,
                                                     **{'interact_func':'random','interact_func_args':interact_dist})] \
                                  for interact_dist in interaction_distributions]
    
three_species_community_chaos = min_species_for_chaos_per_dist[10][1][1]

plt.plot(three_species_community_chaos.ODE_sols['lineage 0'].t[1000:2000],
         three_species_community_chaos.ODE_sols['lineage 0'].y[:,1000:2000].T)

four_species_community_chaos1 = min_species_for_chaos_per_dist[13][1][1]
four_species_community_chaos2 = min_species_for_chaos_per_dist[15][1][1]

plt.plot(four_species_community_chaos1.ODE_sols['lineage 0'].t[1000:2000],
         four_species_community_chaos1.ODE_sols['lineage 0'].y[:,1000:2000].T)

plt.plot(four_species_community_chaos2.ODE_sols['lineage 0'].t[1000:2000],
         four_species_community_chaos2.ODE_sols['lineage 0'].y[:,1000:2000].T)

pickle_dump('three_species_community_chaos.pkl', three_species_community_chaos)
pickle_dump('four_species_community_chaos1.pkl', four_species_community_chaos1)
pickle_dump('four_species_community_chaos2.pkl', four_species_community_chaos2)


