# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:07:21 2024

@author: jamil
"""

######################

# Jamila: for console - cd "Documents/PhD for github/Ecological dynamics and community selection"

#########################

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from copy import deepcopy
from ast import literal_eval

from community_dynamics_and_properties import *
from community_selection_protocols_and_analysis import *

################################## Functions ##########################

def identify_ecological_dynamics(data,le_mean_col,le_sigma_col,
                                 predicted_dynamics_col="Predicted dynamics"):
    
    stable_boundary = -1
    chaos_oscillations_boundary = np.round(0.1*np.sqrt(50),1)
    divergence_threshold = 15
    
    eco_dynamics = ["stable","oscillations","chaotic-like"]
    eco_dyn_conditions = [(data[le_mean_col] <= stable_boundary),
                          (data[le_mean_col] > stable_boundary) & \
                          (np.round(data[le_sigma_col],1) < chaos_oscillations_boundary),
                          (data[le_mean_col] > stable_boundary) & \
                          (np.round(data[le_sigma_col],1) >= chaos_oscillations_boundary)]
        
    data[predicted_dynamics_col] = np.select(eco_dyn_conditions,eco_dynamics)
    
    data.drop(data[data[le_mean_col] >= divergence_threshold].index,inplace=True)

    return data

################################ Tidying data structures #####################

le_across_interact_dists = pd.read_csv("testing_criteria_for_eco_dyn.csv")
community_selection_dynamics = pd.read_csv("community_selection_dynamics2.csv")

community_selection_chaotic = pd.read_pickle("community_selection_chaotic.pkl")
community_selection_oscillating = pd.read_pickle("community_selection_oscillating.pkl")
community_selection_stable = pd.read_pickle("community_selection_stable.pkl")

##################

# convert standard error to standard deviation
le_across_interact_dists["Lyapunov Exponent std error"] *= np.sqrt(50)
le_across_interact_dists.rename(columns={"Lyapunov Exponent std error":"Lyapunov Exponent std deviation"},
                                inplace=True)

#############

community_selection_dynamics["Lyapunov Exponent (cycle 5)"] = \
    community_selection_dynamics['Lyapunov Exponent (cycle 5)'].apply(lambda x: literal_eval(x) if "[" in x else x)
community_selection_dynamics["Lyapunov Exponent (cycle 12)"] = \
    community_selection_dynamics['Lyapunov Exponent (cycle 12)'].apply(lambda x: literal_eval(x) if "[" in x else x)

community_selection_dynamics[["Lyapunov Exponent mean (cycle 5)",
                              "Lyapunov Exponent std deviation (cycle 5)"]] = \
    pd.DataFrame(community_selection_dynamics['Lyapunov Exponent (cycle 5)'].to_list())
community_selection_dynamics[["Lyapunov Exponent mean (cycle 12)",
                              "Lyapunov Exponent std deviation (cycle 12)"]] = \
    pd.DataFrame(community_selection_dynamics['Lyapunov Exponent (cycle 12)'].to_list())

community_selection_dynamics.drop(['Lyapunov Exponent (cycle 5)','Lyapunov Exponent (cycle 12)'],
                                  axis=1,inplace=True)

community_selection_dynamics["Lyapunov Exponent std deviation (cycle 5)"] *= np.sqrt(50)
community_selection_dynamics["Lyapunov Exponent std deviation (cycle 12)"] *= np.sqrt(50)

############################# Assigning ecological dynamics ##################

identify_ecological_dynamics(community_selection_dynamics,
                             "Lyapunov Exponent mean (cycle 5)",
                             "Lyapunov Exponent std deviation (cycle 5)",
                             predicted_dynamics_col="Predicted dynamics (cycle 5)")

identify_ecological_dynamics(community_selection_dynamics,
                             "Lyapunov Exponent mean (cycle 12)",
                             "Lyapunov Exponent std deviation (cycle 12)",
                             predicted_dynamics_col="Predicted dynamics (cycle 12)")

############################## Assessing heritability and variation #############

############ Heritability ##########

no_communities = 20

heritability_comm = community_selection_dynamics.groupby(["Ecological Dynamics","Predicted dynamics (cycle 12)"])\
    [,,"Community Function"].apply(lambda x : heritability_during_selection(x,no_communities))

heritability_comm = heritability_comm.to_frame()
heritability_comm.reset_index(inplace=True)

heritability_comm_df = pd.DataFrame(np.vstack([np.concatenate([np.repeat(dyn,no_selection_cycles-1) \
                                                  for dyn in heritability_comm['Ecological Dynamics']]),
                                  np.concatenate(heritability_comm['Community Function']).T]).T,
                                  columns=["Ecological Dynamics","Cycle","Community Function Heritability"])

heritability_comm_df['Cycle'] = heritability_comm_df['Cycle'].astype(float).astype(int)
heritability_comm_df['Community Function Heritability'] = \
    heritability_comm_df['Community Function Heritability'].astype(float)
