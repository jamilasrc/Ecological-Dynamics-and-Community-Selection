# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:44:44 2024

@author: jamil
"""

######################

# Jamila: for console - cd "Documents/PhD for github/Ecological dynamics and community selection"

#########################

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from ast import literal_eval

from community_dynamics_and_properties import *
from community_selection_protocols_and_analysis import *

#################### Functions ##########################

def identify_community_phenotypes(data_groupby_ecodyn):
    
    breakpoint()
    
    total_comp_over_cycles = data_groupby_ecodyn.groupby(['Lineage'])['Final Composition'].apply(\
                                        lambda x : np.unique(np.concatenate(x)))

################################ Load Data Structures ##########################

le_across_interact_dists = pd.read_csv("testing_criteria_for_eco_dyn.csv",index_col=0)
community_selection_dynamics = pd.read_csv("community_selection_dynamics2.csv",index_col=0)

###

################### Plotting community function over selection cycles ######

community_selection_dynamics.groupby(['Species pool','Predicted dynamics (cycle 5)']).apply(identify_community_phenotypes)

fig, ax = plt.subplots()

sns.lineplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'chaos')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             linewidth=2.5,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)
             #,errorbar=('pi',100),err_style='bars')

sns.scatterplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'chaos')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             s=25,palette=sns.color_palette('viridis',n_colors=3))
    
plt.legend(title='Ecological dynamics',loc='best',labels=['Chaotic(-like)','Oscillations','Stable'])

##########

hue_order = community_selection_dynamics['Predicted dynamics (cycle 5)'].unique()
sns.set_style('whitegrid')

####

fig, ax = plt.subplots()

sns.lineplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'chaos')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)
             
sns.scatterplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'chaos')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             s=25,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3))

plt.xlabel('Selection Cycle',fontsize=14)
plt.ylabel('Community Function',fontsize=14)
plt.title('Species Pool 1',fontsize=18)
ax.legend([],[], frameon=False)

plt.savefig("Figures/communityfunction_chaos_speciespool.png", dpi=300)

#####

fig, ax = plt.subplots()

sns.lineplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'oscillations')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)
             #,errorbar=('pi',100),err_style='bars')

sns.scatterplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'oscillations')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             s=25,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3))

ax.legend([],[], frameon=False)
plt.xlabel('Selection Cycle',fontsize=14)
plt.ylabel('Community Function',fontsize=14)
plt.title('Species Pool 2',fontsize=18)
ax.legend([],[], frameon=False)

plt.savefig("Figures/communityfunction_osc_speciespool.png", dpi=300)


#####

community_selection_dynamics.iloc[\
                                  np.where((community_selection_dynamics['Species pool'] == 'stable')\
                                           & (community_selection_dynamics['Cycle'] == 10))][\
                                  ['Lineage','Community Function']]
# phenotype 1 = lineage 9
# phenotype 2 = else
                                                                                             
fig, ax = plt.subplots()

sns.lineplot(data=community_selection_dynamics.iloc[\
                                    np.where((community_selection_dynamics['Species pool'] == 'stable')\
                                             & (community_selection_dynamics['Lineage'] != 9))],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)

sns.lineplot(data=community_selection_dynamics.iloc[\
                                    np.where((community_selection_dynamics['Species pool'] == 'stable')\
                                             & (community_selection_dynamics['Lineage'] == 9))],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)
             
sns.scatterplot(data=community_selection_dynamics.iloc[\
                                    np.where(community_selection_dynamics['Species pool'] == 'stable')],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             s=25,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3))
    
ax.legend([],[], frameon=False)
plt.xlabel('Selection Cycle',fontsize=14)
plt.ylabel('Community Function',fontsize=14)
plt.title('Species Pool 3',fontsize=18)
ax.legend([],[], frameon=False)

plt.savefig("Figures/communityfunction_stable_speciespool.png", dpi=300)

##########

sns.set_style('white')
fig, ax = plt.subplots()

sns.lineplot(data=community_selection_dynamics.iloc[\
                                    np.where((community_selection_dynamics['Species pool'] == 'stable')\
                                             & (community_selection_dynamics['Lineage'] == 9))],
             x='Cycle',y='Community Function',hue='Predicted dynamics (cycle 5)',
             ax=ax,
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3),
             estimator='mean',errorbar=None)

plt.savefig("Figures/communityfunction_legend.png", dpi=300)
               
########### Variation ########

variation_comm = community_selection_dynamics.groupby(["Species pool",
                                                       "Predicted dynamics (cycle 5)",
                                                       "Cycle"])["Community Function"].var()
variation_comm = variation_comm.to_frame()
variation_comm.reset_index(inplace=True)
variation_comm.rename(columns={"Community Function":"Variation in Community Function"},inplace=True)

#### Plotting ####

sns.lineplot(data=variation_comm.iloc[np.where(variation_comm["Species pool"] == "stable")],
             x="Cycle",y="Variation in Community Function",hue="Predicted dynamics (cycle 5)")
