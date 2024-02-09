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

########## Create Legend ##################

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

################ Zoom in on cycle 12 ##################

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
ax.set_xticks(range(len(community_selection_dynamics['Cycle'].unique())),labels=range(15))
ax.legend([],[], frameon=False)
ax.set_xlim(10,14)
ax.set_ylim(-450,-350)

plt.savefig("Figures/communityfunction_chaos_zoominon12.png", dpi=300)

               
##################### Variation #####################

variation_comm = community_selection_dynamics.groupby(["Species pool",
                                                       "Predicted dynamics (cycle 5)",
                                                       "Cycle"])["Community Function"].var()
variation_comm = variation_comm.to_frame()
variation_comm.reset_index(inplace=True)
variation_comm.rename(columns={"Community Function":"Variation in Community Function"},inplace=True)

variation_comm.replace({'Species pool':{'chaos':'1','oscillations':'2','stable':'3'}},
                       inplace=True)

#### Plotting ####

hue_order = community_selection_dynamics['Predicted dynamics (cycle 5)'].unique()
sns.set_style('whitegrid')

ax = sns.lineplot(data=variation_comm,
             x="Cycle",y="Variation in Community Function",hue="Predicted dynamics (cycle 5)",
             style='Species pool',
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3))
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.xlabel('Selection Cycle',fontsize=14)
plt.ylabel('Variation in Community Function',fontsize=14)
plt.title('Variation in community function across ecological dynamics',fontsize=18)
plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

plt.savefig("Figures/variation_in_communityfunction.png", dpi=300)

############################### Heritability ################################

no_cycles = 15

heritability_comm = community_selection_dynamics.groupby(["Species pool","Predicted dynamics (cycle 5)"])\
    ["Community Function"].apply(lambda x : heritability_during_selection(x,no_cycles))
heritability_comm = heritability_comm.to_frame()
heritability_comm.reset_index(inplace=True)

groups, g_counts = np.unique(heritability_comm["Species pool"],return_counts=True)

heritability_comm_df = pd.DataFrame( \
                                    np.vstack([\
                                    np.concatenate([np.repeat(group,(g_count*(no_cycles-1)).astype(int)) for group, g_count in zip(groups,g_counts)]),
                                    np.concatenate([np.repeat(dyn,no_cycles-1) for dyn in heritability_comm['Predicted dynamics (cycle 5)']]),
                                    np.concatenate(heritability_comm['Community Function']).T]).T,
                                    columns=["Species pool",'Predicted dynamics (cycle 5)',
                                             "Cycle","Community Function Heritability"])   

heritability_comm_df['Cycle'] = heritability_comm_df['Cycle'].astype(float).astype(int)
heritability_comm_df['Community Function Heritability'] = \
    heritability_comm_df['Community Function Heritability'].astype(float)
heritability_comm.replace({'Species pool':{'chaos':'1','oscillations':'2','stable':'3'}},
                       inplace=True)

#### Plotting ####

hue_order = community_selection_dynamics['Predicted dynamics (cycle 5)'].unique()
sns.set_style('whitegrid')

ax = sns.lineplot(data=heritability_comm_df,
             x="Cycle",y="Community Function Heritability",hue="Predicted dynamics (cycle 5)",
             style='Species pool',
             linewidth=2.5,hue_order=hue_order,palette=sns.color_palette('viridis',n_colors=3))
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.xlabel('Selection Cycle',fontsize=14)
plt.ylabel('Heritability of Community Function',fontsize=14)
plt.title('Heritability community function across ecological dynamics',fontsize=18)
plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

plt.savefig("Figures/heritability_of_communityfunction.png", dpi=300)
