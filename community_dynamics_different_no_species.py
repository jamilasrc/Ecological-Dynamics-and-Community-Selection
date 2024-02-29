# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:53:41 2024

@author: jamil
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from copy import deepcopy
from scipy.stats import linregress

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.discrete.discrete_model import MNLogit

from community_dynamics_and_properties_v2 import *

#################################################

min_species = 4
max_species = 50
no_species_to_test = np.arange(min_species,max_species,3)

no_communities = 6
no_lineages = 5
interaction_distribution = {'mu_a':0.9,'sigma_a':0.15}
t_end = 10000

community_dynamics_dict = {}

for no_species in no_species_to_test:
    
    print(str(no_species)+' species',end='\n')
    community_no_spec = []
    
    for community_i in range(no_communities):
        
        community_dynamics = community(no_species,"fixed", None,'random',interaction_distribution)
        community_dynamics.simulate_community(t_end,'Default',np.arange(no_lineages),
                                              init_cond_func_name='Mallmin')
        community_dynamics.calculate_lyapunov_exponents(np.arange(no_lineages),dt=t_end)
        
        community_no_spec.append(community_dynamics)
        print('Community ',str(community_i),' simulations are complete.',end='\n')
        
    community_dynamics_dict[str(no_species)+' species'] = community_no_spec
    
pickle_dump('community_dynamics_different_no_species.pkl', community_dynamics_dict)   
        
############################ Recalculating lyapunov exponents #######################

community_dynamics_dict = pd.read_pickle('community_dynamics_different_no_species.pkl')

for no_species, community_per_species in community_dynamics_dict.items():
    
    print(str(no_species),end='\n')
    
    for community_i in community_per_species:
        
        community_i.calculate_lyapunov_exponents(np.arange(no_lineages),dt=1000,
                                                 separation=1e-3)
        print('Community',str(community_i),'max. lyapunov exponents have been calculated.',
              end='\n')

pickle_dump('community_dynamics_different_no_species.pkl', community_dynamics_dict)   

no_species_column = np.repeat(no_species_to_test,no_communities*no_lineages)
community_column = np.tile(np.repeat(np.arange(no_communities),no_lineages),len(no_species_to_test))
lineage_column = np.tile(np.tile(np.arange(no_lineages),no_communities),len(no_species_to_test))
mean_lyapunov_exponent_column = []
std_lyapunov_exponent_column = []

#################### Dataframe construction and analysis ##################

for no_species, community_per_species in community_dynamics_dict.items():
    
    for i, community_i in enumerate(community_per_species):
        
        community_extract_le = deepcopy(community_i)
        
        for le in community_extract_le.lyapunov_exponents.values():
            
            mean_lyapunov_exponent_column.append(deepcopy(le[0]))
            std_lyapunov_exponent_column.append(deepcopy(le[1]))

community_dynamics_df = pd.DataFrame(np.stack((no_species_column,community_column,
                                               lineage_column,mean_lyapunov_exponent_column,
                                               std_lyapunov_exponent_column)).T,
                                     columns=['No_Species','Community','Lineage',
                                              'Le_mean','Le_std'])

sns.scatterplot(data=community_dynamics_df,x='No_Species',y='Le_mean')
res = linregress(community_dynamics_df['No_Species'],community_dynamics_df['Le_mean'])
plt.plot(community_dynamics_df['No_Species'],
         res.intercept + res.slope*community_dynamics_df['No_Species'],
         'r', label='fitted line')
res2 = linregress(community_dynamics_df['No_Species'].iloc[np.where(\
                                            community_dynamics_df['Le_mean'] < 0)],
           community_dynamics_df['Le_mean'].iloc[np.where(\
                                                       community_dynamics_df['Le_mean'] < 0)])

sns.scatterplot(data=community_dynamics_df,x='No_Species',y='Le_std')
linregress(community_dynamics_df['No_Species'],community_dynamics_df['Le_std'])

diversity_column = []

for no_species, community_per_species in community_dynamics_dict.items():
    
    for i, community_i in enumerate(community_per_species):
        
        community_extract_le = deepcopy(community_i)
        diversity_column.append(list(community_extract_le.diversity.values()))
        
community_dynamics_df['Diversity'] = np.concatenate(diversity_column)

linregress(community_dynamics_df['Le_mean']*1e4,community_dynamics_df['Diversity'])

res3 = ols('Diversity ~ No_Species + Le_mean',
              data=community_dynamics_df).fit()
print(res3.summary())
plt.plot(community_dynamics_df['No_Species'],
         res2.intercept + res2.slope*community_dynamics_df['No_Species'],
         'b', label='fitted line')
print(anova_lm(res3))

##################################### Labelling with ecological dynamics ##################

community_dynamics_df['Ecological_Dynamics'] = np.nan

community_ode_sol_as_list = list(community_dynamics_dict['49 species'][1].ODE_sols.values())

fig, axs = plt.subplots(3,2,figsize=(5,2))

for i, ax in enumerate(axs.flatten()):
    
    ax.plot(community_ode_sol_as_list[i].t,community_ode_sol_as_list[i].y.T)
    
plt.tight_layout()
plt.show()

plt.plot(community_ode_sol_as_list[4].t,
         community_ode_sol_as_list[4].y.T)

community_dynamics_df.loc[475:479,'Ecological_Dynamics'] = \
    ['stable','stable','stable','stable','stable']
    
community_dynamics_df.loc[np.where(community_dynamics_df['Ecological_Dynamics'] == 'chaos')[0],
                          'Ecological_Dynamics'] = 'chaotic-like'

community_dynamics_df.to_csv('community_dynamics_different_no_species.csv')

################################## More analysis #############################

sns.boxplot(data=community_dynamics_df,x='Ecological_Dynamics',y='Le_mean')

res_le_eco = ols('Le_mean ~ Ecological_Dynamics',data=community_dynamics_df).fit()
print(res_le_eco.summary())
print(anova_lm(res_le_eco))

print(pairwise_tukeyhsd(community_dynamics_df['Le_mean'],community_dynamics_df['Ecological_Dynamics']))

sns.boxplot(data=community_dynamics_df,x='Ecological_Dynamics',y='Le_std')

print(pairwise_tukeyhsd(community_dynamics_df['Le_std'],community_dynamics_df['Ecological_Dynamics']))

res_div_by_le = ols('Diversity ~ No_Species + Le_mean',data=community_dynamics_df).fit()
print(res_div_by_le.summary())
print(anova_lm(res_div_by_le))

res_div_by_eco = ols('Diversity ~ No_Species + Ecological_Dynamics',data=community_dynamics_df).fit()
print(res_div_by_eco.summary())
print(anova_lm(res_div_by_eco))

res_eco_le = MNLogit(community_dynamics_df['Ecological_Dynamics'],
                     community_dynamics_df[['Le_mean','Le_std']]).fit()
print(res_eco_le.summary())
print(anova_lm(res_eco_le))

ecological_dynamics_probabilites = res_eco_le.predict(community_dynamics_df[['Le_mean','Le_std']])
ecological_dynamics_probabilites.rename(columns={0:'chaotic-like',1:'oscillations',2:'stable'},
                                     inplace=True)

predicted_ecological_dynamics = np.ma.choose(np.argmax(ecological_dynamics_probabilites.to_numpy(),axis=1),
                                             np.array(['chaotic-like','oscillations','stable']))

percent_match = np.count_nonzero(predicted_ecological_dynamics == \
                                 community_dynamics_df['Ecological_Dynamics'].to_numpy()\
                                     )/len(predicted_ecological_dynamics)*100

####################

res_eco_lem = MNLogit(community_dynamics_df['Ecological_Dynamics'],
                     community_dynamics_df['Le_mean']).fit()

ecological_dynamics_probabilites2 = res_eco_lem.predict(community_dynamics_df[['Le_mean']])

predicted_ecological_dynamics2 = np.ma.choose(np.argmax(ecological_dynamics_probabilites2.to_numpy(),axis=1),
                                             np.array(['chaotic-like','oscillations','stable']))

percent_match2 = np.count_nonzero(predicted_ecological_dynamics2 == \
                                 community_dynamics_df['Ecological_Dynamics'].to_numpy()\
                                     )/len(predicted_ecological_dynamics2)*100
print(percent_match2,end='\n')

####################

res_eco_lestd = MNLogit(community_dynamics_df['Ecological_Dynamics'],
                     community_dynamics_df['Le_std']).fit()

ecological_dynamics_probabilites3 = res_eco_lestd.predict(community_dynamics_df[['Le_std']])

predicted_ecological_dynamics3 = np.ma.choose(np.argmax(ecological_dynamics_probabilites3.to_numpy(),axis=1),
                                             np.array(['chaotic-like','oscillations','stable']))

percent_match3 = np.count_nonzero(predicted_ecological_dynamics3 == \
                                 community_dynamics_df['Ecological_Dynamics'].to_numpy()\
                                     )/len(predicted_ecological_dynamics3)*100
print(percent_match3,end='\n')



