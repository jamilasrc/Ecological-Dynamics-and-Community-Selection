# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:13:48 2024

@author: Jamila
"""

################## Packages #############

import numpy as np
from scipy.integrate import solve_ivp
from scipy import stats
from copy import deepcopy
from matplotlib import pyplot as plt

########################################

################ Classes ########################

class community_parameters:
    
    '''
    
    Create parameters for generalised Lotka-Volterra models. 
        The class has methods for generating growth rates and interaction matrices.
    
    '''
    
    def __init__(self,
                 no_species,
                 growth_func_name,growth_args,
                 interact_func_name,interact_args,
                 usersupplied_growth,usersupplied_interactmat,
                 dispersal):
        
        self.no_species = no_species
        
        ###############
        
        if usersupplied_growth is None:
            
            if growth_args is not None:
            
                for key, value in growth_args.items():
                    
                    setattr(self,key,value)
                
            growth_func = {"fixed": self.growth_rates_fixed,
                           "normal": self.growth_rates_norm}[growth_func_name]
            
            self.growth_rates = self.call_func_noarg(growth_func)
            
        else:
            
            self.growth_rates = usersupplied_growth
            
        ############
        
        for key, value in interact_args.items():
            
            setattr(self,key,value)
        
        if usersupplied_interactmat is None:
            
            interaction_func = {"random": self.random_interaction_matrix,
                                "random normalised by K": 
                                    self.random_interaction_matrix_norm_by_K}[interact_func_name]
            
            self.interaction_matrix = self.call_func_noarg(interaction_func)
            
        else: 
            
            self.interaction_matrix = usersupplied_interactmat
            
            
        self.dispersal = dispersal
        
        
########### Class methods #################

    def call_func_noarg(self,func):
        
        return func()
    
    ###### Growth Rates ###########
    
    def growth_rates_norm(self):
        
        '''
        
        Draw growth rates for n species from normal(mu,sigma) distribution

        Parameters
        ----------
        mu_g : float
            Mean growth rate.
        sigma_g : float
            Standard deviation in growth rate.
        no_species : int
            Number of species (n).

        Returns
        -------
        growth_r : np.array of float64.
            array of growth rates for each species drawn from normal(mu_g,sigma_g).

        '''
        
        growth_r = self.mu_g + self.sigma_g*np.random.rand(self.no_species)
        
        return growth_r

    def growth_rates_fixed(self):
        
        '''
        
        Generate array of growth rates all fixed to 1.
        
        Parameters
        ----------
        no_species : int
            number of species.

        Returns
        -------
        growth_r : np.array of float64.
            array of growth rates, all entries = 1.0.

        '''
        
        growth_r = np.ones((self.no_species,))
        
        return growth_r
    
        
    ###### Interaction Matrix ######
            
    def random_interaction_matrix(self):
        
        '''
    
        Parameters
        ----------
        mu_a : float
            mean interaction strength.
        sigma_a : float
            interaction strength standard deviation.
         no_species : int
             number of species (n).
    
        Returns
        -------
        interact_mat : np.array of size (n,n)
            Interaction matrix. 
    
        '''
        
        # generate interaction matrix drawn from normal(mu_a,sigma_a)
        interact_mat = self.mu_a + self.sigma_a*np.random.randn(self.no_species,self.no_species)
        # set a_ij = -1 for i = j/self-interaction to prevent divergence
        np.fill_diagonal(interact_mat, 1)
        
        return interact_mat
    
    def random_interaction_matrix_norm_by_K(self):
        
        '''
        
        Random interaction matrix weighted by carrying capacity. From Hu et al. (2022) github
    
        Parameters
        ----------
        mu_a : float
            mean interaction strength.
        sigma_a : float
            interaction strength standard deviation.
         no_species : int
             number of species (n).
    
        Returns
        -------
        interact_mat_norm : np.array of size (n,n)
            Interaction matrix. 
    
        '''
        
        # Generate random interaction matrix
        interact_mat = self.mu_a + self.sigma_a*np.random.randn(self.no_species,self.no_species)
        
        # Generate species-specific carrying capacities (K)
        carrying_capacity = 1 + (1/12)*np.random.randn(self.no_species,)
        # Construct n x n matrix of K_i x K_j, 
        #   where i is a row index (species index) and j is a column index (species index)
        K_matrix = np.dot((1/carrying_capacity).reshape(len(carrying_capacity),1),
                          carrying_capacity.reshape(1,len(carrying_capacity)))
        
        # Element-wise multiplication of 
        interact_mat_norm = interact_mat * K_matrix
        
        # set a_ij = -1 for i = j/self-interaction to prevent divergence
    
        np.fill_diagonal(interact_mat_norm, 1) 
        
        return interact_mat_norm
    
    ######################
    
    def generate_community_function(self,usersupplied_community_function=None):
        '''
        
        '''
    
        if usersupplied_community_function is None:
            
            self.species_contribute_community_function = self.species_contribution_to_community_function()
        
        else: 
            
            self.species_contribute_community_function = usersupplied_community_function
        
       
    def species_contribution_to_community_function(self,
                                                   mu_contribution=0,sigma_contribution=1):
        
        '''
        
        Generate parameters for species contribution to community function, or species function.
            Inspired by Chang et al. (2021), "Engineering complex communities by directed evolution".
            All species had a fixed species function, rather than community function
            being emergent from dynamic mechanistic interactions.
            Species contribution to community function is drawn from 
            normal(mu_contribution,sigma_contribution)
            
        Parameters
        ----------
        no_species : int
            Number of species.
        mean_contribution : float
            Mean species function.
        function_std : float
            Standard deviation for species function.
        
        Returns
        -------
        species_function : np.array of floats, size (no_species,)
            Array of individual species functions, drawn from distribution normal(0,function_std).
        
        '''
        
        species_function = mu_contribution + sigma_contribution*np.random.randn(self.no_species)
        
        return species_function

################

class gLV:
    
    '''
    
    Run gLV simulations from initial conditions.
        Takes model parameters as arguments.
        Has class methods for generating initial species abundances and running gLV ODE simulations.
    
    '''
    
    def __init__(self,
                 community_parameters_object,
                 init_cond_func_name,
                 usersupplied_init_cond=None,
                 t_end = 7000): 
        
        self.growth_rates = community_parameters_object.growth_rates
        self.interaction_matrix = community_parameters_object.interaction_matrix
        
        dispersal = community_parameters_object.dispersal
        
        if usersupplied_init_cond is None:
        
            init_cond_func_info = {"Hu":{"func":self.initial_abundances_hu,
                                    "args":[community_parameters_object.no_species,
                                            community_parameters_object.mu_a]},
                              "Mallmin":{"func":self.initial_abundances_mallmin,
                                         "args":[community_parameters_object.no_species
                                                 ,community_parameters_object.dispersal]}}[init_cond_func_name]
            self.initial_abundances = self.initial_abundance(init_cond_func_info["func"],
                                                             init_cond_func_info["args"])
            
        else: 
            
            self.initial_abundances = usersupplied_init_cond
        
        self.ODE_sol = self.gLV_simulation(dispersal,t_end)
      
    ########## Functions for generating initial conditions ############
    
    def initial_abundance(self,init_cond_func,init_cond_args):
        
        '''
        
        Instruct class on which class method for generating initial conditions should be called.
        
        '''
        return init_cond_func(*init_cond_args)
    
    def initial_abundances_mallmin(self,no_species,dispersal):
        
        '''
        
        Generate initial species abundances, based on the function from Mallmin et al. (2023).
        
        Parameters
        ----------
        no_species : int
            Number of species.
        dispersal : float.
            Dispersal or migration rate.
        
        Returns
        -------
        np.array of float64, size (n,). Drawn from uniform(min=dispersal,max=2/no_species)
        
        '''
        
        return np.random.uniform(dispersal,2/no_species,no_species)

    def initial_abundances_hu(self,no_species,mu_a):
        
        '''
        
        Generate initial species abundances, based on the function from Hu et al. (2022).
        
        Parameters
        ----------
        no_species : int
            Number of species.
         mu_a : float
             mean interaction strength.
        
        Returns
        -------
        np.array of float64, size (n,). Drawn from uniform(min=0,max=2*mu_a)
        
        '''
        
        return np.random.uniform(0,2*mu_a,no_species)

    ####### Simulate dynamics ###############
    
    def gLV_simulation(self,dispersal,t_end):
        
        '''
        
        Simulate generalised Lotka-Volterra dynamics.

        Parameters
        ----------
        growth_r : np.array of float64, size (n,)
            Array of species growth rates.
        interact_mat : np.array of float64, size (n,n)
            Interaction maitrx.
        dispersal : float.
            Dispersal or migration rate.
        t_end : int or float
            Time for end of simulation.
        init_abundance : np.array of float64, size (n,)
            Initial species abundances.

        Returns
        -------
         OdeResult object of scipy.integrate.solve_ivp module
            (Deterministic) Solution to gLV ODE system.

        '''
        
        return solve_ivp(gLV_ode,[0,t_end],self.initial_abundances,
                         args=(self.growth_rates,self.interaction_matrix,dispersal))
    
    ########### Community properties #############
    
    def identify_community_properties(self,t_end=7000):
        
        ###### Calculate diversity-related properties ###########
        
        final_popdyn = self.species_diversity(1e-4,-1)
        
        self.final_diversity = final_popdyn[1]
        self.final_composition = np.concatenate((final_popdyn[0],
                                                 np.zeros(self.ODE_sol.y.shape[0]-self.final_diversity)))
        
        ########## Determine if the community is fluctuating ###############
       
        self.fluctuations = self.detect_fluctuations_timeframe(5e-2,[t_end-500,t_end])
        
    def species_diversity(self,extinct_thresh,ind):
        
        '''
        
        Calculate species diversity at a given time.
        
        Parameters
        ----------
        extinct_thresh : float
            Species extinction threshold.
        ind : int
            Index of time point to calculate species diversity (to find species populations at the right time)
        simulations : OdeResult object of scipy.integrate.solve_ivp module
            (Deterministic) Solution to gLV ODE system.

        Returns
        -------
        Species present, species diversity (no. species), species abundances

        '''
        
        # find species that aren't extinct aka species with abundances greater than the extinction threshold.
        present_species = np.where(self.ODE_sol.y[:,ind] > extinct_thresh)
        
        # calculate species diversity (aka no. of species that aren't extinct, 
        #   or the length of the array of present species abundances)
        diversity = present_species[0].shape[0]
         
        return [present_species[0],diversity]
        
    def detect_fluctuations_timeframe(self,fluctuation_thresh,timeframe,extinct_thresh=1e-4):
        
        '''
        
        Detect whether a community exhibts fluctuating dynamics within a given timeframe.

        Parameters
        ----------
        fluctuation_thresh : float
            The threshold for the average coefficient of variation, 
                which determines whether or not a community is fluctuating.
        timeframe : list of ints, length 2.
            The start and end time to detect fluctuations in.
        simulations : OdeResult object of scipy.integrate.solve_ivp module
            (Deterministic) Solution to gLV ODE system.
        extinct_thresh : float
            Species extinction threshold.

        Returns
        -------
        fluctuations_truefalse : TYPE
            DESCRIPTION.

        '''
        
        # deepy copy object attribute so the class object doesn't get mutated
        simulations_copy = deepcopy(self.ODE_sol)
        simulations_copy.y = simulations_copy.y[np.where(simulations_copy.y[:,-1] > extinct_thresh)[0],:]
        
        # find the indices of the nearest time to the times supplied in timeframe
        indices = find_nearest_in_timeframe(timeframe,simulations_copy.t)    
        
        # detect if a community exhibits fluctuating species dynamics
        proportion_of_species_fluctuating = \
            self.detect_fluctuations(fluctuation_thresh,indices,simulations_copy.y)
        
        return proportion_of_species_fluctuating

    def detect_fluctuations(self,fluctuation_thresh,indices,pop_dyn):
        
        '''
        
        Detect whether a community is fluctuating or not.
        Based on the function from Hu et al. (2022).
        
        The average coefficient of variation for each species is calculated 
            (standard deviation in species dynamics weighted by average species 
             abundance), then the proportion of the community exhibiting fluctuating dynamics is calculated. 

        Parameters
        ----------
        fluctuation_thresh : float
            The threshold for the average coefficient of variation, 
                which determines whether or not a community is fluctuating.
        indices : int
            Index of nearest times to the timeframe supplied.
        pop_dyn : .y from OdeResult object of scipy.integrate.solve_ivp module
            Population dynamics ONLY from (deterministic) solution to gLV ODE system.
        
        Returns
        -------
        Boolean
            True = the community is fluctuating, False = the community is at a fixed point.

        '''
        
        # calculate the standard deviation in species population dynamics, 
        #   normalised by average species abundance, for all species.
        average_variation_coeff_per_spec = np.std(pop_dyn[:,indices[0]:indices[1]],axis=1)/np.mean(pop_dyn[:,indices[0]:indices[1]],axis=1)
        
        # find the species with the average CV greater than the flucutation threshold
        species_fluctutating = np.where(average_variation_coeff_per_spec > fluctuation_thresh)[0]
        
        # find the proportion of species with fluctuating dynamics in the whole community.
        fluct_prop = len(species_fluctutating)/pop_dyn.shape[0]
        
        return fluct_prop
    ############# Community function ################
        
    def call_community_function(self,comm_parms_attr_contribution_comm_func):
        
        self.species_contribute_community_function = \
            comm_parms_attr_contribution_comm_func
        
        self.community_function = self.community_function_totalled_over_maturation()
      
    def community_function_totalled_over_maturation(self):
        
        '''
        
        Parameters
        ----------
        species_function : np.array of floats, size (no_species,)
            Species contribution to community function.
        species_abundances_over_time : .y attribute from OdeResult object of scipy.integrate.solve_ivp module
            Species abundances over time.

        Returns
        -------
        community_function : TYPE
            DESCRIPTION.

        '''
        
        summed_abundances = np.sum(self.ODE_sol.y,axis=1)
        
        community_function = np.sum(np.multiply(self.species_contribute_community_function,
                                                summed_abundances))
        
        return community_function

################

class community(community_parameters):
    
   def __init__(self,
                no_species,
                growth_func_name, growth_args,
                interact_func_name, interact_args,
                usersupplied_growth=None,usersupplied_interactmat=None,
                dispersal=1e-8):
       
       super().__init__(no_species,
                        growth_func_name, growth_args,
                        interact_func_name, interact_args,
                        usersupplied_growth,usersupplied_interactmat,
                        dispersal)
       
       self.initial_abundances = {}
       self.ODE_sols = {}
       
       ######### Emergent properties ###########
       self.no_unique_compositions = None
       
       self.final_composition = {}
       self.fluctuations = {}
       self.diversity = {}
       
   ########################
   
   def call_func(self,func_name,func_args):

        return func_name(*func_args)    
      
   def simulate_community(self,
                          func_name,lineages,
                          init_cond_func_name=None,array_of_init_conds=None):
       
       repeat_simulations_info = {"Default":{"func":self.repeat_simulations,
                                             "args":[lineages,init_cond_func_name]},
                                  "Supply initial conditions":{"func":self.repeat_simulations_supply_initcond,
                                                               "args":[lineages,
                                                               array_of_init_conds]}}[func_name]
       
       self.call_func(repeat_simulations_info["func"],repeat_simulations_info["args"])
    
   def simulate_community_with_community_function(self,
                                                  func_name,lineages,
                                                  init_cond_func_name=None,array_of_init_conds=None):
       
       self.community_functions = {}
       
       repeat_simulations_info = {"Default":{"func":self.repeat_simulations,
                                             "args":[lineages,init_cond_func_name,
                                                     "call_gLV_with_community_function"]},
                                  "Supply initial conditions":{"func":self.repeat_simulations_supply_initcond,
                                                               "args":[lineages,
                                                               array_of_init_conds,
                                                               "call_gLV_with_community_function"]}}[func_name]
       
       self.call_func(repeat_simulations_info["func"],repeat_simulations_info["args"])
        
   def repeat_simulations(self,
                          lineages,
                          init_cond_func_name,
                          call_gLV_func="call_gLV"):
       
       '''
       
       Run repeat gLV simulations from different initial conditions.
       
       '''
       
       call_gLV_info = {"call_gLV":self.call_gLV,
                        "call_gLV_with_community_function":self.call_gLV_with_community_function}[call_gLV_func]
       
       for lineage in lineages:
           
           self.call_func(call_gLV_info,[lineage,init_cond_func_name])
           
       self.no_unique_compositions = self.unique_compositions()
       
   def repeat_simulations_supply_initcond(self,
                                          lineages,
                                          array_of_init_conds,
                                          call_gLV_func="call_gLV"):
    
       '''
       
       Run repeat gLV simulations from different initial conditions.
       
       '''
       
       call_gLV_info = {"call_gLV":self.call_gLV,
                        "call_gLV_with_community_function":self.call_gLV_with_community_function}[call_gLV_func]
       
       for count, lineage in enumerate(lineages):
           
         self.call_func(call_gLV_info,[lineage,None,array_of_init_conds[:,count]])
           
       self.no_unique_compositions = self.unique_compositions()
       
   def call_gLV(self,
                lineage,
                init_cond_func_name=None,
                usersupplied_init_cond=None):
       
       '''
       
       Generate initial species abundances. run gLV, and extract community properties.
       
       '''
       
       ########## run gLV from initial conditions #######
       
       gLV_res = gLV(self,init_cond_func_name,usersupplied_init_cond)
       gLV_res.identify_community_properties()
       
       ############ Assign values to relevent class attributes ##############
       
       dict_key = "lineage " + str(lineage)
      
       self.initial_abundances[dict_key] = gLV_res.initial_abundances
       self.ODE_sols[dict_key] = gLV_res.ODE_sol
       
       self.final_composition[dict_key] = gLV_res.final_composition
       self.diversity[dict_key] = gLV_res.final_diversity
       self.fluctuations[dict_key] = gLV_res.fluctuations
           
   def call_gLV_with_community_function(self,
                                        lineage,
                                        init_cond_func_name=None,
                                        usersupplied_init_cond=None):
       
       '''
       
       Generate initial species abundances. run gLV, and extract community properties.
       Includes community function
       
       '''
       
       ########## run gLV from initial conditions #######
        
       gLV_res = gLV(self,init_cond_func_name,usersupplied_init_cond)
       gLV_res.identify_community_properties()
       gLV_res.call_community_function(self.species_contribute_community_function)
       
       ############ Assign values to relevent class attributes ##############
       
       dict_key = "lineage " + str(lineage)
      
       self.initial_abundances[dict_key] = gLV_res.initial_abundances
       self.ODE_sols[dict_key] = gLV_res.ODE_sol
       
       self.final_composition[dict_key] = gLV_res.final_composition
       self.diversity[dict_key] = gLV_res.final_diversity
       self.fluctuations[dict_key] = gLV_res.fluctuations
       
       self.community_functions[dict_key] = gLV_res.community_function
       
   def unique_compositions(self):
       
       '''
       
       Calculate the number of unique final species compositions in a community.
       
       Parameters
       ----------
       obj : community object of __main__ module
       
       Returns
       -------
       no_uniq_comp : int
           Number of unique compositions
       
       '''
       
       all_compositions = np.vstack(list(self.final_composition.values()))
       no_uniq_comp = len(np.unique(all_compositions,axis=0))
       
       return no_uniq_comp
    
   def repeat_lyapunov(self,
                       lineages):
 
    '''
    
    Run repeat gLV simulations from different initial conditions.
    
    '''
    
    self.lyapunov_exponents = {}
    
    for lineage in lineages:
        
        dict_key = "lineage " + str(lineage)
        
        self.lyapunov_exponents[dict_key] = self.gLV_lyapunov_exponent(dict_key)
  
   def gLV_lyapunov_exponent(self,dict_key,n=50,dt=7000,separation=1e-2,extinct_thresh=1e-4):
       
       log_d1d0_list = []
       
       pop_dyn = self.ODE_sols[dict_key].y
       initial_conditions = pop_dyn[:,np.round(pop_dyn.shape[1]/2,0).astype(int)]
       
       initial_conditions_no_sep = deepcopy(initial_conditions)
       
       initial_conditions_sep = deepcopy(initial_conditions)
       species_to_perturbate = np.where(initial_conditions_sep > extinct_thresh)[0][0]
       initial_conditions_sep[species_to_perturbate] += separation
        
       for step in range(n):
            
           gLV_res = gLV(self,None,initial_conditions_no_sep,t_end=dt)
           gLV_res_separation = gLV(self,None,initial_conditions_sep,t_end=dt)
           
           species_abundances_end = gLV_res.ODE_sol.y[:,-1]
           species_abundances_end_sep = gLV_res_separation.ODE_sol.y[:,-1]
           
           separation_dt = np.sqrt(np.sum((species_abundances_end - species_abundances_end_sep)**2))
           
           log_d1d0 = np.log(np.abs(separation_dt/separation))
           log_d1d0_list.append(log_d1d0)
           
           initial_conditions_no_sep = species_abundances_end
           initial_conditions_sep = species_abundances_end + \
               (separation/separation_dt)*(species_abundances_end_sep-species_abundances_end)
               
       max_lyapunov_exponent = mean_std_deviation(np.array(log_d1d0_list))
       
       return max_lyapunov_exponent
           
       
####################### Functions #####################

############################ gLV simulations ##################

def gLV_ode(t,spec,growth_r,interact_mat,dispersal):
    
    '''
    
    ODE system from generalised Lotka-Volterra model

    Parameters
    ----------
    t : float
        time.
    spec : float
        Species population dynamics at time t.
    growth_r : np.array of float64, size (n,)
        Array of species growth rates.
    interact_mat : np.array of float64, size (n,n)
        Interaction maitrx.
    dispersal : float.
        Dispersal or migration rate.

    Returns
    -------
    dSdt : np.array of float64, size (n,)
        array of change in population dynamics at time t aka dS/dt.

    '''
    
    dSdt = np.multiply(growth_r - np.matmul(interact_mat,spec), spec) + dispersal
    
    return dSdt

####################### Random Global Functions ###############

def generate_distribution(mu_maxmin,std_maxmin,mu_step=0.1,std_step=0.05):
    
    '''
    

    Parameters
    ----------
    mu_maxmin : TYPE
        DESCRIPTION.
    std_maxmin : TYPE
        DESCRIPTION.
    mu_step : TYPE, optional
        DESCRIPTION. The default is 0.1.
    std_step : TYPE, optional
        DESCRIPTION. The default is 0.05.

    Returns
    -------
    distributions : TYPE
        DESCRIPTION.

    '''
    
    mu_min = mu_maxmin[0]
    mu_max = mu_maxmin[1]
    
    std_min = std_maxmin[0]
    std_max = std_maxmin[1]
    
    mu_range = np.arange(mu_min,mu_max,mu_step)
    std_range = np.arange(std_min,std_max,std_step)

    mu_rep = np.repeat(mu_range,len(std_range))
    std_rep = np.tile(std_range,len(mu_range))
    
    distributions = np.vstack((mu_rep,std_rep)).T
    distributions = distributions.tolist()
    
    return distributions


    
def find_nearest_in_timeframe(timeframe,simulation_times):
    
    '''
    
    Find the index of the nearest times to those in timeframe 
        (for extracting population dynamics at a given time).
    

    Parameters
    ----------
    timeframe : list of ints or floats
        List of times.
    simulation_times : .t from OdeResult object of scipy.integrate.solve_ivp module
        Simulation times ONLY from (deterministic) solution to gLV ODE system.
    

    Returns
    -------
    indices : int
        indices of times in simulation_times with value

    '''
    
    indices = find_nearest_multivalues(timeframe,simulation_times)
    
    return indices

def find_nearest_multivalues(array_of_values,find_in):
    
    '''
    
    Find nearest value in array for multiple values. Vectorised.
    
    Parameters
    ----------
    array_of_values : np.array of floats or inds
        array of values.
    find_in : np.array of floats or inds
        array where we want to find the nearest value (from array_of_values).
    
    Returns
    -------
    fi_ind[sorted_idx-mask] : np.array of inds
        indices of elements from find_in closest in value to values in array_of_values.
    
    '''
     
    L = find_in.size # get length of find_in
    fi_ind = np.arange(0,find_in.size) # get indices of find_in
    
    sorted_idx = np.searchsorted(find_in, array_of_values)
    sorted_idx[sorted_idx == L] = L-1
    
    mask = (sorted_idx > 0) & \
    ((np.abs(array_of_values - find_in[sorted_idx-1]) < np.abs(array_of_values - find_in[sorted_idx])))
    
    return fi_ind[sorted_idx-mask]

def mean_stderror(data):
    
    '''
    
    Calculate the mean and standard error of a dataset.
    
    Parameters
    ----------
    data : np.array
        Dataset.
    Returns
    -------
    [mean, std_error] : list of floats
        The mean and standard error of data.
    
    '''
    
    mean = np.mean(data)
    
    std_error = stats.sem(data)
    
    return [mean, std_error]

def mean_std_deviation(data):
    
    '''
    
    Calculate the mean and standard error of a dataset.
    
    Parameters
    ----------
    data : np.array
        Dataset.
    Returns
    -------
    [mean, std_error] : list of floats
        The mean and standard error of data.
    
    '''
    
    mean = np.mean(data)
    
    std_deviation = np.std(data)
    
    return [mean, std_deviation]
