###This module is a module like exo.py but an exogenous mortality factor affects both x and y.
###It's the generalisation of exo.py. It can be used instead of exo.py with only setting func_h_y as return_zero_y.
###It's even possible to have no harvesting by setting both func_h_x and func_h_y as respectively return_zero_x and return_zero_y.
###It's even possible to have a generalisation of models.py
###by setting func_h_x as return_zero_x, E_y as -mu*T and func_h_y as return_one_y.

#Dependancies
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import Callable
from matplotlib.colors import TwoSlopeNorm

#1 Pre-implemented functions that can be used in the models (like in exo.py)

##1.1 Functions with both x and y as two first arguments
def identity(x, y, z: float):
    '''This function returns the argument z itself
    x: pest population that is not used by the function but will be needed for the general model
    y: predator population that is not used by the function but will be needed for the general model
    z: the argument will be returned'''
    return z

def id_plus_E(x, y, z:float, E:float):
    '''This function returns the argument z with E substracted from it.
    It is used for example to have r - E_x or m - E_y
    x: pest population that is not used by the function but will be needed for the general model
    y: predator population that is not used by the function but will be needed for the general model
    z: the first term
    E: the term substracted from z'''
    return z + E

def multiply_x(x, y, z:float):
    '''This function multiplies x and z
    x: pest population 
    y: predator population that is not used by the function but will be needed for the general model
    z: the factor by which x is multiplied'''
    return z * x

def return_one(x,y):
    '''This function returns 1
    x: pest population that is not used by the function but will be needed for the general model
    y: predator population that is not used by the function but will be needed for the general model
    '''
    return 1

def return_zero(x,y):
    '''This function returns 0
    x: pest population that is not used by the function but will be needed for the general model
    y: predator population that is not used by the function but will be needed for the general model
    '''
    return 0

def return_x(x,y):
    '''This function returns x when it's sometimes needed
    x: pest population 
    y: predator population that is not used by the function but will be needed for the general model
    '''
    return x

def return_y(x,y):
    '''This function returns y when it's sometimes needed
    x: pest population that is not used by the function but will be needed for the general model
    y: predator population 
    '''
    return y

def logistic_model(
    x,
    y,
    r: float,
    K: float        
):
    '''This model is a continuous model that describes a logistic growth 
    
    Param:
        x: the initial value of x
        y: predator population that is not used by the function but will be needed for the general model
        r: growth rate
        K: carrying capacity
        a: search rate
        
    Return:
        dx: a list of the population size of x'''
    
    #Continuous part of the model
    dx = r * (1 - x/K)

    return dx

def no_int_f(
      x,
      y,
      a: float,
      c: float  
):
    '''This function is the functional response of the no-interaction model
    
    Param:
        x: pest population 
        y: predator population
        a: search rate
        c: half-saturation constant'''
    
    return a*x/(c + x)

def bda_f(
        x,
        y,
        a: float,
        c: float,
        b: float
):
    '''This function is the functional response of Beddington-DeAngelis model
    
    Param:
        x: pest population 
        y: predator population
        a: search rate
        c: half-saturation constant
        b: penalty coefficient of the predator efficiency'''
    
    return a*x/(c + x + b*y)

def squabbling_m(
        x,
        y,
        m: float,
        q: float
        
):
    '''This function is the mortality functin of squabbling model
    
    Param:
        x: pest population 
        y: predator population
        m: death rate
        q: squabbling coefficient'''
    
    return m + q*y

##1.2 Functions with x as first argument but not y
def identity_x(x, z: float):
    '''This function returns the argument z itself
    x: pest population that is not used by the function but will be needed for the general model
    z: the argument will be returned'''
    return z

def return_zero_x(x):
    '''This function returns 0
    x: pest population that is not used by the function but will be needed for the general model
    '''
    return 0

def return_one_x(x):
    '''This function returns 1
    x: pest population that is not used by the function but will be needed for the general model
    '''
    return 1

def return_x_x(x):
    '''This function returns x when it's sometimes needed
    x: pest population 
    '''
    return x

def multiply_x_x(x, z:float):
    '''This function multiplies x and z
    x: pest population 
    z: the factor by which x is multiplied'''
    return z * x

def logistic_model_x(
    x,
    r: float,
    K: float        
):
    '''This model is a continuous model that describes a logistic growth 
    
    Param:
        x: the initial value of x
        r: growth rate
        K: carrying capacity
        
    Return:
        dx: a list of the population size of x'''
    
    #Continuous part of the model
    dx = r * (1 - x/K)

    return dx

def logistic_sub_E_x(
    x,
    r: float,
    K: float,
    E_x:float
):
    '''This function is the logistic growth with an additional term E_x substracted from it.
    It can even be used instead of the basic logistic model with E_x = 0
    
    Param:
        x: pest population
        r: growth rate
        K: carrying capacity
        E_x: substracted term (taking effort)
        
    '''

    return r * (1 - x/K) - E_x

##1.3 Functions with y as first argument but not x
def identity_y(y, z: float):
    '''This function returns the argument z itself
    y: predator population that is not used by the function but will be needed for the general model
    z: the argument will be returned'''
    return z

def return_one_y(y):
    '''This function returns 1
    y: predator population that is not used by the function but will be needed for the general model
    '''
    return 1

def return_zero_y(y):
    '''This function returns 0
    y: predator population that is not used by the function but will be needed for the general model
    '''
    return 0

def return_y_y(y):
    '''This function returns y when it's sometimes needed
    y: predator population 
    '''
    return y

def multiply_y__y(y, z:float):
    '''This function multiplies x and z
    y: predator population 
    z: the factor by which y is multiplied'''
    return z * y

#2 Functions to solve the model. 

##2.1 Generality

def predator_prey_model(
    xyI,
    t,
    gamma: float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float],
):
    '''This function aims to be a general predator-prey model where the user decides on the functions that will be used in the model
    
    Param:
        xyI: a list of values of [x,y,I] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        func_g: the growth rate function. The first argument of func_g is x
        kwargs_g: a dictionnary of the other arguments of func_g
        func_f: the response function. The first two arguments of func_f are x and y
        kwargs_f: a dictionnary of the other arguments of func_f
        func_m: mortality rate function. The first two arguments of func_m are x and y
        kwargs_m: a dictionnary of the other arguments of func_m

    Example of run:
        predator_prey_model(
        xy=xy,
        t=t,
        gamma=gamma,
        func_g=identity_x,
        kwargs_g={'z': r},
        func_f=multiply_x,
        kwargs_f={'z': a} 
        func_m=identity,
        kwargs_m={'z': m}
        )

    Return: 
        dx, dy: a list of the two population size of x and y at time t_{n+1}'''
    
    #Initialisation
    x = xyI[0]
    y = xyI[1]
    I = xyI[2]

    #Continuous part of the model
    dx = func_g(x, **kwargs_g)*x - func_f(x, y, **kwargs_f)*y
    dy = gamma * func_f(x, y, **kwargs_f) * y - func_m(x, y, **kwargs_m)*y
    dI = x
    
    return dx, dy, dI

def solve_predator_prey_model(
    xyI,
    t,
    gamma: float,
    E_x: float,
    E_y: float,
    T: float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float],
    func_h_x: Callable[..., float],
    kwargs_h_x: dict[str, float], 
    func_h_y: Callable[..., float],
    kwargs_h_y: dict[str, float],
    t_0: float,
    t_n: float
):
    '''This function aims to be a general predator-prey model where the user decides on the functions that will be used in the model.
    The first impulsion is at t=T
    
    Param:
        xyI: a list of values of [x,y,I] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_x: taking effort for pests
        E_y: taking effort for predators
        T: release period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        func_h_x: harvesting function for x
        kwargs_h_x: a dictionnary of the arguments of the other arguments of func_h that are not x(nT). x(nT) is the first argument of func_h
        func_h_y: harvesting function for y
        kwargs_h_y: a dictionnary of the arguments of the other arguments of func_h that are not y(nT). y(nT) is the first argument of func_h
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain

    Example of run:
        double_exo.solve_predator_prey_model(
        xy=x0y0_a,
        t=tt,
        gamma=gamma,
        E_x=E_x,
        E_y=E_y,
        T=T,
        func_g=double_exo.identity_x,
        kwargs_g={'z': r},
        func_f=double_exo.multiply_x,
        kwargs_f={'z': a}, 
        func_m=double_exo.identity,
        kwargs_m={'z': m},
        func_h_x=double_exo.return_x_x,
        kwargs_h_x={},
        func_h_y=double_exo.return_y_y,
        kwargs_h_y={},
        t_0=t_0,
        t_n=t_n  
        )

    Return: 
        x, y: values of the solution (x, y) of the ODE
        t: the time vector that has the same shape as x and y'''

    #Store solution in lists
    x = [] #empty list
    y = [] #empty list
    I = [] #empty list

    #Record initial conditions
    x.append(xyI[0])
    y.append(xyI[1])
    I.append(xyI[2])

    #Time points
    t = [t_0]

    #Solve ODE
    intervals = np.arange(t_0, t_n, T) #divide the domain in intervals on length T
    intervals = np.append(intervals, t_n) #add t_n to intervals because t_n is not reached by arange
    ##Initial values before entring into the loop
    x_kT_plus = xyI[0]
    y_kT_plus = xyI[1]
    
    for i in range(1,len(intervals)):
        xyI_kT_plus = [x_kT_plus,y_kT_plus,I[-1]] #the initial value in a period is [x(kT+), y(kT+), I(kT+)]
        ##Span for this period
        tspan = np.linspace(intervals[i-1], intervals[i], 101)
        tspan = np.append(tspan, intervals[i]) 
        ##Solve for this period
        xy_step = odeint(predator_prey_model, xyI_kT_plus, tspan, 
                         args=(gamma, func_g, kwargs_g, func_f, kwargs_f, func_m, kwargs_m)) 
        x.extend(xy_step.T[0]) #Continuous part of x
        x_kT_plus = xy_step.T[0][-1] - E_x*func_h_x(xy_step.T[0][-1], **kwargs_h_x) #Applying func_h_x to x(nT)
        y.extend(xy_step.T[1]) #Continuous part of y
        y_kT_plus = xy_step.T[1][-1] - E_y*func_h_y(xy_step.T[1][-1], **kwargs_h_y) #Applying func_h_y to y(nT)
        I.extend(xy_step.T[2]) #Integral of x
        t.extend(tspan)

    return t, x, y, I

##2.2 Choice of the first time of pulse

def solve_pp_model_mortality_at_t_0(
    xyI,
    t,
    gamma: float,
    E_x: float,
    E_y: float,
    T: float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float],
    func_h_x: Callable[..., float],
    kwargs_h_x: dict[str, float], 
    func_h_y: Callable[..., float],
    kwargs_h_y: dict[str, float],
    t_0: float,
    t_n: float
):
    '''This function aims to be a general predator-prey model where the user decides on the functions that will be used in the model.
    There is an impulsive exogenous mortality at t_0
    
    Param:
        xyI: a list of values of [x,y,I] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_x: taking effort for pests
        E_y: taking effort for predators
        T: release period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        func_h_x: harvesting function for x
        kwargs_h_x: a dictionnary of the arguments of the other arguments of func_h that are not x(nT). x(nT) is the first argument of func_h
        func_h_y: harvesting function for y
        kwargs_h_y: a dictionnary of the arguments of the other arguments of func_h that are not y(nT). y(nT) is the first argument of func_h
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain

    Example of run:
        double_exo.solve_pp_model_mortality_at_t_0(
        xy=x0y0_a,
        t=tt,
        gamma=gamma,
        E_x=E_x,
        E_y=E_y,
        T=T,
        func_g=double_exo.identity_x,
        kwargs_g={'z': r},
        func_f=double_exo.multiply_x,
        kwargs_f={'z': a}, 
        func_m=double_exo.identity,
        kwargs_m={'z': m},
        func_h_x=double_exo.return_x_x,
        kwargs_h_x={},
        func_h_y=double_exo.return_y_y,
        kwargs_h_y={},
        t_0=t_0,
        t_n=t_n  
        )

    Return: 
        x, y: values of the solution (x, y) of the ODE
        t: the time vector that has the same shape as x and y'''

    #Store solution in lists
    x = [] #empty list
    y = [] #empty list
    I = [] #empty list

    #Record initial conditions
    x.append(xyI[0])
    y.append(xyI[1])
    I.append(xyI[2])

    #Time points
    t = [t_0]

    #Solve ODE
    intervals = np.arange(t_0, t_n, T) #divide the domain in intervals on length T
    intervals = np.append(intervals, t_n) #add t_n to intervals because t_n is not reached by arange
    for i in range(1,len(intervals)):
        x_kT_plus = x[-1] - E_x*func_h_x(x[-1], **kwargs_h_x) #Applying func_h_x to x(nT) to have x(nT+)
        y_kT_plus = y[-1] - E_y*func_h_y(y[-1], **kwargs_h_y) #Applying func_h_y to y(nT) to have y(nT+)
        xyI_kT_plus = [x_kT_plus,y_kT_plus,I[-1]] #the initial value in a period is [x(kT+), y(kT+), I(kT+)]
        ##Span for this period
        tspan = np.linspace(intervals[i-1], intervals[i], 101)
        tspan = np.append(tspan, intervals[i]) 
        ##Solve for this period
        xy_step = odeint(predator_prey_model, xyI_kT_plus, tspan, 
                         args=(gamma, func_g, kwargs_g, func_f, kwargs_f, func_m, kwargs_m)) 
        x.extend(xy_step.T[0]) #Continuous part of x
        y.extend(xy_step.T[1]) #Continuous part of y
        I.extend(xy_step.T[2]) #Integral of x
        t.extend(tspan)

    return t, x, y, I

def solve_pp_choose_first_impulsion(
    xyI,
    t,
    gamma: float,
    E_x: float,
    E_y: float,
    T: float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float],
    func_h_x: Callable[..., float],
    kwargs_h_x: dict[str, float], 
    func_h_y: Callable[..., float],
    kwargs_h_y: dict[str, float],
    t_0: float,
    t_n: float,
    t_pulse: float
):
    '''This function aims to be a general predator-prey model where the user decides on the functions that will be used in the model.
    The user chooses the first impulsion time.
    
    Param:
        xyI: a list of values of [x,y,I] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_x: taking effort for pests
        E_y: taking effort for predators
        T: release period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        func_h_x: harvesting function for x
        kwargs_h_x: a dictionnary of the arguments of the other arguments of func_h that are not x(nT). x(nT) is the first argument of func_h
        func_h_y: harvesting function for y
        kwargs_h_y: a dictionnary of the arguments of the other arguments of func_h that are not y(nT). y(nT) is the first argument of func_h
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        t_pulse: first impulsion time

    Example of run:
        double_exo.solve_pp_choose_first_impulsion(
        xy=x0y0_a,
        t=tt,
        gamma=gamma,
        E_x=E_x,
        E_y=E_y,
        T=T,
        func_g=double_exo.identity_x,
        kwargs_g={'z': r},
        func_f=double_exo.multiply_x,
        kwargs_f={'z': a}, 
        func_m=double_exo.identity,
        kwargs_m={'z': m},
        func_h_x=double_exo.return_x_x,
        kwargs_h_x={},
        func_h_y=double_exo.return_y_y,
        kwargs_h_y={},
        t_0=t_0,
        t_n=t_n  
        )

    Return: 
        x, y: values of the solution (x, y) of the ODE
        t: the time vector that has the same shape as x and y'''

    #Store solution in lists
    x = [] #empty list
    y = [] #empty list
    I = [] #empty list

    #Record initial conditions
    x.append(xyI[0])
    y.append(xyI[1])
    I.append(xyI[2])

    #Time points
    t = [t_0]

    #Solve ODE
    ##Define the different intervals
    #t_before_pulse = np.array([t_0,t_pulse]) #to get the time points before the first impulsion
    intervals = np.arange(t_pulse, t_n, T) #divide the domain in intervals on length T
    intervals = np.append(intervals, t_n) #add t_n to intervals because t_n is not reached by arange
    #intervals = np.concatenate((first_interval,intervals))

    ##Solve before the first impulsion
    tspan = np.linspace(t_0,t_pulse,101) #tspan before the first impulsion
    xy_step = odeint(predator_prey_model, xyI, tspan, #The initial value is the initial value of whole domain
                     args=(gamma, func_g, kwargs_g, func_f, kwargs_f, func_m, kwargs_m))
    x.extend(xy_step.T[0]) #x before first impulsion
    x_kT_plus = xy_step.T[0][-1] - E_x*func_h_x(xy_step.T[0][-1], **kwargs_h_x) #First impulsion on x
    y.extend(xy_step.T[1]) #y before first impulsion
    y_kT_plus = xy_step.T[1][-1] - E_y*func_h_y(xy_step.T[1][-1], **kwargs_h_y) #First impulsion on x
    I.extend(xy_step.T[2]) #Integral of x
    t.extend(tspan)

    ##Solve after the first impulsion
    for i in range(1,len(intervals)):
        xyI_kT_plus = [x_kT_plus,y_kT_plus,I[-1]] #the initial value in a period is [x(kT+), y(kT+), I(kT+)]
        ###Span for this period
        tspan = np.linspace(intervals[i-1], intervals[i], 101)
        tspan = np.append(tspan, intervals[i]) 
        ###Solve for this period
        xy_step = odeint(predator_prey_model, xyI_kT_plus, tspan, 
                         args=(gamma, func_g, kwargs_g, func_f, kwargs_f, func_m, kwargs_m)) 
        x.extend(xy_step.T[0]) #Continuous part of x
        x_kT_plus = xy_step.T[0][-1] - E_x*func_h_x(xy_step.T[0][-1], **kwargs_h_x) #Applying func_h_x to x(nT)
        y.extend(xy_step.T[1]) #Continuous part of y
        y_kT_plus = xy_step.T[1][-1] - E_y*func_h_y(xy_step.T[1][-1], **kwargs_h_y) #Applying func_h_y to y(nT)
        I.extend(xy_step.T[2]) #Integral of x
        t.extend(tspan)

    return t, x, y, I

#3 Functions for the comparison between the continuous model and the impulsive model

##3.1 Function to help when plotting the comparison between continuous and impulsive models
def modify_func_Ec(func_g):
    '''This function creates a new function.
    The new function is the same as in parameter...
    ...but with a supplementary parameter E_c...
    ...which is substracted from the result of the function in parameter
    
    Param: 
        func_g: a growth function which has x as first argument
    
    Return: 
        new_func_g: a new function that calculates the result of func_g minus a given E_c.
                    It has got E_c as new argument'''
    def new_func_g(*args, E_c, **kwargs):
        '''This function returns the result of func_g with all its agruments minus E_c'''
        # Calculate the result of func_g
        result = func_g(*args, **kwargs)
        # Substract E_c from the resulte
        return result - E_c
    return new_func_g

##3.2 Functions to plot the dynamics of the two models and to compare the criteria

def plot_pop_mortality_on_x_logistic_LV(
    xyI,
    t,
    r: float,
    K: float,
    a: float,
    gamma: float,
    m: float,
    E_x_c: float,
    T: float, 
    t_0: float,
    t_n: float
):
    '''This function plots the population size of x and y for two models
    (the model with continuous mortality on x;
    the model with impulsive mortality on x)
    The two models have the same initial values.
    We remark that there is mortality only on x.
    The model used is Lotka-Volterra with logistic growth
    
    In can be generalized later.
    The aim of the first version in to enter only a few parameters (especially initial values, E and T)
    
    
    Param:
        xyI: a list of values of [x,y,I] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        r: growth rate
        K: carrying capacity
        a: search rate
        gamma: conversion factor
        m: intrinsic death rate
        E_x_c: continuous taking effort for pests
        T: release period
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain'''
    
    #Solve ODE
    ##Continuous
    xyI_cont = solve_predator_prey_model(
            xyI=xyI,
            t=t,
            gamma=gamma,
            E_x=0, #useless because it will be multiplied by 0. It's just to not lose the E
            E_y=0, #useless because it will be multiplied by 0. It's just to not lose the E
            T=T,
            func_g=logistic_sub_E_x,
            kwargs_g={'r':r, 'K':K, 'E_x':E_x_c},
            func_f=multiply_x,
            kwargs_f={'z': a}, 
            func_m=identity,
            kwargs_m={'z': m},
            func_h_x=return_zero_x,
            kwargs_h_x={},
            func_h_y=return_zero_y, 
            kwargs_h_y={},
            t_0=t_0,
            t_n=t_n  
            )
    x_cont = xyI_cont[1]
    y_cont = xyI_cont[2]
    I_cont = xyI_cont[3]

    t = xyI_cont[0]

    ##Impulsive
    xyI_imp = solve_predator_prey_model(
        xyI=xyI,
        t=t,
        gamma=gamma,
        E_x= 1 - np.exp(-E_x_c*T), #E for impulsive
        E_y=0, #useless because it will be multiplied by 0. It's just to not lose the E
        T=T,
        func_g=logistic_model_x,
        kwargs_g={'r':r, 'K':K},
        func_f=multiply_x,
        kwargs_f={'z': a}, 
        func_m=identity,
        kwargs_m={'z': m},
        func_h_x=return_x_x,
        kwargs_h_x={},
        func_h_y=return_zero_y, 
        kwargs_h_y={},
        t_0=t_0,
        t_n=t_n  
        )
    x_imp = xyI_imp[1]
    y_imp = xyI_imp[2]
    I_imp = xyI_imp[3]

    #Plot results
    ##Evolution of the populations
    plt.figure()
    plt.plot(t, x_cont, color = (0,0,0.9), linestyle='-', label=f'x_cont with {xyI} as initial value')
    plt.plot(t, y_cont, color = (0.9,0,0), linestyle='-', label=f'y_cont with {xyI} as initial value')
    plt.plot(t, x_imp, color = (0,0,0.9), linestyle='--', label=f'x_imp with {xyI} as initial value')
    plt.plot(t, y_imp, color = (0.9,0,0), linestyle='--', label=f'y_imp with {xyI} as initial value')
    plt.xlabel('time')
    plt.ylabel('Population size')
    plt.title(f'Population of pests and predators with continuous and impulsive exogenous mortality on pests with {xyI} as initial value')
    plt.suptitle(f'{r = }, {K = }, {a = }, {gamma = }, {m = }, {E_x_c = }, {T =}')
    plt.legend(loc= 'upper left', bbox_to_anchor=(1,1))
    plt.grid()
    plt.show()

    ##Integral of x
    plt.figure()
    plt.plot(t, I_cont, linestyle='-', label=f'x_cont with {xyI} as initial value')
    plt.plot(t, I_imp, linestyle='--', label=f'y_imp with {xyI} as initial value')
    plt.xlabel('time')
    plt.ylabel('Pests population size')
    plt.title(f'Integral of x continuous and impulsive exogenous mortality on pests with {xyI} as initial value')
    plt.suptitle(f'{r = }, {K = }, {a = }, {gamma = }, {m = }, {E_x_c = }, {T =}')
    plt.legend(loc= 'upper left', bbox_to_anchor=(1,1))
    plt.grid()
    plt.show()

def compare_cont_imp_proportional_mortality_on_x_T(
    xyI0_imp,
    xyI0_cont,
    t,
    gamma: float,
    E_c: float,
    T: float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float,
    t_n: float,
    eps: float = 0.01,
    plot_population: bool = False,
    plot_x: bool = True,
    plot_y: bool = True
):
    
    '''This function returns the values of the criteria to compare between impulsive and continuous model.
    This function also plots (if wanted) the population size of x and y for two models:
    (the model with proportional continuous mortality on x;
    the model with proportional impulsive mortality on x)
    
    We remark that the mortality is only on x.
    That's why the arguments E_y, func_h_y and kwargs_h_y are not there.
    It also returns the values of the criteria to compare between impulsive and continuous model.

    The first event of mortality is at t=T
    
    Param:
        xyI0_imp: a list of values of [x_imp,y_imp,I_imp] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        xyI0_cont: a list of values of [x_cont,y_cont,I_cont] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_c: continuous taking effort for pests
        T: release period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        eps: the threshold below which we want to have the population of pests
        plot_population: to precise if we want to plot the population size
        plot_y: to precise if we want to plot the population y. This parameter is used only when plot_population == True
        
    Return:
        A dictionnary with the following values:
            T: period (already in the arguments)
            I_cont_final: integral of x_cont at the final time
            I_imp_final: integral of x_imp at the final time
            eps: the threshold below which we want to have the population of pests (already in the arguments)
            t_pulse: time of first impulsion (T for this function)
            t_eta_cont:the smallest t_eta such that for all t > t_eta, x_cont(t) < eps
            t_eta_imp:the smallest t_eta such that for all t > t_eta, x_imp(t) < eps
            t_eta_imp - t_eta_cont: difference between t_eta_imp and t_eta_cont. Can be negative if t_eta_cont > t_eta_imp (only if the threshold is reached)'''
    
    #Solve ODE
    ##Impulsive
    xyI_imp = solve_predator_prey_model( #The one thing that changes is the solver used
        xyI=xyI0_imp,
        t=t,
        gamma=gamma,
        E_x= 1 - np.exp(-E_c*T), #Impulsive E
        E_y=0, #No exogenous mortality on y
        T=T,
        func_g=func_g,
        kwargs_g=kwargs_g,
        func_f=func_f,
        kwargs_f=kwargs_f, 
        func_m=func_m,
        kwargs_m=kwargs_m,
        func_h_x=return_x_x, #Proportional exogenous mortality
        kwargs_h_x={}, #Proportional exogenous mortality
        func_h_y=return_zero_y, #No exogenous mortality on y
        kwargs_h_y={}, #No exogenous mortality on y
        t_0=t_0,
        t_n=t_n  
        )
    x_imp = xyI_imp[1]
    y_imp = xyI_imp[2]
    I_imp = xyI_imp[3]
    
    ##Continuous
    func_g_sub_Ec = modify_func_Ec(func_g) #Create the new function g with continuous exogenous mortality
    kwargs_g_sub_Ec = {**kwargs_g, 'E_c' : E_c} #The argument of the new function func_g_sub_Ec.
                                                # **kwargs_g: the arguments of the previous func_g; 
                                                #E_c: the supplementary argument of func_g_sub_Ec

    xyI_cont = solve_predator_prey_model( #The one thing that changes is the solver used
            xyI=xyI0_cont,
            t=t,
            gamma=gamma,
            E_x=0, #Continuous exogenous mortality, so no impulsive part
            E_y=0, #No exogenous mortality on y
            T=T,
            func_g=func_g_sub_Ec,
            kwargs_g=kwargs_g_sub_Ec,
            func_f=func_f,
            kwargs_f=kwargs_f, 
            func_m=func_m,
            kwargs_m=kwargs_m,
            func_h_x=return_zero_x, #Continuous exogenous mortality, so no impulsive part
            kwargs_h_x={}, #Continuous exogenous mortality, so no impulsive part
            func_h_y=return_zero_y, #No exogenous mortality on y
            kwargs_h_y={}, #No exogenous mortality on y
            t_0=t_0,
            t_n=t_n  
            )
    x_cont = xyI_cont[1]
    y_cont = xyI_cont[2]
    I_cont = xyI_cont[3]

    t = xyI_cont[0]

    #Plot results
    ##Evolution of the populations
    if plot_population:
        plt.figure()
        if plot_x:
            plt.plot(t, x_cont, color = (0,0,0.9), linestyle='-', label=f'x_cont with {xyI0_cont} as initial value')
            plt.plot(t, x_imp, color = (0,0,0.9), linestyle='--', label=f'x_imp with {xyI0_imp} as initial value')
        if plot_y:
            plt.plot(t, y_cont, color = (0.9,0,0), linestyle='-', label=f'y_cont with {xyI0_cont} as initial value')
            plt.plot(t, y_imp, color = (0.9,0,0), linestyle='--', label=f'y_imp with {xyI0_imp} as initial value')
        plt.xlabel('time')
        plt.ylabel('Population size')
        plt.title(f'Population of pests and predators with continuous and impulsive exogenous mortality on pests and the first impulsive exogenous mortality at t = T')
        plt.suptitle(f'{kwargs_g}, {E_c = }, {T = }')
        plt.legend(loc= 'upper left', bbox_to_anchor=(1,1))
        plt.grid()
        plt.show()

    #Evaluate the criteria
    ##Integral of x at final time
    I_cont_final = I_cont[-1]
    I_imp_final = I_imp[-1]

    ##Time t_eta when epsilon is reached
    t_eta_cont = None #Default value if epsilon is never reached
    t_eta_imp = None #Default value if epsilon is never reached

    x_cont = np.array(x_cont, dtype=float) #Convert x_cont into array
    x_imp = np.array(x_imp, dtype=float) #Convert x_cont into array
    for i in range(len(x_cont)):
        if np.all(x_cont[i:] < eps): #Test if all value after i is below epsilon
            t_eta_cont = t[i] #Store the first occurence
            break 

    for i in range(len(x_imp)):
        if np.all(x_imp[i:] < eps): #Test if all value after i is below epsilon
            t_eta_imp = t[i] #Store the first occurence
            break 

    if t_eta_cont == None or t_eta_imp == None: #A substraction with a None is impossible
        return {'T': T, 'I_cont_final':I_cont_final, 'I_imp_final':I_imp_final,
                'eps': eps, 't_pulse': T, 't_eta_cont': t_eta_cont, 't_eta_imp':t_eta_imp}
    else:
        return {'T': T, 'I_cont_final':I_cont_final, 'I_imp_final':I_imp_final,
                'eps': eps, 't_pulse': T, 't_eta_cont': t_eta_cont, 't_eta_imp':t_eta_imp, 't_eta_imp - t_eta_cont':t_eta_imp - t_eta_cont}

def compare_cont_imp_proportional_mortality_on_x_0(
    xyI0_imp,
    xyI0_cont,
    t,
    gamma: float,
    E_c: float,
    T: float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float,
    t_n: float,
    eps: float = 0.01,
    plot_population: bool = False,
    plot_y: bool = True
):
    
    '''This function returns the values of the criteria to compare between impulsive and continuous model.
    This function also plots (if wanted) the population size of x and y for two models:
    (the model with proportional continuous mortality on x;
    the model with proportional impulsive mortality on x)
    
    We remark that the mortality is only on x.
    That's why the arguments E_y, func_h_y and kwargs_h_y are not there.
    It also returns the values of the criteria to compare between impulsive and continuous model.

    The first event of mortality is at t=0
    
    Param:
        xyI0_imp: a list of values of [x_imp,y_imp,I_imp] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        xyI0_cont: a list of values of [x_cont,y_cont,I_cont] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_c: continuous taking effort for pests
        T: release period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        eps: the threshold below which we want to have the population of pests
        plot_population: to precise if we want to plot the population size
        plot_y: to precise if we want to plot the population y. This parameter is used only when plot_population == True
        
    Return:
        A dictionnary with the following values:
            T: period (already in the arguments)
            I_cont_final: integral of x_cont at the final time
            I_imp_final: integral of x_imp at the final time
            eps: the threshold below which we want to have the population of pests (already in the arguments)
            t_pulse: time of first impulsion (0 for this function)
            t_eta_cont:the smallest t_eta such that for all t > t_eta, x_cont(t) < eps
            t_eta_imp:the smallest t_eta such that for all t > t_eta, x_imp(t) < eps
            t_eta_imp - t_eta_cont: difference between t_eta_imp and t_eta_cont. Can be negative if t_eta_cont > t_eta_imp (only if the threshold is reached)'''
    
    #Solve ODE
    ##Impulsive
    xyI_imp = solve_pp_model_mortality_at_t_0( #The one thing that changes is the solver used
        xyI=xyI0_imp,
        t=t,
        gamma=gamma,
        E_x= 1 - np.exp(-E_c*T), #Impulsive E
        E_y=0, #No exogenous mortality on y
        T=T,
        func_g=func_g,
        kwargs_g=kwargs_g,
        func_f=func_f,
        kwargs_f=kwargs_f, 
        func_m=func_m,
        kwargs_m=kwargs_m,
        func_h_x=return_x_x, #Proportional exogenous mortality
        kwargs_h_x={}, #Proportional exogenous mortality
        func_h_y=return_zero_y, #No exogenous mortality on y
        kwargs_h_y={}, #No exogenous mortality on y
        t_0=t_0,
        t_n=t_n  
        )
    x_imp = xyI_imp[1]
    y_imp = xyI_imp[2]
    I_imp = xyI_imp[3]
    
    ##Continuous
    func_g_sub_Ec = modify_func_Ec(func_g) #Create the new function g with continuous exogenous mortality
    kwargs_g_sub_Ec = {**kwargs_g, 'E_c' : E_c} #The argument of the new function func_g_sub_Ec.
                                                # **kwargs_g: the arguments of the previous func_g; 
                                                #E_c: the supplementary argument of func_g_sub_Ec

    xyI_cont = solve_pp_model_mortality_at_t_0( #The one thing that changes is the solver used
            xyI=xyI0_cont,
            t=t,
            gamma=gamma,
            E_x=0, #Continuous exogenous mortality, so no impulsive part
            E_y=0, #No exogenous mortality on y
            T=T,
            func_g=func_g_sub_Ec,
            kwargs_g=kwargs_g_sub_Ec,
            func_f=func_f,
            kwargs_f=kwargs_f, 
            func_m=func_m,
            kwargs_m=kwargs_m,
            func_h_x=return_zero_x, #Continuous exogenous mortality, so no impulsive part
            kwargs_h_x={}, #Continuous exogenous mortality, so no impulsive part
            func_h_y=return_zero_y, #No exogenous mortality on y
            kwargs_h_y={}, #No exogenous mortality on y
            t_0=t_0,
            t_n=t_n  
            )
    x_cont = xyI_cont[1]
    y_cont = xyI_cont[2]
    I_cont = xyI_cont[3]

    t = xyI_cont[0]

    #Plot results
    ##Evolution of the populations
    if plot_population:
        plt.figure()
        plt.plot(t, x_cont, color = (0,0,0.9), linestyle='-', label=f'x_cont with {xyI0_cont} as initial value')
        plt.plot(t, x_imp, color = (0,0,0.9), linestyle='--', label=f'x_imp with {xyI0_imp} as initial value')
        if plot_y:
            plt.plot(t, y_cont, color = (0.9,0,0), linestyle='-', label=f'y_cont with {xyI0_cont} as initial value')
            plt.plot(t, y_imp, color = (0.9,0,0), linestyle='--', label=f'y_imp with {xyI0_imp} as initial value')
        plt.xlabel('time')
        plt.ylabel('Population size')
        plt.title(f'Population of pests and predators with continuous and impulsive exogenous mortality on pests and the first impulsive exogenous mortality at t = 0')
        plt.suptitle(f'{kwargs_g}, {E_c = }, {T = }')
        plt.legend(loc= 'upper left', bbox_to_anchor=(1,1))
        plt.grid()
        plt.show()

    #Evaluate the criteria
    ##Integral of x at final time
    I_cont_final = I_cont[-1]
    I_imp_final = I_imp[-1]

    ##Time t_eta when epsilon is reached
    t_eta_cont = None #Default value if epsilon is never reached
    t_eta_imp = None #Default value if epsilon is never reached

    x_cont = np.array(x_cont, dtype=float) #Convert x_cont into array
    x_imp = np.array(x_imp, dtype=float) #Convert x_cont into array
    for i in range(len(x_cont)):
        if np.all(x_cont[i:] < eps): #Test if all value after i is below epsilon
            t_eta_cont = t[i] #Store the first occurence
            break 

    for i in range(len(x_imp)):
        if np.all(x_imp[i:] < eps): #Test if all value after i is below epsilon
            t_eta_imp = t[i] #Store the first occurence
            break 

    if t_eta_cont == None or t_eta_imp == None: #A substraction with a None is impossible
        return {'T': T, 'I_cont_final':I_cont_final, 'I_imp_final':I_imp_final,
                'eps': eps, 't_pulse': 0, 't_eta_cont': t_eta_cont, 't_eta_imp':t_eta_imp}
    else:
        return {'T': T, 'I_cont_final':I_cont_final, 'I_imp_final':I_imp_final,
                'eps': eps, 't_pulse': 0, 't_eta_cont': t_eta_cont, 't_eta_imp':t_eta_imp, 't_eta_imp - t_eta_cont':t_eta_imp - t_eta_cont}


    
def compare_cont_imp_proportional_mortality_on_x(
    xyI0_imp,
    xyI0_cont,
    t,
    gamma: float,
    E_c: float,
    T: float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float,
    t_n: float,
    t_pulse:float,
    eps: float = 0.01,
    plot_population: bool = False,
    plot_x:bool = True,
    plot_y: bool = True,
    plot_bound: bool = False,
    t_bound = 0,
    plot_eps: bool = False,
    pop_in_log: bool = False
):
    
    '''This function returns the values of the criteria to compare between impulsive and continuous model.
    This function also plots (if wanted) the population size of x and y for two models:
    (the model with proportional continuous mortality on x;
    the model with proportional impulsive mortality on x)

    We remark that the mortality is only on x.
    That's why the arguments E_y, func_h_y and kwargs_h_y are not there.
    It also returns the values of the criteria to compare between impulsive and continuous model.

    The first event of mortality is at t=t_pulse
    
    Param:
        xyI0_imp: a list of values of [x_imp,y_imp,I_imp] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        xyI0_cont: a list of values of [x_cont,y_cont,I_cont] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_c: continuous taking effort for pests
        T: period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        t_pulse: time of first impulsion
        eps: the threshold below which we want to have the population of pests
        plot_population: to precise if we want to plot the population size
        plot_x: to precise if we want to plot the population x. This parameter is used only when plot_population == True
        plot_y: to precise if we want to plot the population y. This parameter is used only when plot_population == True
        store_bound: a bool to say if we return the time points where it takes at least one period more or less to reach eps
        t_bound: the time points where it takes at least one period more or less to reach eps
        plot_eps: a bool to say if we plot the control level epsilon
        pop_in_pol: a bool to say if we rescale the y-axis (the population dynamics) in log 
        
    Return:
        A dictionnary with the following values:
            T: period (already in the arguments)
            I_cont_final: integral of x_cont at the final time
            I_imp_final: integral of x_imp at the final time
            eps: the threshold below which we want to have the population of pests (already in the arguments)
            t_pulse: time of first impulsion (already in the arguments)
            t_eta_cont:the smallest t_eta such that for all t > t_eta, x_cont(t) < eps
            t_eta_imp:the smallest t_eta such that for all t > t_eta, x_imp(t) < eps
            t_eta_imp - t_eta_cont: difference between t_eta_imp and t_eta_cont. Can be negative if t_eta_cont > t_eta_imp (only if the threshold is reached)'''
    
    #Solve ODE
    ##Impulsive
    xyI_imp = solve_pp_choose_first_impulsion( #The one thing that changes is the solver used
        xyI=xyI0_imp,
        t=t,
        gamma=gamma,
        E_x= 1 - np.exp(-E_c*T), #Impulsive E
        E_y=0, #No exogenous mortality on y
        T=T,
        func_g=func_g,
        kwargs_g=kwargs_g,
        func_f=func_f,
        kwargs_f=kwargs_f, 
        func_m=func_m,
        kwargs_m=kwargs_m,
        func_h_x=return_x_x, #Proportional exogenous mortality
        kwargs_h_x={}, #Proportional exogenous mortality
        func_h_y=return_zero_y, #No exogenous mortality on y
        kwargs_h_y={}, #No exogenous mortality on y
        t_0=t_0,
        t_n=t_n,
        t_pulse=t_pulse  
        )
    x_imp = xyI_imp[1]
    y_imp = xyI_imp[2]
    I_imp = xyI_imp[3]
    
    ##Continuous
    func_g_sub_Ec = modify_func_Ec(func_g) #Create the new function g with continuous exogenous mortality
    kwargs_g_sub_Ec = {**kwargs_g, 'E_c' : E_c} #The argument of the new function func_g_sub_Ec.
                                                # **kwargs_g: the arguments of the previous func_g; 
                                                #E_c: the supplementary argument of func_g_sub_Ec

    xyI_cont = solve_pp_choose_first_impulsion( #The one thing that changes is the solver used
            xyI=xyI0_cont,
            t=t,
            gamma=gamma,
            E_x=0, #Continuous exogenous mortality, so no impulsive part
            E_y=0, #No exogenous mortality on y
            T=T,
            func_g=func_g_sub_Ec,
            kwargs_g=kwargs_g_sub_Ec,
            func_f=func_f,
            kwargs_f=kwargs_f, 
            func_m=func_m,
            kwargs_m=kwargs_m,
            func_h_x=return_zero_x, #Continuous exogenous mortality, so no impulsive part
            kwargs_h_x={}, #Continuous exogenous mortality, so no impulsive part
            func_h_y=return_zero_y, #No exogenous mortality on y
            kwargs_h_y={}, #No exogenous mortality on y
            t_0=t_0,
            t_n=t_n,
            t_pulse=t_pulse  
            )
    x_cont = xyI_cont[1]
    y_cont = xyI_cont[2]
    I_cont = xyI_cont[3]

    t = xyI_cont[0]

    #Plot results
    ##Evolution of the populations
    if plot_population:
        plt.figure()
        if plot_x:
            plt.plot(t, x_cont, color = (0,0,0.9), linestyle='-', label=f'x_cont with {xyI0_cont} as initial value')
            plt.plot(t, x_imp, color = (0,0,0.9), linestyle='--', label=f'x_imp with {xyI0_imp} as initial value')
        if plot_y:
            plt.plot(t, y_cont, color = (0.9,0,0), linestyle='-', label=f'y_cont with {xyI0_cont} as initial value')
            plt.plot(t, y_imp, color = (0.9,0,0), linestyle='--', label=f'y_imp with {xyI0_imp} as initial value')
        if pop_in_log:
            plt.yscale('log')
        if plot_bound:
            for bound in t_bound:
                plt.axvline(x=bound, color = 'gray', linestyle='dotted')
        if plot_eps:
            plt.plot(t, eps*np.ones_like(t), color = 'orange', linestyle=':')
            plt.text(t[-1]*0.95, eps * 1.02, r'$\varepsilon$', color='orange', fontsize=14)
        plt.xlabel('time')
        plt.ylabel('Population size')
        plt.title(f'Population of pests and predators with exogenous mortality on pests \n and the first impulsive exogenous mortality at t = {t_pulse} \n {kwargs_g}, {E_c = }, {T = }')
        plt.legend(loc= 'upper left', bbox_to_anchor=(1,1))
        plt.grid()
        plt.show()

    #Evaluate the criteria
    ##Integral of x at final time
    I_cont_final = I_cont[-1]
    I_imp_final = I_imp[-1]

    ##Time t_eta when epsilon is reached
    t_eta_cont = None #Default value if epsilon is never reached
    t_eta_imp = None #Default value if epsilon is never reached

    x_cont = np.array(x_cont, dtype=float) #Convert x_cont into array
    x_imp = np.array(x_imp, dtype=float) #Convert x_cont into array
    for i in range(len(x_cont)):
        if np.all(x_cont[i:] < eps): #Test if all value after i is below epsilon
            t_eta_cont= t[i] #Store the smallest t_eta such that for all t > t_eta, x_cont(t) < eps
            break 

    for i in range(len(x_imp)):
        if np.all(x_imp[i:] < eps): #Test if all value after i is below epsilon
            t_eta_imp = t[i] #Store the smallest t_eta such that for all t > t_eta, x_imp(t) < eps
            break 
    if t_eta_cont == None or t_eta_imp == None: #A substraction with a None is impossible
        return {'T': T, 'I_cont_final':I_cont_final, 'I_imp_final':I_imp_final,
                'eps': eps, 't_pulse': t_pulse, 't_eta_cont': t_eta_cont, 't_eta_imp':t_eta_imp}
    else:
        return {'T': T, 'I_cont_final':I_cont_final, 'I_imp_final':I_imp_final,
                'eps': eps, 't_pulse': t_pulse, 't_eta_cont': t_eta_cont, 't_eta_imp':t_eta_imp, 't_eta_imp - t_eta_cont':t_eta_imp - t_eta_cont}
    
##3.3 Functions to plot graphs of the criteria
    
###Those functions simulate the criteria for many values of parameters using for loops
###All of those functions compare two models : impulsive and continuous with proportional harvesting
    
def plot_t_eta_of_eps_prop_mortality_on_x(
    xyI0_imp,
    xyI0_cont,
    t,
    gamma: float,
    E_c: float,
    T: float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float,
    t_n: float,
    t_pulse:float,
    eps_start: float,
    eps_stop: float,
    eps_num: int = 100
):
    '''This function plots t_eta with respect to a range of different epsilon.
    t_eta is the time until reaching the threshold epsilon
    Two graphs (one for impulsive and one for continuous) are on the same plot.
    
    Param:
        xyI0_imp: a list of values of [x_imp,y_imp,I_imp] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        xyI0_cont: a list of values of [x_cont,y_cont,I_cont] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_c: continuous taking effort for pests
        T: period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        t_pulse: time of first impulsion
        eps_start: beginning of the epsilon array
        eps_stop: end of the epsilon array
        eps_num: number of point in the epsilon array'''
    
    #The array of epsilons
    eps_array = np.linspace(eps_start, eps_stop, eps_num)
    #The array of t_etas
    t_eta_cont_array = np.zeros_like(eps_array) #array for the t_eta_cont with respect for each eps
    t_eta_imp_array = np.zeros_like(eps_array) #array for the t_eta_cont with respect for each eps
    for i in range(len(eps_array)):
        criteria = compare_cont_imp_proportional_mortality_on_x( #Store the criteria dictionnary in a variable
            xyI0_imp= xyI0_imp,
            xyI0_cont= xyI0_cont,
            t=t,
            gamma=gamma,
            E_c=E_c,
            T=T,
            func_g=func_g,
            kwargs_g=kwargs_g,
            func_f=func_f,
            kwargs_f=kwargs_f, 
            func_m=func_m,
            kwargs_m=kwargs_m,
            t_0=t_0,
            t_n=t_n,
            t_pulse=t_pulse,
            eps=eps_array[i],
            plot_population=False
            )
        if criteria['t_eta_cont'] is not None: #Verify if eps is reached 
            t_eta_cont_array[i] = criteria['t_eta_cont']
        else:
            t_eta_cont_array[i] = np.nan #If not, this value of epsilon will be ignored
        if criteria['t_eta_imp'] is not None:
            t_eta_imp_array[i] = criteria['t_eta_imp'] #Verify if eps is reached 
        else:
            t_eta_imp_array[i] = np.nan #If not, this value of epsilon will be ignored
    #Plot the graph got
    plt.figure()
    plt.plot(eps_array, t_eta_cont_array, color = (0.3,0.4,1), linestyle = '-', label= 'Time for the continuous model to reach the threshold')
    plt.plot(eps_array, t_eta_imp_array, color = (0.3,0.4,1), linestyle = '--', label= 'Time for the impulsive model to reach the threshold')
    plt.xlabel('epsilon')
    plt.ylabel('time to reach epsilon')
    plt.suptitle(f'{T=}, {t_pulse=}, initial value for impulsive model: {xyI0_imp}, initial value for continuous model: {xyI0_cont}')
    plt.title(f'Time to reach epsilon with respect to epsilon')
    plt.legend(loc= 'upper left', bbox_to_anchor=(1,1))
    plt.grid()
    plt.show() 

def plot_diff_t_eta_of_t_pulse_prop_mortality_on_x(
    xyI0_imp,
    xyI0_cont,
    t,
    gamma: float,
    E_c: float,
    T: float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float,
    t_n: float,
    eps: float = 0.01
):
    '''This function plots t_eta_imp - t_eta_cont with respect to a range of t_pulse from 0 to T.
    t_eta_imp is the time for the impulsive model to reach the threshold epsilon.
    t_eta_cont is the time for the continuous model to reach the threshold epsilon.
    Two graphs (one for impulsive and one for continuous) are on the same plot.
    
    Param:
        xyI0_imp: a list of values of [x_imp,y_imp,I_imp] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        xyI0_cont: a list of values of [x_cont,y_cont,I_cont] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_c: continuous taking effort for pests
        T: period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        eps: the threshold below which we want to have the population of pests'''
    
    #The array of t_pulse
    t_pulse_array = np.linspace(0, T, 100)
    #The array of t_eta_imp - t_eta_cont
    diff_t_eta_array = np.zeros_like(t_pulse_array) #array for t_eta_imp - t_eta_cont with respect for each t_pulse_array
    for i in range(len(t_pulse_array)):
        criteria = compare_cont_imp_proportional_mortality_on_x( #Store the criteria dictionnary in a variable
            xyI0_imp= xyI0_imp,
            xyI0_cont= xyI0_cont,
            t=t,
            gamma=gamma,
            E_c=E_c,
            T=T,
            func_g=func_g,
            kwargs_g=kwargs_g,
            func_f=func_f,
            kwargs_f=kwargs_f, 
            func_m=func_m,
            kwargs_m=kwargs_m,
            t_0=t_0,
            t_n=t_n,
            t_pulse=t_pulse_array[i],
            eps=eps,
            plot_population=False
            )
        if criteria['t_eta_imp'] is not None and  criteria['t_eta_cont'] is not None: #Verify if eps is reached for both model
            diff_t_eta_array[i] = criteria['t_eta_imp - t_eta_cont']
        else:
            return "epsilon is not reached for at least one t_pulse"
        
    #Plot the graph got
    plt.figure()
    plt.plot(t_pulse_array, diff_t_eta_array, color = (0.3,0.4,1), linestyle = '-', label= 'Difference of time to reach the threshold')
    plt.xlabel('t_pulse')
    plt.ylabel('t_imp - t_cont')
    plt.suptitle(f'{T=}, {eps=}, initial value for impulsive model: {xyI0_imp}, initial value for continuous model: {xyI0_cont}')
    plt.title(f'Difference of time to reach epsilon with respect to t_pulse')
    plt.legend(loc= 'upper left', bbox_to_anchor=(1,1))
    plt.grid()
    plt.show()

def plot_diff_t_eta_of_t_pulse_large_prop_mortality_on_x(
    xyI0_imp,
    xyI0_cont,
    t,
    gamma: float,
    E_c: float,
    T: float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float,
    t_n: float,
    t_pulse_start: float,
    t_pulse_stop: float,
    t_pulse_num: int = 100,
    eps: float = 0.01,
    store_bound: bool = False
):
    '''This function plots t_eta_imp - t_eta_cont with respect to a range of t_pulse whose start and end are chosen by the user.
    t_eta_imp is the time for the impulsive model to reach the threshold epsilon.
    t_eta_cont is the time for the continuous model to reach the threshold epsilon.
    Two graphs (one for impulsive and one for continuous) are on the same plot.
    
    Param:
        xyI0_imp: a list of values of [x_imp,y_imp,I_imp] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        xyI0_cont: a list of values of [x_cont,y_cont,I_cont] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_c: continuous taking effort for pests
        T: period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        t_pulse_start: beginning of the t_pulse array
        t_pulse_stop: end of the t_pulse array
        t_pulse_num: number of point in the t_pulse array
        eps: the threshold below which we want to have the population of pests
        store_bound: a bool to say if we return the time points where it takes at least one period more or less to reach eps
        
    Return if plot_bound:
        t_bound the time points where it takes at least one period more or less to reach eps'''
    
    
    #The array of t_pulse
    t_pulse_array = np.linspace(t_pulse_start, t_pulse_stop, t_pulse_num)
    #The array of t_eta_imp - t_eta_cont
    diff_t_eta_array = np.zeros_like(t_pulse_array) #array for t_eta_imp - t_eta_cont with respect for each t_pulse_array
    for i in range(len(t_pulse_array)):
        criteria = compare_cont_imp_proportional_mortality_on_x( #Store the criteria dictionnary in a variable
            xyI0_imp= xyI0_imp,
            xyI0_cont= xyI0_cont,
            t=t,
            gamma=gamma,
            E_c=E_c,
            T=T,
            func_g=func_g,
            kwargs_g=kwargs_g,
            func_f=func_f,
            kwargs_f=kwargs_f, 
            func_m=func_m,
            kwargs_m=kwargs_m,
            t_0=t_0,
            t_n=t_n,
            t_pulse=t_pulse_array[i],
            eps=eps,
            plot_population=False
            )
        if criteria['t_eta_imp'] is not None and  criteria['t_eta_cont'] is not None: #Verify if eps is reached for both model
            diff_t_eta_array[i] = criteria['t_eta_imp - t_eta_cont']
        else:
            return "epsilon is not reached for at least one t_pulse"
    
    #print(diff_t_eta_array)

    #Plot the graph got
    plt.figure()
    plt.plot(t_pulse_array, diff_t_eta_array, color = (0.3,0.4,1), linestyle = '-', label= 'Difference of time to reach the threshold')
    plt.xlabel('t_pulse')
    plt.ylabel('t_eta_imp - t_eta_cont')
    plt.title(f'Difference of time to reach epsilon with respect to t_pulse\n{T=}, {eps=}, initial value for imp model: {xyI0_imp}, initial value for cont model: {xyI0_cont}')
    plt.legend(loc= 'upper left', bbox_to_anchor=(1,1))
    plt.grid()
    plt.show()

    #A special part for the "bounds"
    if store_bound:
        t_bound = [] #The time points where it takes at least one period more or less to reach eps
        for i in range(1, len(diff_t_eta_array)):
            if np.abs(diff_t_eta_array[i] - diff_t_eta_array[i-1]) >= 0.9*T:
                t_bound.append(t_pulse_array[i])
        return t_bound
    



def plot_t_eta_contour_from_t_pulse_over_T_and_T_prop_mortality_on_x(
    xyI0_imp,
    xyI0_cont,
    t,
    gamma: float,
    E_c: float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float,
    t_n: float,
    eps: float,
    T_start: float = 1,
    T_stop: float = 20,
    T_num: int = 100,
    t_pulse_over_T_num: int = 100,
    plot_function: str = 'pcolormesh',
    levels=None,
    shading='auto',
    alpha: float = 1,
    cmap = 'bwr',    
):
    '''This function gives the contour plot of t_eta_imp - t_eta_cont with respect to T and t_pulse/T.
    t_eta_imp is the time for the impulsive model to reach the threshold epsilon.
    t_eta_cont is the time for the continuous model to reach the threshold epsilon.
    
    Param:
        xyI0_imp: a list of values of [x_imp,y_imp,I_imp] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        xyI0_cont: a list of values of [x_cont,y_cont,I_cont] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_c: continuous taking effort for pests
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        eps: the threshold below which we want to have the population of pests
        T_start: beginning of the T array. Do not make it too small to not make the code too slow (because of the small part: too much tspan)
        T_stop: end of the T array
        T_num: number of point in the T array. Do not make it too big to not make the code too slow (because of the for loop)
        t_pulse_over_T_array_num: number of point in the t_pulse/T array. Do not make it too big to not make the code too slow (because of the for loop)

        Parameters for contourf:
            plot_function: the 2D plotting function used. It must be 'contourf' or 'pcolormesh'
            levels: number and positions of the contour lines / regions in contourf is used
            shading: fill style for the quadrilateral if pcolormesh is used
            alpha: opacicty
            cmap: colormap name used to map scalar data to colors
            '''
    #Test if the plot function is among 'contourf' and 'pcolormesh'
    if plot_function not in ['contourf', 'pcolormesh']:
        return 'The argument plot_function must be "contourf" or "pcolormesh"'

    #Range of t_pulse/T (always between 0 and 1):
    t_pulse_over_T_array = np.linspace(0,1,t_pulse_over_T_num)

    #Range of T:
    T_array = np.linspace(T_start,T_stop,T_num)

    #Coordinates of the contourplot
    X, Y = np.meshgrid(t_pulse_over_T_array, T_array)

    #Calculation of the difference between t_eta_imp and t_eta_cont
    ##Matrix with the same shape as the coordinates
    diff_t_eta_matrix = np.zeros_like(X)

    ##For loops using the criteria calculated
    for i in range(len(T_array)):
        for j in range(len(t_pulse_over_T_array)):
            criteria = compare_cont_imp_proportional_mortality_on_x(
                xyI0_imp= xyI0_imp,
                xyI0_cont= xyI0_cont,
                t=t,
                gamma=gamma,
                E_c=E_c,
                T=T_array[i],
                func_g=func_g,
                kwargs_g=kwargs_g,
                func_f=func_f,
                kwargs_f=kwargs_f, 
                func_m=func_m,
                kwargs_m=kwargs_m,
                t_0=t_0,
                t_n=t_n,
                t_pulse=t_pulse_over_T_array[j] * T_array[i],
                eps=eps,
                plot_population=False
                )
            if criteria['t_eta_imp'] is not None and  criteria['t_eta_cont'] is not None: #Verify if eps is reached for both model
                diff_t_eta_matrix[i][j] = criteria['t_eta_imp - t_eta_cont']
            else:
                return "epsilon is not reached at least once" 
    
    #Contour plot
    ##Center on 0
    norm = TwoSlopeNorm(vmin = diff_t_eta_matrix.min(), vcenter=0, vmax = diff_t_eta_matrix.max())
    ##Plot
    if plot_function == 'contourf':
        contour_plot = plt.contourf(X, Y, diff_t_eta_matrix, levels=levels, alpha=alpha, cmap=cmap, norm=norm)
    elif plot_function == 'pcolormesh':
        contour_plot = plt.pcolormesh(X, Y, diff_t_eta_matrix, shading=shading, alpha=alpha, cmap=cmap, norm=norm)
    plt.colorbar(contour_plot, label = 't_imp - t_cont')  
    plt.title(f'Difference of time to reach epsilon with respect to T and t_pulse/T with \n {eps=}, {xyI0_imp} as initial value for impulsive model and {xyI0_cont} as initial value for continuous model')
    plt.xlabel('t_pulse / T')
    plt.ylabel('T')
    plt.show()

def plot_t_eta_contour_from_t_pulse_and_T_prop_mortality_on_x(
    xyI0_imp,
    xyI0_cont,
    t,
    gamma: float,
    E_c: float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float,
    t_n: float,
    eps: float,
    T_start: float = 1,
    T_stop: float = 20,
    T_num: int = 100,
    t_pulse_start: float = 0,
    t_pulse_stop: float = 20,
    t_pulse_num: int = 100,
    plot_function: str = 'pcolormesh',
    levels=None,
    shading='auto',
    alpha: float = 1,
    cmap = 'bwr',  
    plot_T_of_T : bool = False  
):  
    '''This function gives the contour plot of t_eta_imp - t_eta_cont with respect to T and t_pulse/T.
    t_eta_imp is the time for the impulsive model to reach the threshold epsilon.
    t_eta_cont is the time for the continuous model to reach the threshold epsilon.
    
    Param:
        xyI0_imp: a list of values of [x_imp,y_imp,I_imp] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        xyI0_cont: a list of values of [x_cont,y_cont,I_cont] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_c: continuous taking effort for pests
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        eps: the threshold below which we want to have the population of pests
        T_start: beginning of the T array. Do not make it too small to not make the code too slow (because of the small part: too much tspan)
        T_stop: end of the T array
        T_num: number of point in the T array. Do not make it too big to not make the code too slow (because of the for loop)
        t_pulse_start: beginning of the t_pulse array. 
        t_pulse_stop: end of the t_pulse array
        t_pulse_num: number of point in the t_pulse array. Do not make it too big to not make the code too slow (because of the for loop)

        Parameters for contourf of pcolormesh:
            plot_function: the 2D plotting function used. It must be 'contourf' or 'pcolormesh'
            levels: number and positions of the contour lines / regions if contourf is used
            shading: fill style for the quadrilateral if pcolormesh is used
            alpha: opacicty (works for both)
            cmap: colormap name used to map scalar data to colors (works for both)
            '''
    #Test if the plot function is among 'contourf' and 'pcolormesh'
    if plot_function not in ['contourf', 'pcolormesh']:
        return 'The argument plot_function must be "contourf" or "pcolormesh"'
    
    #A message to inform that t_pulse_array_stop < T_array_stop. That would mean that for the longest periods T, the highest possible values of t_pulse are not simulated. This message doesn't avoid the code to run.
    if t_pulse_stop < T_stop:
        print('For the longest periods T, the highest possible values of t_pulse are not simulated')

    #Range of t_pulse:
    t_pulse_array = np.linspace(t_pulse_start,t_pulse_stop,t_pulse_num)

    #Range of T:
    T_array = np.linspace(T_start,T_stop,T_num)

    #Coordinates of the contourplot
    X, Y = np.meshgrid(t_pulse_array, T_array)

    #Calculation of the difference between t_eta_imp and t_eta_cont
    ##Matrix with the same shape as the coordinates
    diff_t_eta_matrix = np.zeros_like(X)

    ##For loops using the criteria calculated
    for i in range(len(T_array)):
        for j in range(len(t_pulse_array)):
            criteria = compare_cont_imp_proportional_mortality_on_x(
                xyI0_imp= xyI0_imp,
                xyI0_cont= xyI0_cont,
                t=t,
                gamma=gamma,
                E_c=E_c,
                T=T_array[i],
                func_g=func_g,
                kwargs_g=kwargs_g,
                func_f=func_f,
                kwargs_f=kwargs_f, 
                func_m=func_m,
                kwargs_m=kwargs_m,
                t_0=t_0,
                t_n=t_n,
                t_pulse=t_pulse_array[j],
                eps=eps,
                plot_population=False
                )
            if criteria['t_eta_imp'] is not None and  criteria['t_eta_cont'] is not None: #Verify if eps is reached for both model
                diff_t_eta_matrix[i][j] = criteria['t_eta_imp - t_eta_cont']
            else:
                return "epsilon is not reached at least once" 
            
    #Contour plot
    ##Center on 0
    norm = TwoSlopeNorm(vmin = diff_t_eta_matrix.min(), vcenter=0, vmax = diff_t_eta_matrix.max())
    ##Plot
    ###Chose the plot function
    if plot_function == 'contourf':
        contour_plot = plt.contourf(X, Y, diff_t_eta_matrix, levels=levels, alpha=alpha, cmap=cmap, norm=norm)
    elif plot_function == 'pcolormesh':
        contour_plot = plt.pcolormesh(X, Y, diff_t_eta_matrix, shading=shading, alpha=alpha, cmap=cmap, norm=norm)
    plt.colorbar(contour_plot, label = 't_imp - t_cont')
    ### Draw T=f(T)
    if plot_T_of_T:
        plt.plot(T_array, T_array, color = 'red', label='T = t_pulse')  
    plt.title(f'Difference of time to reach epsilon with respect to T and t_pulse with \n {eps=}, {xyI0_imp} as initial value for impulsive model and {xyI0_cont} as initial value for continuous model')
    plt.xlabel('t_pulse')
    plt.ylabel('T')
    plt.show()
    

#4 Functions to retrieve the values of the periodic solution.
    
def give_init_value_last_period_prop_mortality_on_x(
    xyI,
    t,
    gamma:float,
    E_c:float,
    T:float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float,
    t_n: float,
    plot_population: bool = False,
    plot_x: bool = True,
    plot_y: bool = True
): 
    '''This function retrieves the initial value of the last period for a proportional mortality on x.
    
    Param:
        xyI: initial value [x0, y0, I0] with I0 always equal to 0 because teh integral of x always begins at 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_c: continuous taking effort for pests. We enter the continuous effort because we always refer at the continuous effort. For example for the stability.
    Later on the code, it will be converted into impulsive effort but it's better to have the continuous effort as an argument
        T: period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        
    Return:
        x_nT_plus_last: initial value of the last period
        '''
    
    #Solve the model to have t and x
    xyI_imp = solve_predator_prey_model(
        xyI=xyI,
        t=t,
        gamma=gamma,
        E_x= 1 - np.exp(-E_c*T), #E for impulsive
        E_y=0, #useless because it will be multiplied by 0. It's just to not lose the argument
        T=T,
        func_g=func_g,
        kwargs_g=kwargs_g,
        func_f=func_f,
        kwargs_f=kwargs_f, 
        func_m=func_m,
        kwargs_m=kwargs_m,
        func_h_x=return_x_x,
        kwargs_h_x={},
        func_h_y=return_zero_y, 
        kwargs_h_y={},
        t_0=t_0,
        t_n=t_n  
        )
    
    t_list = xyI_imp[0] #time points
    x_list = xyI_imp[1] #population size with respect to time

    #Find the last pulse time: it's the last time point that appears twice
    unique, counts = np.unique(t_list, return_counts=True) #All the time points with their number of appearences in t
    doubles = unique[counts > 1] #All the time points that appear more than once (i.e. every time of pulse)

    if len(doubles) == 0:
        return "No duplicates found in t." #An error message if there is no double (for example if t_n < T)

    last_pulse_time = doubles[-2] #Because t_n is already a double by the way solve_predator_prey is coded

    # The indexes where t_list == last_pulse_time
    indexes = np.where(t_list == last_pulse_time )[0]

    # Last index of the last pulse time (i.e. the index of the last nT_plus)
    nT_plus_last_index = indexes[-1]

    # Retrieve x at this index
    x_nT_plus_last = x_list[nT_plus_last_index]

    if plot_population:
        compare_cont_imp_proportional_mortality_on_x(
        xyI0_imp= xyI,
        xyI0_cont= xyI,
        t=t,
        gamma=gamma,
        E_c=E_c,
        T=T,
        func_g=func_g,
        kwargs_g=kwargs_g,
        func_f=func_f,
        kwargs_f=kwargs_f, 
        func_m=func_m,
        kwargs_m=kwargs_m,
        t_0=t_0,
        t_n=t_n,
        t_pulse=T,
        eps=0.01,
        plot_population=True,
        plot_x=plot_x,
        plot_y=plot_y
    )

    return x_nT_plus_last

def find_x_y_p_0_with_error_prop_mortality_on_x(
    xyI,
    t,
    gamma:float,
    E_c:float,
    T:float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float = 0,
    t_n: float = 1000,
    error:float = 1e-3,
    plot_population: bool = False
):
    '''This function retrieves the initial value of the periodic solution for both x and y and for the impulsive model with proportional mortality on x.
    
    Param:
        xyI: initial value [x0, y0, I0] with I0 always equal to 0 because teh integral of x always begins at 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_c: continuous taking effort for pests. We enter the continuous effort because we always refer at the continuous effort. For example for the stability.
    Later on the code, it will be converted into impulsive effort but it's better to have the continuous effort as an argument
        T: period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain that may not be reached because the periodic solution is reached sooner. It's adviced to make it high
        error: the difference tolerated between the initial value of a period and the initial value of the next one
        
    Return:
        x_p_0: estimated initial value of x_p
        y_p_0: estimated initial value of y_p
        '''
    
    #Initialisation

    ##Initial values of the periodic solution that will be estimated
    x_p_0 = None
    y_p_0 = None
    ##Initial values of the current period
    x_nT_plus = xyI[0]
    y_nT_plus = xyI[1]
    ##Initial values of the next period
    x_nT_plus_next = None
    y_nT_plus_next = None
    ##Vector for the plot
    x_plot = [xyI[0]]
    y_plot = [xyI[1]]
    t_plot = [t_0]
    ##Index of the number of the current period - 1
    n=0

    #Recursivity
    for _ in range(t_0, t_n, T): #period by period 
        txyI_span = solve_predator_prey_model(
            xyI=[x_nT_plus,y_nT_plus,0], #The current function doesn't need the integral
            t=t,
            gamma=gamma,
            E_x= 1 - np.exp(-E_c*T), #E for impulsive
            E_y=0, #useless because it will be multiplied by 0. It's just to not lose the argument
            T=T,
            func_g=func_g,
            kwargs_g=kwargs_g,
            func_f=func_f,
            kwargs_f=kwargs_f, 
            func_m=func_m,
            kwargs_m=kwargs_m,
            func_h_x=return_x_x,
            kwargs_h_x={},
            func_h_y=return_zero_y, 
            kwargs_h_y={},
            t_0=t_0 + n*T,
            t_n=t_0 + (n+1)*T #The ODE is solve at the current period
            )
        ##The spans and the initial value of the next period
        x_span = txyI_span[1]
        x_nT_plus_next = np.exp(-E_c*T) * x_span[-1]
        y_span = txyI_span[2]
        y_nT_plus_next = y_span[-1]
        t_span = txyI_span[0]

        ##Extend the vectors for the plot
        x_plot.extend(x_span)
        y_plot.extend(y_span)
        t_plot.extend(t_span)
        
        ##Verify if the initial values of the current and the next period are close enough, and the same for (x(nT), y(nT)) and (x((n+1)T), y((n+1)T))
        if (np.abs(x_nT_plus - x_nT_plus_next) < error and
            np.abs(y_nT_plus - y_nT_plus_next) < error):
            x_p_0 = x_nT_plus_next
            y_p_0 = y_nT_plus_next
            break #Stop if the initial value of the periodic solution is found
        else: #Continue the simulation if the initial value of the periodic solution is not found
            x_nT_plus = x_nT_plus_next
            y_nT_plus = y_nT_plus_next #The new initial values of the period
            n+=1 #Next period
    
    #Plot
    if plot_population:
        plt.figure()
        plt.plot(t_plot, x_plot, color = (0,0,0.9), linestyle='-', label=f'Pest population x with {xyI} as initial value')
        plt.plot(t_plot, y_plot, color = (0.9,0,0), linestyle='-', label=f'Predator population y with {xyI} as initial value')
        plt.xlabel('time')
        plt.ylabel('Population size')
        plt.title(f'Population of pests and predators with exogenous mortality on pests \n with {xyI} as initial value, E = {E_c}, {T = }')
        plt.legend(loc= 'upper left', bbox_to_anchor=(1,1))
        plt.grid()
        plt.show()

    #Return
    if x_p_0 == None or y_p_0 == None:
        print(f'Periodic solution not found: {x_nT_plus_next = }, {y_nT_plus_next = }. \n Increase t_n or the error')
    else:
        return x_p_0, y_p_0
    
def find_x_y_p_T_with_error_prop_mortality_on_x(
    xyI,
    t,
    gamma:float,
    E_c:float,
    T:float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float = 0,
    t_n: float = 1000,
    error:float = 1e-3,
    plot_population: bool = False
):
    '''This function retrieves (x_p(T),y_p(T)) for the impulsive model with proportional mortality on x.
    
    Param:
        xyI: initial value [x0, y0, I0] with I0 always equal to 0 because teh integral of x always begins at 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_c: continuous taking effort for pests. We enter the continuous effort because we always refer at the continuous effort. For example for the stability.
    Later on the code, it will be converted into impulsive effort but it's better to have the continuous effort as an argument
        T: period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain that may not be reached because the periodic solution is reached sooner. It's adviced to make it high
        error: the difference tolerated between the initial value of a period and the initial value of the next one
        
    Return:
        x_p_T: estimated value of x_p(T)
        y_p_T: estimated value of y_p(T)
        '''
    
    #Initialisation

    ##(x_p(T),y_p(T)) that will be estimated
    x_p_T = None
    y_p_T = None
    ##Index of the number of the current period - 1
    n=0

    ##Get the first (x(T), y(T))
    txyI_span = solve_predator_prey_model(
            xyI=xyI, #The initial value
            t=t,
            gamma=gamma,
            E_x= 1 - np.exp(-E_c*T), #E for impulsive
            E_y=0, #useless because it will be multiplied by 0. It's just to not lose the argument
            T=T,
            func_g=func_g,
            kwargs_g=kwargs_g,
            func_f=func_f,
            kwargs_f=kwargs_f, 
            func_m=func_m,
            kwargs_m=kwargs_m,
            func_h_x=return_x_x,
            kwargs_h_x={},
            func_h_y=return_zero_y, 
            kwargs_h_y={},
            t_0=t_0 + n*T,
            t_n=t_0 + (n+1)*T #The ODE is solve at the current period
            )
    x_span = txyI_span[1]
    x_nT = x_span[-1]
    y_span = txyI_span[2]
    y_nT = y_span[-1]
    t_span = txyI_span[0]

    ##(x(T), y(T)) for the next period
    x_nT_next = None
    y_nT_next = None

    ##Vectors for the plot
    x_plot = []
    x_plot.extend(x_span)
    y_plot = []
    y_plot.extend(y_span)
    t_plot = []
    t_plot.extend(t_span)

    ##Next initial value
    x_nT_plus = np.exp(-E_c*T) * x_span[-1]
    y_nT_plus = y_span[-1]
    ##Next period
    n+=1

    #Recursivity
    for _ in range(t_0, t_n, T): #period by period 
        txyI_span = solve_predator_prey_model(
            xyI=[x_nT_plus,y_nT_plus,0], #The current function doesn't need the integral
            t=t,
            gamma=gamma,
            E_x= 1 - np.exp(-E_c*T), #E for impulsive
            E_y=0, #useless because it will be multiplied by 0. It's just to not lose the argument
            T=T,
            func_g=func_g,
            kwargs_g=kwargs_g,
            func_f=func_f,
            kwargs_f=kwargs_f, 
            func_m=func_m,
            kwargs_m=kwargs_m,
            func_h_x=return_x_x,
            kwargs_h_x={},
            func_h_y=return_zero_y, 
            kwargs_h_y={},
            t_0=t_0 + n*T,
            t_n=t_0 + (n+1)*T #The ODE is solve at the current period
            )
        ##The spans, the initial value of the next period, and the next (x(nT), y(nT))
        x_span = txyI_span[1]
        x_nT_plus_next = np.exp(-E_c*T) * x_span[-1]
        x_nT_next = x_span[-1]

        y_span = txyI_span[2]
        y_nT_plus_next = y_span[-1]
        y_nT_next = y_span[-1]

        t_span = txyI_span[0]

        ##Extend the vectors for the plot
        x_plot.extend(x_span)
        y_plot.extend(y_span)
        t_plot.extend(t_span)
        
        ##Verify if (x(nT), y(nT)) and (x((n+1)T), y((n+1)T)) are close enough
        if (np.abs(x_nT - x_nT_next) < error and
            np.abs(y_nT - y_nT_next) < error):
            x_p_T = x_nT_next
            y_p_T = y_nT_next
            break #Stop if (x_p(T), y_p(T)) is found
        else: #Continue the simulation if not
            x_nT = x_nT_next
            y_nT = y_nT_next #The new (x(nT), y(nT))
            x_nT_plus = x_nT_plus_next
            y_nT_plus = y_nT_plus_next #The new initial values of the period
            n+=1 #Next period
    
    #Plot
    if plot_population:
        plt.figure()
        plt.plot(t_plot, x_plot, color = (0,0,0.9), linestyle='-', label=f'Pest population x with {xyI} as initial value')
        plt.plot(t_plot, y_plot, color = (0.9,0,0), linestyle='-', label=f'Predator population y with {xyI} as initial value')
        plt.xlabel('time')
        plt.ylabel('Population size')
        plt.title(f'Population of pests and predators with exogenous mortality on pests \n with {xyI} as initial value, E = {E_c}, {T = }')
        plt.legend(loc= 'upper left', bbox_to_anchor=(1,1))
        plt.grid()
        plt.show()

    #Return
    if  x_p_T == None or y_p_T == None:
        print(f'Periodic solution not found\n Last estimated values:  {x_nT_next = }, {y_nT_next = }. \n Increase t_n or the error')
    else:
        return x_p_T, y_p_T

def find_x_y_p_with_error_prop_mortality_on_x(
    xyI,
    t,
    gamma:float,
    E_c:float,
    T:float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float = 0,
    t_n: float = 1000,
    error:float = 1e-3,
    plot_population: bool = False
):
    '''This function retrieves (x_p(0),y_p(0)) and (x_p(T),y_p(T)) for the impulsive model with proportional mortality on x.
    
    Param:
        xyI: initial value [x0, y0, I0] with I0 always equal to 0 because teh integral of x always begins at 0
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        E_c: continuous taking effort for pests. We enter the continuous effort because we always refer at the continuous effort. For example for the stability.
    Later on the code, it will be converted into impulsive effort but it's better to have the continuous effort as an argument
        T: period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain that may not be reached because the periodic solution is reached sooner. It's adviced to make it high
        error: the difference tolerated between the initial value of a period and the initial value of the next one
        
    Return:
        x_p_0: estimated initial value of x_p
        y_p_0: estimated initial value of y_p
        x_p_T: estimated value of x_p(T)
        y_p_T: estimated value of y_p(T)
        '''
    
    #Initialisation

    ##(x_p(0),y_p(0)) and (x_p(T),y_p(T)) that will be estimated
    x_p_0 = None
    y_p_0 = None
    x_p_T = None
    y_p_T = None
    ##Index of the number of the current period - 1
    n=0

    ##Get the first (x_p(0),y_p(0)) and (x_p(T),y_p(T))
    txyI_span = solve_predator_prey_model(
            xyI=xyI, #The initial value
            t=t,
            gamma=gamma,
            E_x= 1 - np.exp(-E_c*T), #E for impulsive
            E_y=0, #useless because it will be multiplied by 0. It's just to not lose the argument
            T=T,
            func_g=func_g,
            kwargs_g=kwargs_g,
            func_f=func_f,
            kwargs_f=kwargs_f, 
            func_m=func_m,
            kwargs_m=kwargs_m,
            func_h_x=return_x_x,
            kwargs_h_x={},
            func_h_y=return_zero_y, 
            kwargs_h_y={},
            t_0=t_0 + n*T,
            t_n=t_0 + (n+1)*T #The ODE is solve at the current period
            )
    x_span = txyI_span[1]
    x_nT = x_span[-1]
    y_span = txyI_span[2]
    y_nT = y_span[-1]
    t_span = txyI_span[0]

    ##(x(T), y(T)) for the next period
    x_nT_next = None
    y_nT_next = None

    ##Vectors for the plot
    x_plot = []
    x_plot.extend(x_span)
    y_plot = []
    y_plot.extend(y_span)
    t_plot = []
    t_plot.extend(t_span)

    ##Next initial value
    x_nT_plus = np.exp(-E_c*T) * x_nT
    y_nT_plus = y_nT
    ##Next period
    n+=1

    #Recursivity
    for _ in range(t_0, t_n, T): #period by period #Non non non
        txyI_span = solve_predator_prey_model(
            xyI=[x_nT_plus,y_nT_plus,0], #The current function doesn't need the integral
            t=t,
            gamma=gamma,
            E_x= 1 - np.exp(-E_c*T), #E for impulsive
            E_y=0, #useless because it will be multiplied by 0. It's just to not lose the argument
            T=T,
            func_g=func_g,
            kwargs_g=kwargs_g,
            func_f=func_f,
            kwargs_f=kwargs_f, 
            func_m=func_m,
            kwargs_m=kwargs_m,
            func_h_x=return_x_x,
            kwargs_h_x={},
            func_h_y=return_zero_y, 
            kwargs_h_y={},
            t_0=t_0 + n*T,
            t_n=t_0 + (n+1)*T #The ODE is solve at the current period
            )
        ##The spans, the initial value of the next period, and the next (x(nT), y(nT))
        x_span = txyI_span[1]
        x_nT_next = x_span[-1]
        x_nT_plus_next = np.exp(-E_c*T) * x_nT_next
        

        y_span = txyI_span[2]
        y_nT_next = y_span[-1]
        y_nT_plus_next = y_nT_next
        

        t_span = txyI_span[0]

        ##Extend the vectors for the plot
        x_plot.extend(x_span)
        y_plot.extend(y_span)
        t_plot.extend(t_span)
        
        ##Verify if (x(nT), y(nT)) and (x((n+1)T), y((n+1)T)) are close enough
        if (np.abs(x_nT - x_nT_next) < error and
            np.abs(x_nT_plus - x_nT_plus_next) < error and
            np.abs(y_nT_plus - y_nT_plus_next) < error):
            x_p_T = x_nT_next
            y_p_T = y_nT_next
            x_p_0 = x_nT_plus_next
            y_p_0 = y_nT_plus_next
            break #Stop if (x(nT), y(nT)) and (x_p(T), y_p(T)) is found
        else: #Continue the simulation if not
            x_nT = x_nT_next
            y_nT = y_nT_next #The new (x(nT), y(nT))
            x_nT_plus = x_nT_plus_next
            y_nT_plus = y_nT_plus_next #The new initial values of the next period
            n+=1 #Next period
    
    #Plot
    if plot_population:
        plt.figure()
        plt.plot(t_plot, x_plot, color = (0,0,0.9), linestyle='-', label=f'Pest population x with {xyI} as initial value')
        plt.plot(t_plot, y_plot, color = (0.9,0,0), linestyle='-', label=f'Predator population y with {xyI} as initial value')
        plt.xlabel('time')
        plt.ylabel('Population size')
        plt.title(f'Population of pests and predators with exogenous mortality on pests \n with {xyI} as initial value, E = {E_c}, {T = }')
        plt.legend(loc= 'upper left', bbox_to_anchor=(1,1))
        plt.grid()
        plt.show()

    #Return
    if x_p_0 == None or y_p_0 == None or x_p_T == None or y_p_T == None:
        print(f'Periodic solution not found:\n Last estimated values {x_nT_next = }, {y_nT_next = }, {x_nT_plus_next = }, {y_nT_plus_next = }. \n Increase t_n or the error')
    else:
        return x_p_0, y_p_0, x_p_T, y_p_T

#To store values of the periodic solution

def store_x_p_0_prop_mortality_on_x(
    xyI,
    t,
    gamma:float,
    T:float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float,
    t_n: float,
    E_c_start: float,
    E_c_stop: float,
    E_c_num: int= 100
):
    '''This function store the initial values of the simulated periodic solution with respect to a range of E_c.
    
    Param:
        xyI: initial value
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        T: period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        E_c_start: beginning of the E_c array
        E_c_stop: end of the E_c array
        E_c_num: number of point in the E_c array
        
    Return: 
        x_p_0_array: an array with all the estimated initial values with respect to E_c'''
    
    #The array of E_c
    E_c_array = np.linspace(E_c_start, E_c_stop, E_c_num) 
    #The array of the initial value of the periodic solution
    x_p_0_array = np.zeros_like(E_c_array) 
    for i in range(len(E_c_array)):
        x_p_0_array[i] = give_init_value_last_period_prop_mortality_on_x(
        xyI=xyI,
        t=t,
        gamma=gamma,
        E_c=E_c_array[i],
        T=T,
        func_g=func_g,
        kwargs_g=kwargs_g,
        func_f=func_f,
        kwargs_f=kwargs_f, 
        func_m=func_m,
        kwargs_m=kwargs_m,
        t_0=t_0,
        t_n=t_n,
        plot_population=False
    )
        
    return E_c_array, x_p_0_array

def store_x_p_0_prop_mortality_on_x_with_error(
    xyI,
    t,
    gamma:float,
    T:float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float,
    t_n: float,
    E_c_start: float,
    E_c_stop: float,
    E_c_num: int= 100,
    error: float = 1e-3
):
    
    '''This function store the initial values of the simulated periodic solution with respect to a range of E_c.
    
    Param:
        xyI: initial value for the first value of E
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        T: period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        E_c_start: beginning of the E_c array
        E_c_stop: end of the E_c array
        E_c_num: number of point in the E_c array
        error: the difference tolerated between the initial value of a period and the initial value of the next one
        
    Return: 
        x_p_0_array: an array with all the estimated x_p_0 with respect to E_c
        y_p_0_array: an array with all the estimated y_p_0 with respect to E_c'''
    
    #The array of E_c
    E_c_array = np.linspace(E_c_start, E_c_stop, E_c_num) 
    #The arrays of the initial value of the periodic solution
    x_p_0_array = np.zeros_like(E_c_array)
    y_p_0_array = np.zeros_like(E_c_array)

    #Fill the arrays of the initial value of the periodic solution
    xyI0_E = xyI #initial value for the first value of E. This variable will change depending of the next E
    for i in range(len(E_c_array)):
        x_y_p_0 = find_x_y_p_0_with_error_prop_mortality_on_x(
            xyI = xyI0_E,
            t=t,
            gamma=gamma,
            E_c=E_c_array[i],
            T=T,
            func_g=func_g,
            kwargs_g=kwargs_g,
            func_f=func_f,
            kwargs_f=kwargs_f,
            func_m=func_m,
            kwargs_m=kwargs_m, 
            t_0=t_0,
            t_n=t_n,
            error=error,
            plot_population = False
        )
        x_p_0_array[i] = x_y_p_0[0] 
        y_p_0_array[i] = x_y_p_0[1] #Store the initial values found for E in the arrays

        xyI0_E = [x_y_p_0[0], x_y_p_0[1], 0] #Change the initial value for the next E to the current initial value
    
    return E_c_array, x_p_0_array, y_p_0_array

def store_x_y_p_prop_mortality_on_x_with_error(
    xyI,
    t,
    gamma:float,
    T:float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float], 
    t_0: float,
    t_n: float,
    E_c_start: float,
    E_c_stop: float,
    E_c_num: int= 100,
    error: float = 1e-3
):
    '''This function store (x_p_0,y_p_0) and (x_p(T),y_p(T)) of the simulated periodic solution with respect to a range of E_c.
    
    Param:
        xyI: initial value for the first value of E
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so we can put whatever we want)
        gamma: conversion factor
        T: period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        E_c_start: beginning of the E_c array
        E_c_stop: end of the E_c array
        E_c_num: number of point in the E_c array
        error: the difference tolerated between the initial value of a period and the initial value of the next one
        
    Return: 
        E_c_array : an array of the different values of E_c
        x_p_0_array: an array with all the estimated x_p_0 with respect to E_c
        y_p_0_array: an array with all the estimated y_p_0 with respect to E_c
        x_p_T_array: an array with all the estimated x_p_T with respect to E_c
        y_p_T_array: an array with all the estimated y_p_T with respect to E_c'''
    
    #The array of E_c
    E_c_array = np.linspace(E_c_start, E_c_stop, E_c_num) 
    #The arrays of the initial value of the periodic solution
    x_p_0_array = np.zeros_like(E_c_array)
    y_p_0_array = np.zeros_like(E_c_array)
    x_p_T_array = np.zeros_like(E_c_array)
    y_p_T_array = np.zeros_like(E_c_array)

    #Fill the arrays of the initial value of the periodic solution
    xyI0_E = xyI #initial value for the first value of E. This variable will change depending of the next E
    for i in range(len(E_c_array)):
        x_y_p = find_x_y_p_with_error_prop_mortality_on_x(
            xyI = xyI0_E,
            t=t,
            gamma=gamma,
            E_c=E_c_array[i],
            T=T,
            func_g=func_g,
            kwargs_g=kwargs_g,
            func_f=func_f,
            kwargs_f=kwargs_f,
            func_m=func_m,
            kwargs_m=kwargs_m, 
            t_0=t_0,
            t_n=t_n,
            error=error,
            plot_population = False
        )
        x_p_0_array[i] = x_y_p[0] 
        y_p_0_array[i] = x_y_p[1]
        x_p_T_array[i] = x_y_p[2] 
        y_p_T_array[i] = x_y_p[3] #Store the initial values found for E in the arrays

        xyI0_E = [x_y_p[0], x_y_p[1], 0] #Change the initial value for the next E to the current initial value
    
    return E_c_array, x_p_0_array, y_p_0_array, x_p_T_array, y_p_T_array