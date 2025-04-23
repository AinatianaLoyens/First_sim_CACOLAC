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

#Pre-implemented functions that can be used in the models (like in exo.py)

##Functions with both x and y as two first arguments
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

##Functions with x as first argument but not y
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

##Functions with y as first argument but not x
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

#General model with a discrete part for both x and y 

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

#Functions to help when plotting the comparison between continuous and impulsive models
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

#Functions to plot results and to compare the criteria

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
    plot_population: bool = False
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
        xyI: a list of values of [x,y,I] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
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
        func_h_x: harvesting function for x
        kwargs_h_x: a dictionnary of the arguments of the other arguments of func_h_x that is not x(nT). x(nT) is the first argument of func_h_x
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        eps: the threshold below which we want to have the population of pests
        plot_population: to precise if we want to plot the population size
        
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
        plt.plot(t, x_cont, color = (0,0,0.9), linestyle='-', label=f'x_cont with {xyI0_cont} as initial value')
        plt.plot(t, y_cont, color = (0.9,0,0), linestyle='-', label=f'y_cont with {xyI0_cont} as initial value')
        plt.plot(t, x_imp, color = (0,0,0.9), linestyle='--', label=f'x_imp with {xyI0_imp} as initial value')
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
    plot_population: bool = False
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
        xyI: a list of values of [x,y,I] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
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
        func_h_x: harvesting function for x
        kwargs_h_x: a dictionnary of the arguments of the other arguments of func_h_x that is not x(nT). x(nT) is the first argument of func_h_x
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        eps: the threshold below which we want to have the population of pests
        plot_population: to precise if we want to plot the population size
        
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
        plt.plot(t, y_cont, color = (0.9,0,0), linestyle='-', label=f'y_cont with {xyI0_cont} as initial value')
        plt.plot(t, x_imp, color = (0,0,0.9), linestyle='--', label=f'x_imp with {xyI0_imp} as initial value')
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
    plot_population: bool = False
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
        xyI: a list of values of [x,y,I] at a time t_i. I must be 0 because its first the integral of x from t_0 to t_0, which is 0
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
        func_h_x: harvesting function for x
        kwargs_h_x: a dictionnary of the arguments of the other arguments of func_h_x that is not x(nT). x(nT) is the first argument of func_h_x
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        t_pulse: time of first impulsion
        eps: the threshold below which we want to have the population of pests
        plot_population: to precise if we want to plot the population size
        
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
        plt.plot(t, x_cont, color = (0,0,0.9), linestyle='-', label=f'x_cont with {xyI0_cont} as initial value')
        plt.plot(t, y_cont, color = (0.9,0,0), linestyle='-', label=f'y_cont with {xyI0_cont} as initial value')
        plt.plot(t, x_imp, color = (0,0,0.9), linestyle='--', label=f'x_imp with {xyI0_imp} as initial value')
        plt.plot(t, y_imp, color = (0.9,0,0), linestyle='--', label=f'y_imp with {xyI0_imp} as initial value')
        plt.xlabel('time')
        plt.ylabel('Population size')
        plt.title(f'Population of pests and predators with continuous and impulsive exogenous mortality on pests and the first impulsive exogenous mortality at t = {t_pulse}')
        plt.suptitle(f'{kwargs_g}, {E_c = }, {T = },')
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