###This module is a module like exo.py but an exogenous mortality factor affects both x and y.

#Dependancies
import numpy as np
from scipy.integrate import odeint
from typing import Callable

#Pre-implemented functions that can be used in the models (like in exo.py)

##Functions with both x and y as two first arguments
def identity(x, y, z: float):
    '''This function returns the argument z itself
    x: pest population that is not used by the function but will be needed for the general model
    y: predator population that is not used by the function but will be needed for the general model
    z: the argument will be returned'''
    return z

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
        a: search rate
        
    Return:
        dx: a list of the population size of x'''
    
    #Continuous part of the model
    dx = r * (1 - x/K)

    return dx
##Functions with y as first argument but not x
def identity_y(y, z: float):
    '''This function returns the argument z itself
    y: predator population that is not used by the function but will be needed for the general model
    z: the argument will be returned'''
    return z

def return_one(y):
    '''This function returns 1
    y: predator population that is not used by the function but will be needed for the general model
    '''
    return 1

def return_zero(y):
    '''This function returns 0
    y: predator population that is not used by the function but will be needed for the general model
    '''
    return 0

def return_y_y(x,y):
    '''This function returns y when it's sometimes needed
    y: predator population 
    '''
    return y

#General model with a discrete part for both x and y 
##All functions depends on x and y
def predator_prey_model(
    xy,
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
        xy: a list of values of [x,y] at a time t_n
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so put whatever you want)
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
    x = xy[0]
    y = xy[1]

    #Continuous part of the model
    dx = func_g(x, **kwargs_g)*x - func_f(x, y, **kwargs_f)*y
    dy = gamma * func_f(x, y, **kwargs_f) * y - func_m(x, y, **kwargs_m)*y
    
    return dx, dy

def solve_predator_prey_model(
    xy,
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
    
    Param:
        xy: a list of values of [x,y] at a time t_n
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so put whatever you want)
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
        exo.solve_predator_prey_model(
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

    #Record initial conditions
    x.append(xy[0])
    y.append(xy[1])

    #Time points
    t = [t_0]

    #Solve ODE
    intervals = np.arange(t_0, t_n, T) #divide the domain in intervals on length T
    intervals = np.append(intervals, t_n) #add t_n to intervals because t_n is not reached by arange
    ##Initial values before entring into the loop
    x_kT_plus = xy[0]
    y_kT_plus = xy[1]
    for i in range(1,len(intervals)):
        xy_kT_plus = [x_kT_plus,y_kT_plus] #the initial value in a period is [x(kT+), y(kT+)]
        ##Span for this period
        tspan = np.linspace(intervals[i-1], intervals[i], 101)
        tspan = np.append(tspan, intervals[i]) 
        ##Solve for this period
        xy_step = odeint(predator_prey_model, xy_kT_plus, tspan, 
                         args=(gamma, func_g, kwargs_g, func_f, kwargs_f, func_m, kwargs_m)) 
        x.extend(xy_step.T[0]) #Continuous part of x
        x_kT_plus = xy_step.T[0][-1] - E_x*func_h_x(xy_step.T[0][-1], **kwargs_h_x) #Applying func_h_x to x(nT)
        y.extend(xy_step.T[1]) 
        y_kT_plus = xy_step.T[0][-1] - E_y*func_h_y(xy_step.T[1][-1], **kwargs_h_y) #Applying func_h_y to y(nT)
        t.extend(tspan)

    return x, y, t
