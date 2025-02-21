###The functions of this module does not have default values anymore because it's way better to crash than to give the wrong result.

#Dependancies
import numpy as np
from scipy.integrate import odeint
from typing import Callable

print('Logistic growth')

#Pre-implemented functions that can be used in the models
#Those functions need to have x and y as two first arguments
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
    dx = r*x * (1 - x/K)

    return dx

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

#Basic Lotka-Volterra model
def basic_lv_model(
        xy,
        t,
        r: float,
        a: float,
        gamma: float,
        m: float):
    '''This model is a continuous Lotka-Volterra model that describes the evolution of a pest population x and a predator population y.
    the growth rate is the simpliest possible as it is only r, even if it is not realistic without carrying capacity.
    
    Param:
        xy: a list of values of [x,y] at a time t_n
        r: growth rate
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so put whatever you want)
        a: search rate
        gamma: conversion factor
        m: death rate
          
    Return:
        dx, dy: a list of the two population size of x and y at time t_{n+1}
    '''

    #Initialisation
    x = xy[0]
    y = xy[1]

    #Continuous part of the model
    dx = r*x * - a*x * y
    dy = gamma * a*x * y - m*y

    return dx, dy

def solve_basic_lv_model(
        xy,
        t,
        r: float,
        a: float,
        gamma: float,
        m: float,
        E: float,
        T: float,
        t_0: float,
        t_n: float
):
    
    '''This function gives the anwser of the semi-discrete ODE system with the basic Lotka-Volterra model 
    
    Param:
        xy: put the initial value here. It will be changed along the for loop
        r: growth rate
        a: search rate
        gamma: conversion factor
        m: death rate
        E: taking effort
        T: release period
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        
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
    x_kT_plus = xy[0] #Initial values before entring into the loop
    for i in range(1,len(intervals)):
        xy_kT_plus = [x_kT_plus,y[-1]] #the initial value in a period is [x(kT+), y(kT+)]
        #Span for this period
        tspan = np.linspace(intervals[i-1], intervals[i], 101)
        tspan = np.append(tspan, intervals[i]) 
        #Solve for this period
        xy_step = odeint(basic_lv_model, xy_kT_plus, tspan, args=(r, a, gamma, m)) 
        x.extend(xy_step.T[0]) #Continuous part of x
        x_kT_plus = xy_step.T[0][-1] - E*xy_step.T[0][-1]#Equation of the discrete part
        y.extend(xy_step.T[1]) 

        t.extend(tspan)

    return x, y, t
    
#Lotka-Volterra model with logistic growth as growth rate
def logistic_lv_model(
        xy,
        t,
        r: float,
        K: float,
        a: float,
        gamma: float,
        m: float):
    '''This model is a continuous Lotka-Volterra model that describes the evolution of a pest population x and a predator population y.
    the growth rate is the logistic growth
    
    Param:
        xy: a list of values of [x,y] at a time t_n
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so put whatever you want)
        r: growth rate
        K: carrying capacity
        a: search rate
        gamma: conversion factor
        m: death rate
          
    Return:
        dx, dy: a list of the two population size of x and y at time t_{n+1}
    '''

    #Initialisation
    x = xy[0]
    y = xy[1]

    #Continuous part of the model
    dx = logistic_model(x,y,r,K)*x * - a*x * y
    dy = gamma * a*x * y - m*y

    return dx, dy

def solve_logistic_lv_model(
        xy,
        t,
        r: float,
        K: float,
        a: float,
        gamma: float,
        m: float,
        E: float,
        T: float,
        t_0: float,
        t_n: float
):
    
    '''This function gives the anwser of the semi-discrete ODE system with the logistic Lotka-Volterra model 
    
    Param:
        xy: put the initial value here. It will be changed along the for loop
        r: growth rate
        K: carrying capacity
        a: search rate
        gamma: conversion factor
        m: death rate
        E: taking effort
        T: release period
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain
        
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
    x_kT_plus = xy[0] #Initial values before entring into the loop
    for i in range(1,len(intervals)):
        xy_kT_plus = [x_kT_plus,y[-1]] #the initial value in a period is [x(kT+), y(kT+)]
        #Span for this period
        tspan = np.linspace(intervals[i-1], intervals[i], 101)
        tspan = np.append(tspan, intervals[i]) 
        #Solve for this period
        xy_step = odeint(logistic_lv_model, xy_kT_plus, tspan, args=(r, K, a, gamma, m)) 
        x.extend(xy_step.T[0]) #Continuous part of x
        x_kT_plus = xy_step.T[0][-1] - E*xy_step.T[0][-1]#Equation of the discrete part
        y.extend(xy_step.T[1]) 

        t.extend(tspan)

    return x, y, t

#General model

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
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m

    Example of run:
        predator_prey_model(
        xy=xy,
        t=t,
        gamma=gamma,
        func_g=identity,
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
    dx = func_g(x, y, **kwargs_g)*x - func_f(x, y, **kwargs_f)*y
    dy = gamma * func_f(x, y, **kwargs_f) * y - func_m(x, y, **kwargs_m)*y
    
    return dx, dy

def solve_predator_prey_model(
    xy,
    t,
    gamma: float,
    E: float,
    T: float,
    func_g: Callable[..., float],
    kwargs_g: dict[str, float],
    func_f: Callable[..., float],
    kwargs_f: dict[str, float],
    func_m: Callable[..., float],
    kwargs_m: dict[str, float],
    func_h: Callable[..., float],
    kwargs_h: dict[str, float], 
    t_0: float,
    t_n: float
):
    '''This function aims to be a general predator-prey model where the user decides on the functions that will be used in the model
    
    Param:
        xy: a list of values of [x,y] at a time t_n
        t: time points (it is not used in the function but we need to put it to make the function usable to the solver, so put whatever you want)
        gamma: conversion factor
        E: taking effort
        T: release period
        func_g: the growth rate function
        kwargs_g: a dictionnary of the arguments of func_g
        func_f: the response function
        kwargs_f: a dictionnary of the arguments of func_f
        func_m: mortality rate function
        kwargs_m: a dictionnary of the arguments of func_m
        func_h: harvesting function
        kwargs_h: a dictionnary of the arguments of the other arguments of func_h that are not x(nT). x(nT) is the first argument of func_h
        t_0: left endpoint of the domain
        t_n: right endpoint of the domain

    Example of run:
        exo.solve_predator_prey_model(
        xy=x0y0_a,
        t=t,
        gamma=gamma,
        E=E,
        T=T,
        func_g=exo.identity,
        kwargs_g={'z': r},
        func_f=exo.multiply_x,
        kwargs_f={'z': a}, 
        func_m=exo.identity,
        kwargs_m={'z': m},
        func_h=exo.return_x,
        kwargs_h={},
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
    x_kT_plus = xy[0] #Initial values before entring into the loop
    for i in range(1,len(intervals)):
        xy_kT_plus = [x_kT_plus,y[-1]] #the initial value in a period is [x(kT+), y(kT+)]
        #Span for this period
        tspan = np.linspace(intervals[i-1], intervals[i], 101)
        tspan = np.append(tspan, intervals[i]) 
        #Solve for this period
        xy_step = odeint(predator_prey_model, xy_kT_plus, tspan, 
                         args=(gamma, func_g, kwargs_g, func_f, kwargs_f, func_m, kwargs_m)) 
        x.extend(xy_step.T[0]) #Continuous part of x
        x_kT_plus = xy_step.T[0][-1] - E*func_h(xy_step.T[0][-1], xy_step.T[0][-1], **kwargs_h) #Applying func_h to x(nT)
        y.extend(xy_step.T[1]) 

        t.extend(tspan)

    return x, y, t
