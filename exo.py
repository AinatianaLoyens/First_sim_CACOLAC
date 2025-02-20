###The functions of this module does not have default values anymore because it's way better to crash than to give the wrong result.

#Dependancies
import numpy as np
from scipy.integrate import odeint

print('With dicrete part')

#Logistic growth model
def logistic_model(
    x,
    t,
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
    dx = r*x * (1 - x/K)

    return dx

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
    
