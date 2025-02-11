#Dependancies
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

###print('xxx')

#Models
##No interference model
def no_int_model(
    xy: list,
    t = np.linspace(0,20,201),
    r: float = 0.5,
    K: float = 10,
    a: float = 20,
    c: float = 20,
    m: float = 0.1,
    gamma: float = 0.8,
) -> list:
    
    '''This model is a continuous model that describes the evolution of a pest population x and a predator population y 
    
    Param:
        xy: a list of values of [x,y] at a time t_n
        r: growth rate
        K: carrying capacity
        a: search rate
        c: half-saturation constant
        m: death rate
        gamma: conversion factor
        
    Return:
        dx, dy: a list of the two population size of x and y at time t_{n+1}'''
    
    #Initialisation
    x = xy[0]
    y = xy[1]

    #Continuous part of the model
    dx = r*x * (1 - x/K) - (a*x/(c + x)) * y
    dy = gamma * (a*x/(c + x)) * y - m*y

    return dx, dy

def solve_no_int_ode(
    xy: list,
    r: float = 0.5,
    K: float = 10,
    a: float = 20,
    c: float = 20,
    m: float = 0.1,
    gamma: float = 0.8,
    mu: float = 1,
    T: float = 5,
    t_0: float = 0,
    t_n: float = 20
        
):
    '''This function gives the anwser of the semi-discrete ODE system with the no-interference model 
    
    Param:
        model: the chosen model
        xy: put the initial value here. It will be changed along the for loop
        r: growth rate
        K: carrying capacity
        a: search rate
        c: half-saturation constant
        m: death rate
        gamma: conversion factor
        mu: release rate
        T: release period
        start: left endpoint of the domain
        end: right endpoint of the domain
        
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
    y_kT_plus = xy[1] #Initial values before entring into the loop
    for i in range(1,len(intervals)):
        xy_kT_plus = [x[-1],y_kT_plus] #the initial value in a period is [x(kT+), y(kT+)] 
        #Span for this period
        tspan = np.linspace(intervals[i-1], intervals[i], 101)
        tspan = np.append(tspan, intervals[i]) 
        #Solve for this period
        xy_step = odeint(no_int_model, xy_kT_plus, tspan, args=(r, K, a, c, m, gamma)) 
        x.extend(xy_step.T[0])
        y.extend(xy_step.T[1]) #Continuous part of y
        y_kT_plus = xy_step.T[1][-1] + mu*T #Equation of the discrete part
        #y.append(y_kT_plus)

        t.extend(tspan)

    

    return x, y, t

##Beddington-DeAngelis model
def bda_model(
    xy: list,
    t = np.linspace(0,20,201),
    r: float = 0.5,
    K: float = 10,
    a: float = 20,
    c: float = 20,
    m: float = 0.1,
    gamma: float = 0.8,
    b: float = 5
) -> list:
    
    '''This model is a continuous model that describes the evolution of a pest population x and a predator population y following the Beddington-DeAngelis model
    
    Param:
        xy: a list of values of [x,y] at a time t_n
        r: growth rate
        K: carrying capacity
        a: search rate
        c: half-saturation constant
        m: death rate
        gamma: conversion factor
        b: penalty coefficient of the predator efficiency
        
    Return:
        dx, dy: a list of the two population size of x and y at time t_{n+1}'''
    
    #Initialisation
    x = xy[0]
    y = xy[1]

    #Continuous part of the model
    dx = r*x * (1 - x/K) - (a*x/(c + x + b*y)) * y
    dy = gamma * (a*x/(c + x + b*y)) * y - m*y

    return dx, dy

def solve_bda_ode(
    xy: list,
    r: float = 0.5,
    K: float = 10,
    a: float = 20,
    c: float = 20,
    m: float = 0.1,
    gamma: float = 0.8,
    b: float = 5,
    mu: float = 1,
    T: float = 5,
    t_0: float = 0,
    t_n: float = 20
        
):
    '''This function gives the anwser of the semi-discrete ODE system with Bedington-DeAngelis model 
    
    Param:
        model: the chosen model
        xy: put the initial value here. It will be changed along the for loop
        r: growth rate
        K: carrying capacity
        a: search rate
        c: half-saturation constant
        m: death rate
        gamma: conversion factor
        b: penalty coefficient of the predator efficiency
        mu: release rate
        T: release period
        start: left endpoint of the domain
        end: right endpoint of the domain
        
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
    y_kT_plus = xy[1] #Initial values before entring into the loop
    for i in range(1,len(intervals)):
        xy_kT_plus = [x[-1],y_kT_plus] #the initial value in a period is [x(kT+), y(kT+)]
        #Span for this period
        tspan = np.linspace(intervals[i-1], intervals[i], 101) 
        tspan = np.append(tspan, intervals[i])
        print(tspan)
        #Solve for this period
        xy_step = odeint(bda_model, xy_kT_plus, tspan, args=(r, K, a, c, m, gamma, b), rtol = 1e-12) 
        x.extend(xy_step.T[0])
        y.extend(xy_step.T[1]) #Continuous part of y
        y_kT_plus = xy_step.T[1][-1] + mu*T #Equation of the discrete part
        t.extend(tspan)

    

    return x, y, t

##Squabbling model
def s_model(
    xy: list,
    t = np.linspace(0,20,201),
    r: float = 0.5,
    K: float = 10,
    a: float = 20,
    c: float = 20,
    m: float = 0.1,
    gamma: float = 0.8,
    q: float = 0.2
) -> list:
    
    '''This model is a continuous model that describes the evolution of a pest population x and a predator population y following the Beddington-DeAngelis model
    
    Param:
        xy: a list of values of [x,y] at a time t_n
        r: growth rate
        K: carrying capacity
        a: search rate
        c: half-saturation constant
        m: death rate
        gamma: conversion factor
        q: squabbling coefficient
        
    Return:
        dx, dy: a list of the two population size of x and y at time t_{n+1}'''
    
    #Initialisation
    x = xy[0]
    y = xy[1]

    #Continuous part of the model
    dx = r*x * (1 - x/K) - (a*x/(c + x)) * y
    dy = gamma * (a*x/(c + x)) * y - (m + q*y) *y

    return dx, dy

def solve_s_ode(
    xy: list,
    r: float = 0.5,
    K: float = 10,
    a: float = 20,
    c: float = 20,
    m: float = 0.1,
    gamma: float = 0.8,
    q: float = 0.2,
    mu: float = 1,
    T: float = 5,
    t_0: float = 0,
    t_n: float = 20
        
):
    '''This function gives the anwser of the semi-discrete ODE system with Bedington-DeAngelis model 
    
    Param:
        model: the chosen model
        xy: put the initial value here. It will be changed along the for loop
        r: growth rate
        K: carrying capacity
        a: search rate
        c: half-saturation constant
        m: death rate
        gamma: conversion factor
        b: penalty coefficient of the predator efficiency
        mu: release rate
        T: release period
        start: left endpoint of the domain
        end: right endpoint of the domain
        
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
    y_kT_plus = xy[1] #Initial values before entring into the loop
    for i in range(1,len(intervals)):
        xy_kT_plus = [x[-1],y_kT_plus] #the initial value in a period is [x(kT+), y(kT+)] that is the last element of [x,y]
        #Span for this period
        tspan = np.linspace(intervals[i-1], intervals[i], 101) 
        tspan = np.append(tspan, intervals[i])
        print(tspan)
        #Solve for this period
        xy_step = odeint(s_model, xy_kT_plus, tspan, args=(r, K, a, c, m, gamma, q)) 
        x.extend(xy_step.T[0])
        y.extend(xy_step.T[1]) #Continuous part of y
        y_kT_plus = xy_step.T[1][-1] + mu*T #Equation of the discrete part
        
        t.extend(tspan)

    return x, y, t

#Functions for the stability conditions
def calculate_mu_a(
    r: float = 0.5,
    K: float = 10,
    a: float = 20,
    c: float = 20,
    m: float = 0.1
):
    '''This function calculate the threshold mu_a above which mu should be to have a GAS in the no-interference model
    
    Param: 
        r: growth rate
        K: carrying capacity
        a: search rate
        c: half-saturation constant
        m: death rate
        
    Return:
        mu_a: value of mu_a'''
    
    if c <= K:
        S = ((K+c)**2) / (4*K)
    elif c > K:
        S = c
    
    mu_a = r*m*S/a

    return mu_a
def calculate_mu_b(
    r: float = 0.5,
    K: float = 10,
    a: float = 20,
    c: float = 20,
    m: float = 0.1, 
    b: float = 5,
    T: float = 5
):
    '''This function calculate the threshold mu_b above which mu should be to have a GAS in the BDA model
    
    Param: 
        r: growth rate
        K: carrying capacity
        a: search rate
        c: half-saturation constant
        m: death rate
        b: penalty coefficient of the predator efficiency
    
    Return:
        result: value of mu_b'''
    
    factor_1 = (c + K)/b
    num_factor_2 = 1 - np.exp( - (r*b/a) * m*T)
    den_factor_2 = np.exp(- (r*b/a) * m*T) - np.exp(-m*T)
    factor_2 = num_factor_2/den_factor_2
    factor_3 = (1 - np.exp(-m*T)) / (T)
    result = factor_1 * factor_2 * factor_3

    return result

def calculate_mu_q(
    r: float = 0.5,
    K: float = 10,
    a: float = 20,
    c: float = 20,
    m: float = 0.1,
    q: float = 0.2
):
    '''This function calculate the threshold mu_b above which mu should be to have a GAS in the squabbling model
    
    Param: 
        r: growth rate
        K: carrying capacity
        a: search rate
        c: half-saturation constant
        m: death rate
        q: squabbling coefficient
    
    Return:
        mu_q: value of mu_q'''
    if c <= K:
        S = ((K+c)**2) / (4*K)
    elif c > K:
        S = c
    mu_q = (r*S/a) * ( (r*S*q/a) + m )
    return mu_q

#Periodic solutions
def y_p_s(
    t = np.linspace(0,20,201),
    r: float = 0.5,
    K: float = 10,
    a: float = 20,
    c: float = 20,
    m: float = 0.1,
    gamma: float = 0.8,
    q: float = 0.2,
    mu: float = 1,
    T: float = 1  
):
    '''This function calculate the periodic solution y for the squabbling model
    
    Param:
        r: growth rate
        K: carrying capacity
        a: search rate
        c: half-saturation constant
        m: death rate
        gamma: conversion factor
        q: squabbling coefficient 
        mu: release rate
        T: release period   
        
    Return:
        y: The value of the periodic solution a time point(s) t'''
    y_star = (1/2) * ( mu*T - (m/q) + np.sqrt( (mu*T - (m/q))**2 + (4*mu*m*T) /( q*(1-np.exp(-m*T))) ) )
    y = (m * y_star * np.exp(-m * (t % T))) / (m + (1 - np.exp(-m * (t % T))) *q*y_star)
    return y