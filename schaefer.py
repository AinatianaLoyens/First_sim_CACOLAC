#Dependancies
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Logistic growth model
def logistic_model(
    x,
    t = np.linspace(0,20,201),
    r: float = 0.5,
    K: float = 10        
):
    '''This model is a continuous model that describes a logistic growth which will be applied to the Schaefer 
    
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

#Pulsed Schaefer model
def solve_schaefer_model(
    x,
    r: float = 0.5,
    K: float = 10,
    E: float = 1,
    T: float = 5,
    t_0: float = 0,
    t_n: float = 20
):
    '''This function gives the anwser of the pulsed Schaefer ODE system

    Param:
        x: the initial value of x. It will be changed along the for loop
        r: growth rate
        K: carrying capacity
        a: search rate
        E: taking effort
        T: period
        
    Return:
        t: time points
        x_sol: values of the solution x of the ODE
        '''
    
    #Initialisation
    x_sol = [x]
    t = [t_0]

    #Solve ODE
    intervals = np.arange(t_0, t_n, T) #divide the domain in intervals on length T
    intervals = np.append(intervals, t_n) #add t_n to intervals because t_n is not reached by arange
    x_kT_plus = x #Initial values before entring into the loop
    for i in range(1,len(intervals)):
        #Span for this period
        tspan = np.arange(intervals[i-1], intervals[i] + 0.01, 0.01) 
        #Solve for this period
        x_step = odeint(logistic_model, x_kT_plus, tspan, args=(r, K))
        x_sol.extend(x_step.T)
        x_kT_plus = np.exp(-E*T)*x_step[-1] #Equation of the discrete part

        t.extend(tspan)

    return t, x_sol