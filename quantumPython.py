"""
This software was made by Kyle Gersbach and developed for a final project
at the University of Washington Bothell's Quantum mechanics I course.

Heavy inspiration was from:
A Python Program for Solving Schrödinger’s Equation in Undergraduate Physical Chemistry
Matthew N. Srnec, Shiv Upadhyay, and Jeffry D. Madura
Journal of Chemical Education 2017 94 (6), 813-815
DOI: 10.1021/acs.jchemed.7b00003

https://pubs.acs.org/doi/10.1021/acs.jchemed.7b00003
"""


import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.models import Select, Slider, Button
from bokeh.plotting import figure, show
from bokeh.io import output_notebook

from bokeh.palettes import Spectral11



class OneDWell:
    """
    The class to instiatiate in order to create solve a 1D well with an electron inside
    
    """
    
    
    def __init__(self):
        """
        Constructor for the class.
        
        call this in a jupyter notebook with:
        [var] = oneDWell()
        
        """
        output_notebook()
        self.HBAR = 1.054571e-34 #J*s
        self.numSteps = 1000
        self.mass = 9.1094e-31
        self.maxE = None
        self._calculated = False
        
    
    def setPotential(self, func):
        """
        A function to set the potential well to be calculated.
        
        This function takes a function as a parameter which must have a single variable input,
        and that variable must be position in meters.
        
        Parameters:
            func (Function): a function which describes your well.
        
        """
        self.potentialFunc = func
    
    
    def setRange(self, minX, maxX, steps = 1000):
        """
        This function sets the range at which the well is calculated. Additionally,
        you can also set how many steps it uses.
        WARNING: increasing steps increases computational time exponentially!
        
        Parameters:
            minX (double): The minimum x value for the well in meters
            maxX (double): The maximum x value for the well in meters
            steps (int): The number of finite steps to compute
        
        """
        self.range = (minX,maxX)
        self.numSteps = steps
        self.dX = (maxX-minX)/steps
        
        
    def setMaxEnergy(self,energy):
        """
        This function can manually set the maximum allowed energy of the well.
        If this function is not used, the program will default to the lowest
        potential energy value of either edge of the computational area.
        
        Parameters:
            energy (float): The maximum allowed energy in Joules
        
        """
        self.maxE = energy
            
        
    def calcWell(self):
        """
        This function will do the heavy lifting and will calculate the Eigenvalue problem.
        This function will only work properly if you have used the following functions:
            setPotential()
            setRange()
        and optionally:
            setMaxEnergy()        
        """
        if (self.potentialFunc == None):
            raise Exception("No potential function given, use setPotential() first!")
        if (self.range == None):
            raise Exception("No range for potential given, used setRange() first!")
        
        #Find max energy level
        if self.maxE == None:
            self.maxE = min(self.potentialFunc(self.range[0]),self.potentialFunc(self.range[1]))
        
        # Calculating Hamiltonian
        # H = -HBAR^2 / 2m * d^2/dx^2  +  V(x)
        # 3-point finite difference method
        secondDeriv = (np.diag(np.ones(self.numSteps-1),-1) -2*np.diag(np.ones(self.numSteps)) + np.diag(np.ones(self.numSteps-1),1))/(self.dX**2)
        
        #Potential on diagonal
        potential = np.zeros((self.numSteps,self.numSteps))
        for i in range(self.numSteps):
            xval = self.range[0] + self.dX*i
            potential[i,i] = self.potentialFunc(xval)
        
        #Create Hamiltonian
        hamiltonian = (-(self.HBAR**2)/(2.0*self.mass))*secondDeriv + potential
        
        #Calculate Eigenvalue function
        self.E, self.V = np.linalg.eigh(hamiltonian)
        
        n = len(self.E)
        for i in range(len(self.E)):
            if self.E[i]<0 or self.E[i]>self.maxE:
                n=i
                break
            
            
        self.E = self.E[0:n]
        self.Psi = self.V[:,0:n]
        
        print("Calculation complete.")
        self._calculated=True
        
        
    def plotPotential(self):
        """
        Plots the potential well that you supplied.
        
        This function requires you to have set both a potenital energy function
        and a range to plot over using the setPotential() and setRange() functions.
        """
        if (self.potentialFunc == None):
            raise Exception("No potential function given, use setPotential() first!")
        if (self.range == None):
            raise Exception("No range for potential given, used setRange() first!")
            
        
        fig = figure(title="Potential well", x_axis_label="Position", y_axis_label="Potential energy")
        xvals = np.arange(self.range[0], self.range[1], (self.range[1]-self.range[0])/self.numSteps)
        yvals = np.zeros(len(xvals))
        for i in range(len(xvals)):
            yvals[i] = self.potentialFunc(xvals[i]) 
            fig.line(x=xvals,y=yvals)
        show(fig)
    
    
    def printEnergy(self):
        """
        Displays the energy values recovered from the Schrodinger equation.
        
        This function requires that you have successfully run the calcWell() function.
        """
        if not self._calculated:
            raise Exception("Well has not been calculated, use calcWell() first!")
        
        if len(self.E)==0:
            print("No Energies!")
        
        for i in range(len(self.E)):
            print("Energy level {}: E={}".format(i,self.E[i]))
    
    
    def plotWavefunction(self,energyLevel=None):
        """
        This function will plot the resulting wavefunctions for the given well.
        For this function to work properly, you can either supply it with a list
        of integers representing the energy levels (i.e [0,1,2] is the 0th, 1st and
        2nd allowed energies). Or if left blank, it will plot all allowed energies.
        
        This function requires that you have successfully run the calcWell() function.
        
        Parameters:
            energyLevel (int): an int or list of ints representing the wavefunctions to be calculated
        """
        if not self._calculated:
            raise Exception("Well has not been calculated, use calcWell() first!")
        
        if type(energyLevel)==int:
            energyLevel = [energyLevel]
        elif energyLevel == None:
            energyLevel = [i for i in range(len(self.E))]
        
        fig = figure(title="Wavefunction", x_axis_label="Position", y_axis_label="Psi(x)")
        
        for e in energyLevel:
            xvals = np.arange(self.range[0], self.range[1], (self.range[1]-self.range[0])/self.numSteps)
            fig.line(x=xvals[1:],y=self.Psi[:,e],color=Spectral11[(e+1)%11],legend_label="E={}".format(self.E[e]))
        
        fig.legend.click_policy="hide"
        show(fig)
        
        
    def plotProbability(self,energyLevel=None):
        """
        This function will plot the resulting probability distributions for the given well.
        For this function to work properly, you can either supply it with a list
        of integers representing the energy levels (i.e [0,1,2] is the 0th, 1st and
        2nd allowed energies). Or if left blank, it will plot all allowed energies.
        
        This function requires that you have successfully run the calcWell() function.
        
        Parameters:
            energyLevel (int): an int or list of ints representing the wavefunctions to be calculated
        """
        if not self._calculated:
            raise Exception("Well has not been calculated, use calcWell() first!")
        
        if type(energyLevel)==int:
            energyLevel = list(energyLevel)
        elif energyLevel == None:
            energyLevel = [i for i in range(len(self.E))]
        
        fig = figure(title="Probability density", x_axis_label="Position", y_axis_label="Probability")
        psiSquare = np.square(self.Psi)
        for e in energyLevel:
            xvals = np.arange(self.range[0], self.range[1], (self.range[1]-self.range[0])/self.numSteps)
            fig.line(x=xvals[1:],y=psiSquare[:,e],color=Spectral11[(e+1)%11],legend_label="E={}".format(self.E[e]))
        
        fig.legend.click_policy="hide"
        show(fig)

        
    def plotTimeEvolution(self,t,energyLevel=None,initialFactors=None):
        """
        This function will plot the resulting wavefunctions with time evolution
        according to the time dependent schrodinger equation. For this function 
        to work properly, you can either supply it with a list of integers representing 
        the energy levels (i.e [0,1,2] is the 0th, 1st and 2nd allowed energies). 
        Or if left blank, it will plot all allowed energies. Additionally, you must either
        supply a list of initial factors for each wavefunction or again leave it blank,
        this time, blank values will normalize each psi_n to be equally probable.
        
        the initial factors represent the C_n values in the equation:
        Psi(x,t) = SUM(C_n * exp(-i*E_n*t/HBAR) *Psi_n(x,0)) 
        
        This function requires that you have successfully run the calcWell() function.
        
        Parameters:
            t (double): The time in seconds to calculate the new wavefunction
            energyLevel (ints): a list of ints representing the wavefunctions to be calculated
            initialFactors (ints): a list of ints represting the factors in front of the psi_n
        """
        if not self._calculated:
            raise Exception("Well has not been calculated, use calcWell() first!")
        
        if type(energyLevel)==int:
            raise Exception("No time evolution with 1 energy level")
        elif energyLevel == None:
            energyLevel = [i for i in range(len(self.E))]
            
        if initialFactors == None:
            initialFactors = [1/np.sqrt(len(energyLevel))]*len(energyLevel)
        
        tpsi = self._timeDependance(energyLevel,initialFactors,t)
        fig = figure(title="Time Evolved Psi", x_axis_label="Position", y_axis_label="Psi(x)")
        xvals = np.arange(self.range[0], self.range[1], (self.range[1]-self.range[0])/self.numSteps)
        fig.line(x=xvals[1:],y=tpsi.real,color=Spectral11[1],legend_label="Psi (real)")
        fig.line(x=xvals[1:],y=tpsi.imag,color=Spectral11[2],legend_label="Psi (Imaginary)")
        square = np.square(tpsi.real)+np.square(tpsi.imag)
        fig.line(x=xvals[1:],y=square,color=Spectral11[3],legend_label="Probability Density")
        fig.legend.click_policy="hide"
        show(fig)
        
        
    def _timeDependance(self,energyLevel,factors,t):
        """
        This function calculates the actual value of psi(x,t) for a given t value.
        It should only be called inside the plotTimeEvolution() function.
        
        This function requires that you have successfully run the calcWell() function.
        
        Parameters:
            energyLevel (ints): a list of ints representing the wavefunctions to be calculated
            factors (ints): a list of ints represting the factors in front of the psi_n
            t (double): the time in seconds to calculate the new wavefunction
        """
        
        tdpsi = None
        for i in range(len(energyLevel)):
            if not type(tdpsi) == type(np.array([0])):
                e = energyLevel[i]
                tdpsi = factors[i]*np.exp(-1j*self.E[i]*t /self.HBAR)*self.Psi[:,e]
            else:
                tdpsi += factors[i]*np.exp(-1j*self.E[e]*t /self.HBAR)*self.Psi[:,e]
        
        return tdpsi
        
        