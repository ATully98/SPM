"""
Python version of SetupErrors_v6
v6.1 includes use of warnings package to print warning instead of simple print statement
"""


import numpy as np
import warnings



def SetupErrors_v6(Tc, Temps, Blab):
	 #Function to define the magnitudes of the uncertainties in paleointensity
	 #experiments
	 #This should be used by all models to ensure consistent uncertainties

	 #Version 6 removes the Blab dependence of Fres and assumes realistic field
	 #strengths (i.e., 10--100 muT)


	if Blab < 5:
		warnings.warn("SetupErrors:Blab; Blab is low, please ensure Blab is in micro tesla")


	#   Temps parsed as list
	#   convert to array for manipulation
	Temps = np.array(Temps)


	# Temperature error introduced from hold time reproducibility
	# Based on polynomial fitted to bootstrap temperature standard deviation
	# Asssumes normally distributed hold times with a mean of 2400s and std of
	# 30s, the time to reach equilibrium is assumed to be ~800s dHT is
	# calculated using an effective hold time of 1600s
	pHT1 = -9.29943135827724e-05
	pHT2 = -0.000387395057716160
	pHT3 =  0.000267750549508746
	pHT4 =  0.000212875979915772


	hold_time = pHT1 *Temps**3 + pHT2*Temps**2 + pHT3*Temps + pHT4  # FIND HOLD TIMES
	hold_time[0]=0												  # Set first hold time to 0
	hold_time[np.where(Temps==1)]=0										   # Where Temps is 1 replace hold time with 0



	# Temperature error introduced from cooling time reproducibility
	# Based on polynomial fitted to bootstrap temperature standard deviation
	# Asssumes normally distributed cool times with a mean and stdev that
	# depend on the set temperature
	pCR1 = -0.00331149651861100
	pCR2 =  0.00317731862087431
	pCR3 = -0.000836205078970524
	pCR4 =  0.000775482701910688
	pCR5 =  0.000204387220911639

	cooling_rate=pCR1*Temps**4 + pCR2*Temps**3 + pCR3*Temps**2 + pCR4*Temps + pCR5
	cooling_rate[0] = 0
	cooling_rate[np.where(Temps==1)]=0


	# Absoulte temperture reproducibility error of 0.138 degree Celcius
	# Determined as the standard deviation from measured data @ 610C
	# Normalized by Tc
	T_repeat = np.ones((1,len(Temps)-1))
	T_repeat = np.insert(T_repeat, (0), 0., axis=1)
	T_repeat=np.transpose(T_repeat*(0.138/Tc))
	# Estimated error due to thermal gradients
	# ASC furnance at the IGGCAS has thermal gradients on the order of
	# ~0.32 deg C per cm, indpendent of set temperature
	# Assuming normality and that 95% of samples are placed with 0.5cm of their
	# original position, sample position can be describe by N(0, (0.25).^2)
	# This translates into a 0.08 deg C temperature standard deviation
	T_grad=np.ones((1,len(Temps)-1))
	T_grad = np.insert(T_grad, (0), 0., axis=1)
	T_grad=np.transpose(T_grad*(0.08/Tc))


	# Total temperature error
	# Since all errors are in the same temperature "units" add absolute errors
	T=(hold_time**2 + cooling_rate**2 + T_repeat**2 + T_grad**2)**(0.5)
	# Absolute field error - based on the resolution of applied current
	# For system used at Kochi resolution is 1 mA. For a 40 muT field ~40 mA
	# current is required. Therefore resolution (and repeatability) is ~0.25#
	# equavalent to <= 100 nT in applied fields <= 40 muT
	# error.F=(0.25/100)*Blab;

	# Applied Field gradients
	# From both IGGCAS and Soton ovens gradient is ~0.03% of the applied field
	# per cm. See thermal gradients for sample postition distribution
	# Taken to be normally distributed with standard deviation of 0.0075%
	B=(0.0075/100)*Blab

	# Total applied field error
	# error.F=sqrt(error.F_repeat.^2 + error,F_grad.^2);

	# Residual field error, 0.05% of Blab, equavalent to <=20 nT in applied
	# fields <= 40 muT
	res=(0.05/100)*40

	# Absolute measurement error, equivalent to 0.5% of the NRM - Determined
	# from AF measurements from 64 samples, which include 3615 xyz measurements
	# ~95.0% have noise equivalent to <=0.36% of the x-y-z(T) Sample size
	# adjusted
	meas=0.36/100

	# Background noise - Determined from background drift from from AF
	# measurements from 151 samples, which include 9858 xyz measurements (data
	# with high drift, >|1e-7| A/m, were excluded as possible flux jumps)
	# Background is described as a # of the inital total NRM and modeled by a
	# Cauchy distribution with parameters:
	# Location = 1.784594921818256e-04
	# Scale = 8.729006142244014e-04
	# The random numbers should be rescaled into remanence values. This is done
	# by multiplying by Blab, which is the same as the total NRM
	background=[1.784594921818256e-04, 8.729006142244014e-04];

	# Angular error in remanence measurement introduced through sample handling
	# The angle in radians follows a Weibull distribution with parameters
	# A=0.033161924863149 and B=1.633275887444545. If the magnitude of the
	# angle changes only parameter A needs to be changed, and in a linear
	# fashion, i.e., to reduce the meadian value by half, divide A by 2.
	# Alternatively, scale the angles after random number generation
	theta=[0.033161924863149, 1.633275887444545];

	# Angular error in applied field direction introduced through sample
	# handling. Same as above.
	phi=theta

	# replaces theta and phi
	kappa = 1600

	return hold_time, cooling_rate, T_repeat, T_grad, T, B, res, meas, background, kappa
