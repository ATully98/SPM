"""
Utility library of the Paleointensity models

Functions:

OpenExperiment
deg2rad
rad2deg
dir2cart
cart2dir
Beta_fits
quad
RndRot - deprecated
PmagPCA
Tmatrix
Anis_mat
PearsonCorr2
calc_angle
common_slope
AraiCurvature
TaubinSVD
LMA
VarCircle
dirot
FisherMeanDir
Polyarea
custom_formatwarning
MDblocking
wbl_fits
field_fit
quad2d
Generate_N_orient
ATRMtensor
designATRM
RndVec
FindP
rotvec2mat
formattime
find_nearest
FishAngleNoise
create_array
dirot_pmag
"""

import numpy as np
import csv
import warnings
from warnings import warn as warning
import scipy as sp
from scipy import stats
from scipy.integrate import quadrature as quadrature
from scipy.integrate import dblquad
from scipy.signal import detrend
from numpy.random import default_rng
import pandas as pd
import time
import os
import ast
from matplotlib import path


def OpenExperiment(Experiment):
	"""
	function to open Experiment file and read in list of temperatures, treatments and Hx, Hy and Hz.

	Input:
			Experiment - str path to experiment file

	Output:
			temps - array of temperatures
			treatment - list of treatment steps
			Hx - array of x components
			Hy - array of y components
			Hz - array of z components
	"""

	temps=[]
	treatment = []
	Hx = []
	Hy = []
	Hz = []

	file = open(Experiment, "r")

	file.readline()
	for a, b, x, y, z in csv.reader(file, delimiter="\t"):
		temps.append(float(a))
		treatment.append(float(b))
		Hx.append(float(x))
		Hy.append(float(y))
		Hz.append(float(z))

	# Create temps array
	temps = np.reshape(temps,(len(temps),1))

	# Lists of H components from final 3 columns in np arrays
	Hx = np.array(Hx)
	Hy = np.array(Hy)
	Hz = np.array(Hz)

	return temps, treatment, Hx, Hy, Hz
	#Usage: temps, treatment, Hx, Hy, Hz =  Utlities.OpenExperiment(Experiment)


def deg2rad(InDegrees):
	"""
	Function to convert from degrees to radians
	Inputs:
			InDegrees - float/array value(s) for angle in degrees

	Outputs:
			InRadians - float/array value(s) for converted angle in radians

	"""

	#Covert an angle in degrees to radians
	InRadians = (np.pi/180) * InDegrees

	return InRadians


def rad2deg(InRadians):
	"""
	Function to convert from radians to degrees

	Inputs:
			InRadians - float/array value(s) for angle in radians

	Outputs:
			InDegrees - float/array value(s) for converted angle in degress

	"""
	#Covert an angle in radians to degrees
	InDegrees = (180/np.pi) * InRadians

	return InDegrees


def dir2cart(Dec, Inc, Mag = None):
	"""
	Function converts a paleomagmetic direction vector to cartesian coordinates
	Inputs:
			Dec - float/array value(s) for declination in degrees
			Inc - float/array value(s) for inclination in degrees
			Mag - float/array value(s) for Magnetisation magnitude

	Outputs:
			x - float/array cartesian co-coordinates
			y - float/array cartesian co-coordinates
			z - float/array cartesian co-coordinates
	"""
	# check if Mag is included and create either an array or float value depending on Dec input
	if Mag is None:
		if isinstance(Dec, int) or isinstance(Dec, float) == True:
			Mag = 1
		else:
			Mag = np.ones((len(Dec),1))

	# convert dec and inc from degrees to radians
	Dec=deg2rad(Dec)
	Inc=deg2rad(Inc)

	#convert from directional to cartesian x,y,z
	x = np.cos(Dec)*np.cos(Inc)*Mag
	y = np.sin(Dec)*np.cos(Inc)*Mag
	z = np.sin(Inc)*Mag

	# check for nan values and set them to zero
	if (np.isnan(np.sum(x)) == True) and (isinstance(x, np.ndarray) == True)  :
		x[np.where(np.isnan(x) == True)] = 0
		y[np.where(np.isnan(y) == True)] = 0
		z[np.where(np.isnan(z) == True)] = 0

	return x ,y, z


def cart2dir(x, y, z):
	"""
	Convert cartesian coordinates to declination/inclination
 	Based on 2G manual

	Inputs:
			x - float/array value(s) for x cartesian coord
			y - float/array value(s) for y cartesian coord
			z - float/array value(s) for z cartesian coord

	Outputs:
			Dec - float/array value(s) for declination in degrees
			Inc - float/array value(s) for inclination in degrees
			R - float/array value(s) for radial magnitude

	"""


	# Find magnitude of vector
	R = (( (x)**2 + (y)**2 + (z)**2 )**(0.5))
	#set up arrays for Dec and Inc
	Dec = create_array((R.size, 1))
	Inc = create_array((R.size, 1))

	# Check for size in case input is scalar value rather than array
	# find scalar dec and inc
	if R.size == 1:
		# if magnitude is zero set Dec and Inc to nan
		if R == 0:
			Dec = np.nan
			Inc = np.nan
		else:
			# find Dec and Inc in degrees

			Dec = rad2deg(np.arctan2(y,x) )
			Inc = rad2deg(np.arcsin(z/R) )

			# ensure dec is between 0 and 360 degrees
			if Dec < 0:
				Dec = Dec +360
			elif Dec > 360:
				Dec = Dec-360

	else:
		# find arrays of dec and inc
		# get correct shape
		R = R.reshape((R.size,1))
		# iterate through array
		for i in range(R.size):
			# check if magnitude is zero and set Dec and Inc to zero if true
			if R[i] == 0:
				#warning('cart2dir:Error;  R==0')

				Dec[i] = np.nan
				Inc[i] = np.nan

			else:
				# find dec and inc in degrees
				Dec[i]=rad2deg(np.arctan2(y[i],x[i]) )
				Inc[i]=rad2deg(np.arcsin(z[i]/R[i]) )

				# ensure Dec is between 0 and 360 degrees
				if Dec[i]<0:
					Dec[i] = Dec[i] +360
				elif Dec[i]>360:
					Dec[i] = Dec[i]-360


	return Dec, Inc, R

def Beta_fits(B_iter):
	"""
	Function to extract random sample of beta fits from file

	Input:
			B_iter - interger of how many times the experiment will be run, and therefore how many fits are needed

	Output:
			Beta_fits - an array of shape (B_iter, 2) containg the random selection of beta a_fits

	"""

	#	Beta File path
	Beta_fits = "./Core_Files/Beta_fits_24-Sept-2013.txt"

	#	set up lists to hold values
	a_fits = []
	b_fits = []

	#open file and read
	file = open(Beta_fits, "r")
	file.readline()

	#extract a and b values to lists
	for a, b,in csv.reader(file, delimiter="\t"):

		a_fits.append(float(a))

		b_fits.append(float(b))

	#	convert lists into arrays
	a_fits = np.reshape(np.array((a_fits)), (len(a_fits),1))
	b_fits = np.reshape(np.array((b_fits)), (len(b_fits),1))

	#join arrays into single double coloumn array
	betas = np.concatenate((a_fits,b_fits), axis = 1)

	#	select random indices
	samp_size = len(betas)
	rng = default_rng()
	sample = rng.integers(1, high = samp_size, size=(B_iter))

	#	select fits at the selected indices
	Beta_fits = betas[sample]

	return Beta_fits

def quad(beta_vals, low, high):
	"""
	Function for guassian quadrature integration of beta distribution
	Inputs:
			beta_vals - array of shape (2,) containg the a and b values decribing the distribution
			lower - value of the lower limit of the integration
			higher - value of the high limit of the integration
	Output:
			quad -	float value of the integral
	"""

	# set a and b value from input
	a = beta_vals[0]
	b = beta_vals [1]

	# ensure limits are floats and not int or arrays
	low = float(low)
	high = float(high)

	# Function for integration
	Pfun = lambda x: stats.beta.pdf(x, a, b)

	# maximum interation completed in calculation
	maxiter = 200

	# Complete integral
	integral = quadrature(Pfun, low, high , maxiter=maxiter)

	# Set integral output and error output
	quad = integral[0]
	err = integral[1]

	return quad

# def RndRot(vec, angle, rotaxis):
#
# 	"""
# 	Previously AngleNoise function, now deprecated and should be left commented
# 	Inputs:
# 			vec - (n, 3) array containing vectors to be roatated
# 			angle - float/array of angle(s) in radians which to the vector by
# 			rotaxis - (n, 3) array containg vectors that form the axis of rotation
#
# 	Output:
# 			vector - (n, 3) array containing vectors that hve been rotated by angle about rotaxis
# 	"""
#
# 	# check rotation axis shape
# 	s=np.shape(rotaxis)
#
# 	if s[0]!=len(angle):
# 		raise ValueError('AngleNoise:VectorLength; Rotaxis and angle vectors must be the same length.')
#
#
# 	# Convert rotation axis into a unit vector
# 	vec_len=(np.sum(rotaxis**2, axis = 1))**(0.5)
# 	for i in range(3):
# 		rotaxis[:,i]=rotaxis[:,i]/vec_len
#
#
# 	##Main function
# 	vector = np.zeros((len(angle), 3))
# 	cAng = np.cos(angle)
# 	sAng = np.sin(angle)
#
# 	# Expand vec to the size of rotaxis if vec is 1x3
# 	s=np.shape(vec)
# 	if s[0]==1:
# 		vec = np.tile(vec, (len(angle),1))
#
#
# 	for j in range(len(angle)):
#
# 		if angle[j] == 0:
# 			vector[j,]=vec[j,]
# 		else:
# 			rotaxis[j,]=rotaxis[j,]   #./norm(rotaxis(j,:));
# 			vector[j,]=vec[j,]*cAng[j] + np.dot(vec[j,],rotaxis[j,])*(1-cAng[j])*rotaxis[j,] + np.cross(rotaxis[j,],vec[j,])*sAng[j]
#
#
# 	return vector


def PmagPCA(Dec, Inc, Int, type = "free"):
	"""
	 Function to determine the best-fit line through demagnetization data
	 Based on PCA methods of Kirschvink (1980)
	 Note: This only fits lines, not planes

	Inputs:
			Dec  - Array of declinations
			Inc  - Array of inclinations
			Int  - Array of magnitudes/intensities
			type - string selecting the fit to be used in fitting routine; current options are "free", "anchored" and "origin"

	Outputs:
			Mdec - Float value for average declination
			Minc - Float value for average inclination
			MAD  - Float value for average MAD
	"""

	# Setup input
	# convert to cartesian co-ords
	X, Y, Z = dir2cart(Dec, Inc, Int)
	input = np.concatenate((X, Y, Z),axis =1)

	# fit dependent set up
	if type.casefold() == "free":	#free fit
		bars = np.array((np.mean(X), np.mean(Y), np.mean(Z)))

	elif type.casefold() == "anc" or type.casefold() == "anchored": # anchored fit
		bars = np.array((0, 0, 0))

	elif type.casefold() == "origin":	# free fit with origin
		input = np.vstack((input, np.array((0, 0, 0))))
		bars = np.array((np.mean(X), np.mean(Y), np.mean(Z)))

	else:
		raise ValueError("PCADir: Fit_Type; Fit type unrecognised. (Recognised types: free, anc/anchored, origin)")

	n = input.shape[0]
	x = np.zeros((n, 3))


	for j in range(n):
		x[j,:] = input[j,:] - bars

	# perform PCA

	#set up T matrix
	T = Tmatrix(x)
	# find eigenvalues and vector

	tau, V = np.linalg.eig(T)
	#find index of maximum tau
	max_tau_ind = np.argmax(tau)
	v1 = -np.transpose(V[:, max_tau_ind])
	v1 = v1.real
	v1imag = v1.imag
	if any(elt != 0 for elt in v1imag):
	    raise ValueError("PmagPCA; Non zero imaginary component detected")

	tau = tau/(np.sum(tau))	# normalize tau
	tau = np.sort(tau)[::-1]	# sort into descending order

	# reference vector for defining the direction of principal component

	# create reference vector for defining the direction of the principal component
	P = np.array((input[0,:],input[-1,:]))
	reference = P[0,:] - P[1,:]
	ref_dot = np.sum(v1*reference)

	if ref_dot <-1:
		ref_dot = -1
	if ref_dot > 1:
		ref_dot = 1

	#Ensure pointing correct way
	if  np.arccos(ref_dot) > np.pi/2:
		v1=-v1


	eps = np.finfo(float).eps
	# MAD is too small to calculate - set to zero
	if tau[1] + tau[2] <= eps:
			MAD = 0
	else:
		MAD = rad2deg(np.arctan( ((tau[1] + tau[2])**(0.5)) / (tau[0]**(0.5)) ))
	# Convert to dec/inc
	Mdec, Minc, R = cart2dir(v1[0],v1[1],v1[2])

	return Mdec, Minc, MAD


def Tmatrix(X):
	"""
	function to create the orientation matrix
	Inputs:
			X - array of shape (n,3)

	Outputs:
			T - T matrix (array) of shape (3,3)
	"""
	#create array
	T = np.empty((3,3))

	T[0,0] = np.sum(X[:,0] *X[:,0])
	T[1,0] = np.sum(X[:,1] *X[:,0])
	T[2,0] = np.sum(X[:,2] *X[:,0])

	T[0,1] = np.sum(X[:,0] *X[:,1])
	T[1,1] = np.sum(X[:,1] *X[:,1])
	T[2,1] = np.sum(X[:,2] *X[:,1])

	T[0,2] = np.sum(X[:,0] *X[:,2])
	T[1,2] = np.sum(X[:,1] *X[:,2])
	T[2,2] = np.sum(X[:,2] *X[:,2])

	return T


def Anis_mat(s):
	"""
	Function to build anisotropy matrix
	Input:
			s - (6, 1) array containing the s tensor
	Output:
			A - (3, 3) array of the anisotpy matrix
	"""

	# Build the anisotropy tensor
	A = np.ones((3,3))

	# fill in values from s tensor
	A[0,0] = s[0]
	A[1,1] = s[1]
	A[2,2] = s[2]
	A[1,0] = s[3]
	A[2,1] = s[4]
	A[2,0] = s[5]
	A[0,1] = A[1,0]
	A[0,2] = A[2,0]
	A[1,2] = A[2,1]

	return A


def PearsonCorr2(X, Y):
	"""
	Function to determine the Pearson linear correlation between 2 input vectors
	Inputs:
			X - Array of shape (n, ) or (n, 1)
			Y - Array of shape (n, ) or (n, 1)
	Output:
			R2 - float value of squared Pearson correlation coefficient
	"""


	# Chweck input shapes
	if X.shape != Y.shape:
		raise ValueError(f"Utilities: PearsonCorr2; input arrays are of shape {X.shape} and {Y.shape}, input arrays must be of the same shape")

	# remove mean from inputs
	Xd = detrend(X, type = "constant", axis = 0) # (x-xbar)
	Yd = detrend(Y, type = "constant", axis = 0) # (y-ybar)

	# Find correlation
	R2 = np.sum((Xd*Yd))**2 / ( np.sum(Xd**2) * np.sum(Yd**2) )

	return R2


def calc_angle(Dir1, Dir2):
	"""
	Function to calculate the angle between two directions specified by their inclinations and declinations
	Inputs:
			Dir1 - array of shape (1, 2) containing the declination as the first column and inclination as the second for the first direction
			Dir2 - array of shape (1, 2) containing the declination as the first column and inclination as the second for the second direction
	Output:
			theta - float value in degrees for the angle between the 2 directions
	"""
	# find the cartesian co-ordinates for the directions and normalize
	x, y, z = dir2cart(Dir1[0,0], Dir1[0,1])
	Dir1_cart = np.array((x, y, z))
	Dir1_cart = Dir1_cart / (np.sum(Dir1_cart**2))*(0.5)
	x, y, z = dir2cart(Dir2[0,0], Dir2[0,1])
	Dir2_cart = np.array((x, y, z))
	Dir2_cart = Dir2_cart / (np.sum(Dir2_cart**2))*(0.5)

	# calculate angle between directions
	theta = np.arctan2(np.linalg.norm(np.cross(Dir1_cart, Dir2_cart)),np.dot(Dir1_cart, Dir2_cart))
	theta = rad2deg(theta)


	return theta 	# Angle between two directions


def common_slope(bhat, varX, varY, varXY, Ns):
	"""
	Minimization Function to determine the probability of a common slope
	Derived from Warton et al. (2006), Bivariate line-fitting methods for
	allomettry, Biol. Rev., 81, 259-291, doi: 10.17/S1464793106007007

	Inputs:
			bhat - array of shape (1,)
			varX - array of shape (2,)
			varY - array of shape (2,)
			varXY - array of shape (2,)
			Ns - array of shape (2,)
	Output:
			return_val - float value for the probability of a common slope
	"""
	Sr = (Ns-1)/(Ns-2) * (varY - 2 *bhat *varXY + (bhat**2) *varX)
	Sf = (Ns-1)/(Ns-2) * (varY + 2 *bhat *varXY + (bhat**2)*varX)
	Srf = (Ns-1)/(Ns-2) * (varY - (bhat**2) *varX)

	return_val = np.sum(Ns *( 1/(Sr) + 1/(Sf) ) *Srf**2)
	return return_val # common slope minimization function


def AraiCurvature(x,y):
	"""
	Function for calculating the radius of the best fit circle to a set of
	x-y coordinates.
	Paterson, G. A., (2011), A simple test for the presence of multidomain
	behaviour during paleointensity experiments, J. Geophys. Res., doi: 10.1029/2011JB008369

	Inputs:
			x - array of shape (n, 1) containg Arai plot x points
			y - array of shape (n, 1) containg Arai plot y points
	Output:
			parameters - array of shape (4, 1):
						parameters[0,0] = k
						parameters[1,0] = a
						parameters[2,0] = b
						parameters[3,0] = SSE (goodness of fit)
	"""
	# Reshape vectors for suitable input
	x = np.reshape(x, (len(x), 1))
	y = np.reshape(y, (len(y), 1))

	# Normalize vectors
	x = x /np.amax(x)
	y = y/np.amax(y)

	# Provide the initial estimate
	# This will be used if number of points <=3
	E1 = TaubinSVD(np.concatenate((x,y),axis = 1))
	estimates = np.array((E1[0,2], E1[0,0], E1[0,1]))

	if len(x) > 3:
		# Determine the iterative solution
		#This needs at least 4 points for calculating the variance
		E2 = LMA(np.concatenate((x,y),axis=1), E1)
		estimates = np.array((E2[2], E2[0], E2[1]))

	else:
		E2 = E1.reshape(E1.size,)


	# Define the function to be minimized and calculate the SSE
	func = lambda v: np.sum((np.sqrt((x-v[1])**2+(y-v[2])**2)-v[0])**2)
	SSE = func(estimates)

	if E2[0] <= np.mean(x) and E2[1] <= np.mean(y):
		k = -1/E2[2];
	else:
		k = 1/E2[2]


	parameters = np.array(([k], [E2[0]], [E2[1]], [SSE]))

	return parameters # Arai plot curvature


def TaubinSVD(XY):
	"""

	Algebraic circle fit by Taubin
	G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
		  		Space Curves Defined By Implicit Equations, With
				Applications To Edge And Range Image Segmentation",
		  	IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)

 	Input:  XY - array of shape (n, 2) of coordinates of n points

	Output: Par - (1, 3) array containing a, b, R for the fitting circle: center (a,b) and radius R
							Par[0,0] - a
							Par[0,1] - b
							Par[0,2] - R


	Note: this is a version optimized for stability, not for speed
	"""
	centroid = np.mean(XY, axis = 0)   # the centroid of the data set

	X = XY[:,0:1] - centroid[0]  #  centering data
	Y = XY[:,1:2] - centroid[1]  #  centering data
	Z = X*X + Y*Y
	Zmean = np.mean(Z)
	Z0 = (Z-Zmean)/(2*(Zmean**(0.5)))
	ZXY = np.concatenate((Z0, X, Y),axis=1)
	U, S, V = np.linalg.svd(ZXY,compute_uv = True)

	A = V[2]
	A[0] = A[0]/(2*(Zmean**(0.5)))

	A = (np.concatenate((A, np.array(([-Zmean*A[0]])) ),axis = 0)).reshape((4,1))
	Par = np.concatenate((( -np.transpose(A[1:3])/A[0]/2+centroid, (np.sqrt(A[1]*A[1]+A[2]*A[2]-4*A[0]*A[3])/np.abs(A[0])/2).reshape(1,1)   )),axis = 1)

	return Par   #  TaubinSVD



def LMA(XY,ParIni):
	"""
	Geometric circle fit (minimizing orthogonal distances)
	based on the Levenberg-Marquardt scheme in the
	"algebraic parameters" A,B,C,D  with constraint B*B+C*C-4*A*D=1
	N. Chernov and C. Lesort, "Least squares fitting of circles",
	J. Math. Imag. Vision, Vol. 23, 239-251 (2005)

	Input:  XY - array of shape (n, 2) of coordinates of n points
		 	ParIni = array containing (a, b, R) is the initial guess (supplied by user)

	Output: Par - (3, ) array containing a, b, R for the fitting circle: center (a,b) and radius R
							Par[0] - a
							Par[1] - b
							Par[2] - R
	"""

	factorUp = 10
	factorDown = 0.04
	lamb0 = 0.01
	epsilon = 0.000001
	IterMAX = 50
	AdjustMax = 20
	Xshift = 0
	Yshift = 0
	dX = 1
	dY = 0

	n = np.shape(XY)[0]	  # number of data points

	# starting with the given initial guess

	anew = ParIni[0,0] + Xshift
	bnew = ParIni[0,1] + Yshift

	Anew = 1/(2*ParIni[0,2])
	aabb = anew*anew + bnew*bnew
	Fnew = (aabb - ParIni[0,2]*ParIni[0,2])*Anew
	Tnew = np.arccos(-anew/np.sqrt(aabb))
	if bnew > 0:
		Tnew = 2*np.pi - Tnew

	VarNew = VarCircle(XY,ParIni)

	#	 initializing lambda and iter
	lamb = lamb0
	finish = 0

	for iter in range(IterMAX):

		Aold = Anew
		Fold = Fnew
		Told = Tnew
		VarOld = VarNew

		H = np.sqrt(1+4*Aold*Fold)
		aold = -H*np.cos(Told)/(Aold+Aold) - Xshift
		bold = -H*np.sin(Told)/(Aold+Aold) - Yshift
		Rold = 1/np.abs(Aold+Aold)

		#  computing moments
		DD = 1 + 4*Aold*Fold
		D = np.sqrt(DD)
		CT = np.cos(Told)
		ST = np.sin(Told)

		H11=0
		H12=0
		H13=0
		H22=0
		H23=0
		H33=0
		F1=0
		F2=0
		F3=0

		for i in range(n):
			Xi = XY[i,0] + Xshift
			Yi = XY[i,1] + Yshift
			Zi = Xi*Xi + Yi*Yi
			Ui = Xi*CT + Yi*ST
			Vi =-Xi*ST + Yi*CT

			ADF = Aold*Zi + D*Ui + Fold
			SQ = np.sqrt(4*Aold*ADF + 1)
			DEN = SQ + 1
			Gi = 2*ADF/DEN
			FACT = 2/DEN*(1 - Aold*Gi/SQ)
			DGDAi = FACT*(Zi + 2*Fold*Ui/D) - Gi*Gi/SQ
			DGDFi = FACT*(2*Aold*Ui/D + 1)
			DGDTi = FACT*D*Vi

			H11 = H11 + DGDAi*DGDAi
			H12 = H12 + DGDAi*DGDFi
			H13 = H13 + DGDAi*DGDTi
			H22 = H22 + DGDFi*DGDFi
			H23 = H23 + DGDFi*DGDTi
			H33 = H33 + DGDTi*DGDTi

			F1 = F1 + Gi*DGDAi
			F2 = F2 + Gi*DGDFi
			F3 = F3 + Gi*DGDTi

		for adjust in range(AdjustMax):

			# Cholesly decomposition

			G11 = np.sqrt(H11 + lamb)
			G12 = H12/G11
			G13 = H13/G11
			G22 = np.sqrt(H22 + lamb - G12*G12)
			G23 = (H23 - G12*G13)/G22
			G33 = np.sqrt(H33 + lamb - G13*G13 - G23*G23)

			D1 = F1/G11
			D2 = (F2 - G12*D1)/G22
			D3 = (F3 - G13*D1 - G23*D2)/G33

			dT = D3/G33
			dF = (D2 - G23*dT)/G22
			dA = (D1 - G12*dF - G13*dT)/G11

			# updating the parameters
			Anew = Aold - dA
			Fnew = Fold - dF
			Tnew = Told - dT

			if 1+4*Anew*Fnew < epsilon and lamb>1:
				Xshift = Xshift + dX
				Yshift = Yshift + dY

				H = np.sqrt(1+4*Aold*Fold)
				aTemp = -H*np.cos(Told)/(Aold+Aold) + dX
				bTemp = -H*np.sin(Told)/(Aold+Aold) + dY
				rTemp = 1/np.abs(Aold+Aold)

				Anew = 1/(rTemp + rTemp)
				aabb = aTemp*aTemp + bTemp*bTemp
				Fnew = (aabb - rTemp*rTemp)*Anew
				Tnew = np.arccos(-aTemp/np.sqrt(aabb))
				if bTemp > 0:
					Tnew = 2*np.pi - Tnew

				VarNew = VarOld
				break

			if 1+4*Anew*Fnew < epsilon:
				lamb = lamb * factorUp
				continue

			DD = 1 + 4*Anew*Fnew
			D = np.sqrt(DD)
			CT = np.cos(Tnew)
			ST = np.sin(Tnew)

			GG = 0;

			for i in range(n):
				Xi = XY[i,0] + Xshift
				Yi = XY[i,1] + Yshift
				Zi = Xi*Xi + Yi*Yi
				Ui = Xi*CT + Yi*ST

				ADF = Anew*Zi + D*Ui + Fnew
				SQ = np.sqrt(4*Anew*ADF + 1)
				DEN = SQ + 1
				Gi = 2*ADF/DEN
				GG = GG + Gi*Gi

			VarNew = GG/(n-3)

			H = np.sqrt(1+4*Anew*Fnew)
			anew = -H*np.cos(Tnew)/(Anew+Anew) - Xshift
			bnew = -H*np.sin(Tnew)/(Anew+Anew) - Yshift
			Rnew = 1/np.abs(Anew+Anew)

			# checking if improvement is gained
			if VarNew <= VarOld:	  #   yes, improvement
				progress = (np.abs(anew-aold) + np.abs(bnew-bold) + np.abs(Rnew-Rold))/(Rnew+Rold)
				if progress < epsilon:
					Aold = Anew
					Fold = Fnew
					Told = Tnew
					VarOld = VarNew
					finish = 1
					break

				lamb = lamb * factorDown
				break
			else:					 #   no improvement
				lamb = lamb * factorUp
				continue

		if finish == 1:
			break

	H = np.sqrt(1+4*Aold*Fold)
	Par_1 = -H*np.cos(Told)/(Aold+Aold) - Xshift
	Par_2 = -H*np.sin(Told)/(Aold+Aold) - Yshift
	Par_3 = 1/np.abs(Aold+Aold)
	Par = np.array(([Par_1,Par_2,Par_3]))

	return Par  # LMA


def VarCircle(XY,Par):

	"""
	Fuction computing the sample variance of distances from data points (XY) to the circle Par = [a b R]
	Inputs:
			XY - array of shape (n, 2) containing X and Y points
			Par - array of shape (1, 3) containing (a, b, R) is the initial guess, where circle center (a,b) and radius R

	Output:
			Var - float value for the variance
	"""

	n = np.shape(XY)[0]	  # number of data points

	Dx = XY[:,0:1] - Par[0,0]
	Dy = XY[:,1:2] - Par[0,1]
	D = np.sqrt(Dx*Dx + Dy*Dy) - Par[0,2]

	Var = (np.transpose(D)@D/(n-3))[0,0]

	return Var  #  VarCircle


def dirot(Dec, Inc, Az, Pl):
	"""
	Converts a direction to geographic coordinates using az,pl as azimuth and
	plunge (inclination) of Z direction
	Based on 2G software which uses the strike for the calculations
	Strike = Az +90

	Inputs:
			Dec - (n, 1) array containg declination values
			Inc - (n, 1) array containg inclination values
			Az - float value for the Azimuth
			Pl - float value for the Plunge
	Output:
			Dgeo - (n, 1) array of the geographic declinations in degrees
			Igeo - (n, 1) array of the geographic inclinations in degrees
	"""
	# convert to cartesian coords
	x, y, z = dir2cart(Dec, Inc, 1)

	# find the strike and plunge in radians
	str_rad = deg2rad(Az+90) # Here add 90 to get the strike
	pl_rad = deg2rad(Pl)


	xp =  (x *np.sin(pl_rad) + z *np.cos(pl_rad)) *np.sin(str_rad) + y *np.cos(str_rad)
	yp = -(x *np.sin(pl_rad) + z *np.cos(pl_rad)) *np.cos(str_rad) + y *np.sin(str_rad)
	zp =  -x *np.cos(pl_rad) + z *np.sin(pl_rad)

	Dgeo, Igeo, R = cart2dir(xp, yp, zp)

	return Dgeo, Igeo # rotate data from core to geographic coords



def FisherMeanDir(Dec, Inc, RFlag = 0):
	"""
	Find fisher mean statistics
	Inputs:
			Dec - (n, 1) array for declination values
			Inc - (n, 1) array for inclination values
			RFlag - int value to signal if directions should be reversed, 1 == True, 0 == False, any other value will raise an error
	Outputs:
			Mdec - (n,1) array for Fisher mean declination values
			Minc - (n,1) array for Fisher mean inclination values
			k - float value
			a95 - float value for a95 of the Fisher mean
			R - float value for the magnitude of the Fisher mean direction
	"""

	if RFlag == 1: # Flip Reverse directions
		Inc[np.where(Dec < 270 and Dec > 90)] = -1*Inc[np.where(Dec < 270 and Dec > 90)]
		Dec[np.where(Dec < 270 and Dec > 90)] = Dec[np.where(Dec < 270 and Dec > 90)] + 180
		Dec[np.where(Dec < 0)] = Dec[np.where(Dec < 0)] + 360
		Dec[np.where(Dec > 360)] = Dec[np.where(Dec > 360)] - 360

	elif RFlag != 0:
		raise ValueError("Utilities, FisherMeanDir;  RFlag value not supported.  RFlag must be 0 or 1")

	# get cartesian coords
	X, Y, Z = dir2cart(Dec, Inc)

	# check for nan values
	X = X[~np.isnan(X)]
	Y = Y[~np.isnan(Y)]
	Z = Z[~np.isnan(Z)]

	# ensure correct shape
	X = X.reshape((len(X),1))
	Y = Y.reshape((len(X),1))
	Z = Z.reshape((len(X),1))

	N = len(X)

	# sum directions
	xsum = np.sum(X)
	ysum = np.sum(Y)
	zsum = np.sum(Z)

	# find the magnitude of the mean direction
	R2 = xsum**2 + ysum**2+ zsum**2
	R = np.sqrt(R2)

	# normalize
	x = xsum/R
	y = ysum/R
	z = zsum/R

	# convert to directional coords
	Mdec = rad2deg(np.arctan2(y,x))
	Minc = rad2deg(np.arcsin(z))

	# check Mdec within 0 and 360
	if Mdec < 0:
		Mdec = Mdec + 360
	if Mdec > 360:
		Mdec = Mdec - 360

	# calculate stats output
	k = (N-1)/(N-R)
	if N == 1:
		pwr = np.inf
	else:
		pwr = 1/(N-1)
	bracket = ((1/0.05)**pwr)-1
	OB = (N-R)/R
	Calpha = 1-(OB*bracket)
	alpha = np.arccos(Calpha)
	a95 = rad2deg(alpha)

	return Mdec, Minc, k, a95, R # get the Fisher mean


def Polyarea(x,y, n = 3):
	"""
	Uses shoelace forumla to find area of polygon - https://en.wikipedia.org/wiki/Shoelace_formula
	Inputs:
			x - array of x points that define enclosed area
			y - array of y points that define enclosed area
			n - int value for number of points
	Output:
			Area - float value for the area enclosed by the input points
	"""
	# x and y need to be 1d e.g. (3,)
	x = x.reshape(n,)
	y = y.reshape(n,)
	# cal area enclosed
	Area = 0.5 *np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

	return Area


def custom_formatwarning(msg, *args, **kwargs):
	"""
	Function gives control over warning output to allow for greater clarity to user
	Inputs:
			msg - str to be used in the warning message
	Output:
			returns formatted warning message
	"""
	# ignore everything except the message
	return str(" WARNING: \n")+str(msg) + '\n'


def MDblocking(Tb1, Tb2, beta_vals, lam, alph4):
	"""
	Blocking function based on Andy Biggin's MD Model P
	Modified to use a beta distribution for reciprocal blocking temperatures

	Tb1 and Tb2 are the distribution blocking temperatures
	a and b are the beta distribution parameters
	lambda is the MD 'scaling' parameter
	alph4 is another scaling factor (I usually set it to zero)

	Inputs:
			Tb1 - float value for the lower temperature bound
			Tb2 - float value for the upper temperature bound
			beta_vals - tuple containg the beta values a and b,  (a,b)
			lam - float value for lambda
			alph4 - float value for scaing factor
	Output:
			chi - function for MD blocking to be used in integration
	"""


	Tbprime = (Tb1+ Tb2)/2

	if Tbprime == 1:
		Tbprime = 1-eps

	a = beta_vals[0]
	b = beta_vals[1]
	# reciprocal function
	alpha = sp.stats.beta.pdf(Tbprime, a, b)
	# end of reciprocal function

	gamma = np.abs(Tb2-Tb1)

	# chi function defined in stages
	chia = gamma / lam
	chib = chia**2
	chic = (1 + chib)**(-1)
	chid = alpha * chic
	chi = chid + alph4

	return chi

def field_fit():
	"""
	Lognorm distribution fitting parameters for Banc values
	Output:
			fit - (3, ) array containing fit parameters for Banc
	"""

	fit = np.array([0.10196503349695824, -126.27590228194933, 173.5119376119369])

	return fit
	# Old weibull fit
	# """
	# Weibull distribution fitting parameters for Banc values
	# Output:
	# 		fit - (2, ) array containing fit parameters for Banc
	# """
	# fit = np.array([48.4561988657695, 2.38677810181194])
	#
	# return fit


def quad2d(blocking, a_lim, b_lim, low_curve, high_curve, rounding_err = False):
	"""
	Calls scipy dblquad function
	Inputs:
			blocking - blocking distribution function
			a_lim - float values for limits of integration (a_lim < b_lim)
			b_lim - float values for limits of integration (a_lim < b_lim)
			low_curve - lambda function or float for the lower boundary curve in y, function must take single float argument and return a float
			high_curve - lambda function or float for the upper boundary curve in y, function must take single float argument and return a float
			Atol - float for absolute tolerance in integration
			Rtol - float for relative tolerance in integration
	Outputs:
			y - float value for the integral
	"""
	Atol = 1e-8
	Rtol = 1e-4


	if rounding_err == True:
		Atol = Atol/1e3
		Rtol = Rtol/1e3

	y, err = dblquad(blocking, a_lim, b_lim, low_curve, high_curve, epsabs = Atol, epsrel = Rtol)

	return y


def Generate_N_orient(B_iter):
	"""
	Creates normalised N_orient (ChRM) vectors with random orientation
	Input:
			B_iter - int value for the number of vectors to create

	Output:
			N_orient - (B_iter, 3) array with each row containing the x, y and z components of the normalised and randomly oriented vector

	"""
	# Number of directions needed, 3 for every row
	n_directions = B_iter*3

	# set up and generate random values
	rng = default_rng()
	vals = rng.standard_normal(n_directions)
	N_vec = np.array((vals))
	# shape into output
	N_vec = N_vec.reshape((B_iter,3))
	# normalize each vector
	vec_len = np.sqrt( np.sum(N_vec**2, axis = 1))
	N_orient = np.empty((B_iter,3))
	N_orient[:] = 0
	for i in range(3):
		N_orient[:,i] = N_vec[:,i] / vec_len

	return N_orient


def ATRMtensor(data, dirs, baseline = False):
	"""
	Function to calculate the anisotropy tensor for a 6 position experiment
		See SPD: https://earthref.org/PmagPy/SPD/spdweb_p8.html
		Inputs:
		  data - 6x3 matrix containing the 6 3D vectors of the TRM
				 measurements
		  dirs - 6x3 matrix containing the 6 3D vectors that define the
				 directiosn used to impart the magnetization
		  baseline - 6x3 matrix containing the 6 3D vectors of the
					 demagnetized sample. Used to subtract residual magnetization
		Outputs:
		  s - The 6 uniques elements of the anisotropy tensor
		  tau - The eigen values of teh anisotropy tensor
		  A - Anisotropy tensor (3, 3)
	"""
	# Check inputs
	S=np.shape(data)
	npos = S[0]

	if baseline == False:
		basline = np.zeros((npos,3))

	if S[1] != 3:
		raise ValueError('ATRM_Tensor:directions; Directions must be three axes')


	if npos != 6:
			raise ValueError('ATRM_Tensor:directions; Only 6 direction positions are supported')


	work = data - baseline
	w = np.reshape(work, (np.size(work),1))
	B = designATRM(dirs)[0] # get the design matrix

	s = B@w
	trace = np.sum(s[0:3])
	s = s/trace

	A = np.empty((3,3))
	A[:] = np.nan
	# Build the 3x3 matrix
	A[0,0] = s[0]
	A[1,1] = s[1]
	A[2,2] = s[2]
	A[1,0] = s[3]
	A[2,1] = s[4]
	A[2,0] = s[5]
	A[0,1] = A[1,0]
	A[0,2] = A[2,0]
	A[1,2] = A[2,1]

	tau = np.linalg.eigvals(A) # returns unnormalized eigen values
	tau=tau/sum(tau)

	return s, tau, A


def designATRM(dirs):
	"""
 	Function to create an anisotropy design matrix for a 6 position experiment

 	See SPD: https://earthref.org/PmagPy/SPD/spdweb_p8.html

 	Inputs:
	   	dirs - 6x3 matrix containing the 6 3D vectors that define the
			  	directiosn used to impart the magnetization

 	Outputs:
	   	B - The transpose/inverted design matrix (multiplied by the TRM
		   	vectors to determine the anisotropy tensor elements)
	   	A - The design matrix
	"""
	# Check inputs
	S = np.shape(dirs)

	if S[1] != 3:
		raise ValueError('designATRM:directions; Directions must be three axes')


	npos = S[0]

	if npos != 6:
		raise ValueError('designATRM:directions; Only 6 direction positions are supported')

	## Get the matrix

	A=np.zeros((npos*3, 6))

	for i in range(npos):
		ind = (i*3)
		A[ind,0]=dirs[i,0]
		A[ind,3]=dirs[i,1]
		A[ind,5]=dirs[i,2]

		ind=(i*3)+1
		A[ind,3]=dirs[i,0]
		A[ind,1]=dirs[i,1]
		A[ind,4]=dirs[i,2]

		ind=(i*3)+2
		A[ind,5]=dirs[i,0]
		A[ind,4]=dirs[i,1]
		A[ind,2]=dirs[i,2]


	At = np.transpose(A)
	ATA= At@A
	B = np.linalg.solve(ATA,At)
	return B, A



def RndVec(size = (1,3)):
	"""
	Generates normalised random vector of specified shape
	Input:
			size - tuple containing the desired shape of the output
	Output:
			unitvec - randomly oriented normalised vector
	"""
	# create random vector
	rng = default_rng()
	vec = rng.random(size = size)

	# normalise
	unitvec = vec/( np.sqrt( np.sum(vec**2)))
	return unitvec


def FindP():
	"""
	Function generates random P value for anisotropy
	Outputs:
			P - float value describing anisotpy of sample
	"""
	# generate 3 random vetors
	k = RndVec(size = (3,3))
	# find magnitude of each
	mag = np.sqrt(np.sum(k**2,aizs=1))

	# Find the largest and smallest vector
	k1 = np.amax(mag)
	k3 = np.amin(mag)

	# P given by ratio of k1 and k3
	P = k1/k3

	return P


def rotvec2mat(r, angle):
	"""
	Find rotation matrix from given rotation axis and angle
	Inputs:
			r - (3, ) array containing vector of rotation axis
			angle - angle of rotation in radians
	Output:
			m - (3, 3) array containg rotation matrix
	"""

	# find value of angle terms
	s = np.sin(angle)
	c = np.cos(angle)
	t = 1 - c

	# normalise rotation axis
	n = r/(np.sqrt(np.sum(r**2)))

	x = n[0]
	y = n[1]
	z = n[2]

	# calculate rotation marix
	m = np.array([[t*x*x + c, t*x*y - s*z,  t*x*z + s*y], [t*x*y + s*z,  t*y*y + c, t*y*z - s*x], [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]] )

	return m


def formattime(time):
	"""
	Funtion formats time from number of seconds to days, hours, minutes and seconds for readability
	Input:
			time - float value of seconds to be converted
	Output:
			days - float value for the number of days
			hours - float value for the number of hours
			minutes - float value for the number of minutes
			seconds - float value for the number of seconds
	"""
	seconds = time

	seconds_in_day = 60 * 60 * 24
	seconds_in_hour = 60 * 60
	seconds_in_minute = 60

	days = round(seconds // seconds_in_day)
	hours = round((seconds - (days * seconds_in_day)) // seconds_in_hour)
	minutes = round((seconds - (days * seconds_in_day) - (hours * seconds_in_hour)) // seconds_in_minute)
	seconds = (seconds - (days * seconds_in_day) - (hours * seconds_in_hour) - (minutes * seconds_in_minute))

	return days, hours, minutes, seconds


def find_nearest(array, value):
	"""
	Finds the value of and index of the value in an array closest to a specified value
	Inputs:
			array - array to be searched
			value - specified value from which to find the closest
	Outputs:
			array[idx] - value in array closest to specified value
			idx - index location of the closest value in array
	"""
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()

	return array[idx], idx


def FishAngleNoise(kappa, N, vec):
	"""
	Function adds on fisher distributed angular noise to a vector
	Inputs:
			kappa - float value
			N - int value for the number of rows in the of the output array
			vec - (n, 3) array with each row containing a vector to which noise is to be added
	Outputs:
			fish_vec - (N, 3) array containing vectors with noise added on
	"""
	# check if noise is turned on
	if kappa == 0:
		# if input vec length is N, then output is vec with no noise
		if len(vec) == N:
			fish_vec = vec
		# if vec contains single vector, repeat vector to specified length
		elif len(vec) == 1:
			fish_vec = np.repeat(vec, N, axis=0)

	# if noise is turned on
	else:
		# extract x, y and z components
		X = vec[:,0]
		Y = vec[:,1]
		Z = vec[:,2]

		# convert to directional coords
		dec_orig, inc_orig, R = cart2dir(X,Y,Z)

		# set up noise array
		k = np.ones((N, 1)) * kappa

		# generate noise
		rng = default_rng()
		R1 = rng.random(size = (N, 1))
		R2 = rng.random(size = (N, 1))
		L = np.exp(-2*k)
		a = R1 * (1-L)+L
		fac = np.sqrt((-np.log(a))/(2*k))
		inc_tmp = 90 -2 * rad2deg(np.arcsin(fac))
		dec_tmp = 2 * R2 * 180


		dec, inc = dirot_pmag(dec_tmp, inc_tmp, dec_orig, 90-inc_orig)

		dec = dec - 180

		dec[np.where(dec<360)] = dec[np.where(dec<360)]+360
		dec[np.where(dec>360)] = dec[np.where(dec>360)]-360

		# get x, y and z components with noise added
		fx, fy, fz = dir2cart(dec, inc, R)
		fish_vec = np.concatenate((fx, fy, fz), axis = 1)

		# clean output
		for r in range(len(vec)):
			if np.sqrt(np.sum(vec[r,:]**2)) == 0:
				fish_vec[r,:] = vec[r,:]


	return fish_vec


def create_array(shape):
	"""
	creaty array containing nan values
	Input:
			shape - desired shape of output array
	Output:
			arr - array of input shape containing nan values at every index
	"""
	# create empty array and populate with nans
	arr = np.empty(shape)
	arr[:] = np.nan

	return arr


def dirot_pmag(Dec, Inc, Az, Plz):
	"""
	Converts a direction to geographic coordinates using az,pl as azimuth and
	plunge of Z direction (from horizontal)
	Based on PyMag by Lisa Tauxe, which uses the plunge of x direction
	"""
	r = len(Dec)

	A1 = create_array((r,1))
	A2 = create_array((r,1))
	A3 = create_array((r,1))
	x, y, z = dir2cart(Dec, Inc)

	if len(Az) != r:
		Az = Az * np.ones((r,1))
		Plz = Plz * np.ones((r,1))

	# set up rotation matrix
	A1 = np.concatenate((dir2cart(Az, Plz)),axis = 1)
	A2 = np.concatenate(dir2cart(Az+90, 0),axis = 1)
	A3 = np.concatenate(dir2cart(Az - 180, 90 - Plz), axis=1)
	# Do rotation
	xp = A1[:,0:1] * x + A2[:,0:1] * y + A3[:,0:1] * z
	yp = A1[:,1:2] * x + A2[:,1:2] * y + A3[:,1:2] * z
	zp = A1[:,2:3] * x + A2[:,2:3] * y + A3[:,2:3] * z

	Dgeo, Igeo = cart2dir(xp, yp, zp)[0:2]

	return Dgeo, Igeo
