import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.signal import detrend
from matplotlib import path
from IZZI_MD_calc import calc_IZZI_MD

#########################
# Yu (2012; JGR) method #
#########################
# get normalised points
def get_Z(Xpts, Ypts, b, seg_min, seg_max):
	Xseg = Xpts[seg_min:seg_max+1]
	Yseg = Ypts[seg_min:seg_max+1]
	Zx_points = Xseg/Ypts[0]
	Zy_points = Yseg/Ypts[0]
	n = len(Xseg)

	# NRM max
	Z_NRM = Zy_points[0]
	# TRM min
	Z_TRM = Zx_points[0]
	# find b at each point. NRM loss / pTRM gained
	bi = (Z_NRM - Zy_points)/(Zx_points-Z_TRM)
	# difference from reference value
	b_p = (bi - np.abs(b))
	# weighting factor
	r = Zx_points/Zx_points[-1]
	# ensure nan
	bi[0] = np.nan

	number = n-1
	Z= np.nansum(b_p*r)/np.sqrt(number)

	return Z
#########################
# Yu (2012; JGR) method #
#########################
# get normalised points
def get_Z_star(Xpts, Ypts, Y_int, b, seg_min, seg_max):
	Xseg = Xpts[seg_min:seg_max+1]
	Yseg = Ypts[seg_min:seg_max+1]
	Zx_points = Xseg/Ypts[0]
	Zy_points = Yseg/Ypts[0]
	z_y_int = (Y_int/Ypts[0]).item()
	Z_NRM = Zy_points[0]
	Z_TRM = Zx_points[0]
	n = len(Xseg)

	# find b at each point. NRM loss / pTRM
	bi = (Z_NRM - Zy_points)/(Zx_points-Z_TRM)
	# ensure nan
	bi[0] = np.nan
	number = n-1
	bi_r = np.abs((bi-np.abs(b))*Zx_points)
	Z_star = 100*np.nansum(bi_r)/(number*z_y_int)
	return Z_star

# Find the scaled length of the best fit line
# normalise points
def get_ziggie(Xpts, Ypts, seg_min, seg_max):
	Xn = Xpts[seg_min:seg_max+1]
	Xn = Xn/Xn[-1]
	Yn = Ypts[seg_min:seg_max+1]
	Yn = Yn/Yn[0]
	n = len(Xn)
	# find best fit line
	U = detrend(Xn, type = "constant", axis = 0) # (Xi-Xbar)
	V = detrend(Yn, type = "constant", axis = 0) # (Yi-Ybar)
	b = np.sign(np.sum(U*V))*np.std(Yn, ddof = 1)/np.std(Xn, ddof = 1);
	Y_int = np.mean(Yn) - b*np.mean(Xn)
	X_int = -Y_int/b

	# Project the data onto the best-fit line
	Rev_x = (Yn - Y_int) / b # The points reflected about the bes-fit line
	Rev_y = b * Xn + Y_int
	x_prime = (Xn + Rev_x)/2   # Average the both sets to get the projected points
	y_prime = (Yn + Rev_y)/2

	# Get the TRM, NRM, and line lengths
	Delta_x_prime = np.abs( np.amax(x_prime)-np.amin(x_prime) )
	Delta_y_prime = np.abs( np.amax(y_prime)-np.amin(y_prime) )
	Line_Len = np.sqrt(Delta_x_prime**2 + Delta_y_prime**2)

	# Set cumulative length to 0
	cum_len = 0.0

	# iterate through pairs of points in Arai plot
	for i in range(0, n-1):

		# find the distance between the two points
		dist = np.sqrt((Xn[i+1,0] - Xn[i,0])**2 + (Yn[i+1,0] - Yn[i,0])**2)
		# Add to the cumulative distance
		cum_len = cum_len + dist

	# calculate the log of the cumulative length over the length of the best fit line
	ziggie = np.log(cum_len/Line_Len)
	return ziggie, cum_len, Line_Len, Xn, Yn, x_prime, y_prime

def get_gradients(Xpts, Ypts, seg):
	X_seg = Xpts[seg]
	Y_seg = Ypts[seg]
	xbar = np.mean(X_seg)
	ybar = np.mean(Y_seg)
	U = detrend(X_seg, type = "constant", axis = 0) # (Xi-Xbar)
	V = detrend(Y_seg, type = "constant", axis = 0) # (Yi-Ybar)
	b = np.sign(np.sum(U*V))*np.std(Y_seg, ddof = 1)/np.std(X_seg, ddof = 1)
	n = len(X_seg)

	sigma_b = np.sqrt( (2*np.sum(V**2)-2*(b)*np.sum(U*V)) / ( (n-2)*np.sum(U**2)) )
	beta = np.abs(sigma_b/b)


	Y_int = np.mean(Y_seg) - b*np.mean(X_seg)
	X_int = -Y_int/b

	return b, Y_int, beta

def get_beta(Xpts, Ypts, seg):
	X_seg = Xpts[seg]
	Y_seg = Ypts[seg]
	n = len(X_seg)
	xbar = np.mean(X_seg)
	ybar = np.mean(Y_seg)
	U = detrend(X_seg, type = "constant", axis = 0) # (Xi-Xbar)
	V = detrend(Y_seg, type = "constant", axis = 0) # (Yi-Ybar)
	b = np.sign(np.sum(U*V))*np.std(Y_seg, ddof = 1)/np.std(X_seg, ddof = 1)
	sigma_b = np.sqrt( (2*np.sum(V**2)-2*(b)*np.sum(U*V)) / ( (n-2)*np.sum(U**2)) )
	beta = np.abs(sigma_b/b)


	Y_int = np.mean(Y_seg) - b*np.mean(X_seg)
	X_int = -Y_int/b

	return beta


def get_points_2d(j, grad):
	grad = -grad
	angle = np.arctan(-grad)
	Ypts = np.arange(0,11,1)
	Ypts = Ypts.reshape((len(Ypts),1))/10
	Xpts = (Ypts - 1)/grad

	Ypts[1:-1:2] += j*np.sin(angle)
	Ypts[2:-1:2] -= j*np.sin(angle)
	Xpts[1:-1:2] += j*np.cos(angle)
	Xpts[2:-1:2] -= j*np.cos(angle)

	Xpts = Xpts[::-1]
	Ypts = Ypts[::-1]


	return Xpts, Ypts



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


def SCAT(b, Xpts, Ypts, seg_min, seg_max, n):

	seg = np.arange(seg_min, seg_max+1,1)
	X_seg = Xpts[seg]
	Y_seg = Ypts[seg]
	xbar = np.mean(X_seg)
	ybar = np.mean(Y_seg)

	## SCAT - uses the user input
	beta_T = 0.1
	sigma_T = beta_T*np.abs(b)
	b1 = b - 2*sigma_T
	b2 = b + 2*sigma_T

	# determine the intercepts
	a1 = ybar-b1*xbar # upper
	a2 = ybar-b2*xbar # lower
	a3 = -a2/b2 # the upper
	a4 = -a1/b1 # and lower x-axis intercepts

	C1 = np.array([[0, a2]]) # lower left corner
	C2 = np.array([[0, a1]]) # upper left corner
	C3 = np.array([[a3, 0]]) # upper right corner
	C4 = np.array([[a4, 0]]) # lower right corner
	SCAT_BOX = np.concatenate((C1, C2, C3, C4, C1),axis = 0)

	Check_points =  create_array((1,1))# the x-, y-coords of all the checks within the SCAT range
	Check_points =  np.delete(Check_points, (0), axis=0)




	# Create an array with the points to test
	if Check_points.size == 0:
		SCAT_points = (np.concatenate((X_seg, Y_seg),axis = 1)) # Add the TRM-NRM Arai plot points

	else:
		quit()

	eps = (np.finfo(float).eps)
	# add tolerance so points on line are included
	for point in SCAT_points:
		if point[0] == 0.0:
			point[0] = point[0]+eps
		if point[1] == 0.0:
			point[1] = point[1] + eps
	p = path.Path(SCAT_BOX)
	IN = p.contains_points(SCAT_points)

	SCAT = np.floor(np.sum(IN)/len(IN)) # The ratio ranges from 0 to 1, the floor command rounds down to nearest integer (i.e., rounds to 0 or 1)

	## Get multiple SCATs - uses a hard coded range of beta thresholds

	# beta_thresh=(0.002:0.002:0.25);
	beta_thresh = np.linspace(0.01, 0.25, 100)
	nthresh = len(beta_thresh)

	tmp_SCAT = np.empty((nthresh,1))
	tmp_SCAT[:] = np.nan

	for i in range(nthresh):
		tmp_beta_T = beta_thresh[i]

		sigma_T = tmp_beta_T*np.abs(b)
		b1 = b - 2*sigma_T
		b2 = b + 2*sigma_T

		# determine the intercepts
		a1 =  ybar-b1* xbar # upper
		a2 =  ybar-b2* xbar # lower
		a3 = -a2/b2 # the upper
		a4 = -a1/b1 # and lower x-axis intercepts

		C1 = np.array([[0, a2]]) # lower left corner
		C2 = np.array([[0, a1]]) # upper left corner
		C3 = np.array([[a3, 0]]) # upper right corner
		C4 = np.array([[a4, 0]]) # lower right corner
		SCAT_BOX_mul = np.concatenate((C1, C2, C3, C4, C1),axis = 0)

		Check_points =  create_array((1,1))# the x-, y-coords of all the checks within the SCAT range
		Check_points =  np.delete(Check_points, (0), axis=0)  # the x-, y-coords of all the checks within the SCAT range

		# Create an array with the points to test
		if Check_points.size == 0:
			SCAT_points = (np.concatenate((X_seg, Y_seg),axis = 1)) # Add the TRM-NRM Arai plot points
		else:
			SCAT_points = vstack((np.concatenate((X_seg, Y_seg),axis = 1), Check_points)) # Add the TRM-NRM Arai plot points

		for point in SCAT_points:
			if point[0] == 0.0:
				point[0] = point[0]+eps
			if point[1] == 0.0:
				point[1] = point[1] + eps

		p = path.Path(SCAT_BOX_mul)
		IN = p.contains_points(SCAT_points)
		tmp_SCAT[i] = np.floor(np.sum(IN)/len(IN))  # The ratio ranges from 0 to 1, the floor command rounds down to nearest integer (i.e., rounds to 0 or 1)

	multi_SCAT = tmp_SCAT
	multi_SCAT_beta = beta_thresh
	return SCAT


def get_ecdf(ecdf_data):
	numbers = ecdf_data.to_list()
	n_p = int(len(numbers))
	y_p = np.linspace(1/n_p, 1, num=n_p)

	numbers.insert(0,0.0)
	numbers.sort()
	y_p = np.insert(y_p,0,0.0)

	return numbers, y_p


def get_IZZI_MD(Xpts,Ypts,Treatment,seg_min,seg_max):
	if seg_min == 0:
		seg_min = 1

	IZZI_MD = calc_IZZI_MD(Xpts,Ypts,Treatment,seg_min,seg_max)
	if np.isnan(IZZI_MD):
		return IZZI_MD

	else:
		return np.float64(IZZI_MD.item())
