import Utilities
import GetPintParams_v7
import numpy as np
import time
import pandas as pd
import scipy as sp
from scipy.signal import detrend
from numpy.random import default_rng
def Run(Mvec, Temps, Treatment, T_orient, Blab, Az, Pl, ChRM, Flags, Anis_vec,  NLT_vec, Bnlt, beta_T, Beta_fits, Bexp,  Simulation, lam = "N/A"):
	"""
	Free Run function takes input from the model and provides the statistical results of the experiment for a randomly selected fits of the Arai plot.  This fit contains sequential points and
	must pass specified criteria on the fraction of the full plot that is used.  Both the minimum fraction and the way in which this calculated can be specified.  In order to do this the GetPintParams
	function is called twice, once intially for the whole plot fit with skeleton values to provide the X and Y points of the plot allowing te fraction of the fit to be calculated and then again to
	 find the statistics of the random fit.
	Inputs:
			Mvec - (n, 3) array, matrix of magnetization values when n is the total number of steps, which includes NRM, TRM, and all check measurements (i.e., the raw data measurements)
			Temps - (n, 1) array of the temperature of each treatment
		  	Treatment - (n, ) array of integer values describing the treatment type at each step, this follows the convention of the ThellierTool
						 (0=NRM demag, 1=TRM remag, 2=pTRM check, 3=pTRM tail check, 4=additivity check, 5=inverse TRM step). If any Treatment is set to '5' all data are treated as a Thellier experiment
		   	T_orient - (1, 3) array for the unit vector containing the x, y, z values for a know Blab orientation.
		    Blab - float value of the strength of the laboratory field in muT
			Az - float value for the sample azimuth in degrees
			Pl - float value for the sample plunge in degrees - angle from horizontal, positive down
			ChRM - a (1, 2) or a (1, 3) array for the vector describing an independent measure of the direction of Banc (can be a known direction).
							 If ChRM is a (1, 2) vector it is assumed to be a Dec/Inc direction and converted to a Cartesian vector.
							 If rotation is applied ChRM is assumed to already be in the final coordinate system
			Flags - list of flags to be used in stats:
							A_flag		- flag for anisotropy correction (0=no correction, 1=apply correction)
							NLT_flag	  - flag for non-linear TRM correction (0=no correction, 1=apply correction)
							NRflag		- flag for Reciprocity correction (0=no correction, 1=apply correction)
							NRM_rot_flag - flag for directional rotation from core coords to stratigraphic coords (0 or []=no rotation, 1=apply rotation)
			NLT_vec - (len(Bnlt)+1, 3) array with each row containing the vector for each NLT check step
		 	Anis_vec - (7, 3) array containing the anis vectors for each oreintation
			Bnlt -	(6, ) array containing the NLT temperture steps, the last value is Blab
			beta_T - a (1, 1) scalar for the beta threshold for defining the SCAT box, if missing the default value is beta_T=0.1
			Beta_fits - (2, ) array containing the parameters a and b
			Bexp - (1, ) array containing the expected value for the ancient field, Banc. This is the Banc passed and used in the calculations in the Model
			lam  - (1,) array containing the lambda value
			Sample - str containing the name of the sample, defaults to "No Sample Name".  This is used when analysing real sample data.

	Outputs:
			Fits - list of dictionary containing all the statistics calculated for each Arai plot fit
			end_time - float for the time taken to calculate and return stats in seconds
	"""
	#Min number of points in fit
	Nmin = 3
	ffitmin = 0.45
	start_time = time.time()

	#number of steps
	UT = np.unique(Temps)
	points = len(UT)+1
	Fits = []
	rng = default_rng()
	# free start point
	start_pt_free = rng.integers(0, high = points - Nmin +1)

	# Find TRMvec and NRMvec points on Arai plot
	tmp_Params = GetPintParams_v7.GetPintParams_v7(Mvec, Temps, Treatment, 0, points-1, T_orient, Blab, [], [], [], [0,0,0,0], [],  [], [], 0.1)


	# find random fit segment
	seg = AraiFitter(tmp_Params["TRMvec"], tmp_Params["NRMvec"], ffitmin, "FRAC", points, Nmin, start_pt_free)

	Params = GetPintParams_v7.GetPintParams_v7(Mvec, Temps, Treatment, seg[0], seg[-1], T_orient, Blab, Az, Pl, ChRM, Flags, Anis_vec,  NLT_vec, Bnlt, beta_T)
	Params["Lambda"] = lam.item()
	Params["blocking"] = Beta_fits
	Params["N_orient"] = ChRM
	Params["Bexp"] = Bexp#.item()
	Params["Accuracy"] = Params["Banc"]/Params["Bexp"]
	Params["Simulation"] = Simulation

	Fits.append(Params)


	end_time = (time.time() - start_time)
	return Fits, end_time




def AraiFitter(TRMvec, NRMvec, fmin, type, points, Nmin, start_pt):
	"""
	Find free fit of the Arai plot that passes minimum fraction criteria
	Inputs:
			TRMvec - (n, 4) array with each row containing the temperature step and the TRM vector components
			NRMvec - (n, 4) array with each row containing the temperature step and the NRM vector components
			fmin - float value specifying the minimum fraction of the Arai plot to be used in the fit
			type - str input specifying the method with which to calculate the Arai plot fraction used.  Currently supported methods are FRAC, Fvds, f.  The default is FRAC, a warning will be raised when this occurs
			points - int value for the number of points in the Arai plot (+1 for indexing in python)
			Nmin - int value for the minimum number of Arai plot points to be used
			start_pt - int for the starting index of the fit
	Output:
			seg - (m, ) array containing the indices of the points to be used in the fit
	"""


	Xpts = np.sqrt(np.sum(TRMvec, axis = 1))
	Ypts = np.sqrt(np.sum(NRMvec[:,1:]**2, axis = 1))

	VDS = np.sum(np.sqrt(np.sum((np.diff(NRMvec[:,1:])**2),axis = 1))) + np.sqrt(np.sum(NRMvec[-1,1:]**2))


	seg_min = start_pt
	seg_max = seg_min + rng.integers(Nmin, points - seg_min+1)
	points = points - 1
	seg =  np.arange(seg_min, seg_max)
	X_seg = Xpts[seg_min:seg_max+1]
	Y_seg = Ypts[seg_min:seg_max+1]

	U = detrend(X_seg, type = "constant", axis = 0) # (Xi-Xbar)
	V = detrend(Y_seg, type = "constant", axis = 0) # (Yi-Ybar)



	slope = np.sign(np.sum(U*V))*np.std(Y_seg, ddof = 1)/np.std(X_seg, ddof = 1)
	Y_int = np.mean(Y_seg) - slope*np.mean(X_seg)


	# Project the data onto the best-fit line
	Rev_y = slope * Xpts + Y_int
	Py = (Ypts + Rev_y) /2

	dyt = np.abs( np.max(Py[seg_min:seg_max+1]) -  np.min( Py[seg_min:seg_max+1] ) )
	sumdy = np.sum( np.diff( Py[seg_min:seg_max+1] )**2)

	gap = 1 - (sumdy/(dyt**2))

	f = np.abs(dyt/Y_int)
	fvds = np.abs(dyt/VDS)
	FRAC = np.sum( np.sqrt( np.sum( (np.diff(NRMvec[seg_min:seg_max+1,1:])**2), axis = 1 ) ) ) /VDS

	if type == "FRAC":
		stat_type = FRAC
	elif type == "Fvds":
		stat_type = fvds
	elif type == "f":
		stat_type = f
	else:
		stat_type = FRAC
		raise warning(f"AraiFitter; {type} Fraction calculation method is not currently available, FRAC will be used")


	while (slope > 0 ) or (stat_type < fmin) or (gap < 0):

		if (seg_max == points) and (seg_min > 0):
			# At the end, but not the start - one extra point at the start
			seg_min = seg_min-1
		elif (seg_min == 0 and seg_max < points):
			# At the start, but not the end - one extra point at the end
			seg_max = seg_max + 1
		else:
			# randomly add to the end or start
			# Random interger: -1, or 1
			# rnd_int(1) decides seg_min/max: -1=min, 1=max
			rnd_int = (1 - -1)*rng.random_sample() + -1
			if rnd_int == -1:
				seg_min = seg_min - 1
			else:
				seg_max = seg_max + 1



		# Make sure things are not going funky
		if (seg_min < 1):
			seg_min = 1

		if (seg_max > points):
			seg_max = points


		if (seg_min == 1) and (seg_max == points):
			# whole segment
			#		 disp(['FRAC = ', sprintf('%1.2f', FRAC), 'SEG = ', sprintf('%d, %d', seg_min, seg_max)])
			seg = np.arange(seg_min, seg_max+1)
			break


		# recalculate the best-fit parameters
		seg =  np.arange(seg_min, seg_max+1)
		X_seg = Xpts[seg_min:seg_max+1]
		Y_seg = Ypts[seg_min:seg_max+1]
		U = detrend(X_seg, type = "constant", axis = 0) # (Xi-Xbar)
		V = detrend(Y_seg, type = "constant", axis = 0) # (Yi-Ybar)
		slope = np.sign(np.sum(U*V))*np.std(Y_seg, ddof = 1)/np.std(X_seg, ddof = 1)
		Y_int = np.mean(Y_seg) - slope*np.mean(X_seg)
		Rev_y = slope * Xpts + Y_int
		Py = (Ypts + Rev_y) /2
		dyt = np.abs( np.max(Py[seg_min:seg_max+1]) -  np.min( Py[seg_min:seg_max+1] ) )
		sumdy = np.sum( np.diff( Py[seg_min:seg_max+1] )**2)
		gap = 1 - (sumdy/(dyt**2))
		f = np.abs(dyt/Y_int)
		fvds = np.abs(dyt/VDS)
		FRAC = np.sum( np.sqrt( np.sum( (np.diff(NRMvec[seg_min:seg_max+1,1:])**2), axis = 1 ) ) ) /VDS
		if type == "FRAC":
			stat_type = FRAC
		elif type == "Fvds":
			stat_type = fvds
		elif type == "f":
			stat_type = f
		else:
			stat_type = "FRAC"
			raise warning(f"AraiFitter; {type} Fraction calculation method is not currently available, FRAC will be used")

	return seg
