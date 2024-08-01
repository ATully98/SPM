import Utilities
import GetPintParams_v7
import numpy as np
import time
import Directional

def Run(Mvec, Temps, Treatment, T_orient, Blab, Az, Pl, ChRM, Flags, Anis_vec,  NLT_vec, Bnlt, beta_T, Beta_fits, Bexp, Simulation, fn, params, lam = "N/A"):

	"""
	Control Run function takes input from the model and provides the statistical results of the experiment for all possible sequential fits of the Arai plot
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
			lam  -  float containing the lambda value
			Sample - str containing the name of the sample, defaults to "No Sample Name".  This is used when analysing real sample data.

	Outputs:
			Fits - list of dictionary containing all the statistics calculated for each Arai plot fit
			end_time - float for the time taken to calculate and return stats in seconds
	"""
	#Min number of points in fit
	Nfitmin = 2
	ffitmin = 0.1
	start_time = time.time()

	#number of steps
	UT = np.unique(Temps)
	points = len(UT)+1
	Fits = []

	blocking_a = Beta_fits[0]
	blocking_b = Beta_fits[1]
	for i in range(points - Nfitmin+1):

		# stop is points for models and points-1 for tdt, must check why
		for j in range(i+ Nfitmin - 1, points):
			# print("Current Fit;   seg start: "+str(i) +", seg end: "+ str(j))
			Params = GetPintParams_v7.GetPintParams_v7(Mvec, Temps, Treatment, i, j, T_orient, Blab, Az, Pl, ChRM, Flags, Anis_vec,  NLT_vec, Bnlt, beta_T)
			#Params = Directional.direction_stats(Mvec, Temps, Treatment, Blab, T_orient, ChRM, Az, Pl,Flags, i, j)
			Params["Lambda"] = lam
			Params["Lam. Dist."] = str(f"{fn} {params}")
			Params["blocking a"] = blocking_a
			Params["blocking b"] = blocking_b
			Params["N_orient"] = ChRM
			Params["Bexp"] = Bexp.reshape((1,)).item()
			Params["Accuracy"] = Params["Banc"]/Params["Bexp"]
			Params["Simulation"] = Simulation

			Fits.append(Params)


	end_time = (time.time() - start_time)
	return Fits, end_time
