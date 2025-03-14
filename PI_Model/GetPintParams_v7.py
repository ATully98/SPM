import numpy as np
from numpy import vstack
from numpy.linalg import norm
from numpy.polynomial import Polynomial as Poly
import scipy as sp
from scipy.signal import detrend
import warnings
from warnings import warn as warning
import Utilities
import matplotlib as mpl
from matplotlib import path
import zig_params
def GetPintParams_v7(Mvec, Temps, Treatment, start_pt, end_pt, Blab_orient, Blab, Az, Pl, ChRM, Flags, Anis_vec,  NLT_vec, Bnlt, beta_T):

	"""
	March 2022 - AWT
	v7  - Inclusion of NLT and Anis into calculations and outputs

	Python conversion of GetPintParams.m
	Nov 2021

	% function to return all paleointensity parameters for a sample
	%
	% LAST UPDATE 28 July 2018
	%
	% 28 July 2018 - GAP
	% 1) If number of points is <=3, Arai Curvature uses the Taubin estimate as
	%	opposed to teh LMA one, which needs at least 4 points to calculate variance.
	%
	% 22 July 2014 - GAP
	% 1) Added output variables for s_tensor for the anisotropy data
	% 2) Added output variables for Meas_Data (Mvec) and Experiment (Temps, and Treatment) and Blab_orient
	% 3) Added S_prime_qual the fitting quality factor of Carvallo & Dunlop (2001, ESPL)
	%
	% 04 Apr. 2014 - GAP
	% 1) Added output for the NRM and TRM vectors
	% 2) Uncommented k_prime code
	% 3) Corrected error in calculation of IZZI_MD, whereby all points were used to determine the length of the ZI line
	%
	% 25 Feb. 2014 - GAP
	% 1) Corrected additivity checks to test pTRM(Ti, T0) instead of pTRM(Tj, Ti). This is the correct convevtion.
	%	Changes applied to input data handling and dAC calcualtion
	%
	% 18 Feb. 2014 - GAP
	% 1) Tidied up the additivity check section, pTRM difference now matches SPD and the calculation no longer uses the ADD_vec array for temperatures
	% 2) Add Plot_Z_Arai and Plot_Z_pTRM_Lines for outputing the z-component data only
	%
	% 09-11 Feb. 2014 - GAP
	% 1) Updated version to v6
	% 2) Corrected R_det to use the segment and not all of the points
	% 3) Corrected the denominator for GAP_MAX
	% 4) Changed theta to use the free-floating PCA fit and not the anchored
	% 5) Corrected calculation for VDS, which affects FRAC and GAP_MAX
	% 7) Added Params.Plot_orth Params.Plot_line to output the NRM vector and best-fit Arai line for plotting
	% 8) Added Params.x_prime, Params.y_prime, Params.Delta_x_prime, and Params.Delta_y_prime, to replace Px, Py, TRM_len, and dyt, respectively.
	% 9) Added beta_T for the SCAT box as an input variable with a default value of 0.1
	% 10) Changed T_orient to Blab_orient
	% 11) Added PearsonCorr2 function to get the square of the Pearson Correlation (R_corr). This removes a dependency on the inbuilt MATLAB function
	% 12) Brought more variable names inline with SPD names
	% 13) Updated the IZZI_MD calculation to follow the pseudo-code outline in SPD (logic rearrangement, no calculation change)
	% 14) Removed Params.alpha(Params.alpha>90)=Params.alpha-90 for alpha, alpha_prime, DANG, and alpha_TRM. Legacy from early version of the models
	%	 when direction of PCA was not constrained in the PmagPCA routine
	% 15) Changed Ints(j) to Params.Ypts(j) in the calculation of CRM(j) for CRM_R. This SPD notation.
	% 16) Made Params.Mean_DRAT calculation explicit (removed the use of CDRAT).
	% 17) Added Params.dpal_ratio (= log(corrected_slope/measured_slope)) and Params.dpal_signed (signed dpal) to test later - I think dpal is biased
	% 18) Change the anisotropy correction to use the free-floating PCA fit instead of the vector mean
	% 19) Simplified the calculation of tstar(j) (part of dt*) in the last IF segment - it included terms that cancelled
	% 20) Corrected parameters that use a normalizer that can be negative. Now they use the absolute value of the normalizer (f, Z, Zstar, NRM_dev, dCK, dt_star, dAC).
	% 21) Added additional comment lines that are added for the SPD.m version of this code
	% 22) Corrected several spelling mistakes in the comments and updated the descriptions for the output statistics
	% 23) v6 of the code will be the basis for the version to made publicly available (as SPD.m). The public version will remove extra functions and
	%	 statistics that are not in SPD (e.g., tests for common slopes/elevations, commented code, etc.). The plotting outputs will also be removed.
	%
	% 23/24 Nov. 2013 - GAP
	% 1) Added Params.pTRM_Lines to output the lines for plotting pTRM checks
	% on Arai plots
	% 2) Clarified orientation convention in subfunction dirot (rotates from core to geographic coords)
	%
	% 25 Sept. 2013 - GAP
	% 1) Added Params.pTRM_sign, the sign of the maximum pTRM check stats
	% 2) Added Params.CpTRM_sign, the sign of the cumulative pTRM check stats
	% 3) Added Params.tail_sign, the sign of the maximum pTRM tail check stats
	%
	% 24 Sept. 2013 - GAP
	% 1)Corrected an error in the handling of Thellier data, due to additivity checks not being added to the
	% data handling routine.The error caused a crash and did not influence any stats.
	% 2) Rearranged the pTRM check data handling for Thellier data to make it more efficient.
	% 3) Added a missing line in Thellier data handling for the calculation of check(%)
	%
	% 28-31 Aug. 2013 - GAP
	% 1) Added rounding of the stats. Placed in a single code block at the end to allow easy adjustments
	% 3) Updated terminology to fit with SPD.
	% 3) Corrected variable name "corr_Params.slope" to "corr_slope" lines 963/4
	% 4) Change Params.MDpoints to output the measure NRM for the tail checks
	% 5) Added Params.ADpoints to output additivity data and overhauled/corrected/updated the additivity check calculations
	% 6) Added Params.PLOT, an array containing the data required to create an Arai plot
	% 7) Updated pTRM checks such that they are only included if both the check temp and the peak experiment temp are <= Tmax
	% 8) Updated IZZI_MD to correct the calculation of the the ZI curve length
	%

	## Input

	Mvec		  - (n, 3) array, matrix of magnetization values when n is the total number of steps, which includes NRM, TRM, and all check measurements (i.e., the raw data measurements)
	Temps		 -  (n, 1) array of the temperature of each treatment
	Treatment	 -  (n, ) array of integer values describing the treatment type at each step, this follows the convention of the ThellierTool
				 (0=NRM demag, 1=TRM remag, 2=pTRM check, 3=pTRM tail check, 4=additivity check, 5=inverse TRM step). If any Treatment is set to '5' all data are treated as a Thellier experiment
	start_pt	  - integer for the start point for the Arai plot best-fit line - ONLY THE INDEX IS CODED, NOT THE TEMPERATURE.
	end_pt		- integer for the end point for the Arai plot best-fit line - ONLY THE INDEX IS CODED, NOT THE TEMPERATURE.
	Blab_orient	  - (1, 3) array for the unit vector containing the x, y, z values for a know Blab orientation.
	Blab		  - float value of the strength of the laboratory field in muT
	Az			- float value for the sample azimuth in degrees
	Pl			- float value for the sample plunge in degrees - angle from horizontal, positive down
	ChRM		  - a (1, 2) or a (1, 3) array for the vector describing an independent measure of the direction of Banc (can be a known direction).
					 If ChRM is a (1, 2) vector it is assumed to be a Dec/Inc direction and converted to a Cartesian vector.
					 If rotation is applied ChRM is assumed to already be in the final coordinate system
	Flags 		- list of flags to be used in stats:
					A_flag		- flag for anisotropy correction (0=no correction, 1=apply correction)
					NLT_flag	  - flag for non-linear TRM correction (0=no correction, 1=apply correction)
					NRflag		- flag for Reciprocity correction (0=no correction, 1=apply correction)
					NRM_rot_flag  - flag for directional rotation from core coords to stratigraphic coords (0 or []=no rotation, 1=apply rotation)
	NLT_vec - (len(Bnlt)+1, 3) array with each row containing the vector for each NLT check step
	Anis_vec - (7, 3) array containing the anis vectors for each oreintation
	Bnlt		- (6, ) array containing the NLT temperture steps, the last value is Blab
	beta_T		- a (1, 1) scalar for the beta threshold for defining the SCAT box, if missing the default value is beta_T = 0.1


	%% Output
	% Output is currently a dictionary with fields listed below.
	% See the Standard Paleointensity Definitions for full details of the statistics.
	%
	% Meas_Data	-   the measurement data (x, y, z)
	% Meas_Temp   -   the experimental temperture steps
	% Meas_Treatment  -   the experimental treatment type
	% Blab_orient  -   the orientation of the lab field
	% Xpts		 -   TRM points on the Arai plot
	% Ypts		 -   NRM points on the Arai plot
	% n			-   number of points for the best-fit line on the Arai plot
	% nmax		 -   the total number of NRM-TRM points on the Arai plot
	% Seg_Ends	 -   the indices for the start and end of the best-fit line [2x1]
	% b			-   the slope of the best-fit line
	% sigma_b	  -   the standard error of the slope
	% beta		 -   sigma_b/b
	% X_int		-   the intercept of the best-fit line of the TRM axis
	% Y_int		-   the intercept of the best-fit line of the NRM axis
	% x_prime	  -   the Arai plot TRM points projected onto the best-fit line
	% y_prime	  -   the Arai plot NRM points projected onto the best-fit line
	% Delta_x_prime -  the TRM length of the best-fit line
	% Delta_y_prime -  the NRM length of the best-fit line
	% VDS		  -   the vector difference sum of the NRM vector
	% f			-   the fraction of NRM used for the best-fit line
	% f_vds		-   the NRM fraction normalized by the VDS
	% FRAC		 -   the NRM fraction with full vector calculation (Shaar & Tauxe, 2013; G-Cubed)
	% gap		  -   the gap factor
	% GAP_MAX	  -   the maximum gap (Shaar & Tauxe, 2013; G-Cubed)
	% qual		 -   the quality factor
	% w			-   the weighting factor of Prevot et al. (1985; JGR)
	% R_corr	  -	Linear correlation coefficient (Pearson correlation)
	% R_det		-   Coefficient of determination of the SMA linear model fit
	% Line_Len	 -   the length of the best-fit line
	% f_vds		-   the fraction of vector difference sum NRM used for the best-fit line
	% k			-   the curvature of the Arai plot following Paterson (2012; JGR)
	% SSE		  -   the fit of the circle used to determine curvature (Paterson, 2012; JGR)
	% k_prime	  -   the curvature of the Arai plot from the best-fit segment Paterson (2012; JGR)
	% S_prime_qual -   the fitting quality factor of Carvallo & Dunlop (2001, ESPL)

	% MAD_anc	  -   the maximum angular deviation of the anchored PCA directional fit
	% MAD_free	 -   the maximum angular deviation of the free-floating PCA directional fit
	% alpha		-   the angle between the anchored and free-floating PCA directional fits
	% alpha_prime  -   the angle between the anchored PCA directional fit and the true NRM direction (assumed to be well known)
	% alpha_TRM	-   the angle between the applied field and the acquire TRM direction (determined as an anchored PCA fit to the TRM vector)
	% DANG		 -   the deviation angle (Tauxe & Staudigel, 2004; G-Cubed)
	% Theta		-   the angle between the applied field and the NRM direction (determined as a free-floating PCA fit to the TRM vector)
	% a95		  -   the alpha95 of the Fisher mean of the NRM direction of the best-fit segment
	% NRM_dev	  -   the intensity deviation of the free-floating principal component from the origin, normalized by Y_int (Tanaka & Kobayashi, 2003; EPS)
	% CRM_R		-   the potential CRM% as defined by Coe et al. (1984; JGR)

	% BY_Z	   -   the zigzag parameter of Ben-Yosef et al. (2008; JGR)
	% Z		  -   the zigzag parameter of Yu & Tauxe (2005; G-Cubed)
	% Z_star	 -   the zigzag parameter of Yu (2012; JGR)
	% IZZI_MD	-   the zigzag parameter of Shaar et al. (2011, EPSL)
	% MD_area	-   the unsigned Arai plot area normalized by the length of the best-fit line (following the triangle method of Shaar et al. (2011, EPSL))

	% n_pTRM	   -   the number of pTRM checks used to the maximum temperature of the best-fit segment
	% PCpoints		-   (N_alt x 3) array. First column is temperature the check is to (Ti), second is the temperature the check is from (Tj), third is total pTRM gained at each pTRM check (= pTRM_check_i,j in SPD)
	% check		   -   the maximum pTRM difference when normalized by pTRM acquired at each check step
	% dCK			 -   pTRM check normalized by the total TRM (i.e., X_int) (Leonhardt et al., 2004; G-Cubed)
	% DRAT			-   pTRM check normalized by the length of the best-fit line (Selkin & Tauxe, 2000; Phil Trans R Soc London)
	% maxDEV		  -   maximum pTRM difference normalized by the length of the TRM segment used for the best-fit slope (Blanco et al., 2012; PEPI)
	% CDRAT		   -   cumulative DRAT (Kissel & Laj, 2004; PEPI)
	% CDRAT_prime	 -   cumulative absolute DRAT
	% DRATS		   -   cumulative pTRM check normalized by the maximum pTRM of the best-fit line segment (REF - Tauxe...)
	% DRATS_prime	 -   cumulative absolute pTRM check normalized by the maximum pTRM of the best-fit line segment
	% mean_DRAT	   -   CDRAT divided by number of checks
	% mean_DRAT_prime -   the average DRAT (Herro-Bervera & Valet, 2009; EPSL)
	% mean_DEV		-   average pTRM difference normalized by the length of the TRM segment used for the best-fit slope (Blanco et al., 2012; PEPI)
	% mean_DEV_prime  -   average absolute pTRM difference normalized by the length of the TRM segment used for the best-fit slope
	% dpal			-   cumulative check of Leonhardt et al. (2004; G-Cubed)

	% n_tail	   -   the number of tail checks used to the maximum temperature of the best-fit segment
	% MDpoints   -   (N_MD x 2) array. First column is temperature, second is the remanence remaining after the demagnetization step for a pTRM tail check (= tail_check_i in SPD)
	% dTR		-   tail check normalized by the total NRM (i.e., Y_int) (Leonhardt et al., 2004; G-Cubed)
	% DRATtail   -   tail check normalized by the length of the best-fit line (Biggin et al., 2007; EPSL)
	% MDvds	  -   tail check normalized by the vector difference sum corrected NRM (REF - Tauxe...)
	% dt_star	-   pTRM tail after correction for angular dependence (Leonhardt et al., 2004; G-Cubed)

	% n_add	   - the number of additivity checks used to the maximum temperature of the best-fit segment
	% ADpoints   - (N_AC x 3) array. First 2 columns are lower and upper temperatures for remaining pTRM, third is the remanence remaining after a repeat demagnetization additivity check (= Mrem in SPD)
	% dAC	   - maximum additivity check normalized by the total TRM (i.e., X_int) (Leonhardt et al., 2004; G-Cubed)

	% SCAT			- SCAT parameters of Shaar & Tauxe (2013). N.B. the beta threshold is hard-wired to 0.1
	% com_slope_pval  - the probability of a common slope when comparing the NRM-TRM slope, with that defined by pTRM and pTRM tail checks (Warton et al., 2006; Biol. Rev.)
	% com_elev_pval   - the probability of a common elevation (intercept) when comparing the NRM-TRM slope, with that defined by pTRM and pTRM tail checks (Warton et al., 2006; Biol. Rev.)

	% Hanc		 -   the unit vector in the direction of Banc as calculated by Selkin et al. (2000; EPSL) for the correction of anisotropy
	% s_tensor	 -   the six unique elements of the anisotropy tensor
	% Anis_c   -   factor used to scale the TRM vectors to correct for the effects of anisotropy

	%	REMOVED (To be checked if needed)	 IMG_flag	 -   flag to determine if non-linear TRM correction returns a complex number (1 if true, 0 if false)


	%% Initial input variable checks
	"""



	# extract flag values
	A_flag = Flags[0]
	NLT_flag = Flags[1]
	NRflag = Flags[2]
	NRM_rot_flag = Flags[3]
	# Checks for inputs, implemenation still to be finalised
	if np.size(Mvec,axis=1)!=3:
		raise ValueError('GetPintParams:Input', 'Input magnetization data must be a [n x 3] magnetization vector')


	# Mvec, Temps, and Treatment should all have the same length
	if np.size(Mvec,axis=0) != len(Temps) or len(Temps) != len(Treatment) or np.size(Mvec, axis=0) != len(Treatment):
		raise ValueError('GetPintParams:Input; Input magnetization matrix, temperature vector, and Treatment vector must be the same length')

	#   start undefined  OR end undefined OR start @ neg temp OR end @ neg temp OR end comes before start
	if (start_pt < 0) or (end_pt < 0) or (end_pt <= start_pt):
		raise ValueError('GetPintParams:Input; Start and end points for best-fit analysis must be properly defined')


	if  Az == [] or Pl == []:
		NRM_rot_flag=0
		if	 NRM_rot_flag == 1:
			warning('GetPintParams:Input; No sample azimuth or plunge defined: no rotation applied')


	if ChRM != []:
		if ChRM.size == 2: # Assume to be Dec, Inc
			tmp_var = ChRM/norm(ChRM)
			ChRM = Utilities.dir2cart(tmp_var[1], tmp_var[2], Mag = 1)
		elif ChRM.size == 3:
			# a 3D unit vector, but normalize just in case it is not a unit vector
			ChRM=ChRM/norm(ChRM)
		else:
			warning('GetPintParams:Input; Unrecognised ChRM format. Statistics that require ChRM will be ignored')
			ChRM = np.arange(1.0,4.0,1)
			ChRM[:]=np.nan


	if not A_flag: # No anisotropy correction
		A_flag=0
		Anis_vec=[]

	if A_flag == 1 and Anis_vec == []:
		A_flag=0
		warning('GetPintParams:Input;  Anisotropy correction is desired, but no tensor is provided. No correction will be applied')

	if not NLT_flag: # No non-linear TRM correction
		NLT_flag=0
		NLT_vec=[]

	if NLT_flag == 1 and NLT_vec == []:
		NLT_flag=0
		warning('GetPintParams:Input;  Non-linear TRM correction is desired, but no linearity estimate is provided. No correction will be applied')

	if not beta_T:
		beta_T=0.1

	# % Set the experimental flag for separating the data
	# % Exp_Flag=0 for Coe, Aitken, and IZZI
	# % Exp_Flag>0 for Thellier
	Exp_Flag = np.count_nonzero(Treatment == 5)

	# %% Input data handling
	# % Separate the data into NRM, TRM, pTRM and tail measurements etc
	# % For the TRM, NRM, and tail check  matrices, the first column is the temperature
	# % For pTRM and additivity checks, the first column is the temperature the check is TO, the second is the temperature the check is FROM

	# % Flag for detecting an extra TRM step at teh end of an incomplete
	# % experiment
	Extra_TRM_Flag = 0

	if Exp_Flag == 0:
		# Coe/Aitken/IZZI Experiment
		# Find the NRM steps
		NRMvec = []
		NRM_mt_Flag = 1

		# solves differences between model and real data - for model data the
		# initial NRM has Axis=1 (generates the initial NRM), but for real data Axis=0
		if Treatment[0] !=0 :
			NRMvec = Mvec[0,:]
			NRMvec = np.insert(NRMvec, 0 ,0).reshape((1,4))
			NRM_mt_Flag = 0

		if NRM_mt_Flag == 0:
			NRMvec = vstack((NRMvec,np.concatenate((Temps[np.where(Treatment==0)], Mvec[np.where(Treatment==0)]),axis=1)))
		# : removed
		elif NRM_mt_Flag == 1:
			NRMvec = np.concatenate((Temps[np.where(Treatment==0)], Mvec[np.where(Treatment==0)]),axis=1)

		else:
			raise ValueError("Error; NRMvec not constructed")
		tail_vec = np.concatenate((Temps[np.where(Treatment==3)], Mvec[np.where(Treatment==3)]),axis=1)

		# % Calculate the tail differences
		# %	 MD_vec=tail_vec;
		# %	 MD_vec(:,2:end)=NaN;
		MD_scalar = np.empty((len(tail_vec),2))
		MD_scalar[:] = np.nan

		for n in range(len(tail_vec)):

			MD_scalar[n,0] = tail_vec[n,0]
			MD_scalar[n,1] = np.sqrt(np.sum( tail_vec[n:n+1, 1:]**2, axis=1)) - np.sqrt(np.sum((NRMvec[np.where(NRMvec[:,0]==tail_vec[n,0]), 1:])**2 ))

		TRMvec = np.array(([NRMvec[0,0],0,0,0])) # Set the first step to zeros
		TRMvec = TRMvec.reshape(1,4)



		pCheck = Utilities.create_array((1,3))	 # the full vector check
		SCAT_check = Utilities.create_array((1,3))	 # a matrix for the SCAT parameter
		ALT_vec = Utilities.create_array((1,5)) # the pTRM check difference
		ALT_scalar = Utilities.create_array((1,3)) # the pTRM check difference as a scalar
		check_pct = Utilities.create_array((1,3)) # the pTRM difference as a percentage of the pTRM at that step
		ADD_vec = Utilities.create_array((1,5)) # the matrix for additivity check data
		ADD_scalar =  Utilities.create_array((1,3))
		for n in range(1,len(Treatment)):

			if Treatment[n] == 1: #TRM step

				# If this sum is zero, then we have 1 more TRM step than NRM,
				# which can happen if an IZZI, Aitken, or Thellier experiment ended early
				if np.sum(np.isin(Temps[n],NRMvec[:,0])) != 0:

					TRMvec = vstack((TRMvec, np.concatenate((Temps[n], Mvec[n,:] -NRMvec[np.where(NRMvec[:,0]==Temps[n]),1:].reshape(3,)))))
				else:
					Extra_TRM_Flag = 1

			if Treatment[n] == 2: # pTRM check

				#In this experiment they are always performed after a
				# demagnetization experiment Axis==0 || ==3

				# Check we have a corresponding TRM step
				# If not skip
				if np.isin(Temps[n], TRMvec[:,0]) == False:

					continue

				if Treatment[n-1] == 0 or Treatment[n-1] == 3:

					# Since it is a repeat TRM we can search TRMvec for the correct
					# temperature
					pCheck_vec = np.array([Mvec[n,:] - Mvec[n-1,:]])
					pCheck =	 vstack((pCheck,pCheck_vec))
					SCAT_check = vstack((SCAT_check,(np.concatenate((Temps[n], Temps[n-1], (np.sum(pCheck_vec**2, axis = 1))**(0.5)))).reshape(1,3)))
					ALT_vec =	vstack((ALT_vec,(np.concatenate((Temps[n], Temps[n-1], (pCheck_vec-TRMvec[np.where(TRMvec[:,0] == Temps[n]), 1:]).reshape(3,)))).reshape(1,5)))
					ALT_scalar = vstack((ALT_scalar, (np.concatenate((Temps[n], Temps[n-1], (np.sqrt(np.sum(pCheck_vec**2, axis = 1))) -( np.sqrt(np.sum( TRMvec[np.where(TRMvec[:,0] == Temps[n]), 1:].reshape(1,3)**2, axis = 1))) )))	 ))
					check_pct =  vstack((check_pct, (np.concatenate((Temps[n], Temps[n-1], ((np.sum(pCheck_vec**2, axis = 1))**(0.5) - (np.sum( TRMvec[np.where(TRMvec[:,0]==Temps[n]), 1:].reshape(1,3)**2, axis = 1) )**(0.5)) / (np.sum( TRMvec[np.where(TRMvec[:,0]==Temps[n]),1:].reshape(1,3)**2, axis = 1))**(0.5) )  ))))


				elif Treatment[n-1] == 2 and Treatment[n-2] == 0:
					# Two back to back pTRM checks
					pCheck_vec = np.array([ Mvec[n,:] - Mvec[n-2,:]])
					pCheck =	 vstack((pCheck, pCheck_vec))
					SCAT_check = vstack((SCAT_check,(np.concatenate((Temps[n], Temps[n-2], (np.sum(pCheck_vec**2, axis = 1))**(0.5)) ))))
					ALT_vec =	vstack((ALT_vec, (np.concatenate((Temps[n], Temps[n-2], (pCheck_vec - TRMvec[np.where(TRMvec[:,0] == Temps[n]),1:]).reshape(3,)))).reshape(1,5)))
					ALT_scalar =  vstack((ALT_scalar, (np.concatenate((Temps[n], Temps[n-2], (np.sqrt(np.sum(pCheck_vec**2, axis = 1))) -( np.sqrt(np.sum( TRMvec[np.where(TRMvec[:,0] == Temps[n]), 1:].reshape(1,3)**2, axis = 1))) )))	 ))
					check_pct =  vstack((check_pct, (np.concatenate((Temps[n], Temps[n-2], ((np.sum(pCheck_vec**2, axis = 1))**(0.5) - (np.sum( TRMvec[np.where(TRMvec[:,0]==Temps[n]), 1:].reshape(1,3)**2, axis = 1) )**(0.5)) / (np.sum( TRMvec[np.where(TRMvec[:,0]==Temps[n]),1:].reshape(1,3)**2, axis = 1))**(0.5) )  ))))

				else:
					print('Temperature step '+ str(Temps[n]))
					raise ValueError('Unsupported experiment')

			if Treatment[n] == 4: # additivity check
				# Additivity check is a repeat demag step

				if Treatment[n-1] == 1: # Previous step was a TRM step

					ADD_vec =  vstack((ADD_vec, (np.concatenate((Temps[n], Temps[n-1], (Mvec[np.where((Temps.flatten()==Temps[n-1].flatten()) & (Treatment==1)), :] - Mvec[n,:]).reshape(3,)))))) # This observed Mrem
					ADD_scalar = vstack((ADD_scalar, (np.concatenate((Temps[n], Temps[n-1],  [(np.sum( Mvec[np.where((Temps.flatten()==Temps[n-1].flatten()) & (Treatment==1)), :]**2 ))**(0.5) - (np.sum(Mvec[n,:]**2))**(0.5)])))))

				if Treatment[n-1] == 2: # Previous step was a pTRM check step

					ADD_vec =  vstack((ADD_vec, (np.concatenate((Temps[n], Temps[n-2],  (Mvec[np.where((Temps.flatten()==Temps[n-2].flatten()) & (Treatment==1)), :] - Mvec[n,:]).reshape(3,)))))) # This observed Mrem
					ADD_scalar =  vstack((ADD_scalar, (np.concatenate((Temps[n], Temps[n-2], [(np.sum( Mvec[np.where((Temps.flatten()==Temps[n-2].flatten()) & (Treatment==1)), :]**2 ) )**(0.5) - (np.sum(Mvec[n,:]**2))**(0.5)])))))

				if Treatment[n-1] == 4:

					if Treatment[n-2] == 1:
						ADD_vec =  vstack((ADD_vec, (np.concatenate((Temps[n], Temps[n-2],   (Mvec[np.where((Temps.flatten()==Temps[n-2].flatten()) & (Treatment==1)), :] - Mvec[n,:]).reshape(3,)))))) # This observed Mrem
						ADD_scalar = vstack((ADD_scalar, (np.concatenate((Temps[n], Temps[n-2],  [(np.sum( Mvec[np.where((Temps.flatten()==Temps[n-2].flatten()) & (Treatment==1)), :]**2))**(0.5) - (np.sum(Mvec[n,:]**2))**(0.5)])))))
					else:
						warning("GetPintParams:Additivity; Ignoring additivity check following an additivity check: this sequence is not yet supported")

				else:
					raise ValueError(f"GetPintParams:Additivity; Unsupported additivity check sequence - Temperature step {Temps[n]}")

	else:
		# Thellier-Thellier Experiment
		Dirvec = np.concatenate((Temps[np.where(Treatment==1)], Mvec[np.where(Treatment==1)]),axis =1)
		Dirvec = np.delete(Dirvec, (0), axis = 0)  #   Remove the first row, which is the initial NRM

		Invvec=np.concatenate((Temps[np.where(Treatment==5)], Mvec[np.where(Treatment==5) ]),axis=1)

		NRMvec = Utilities.create_array((1,2))   # set up array

		NRMvec= np.concatenate((Dirvec[:,0:1],(Dirvec[:,1:]+Invvec[:,1:])/2),axis=1)
		NRMvec = vstack((np.concatenate((np.zeros((1,1)), Mvec[0:1,:]),axis=1),NRMvec))	#Set the first step to zeros

		TRMvec= np.concatenate((Dirvec[:,0:1], (Dirvec[:,1:] - Invvec[:,1:])/2),axis = 1)
		TRMvec = vstack((np.zeros((1,4)),TRMvec))   #Set the first step to zeros

		# Find the pTRM tail checks
		tail_vec=np.concatenate((Temps[np.where(Treatment==3)], Mvec[np.where(Treatment==3)]), axis = 1)
		# Calculate the tail differences

		MD_scalar=np.empty((len(tail_vec),2))
		ADD_vec= [] # the matrix for additivity check vectors

		for n in range(len(tail_vec)):

			MD_scalar[n,0] = tail_vec[n,0]
			MD_scalar[n,1] = (np.sum( tail_vec[n:n+1, 1:]**2, axis=1))**(0.5) - (np.sum( NRMvec[np.where(NRMvec[:,0]==tail_vec[n,0]), 1: ][0]**2, axis=1) )**(0.5)



		pCheck = Utilities.create_array((1,3))	# the full vector check
		SCAT_check = Utilities.create_array((1,3))	   # a matrix for the SCAT parameter
		ALT_vec = Utilities.create_array((1,5))	  # the pTRM check difference
		ALT_scalar = Utilities.create_array((1,3))	 # the pTRM check difference as a scalar
		check_pct = Utilities.create_array((1,3))   # the pTRM difference as a percentage of the pTRM at that step
		ADD_vec = Utilities.create_array((1,5)) # the matrix for additivity check data
		ADD_scalar =  Utilities.create_array((1,3))

		for n in range(1,len(Treatment)):

			if Treatment[n] == 2: # pTRM check
				# % In this experiment they are always performed after an inverse
				# % TRM acquisition Axis==5 or a pTRM tail check Axis==3
				# % Since it is a repeat TRM we can search TRMvec for the correct
				# % temp
				if Treatment[n-1] == 5:
					pCheck_vec = (Mvec[n,:] - Mvec[n-1,:])/2
				elif Treatment[n-1] == 3:
					pCheck_vec = Mvec[n,:] - Mvec[n-1,:]
				else:
					raise ValueError("GetPintParams:Thellier_pTRM; Should not be here!\nTreatments: "+str(Treamtent[n])+" and "+str(Treatment[n-1])+"\nTemperature: "+str(Temps[n])+" and "+ str(Temps[n-1]))

				ALT_vec = vstack((ALT_vec,	   (np.concatenate((Temps[n], Temps[n-1], pCheck_vec-(TRMvec[np.where(TRMvec[:,0] == Temps[n]),1:]).reshape(3,))))))
				ALT_scalar = vstack((ALT_scalar, (np.concatenate((Temps[n], Temps[n-1],  np.array(([(np.sum(pCheck_vec**2))**(0.5) - (np.sum( TRMvec[np.where(TRMvec[:,0] == Temps[n]), 1:]**2))**(0.5) ])))))))
				check_pct = vstack((check_pct,   (np.concatenate((Temps[n], Temps[n-1], np.array(([((np.sum(pCheck_vec**2))**(0.5) - (np.sum( TRMvec[np.where(TRMvec[:,0] == Temps[n]) ,1:]**2))**(0.5)) / (np.sum( TRMvec[np.where(TRMvec[:,0] == Temps[n]), 1:]**2))**(0.5)])))))))
				pCheck = vstack((pCheck, pCheck_vec))
				SCAT_check = vstack((SCAT_check, (np.concatenate((Temps[n], Temps[n-1], np.array(([(np.sum(pCheck_vec**2))**(0.5)]))))))) # a matrix for the SCAT parameter

			elif Treatment[n]==4: # additivity check
				warning('GetPintParams:Additivity; Additivity checks not yet supported for Thellier-Thellier routine  - if required contact GAP')

				# Skip over the direct and inverse steps, which are already taken care of

	# Remove nan rows created during constructions
	arrays = [pCheck,SCAT_check,ALT_vec,ALT_scalar,check_pct,ADD_vec,ADD_scalar]

	for i in range(len(arrays)):
		arrays[i] = np.delete(arrays[i], (0), axis=0)

	pCheck = arrays[0]
	SCAT_check = arrays[1]
	ALT_vec = arrays[2]
	ALT_scalar = arrays[3]
	check_pct = arrays[4]
	ADD_vec = arrays[5]
	ADD_scalar = arrays[6]

	# Get the experimental info
	Meas_Data = Mvec
	Meas_Treatment = Treatment
	Meas_Temp = Temps

	# Get the NRM points in the TRM vector and vice versa
	NRM_in_TRM = np.isin(NRMvec[:,0], TRMvec[:,0])
	# TRM_in_NRM = ismember(TRMvec(:,1), NRMvec(:,1));

	Xpts=(np.sum(TRMvec[:,1:]**2, axis = 1))**(0.5)
	Xpts = Xpts.reshape((len(Xpts),1))
	Ypts=(np.sum((NRMvec[np.where(NRM_in_TRM==True),1:].reshape((np.count_nonzero(NRM_in_TRM == 1),3)))**2, axis = 1))**(0.5)
	Ypts = Ypts.reshape((len(Ypts),1))
	# print(f"NRM {NRMvec}")
	# print(f"TRM {TRMvec}")
	# print(f"NinT {NRM_in_TRM}")
	#
	# print(f"Xpts {Xpts}")
	# print(f"Ypts {Ypts}")
	nmax=len(Xpts)
	Temp_steps=TRMvec[:,0]

	# Output the various check points
	if pCheck.size == 0:
		PCpoints= []
	else:
		PCpoints = np.concatenate((ALT_scalar[:,0:1], ALT_scalar[:,1:2], ((np.sum(pCheck**2, axis = 1)).reshape((len(pCheck),1))**(0.5))), axis =1 )

	if tail_vec.size == 0:
		MDpoints=[]
	else:
		MDpoints= np.concatenate((tail_vec[:,0:1], ((np.sum(tail_vec[:,1:]**2, axis = 1)).reshape((len(tail_vec[:,1:]),1))**(0.5))),axis=1)


	if ADD_vec.size == 0:
		ADpoints=[]
	else:
		ADpoints = np.concatenate((ADD_vec[:,0:2], (np.sum(ADD_vec[:,2:]**2, axis = 1))**(0.5)),axis = 1)

	# Define the indices of the best-fit segment
	seg_min = start_pt
	seg_max = end_pt
	seg = np.arange(seg_min, seg_max+1,1)

	## Anisotropy correction
	Anis_c = np.nan
	Hanc = np.empty((1,3))
	Hanc[:] = np.nan

	# If anisotropy on
	if A_flag == 1:

		# create oreintation matrix
		B_axes = np.vstack((np.eye(3), -np.eye(3)))
		B_axes = np.append(B_axes ,B_axes[0:1,:], axis = 0)
		# find Anisotropy tensor
		s_tensor, eigval = Utilities.ATRMtensor(Anis_vec[0:-1,:], B_axes[0:-1,:])[0:2]
		dTRM_Anis = 100 *(np.linalg.norm(Anis_vec[0,:] - Anis_vec[-1, :]))/np.linalg.norm(Anis_vec[0,:])
		# Follows the method of Veitch et al(1984; Arch. Sci., 37, 359-373) as recommended by
		# Paterson (2013; Geophys. J. Int.; 193, 684-710, doi: 10.1093/gji/ggt033)

		# Find the anisotropy corrected NRM direction
		# Get the NRM Decs and Incs then PCA the mean unit vector  (mX, mY, mZ)
		Ds, Is, Int = Utilities.cart2dir(NRMvec[seg,1:2], NRMvec[seg,2:3], NRMvec[seg,3:4])

		mD, mI, MAD = Utilities.PmagPCA(Ds, Is, Int, type = 'free')
		mX, mY, mZ = Utilities.dir2cart(mD, mI, Mag = 1) #Mhat_ChRM

		Mhat_ChRM = np.array((mX, mY, mZ))


		A = Utilities.Anis_mat(s_tensor) # The anisotropy tensor


		Hanc = np.transpose(np.linalg.solve(A,np.transpose(Mhat_ChRM)))

		Hanc = Hanc/np.linalg.norm(Hanc)   # Unit vector in dircetion of the ancient field
		Manc = np.transpose(A*np.transpose(Hanc))
		Mlab = np.transpose(A*np.transpose(Blab_orient))
		Anis_c = np.linalg.norm(Mlab)/np.linalg.norm(Manc)


	# if anisotpy off
	else:
		s_tensor = []
		dTRM_Anis = np.nan
	## Arai stats

	n = len(seg)
	Seg_Ends = np.array((seg_min, seg_max))
	# Max and min temperatures
	Tmin = TRMvec[seg_min, 0]
	Tmax = TRMvec[seg_max, 0]
	# segment points
	X_seg = Xpts[seg]
	Y_seg = Ypts[seg]
	xbar = np.mean(X_seg)
	ybar = np.mean(Y_seg)
	U = detrend(X_seg, type = "constant", axis = 0) # (Xi-Xbar)
	V = detrend(Y_seg, type = "constant", axis = 0) # (Yi-Ybar)

	U_full = detrend(Xpts, type = "constant", axis = 0) # (Xi-Xbar)
	V_full = detrend(Ypts, type = "constant", axis = 0) # (Yi-Ybar)

	# if NLT on
	if NLT_flag == 1:

		# If last row is Blab
		if Blab == Bnlt[-1]:
			dtrm_vec = NLT_vec[-1,:]
			NLT_vec = NLT_vec[0:-1,:]
			Bnlt = Bnlt[0:-1]
		# NLT Data
		NLT_data = np.transpose(np.sqrt( np.sum(NLT_vec**2, axis = 1)))
		NLT_meas = np.transpose(NLT_data)

		nlt_corr = Utilities.PearsonCorr2(Bnlt, NLT_data)
		if nlt_corr > 0.999:
			NLT_a_est = 1e4
			NLT_b_est = 1/1e4
		else:
			#initial guess
			guess1 = NLT_data[-1]*0.9
			guess2 = 1/guess1

			NLT_func = lambda x,c0,c1: c0*np.tanh(c1*x)
			out, outcov = sp.optimize.curve_fit(NLT_func, Bnlt, NLT_data, p0 = [guess1, guess2] )
			NLT_a_est = out[0]
			NLT_b_est = out[1]


		dTRM_NLT = 100 *(np.linalg.norm(dtrm_vec - TRMvec[-1, 1:]))/np.linalg.norm(TRMvec[-1,1:])
	else:
		dTRM_NLT = np.nan
		NLT_meas = np.nan
		NLT_a_est = np.nan
		NLT_b_est = np.nan
		NLT_data = np.nan

	# Get the paleointensity estimate
	b = np.sign(np.sum(U*V))*np.std(Y_seg, ddof = 1)/np.std(X_seg, ddof = 1)
	b_full = np.sign(np.sum(U_full*V_full))*np.std(Ypts, ddof = 1)/np.std(Xpts, ddof = 1)
	if A_flag == 1 and NLT_flag == 0:
		Banc = Blab * np.abs( Anis_c * b )
		#Banc = Blab * np.abs( b )

	elif A_flag == 0 and NLT_flag == 1:
		Banc = np.real( np.arctanh( np.abs(b) * np.tanh(NLT_b_est*Blab) ) / NLT_b_est )

	elif A_flag == 1 and NLT_flag == 1:
		Banc = np.real( np.arctanh( np.abs(Anis_c*b) * np.tanh(NLT_b_est*Blab) ) / NLT_b_est )

	else:
		Banc = np.abs(b)*Blab


	sigma_b = np.sqrt( (2*np.sum(V**2)-2*(b)*np.sum(U*V)) / ( (n-2)*np.sum(U**2)) )
	beta = np.abs(sigma_b/b)
	sigma_B = Blab*sigma_b


	Y_int = np.mean(Y_seg) - b*np.mean(X_seg)
	X_int = -Y_int/b
	Y_int_full = np.mean(Ypts) - b_full*np.mean(Xpts)
	X_int_full = -Y_int_full/b_full
	# Project the data onto the best-fit line
	Rev_x = (Ypts - Y_int) / b # The points reflected about the bes-fit line
	Rev_y = b * Xpts + Y_int
	x_prime = (Xpts + Rev_x)/2   # Average the both sets to get the projected points
	y_prime = (Ypts + Rev_y)/2

	# Get the TRM, NRM, and line lengths
	Delta_x_prime = np.abs( np.amax(x_prime[seg])-np.amin(x_prime[seg]) )
	Delta_y_prime = np.abs( np.amax(y_prime[seg])-np.amin(y_prime[seg]) )
	Line_Len = np.sqrt(Delta_x_prime**2 + Delta_y_prime**2)

	# Get the VDS and fraction related stuff
	VDS = np.sum(np.sqrt(np.sum((np.diff(NRMvec[:,1:], axis = 0)**2),axis = 1))) + np.sqrt(np.sum(NRMvec[-1,1:]**2))
	sumdy = np.sum( np.diff( y_prime[seg], axis = 0 )**2)

	f = np.abs(Delta_y_prime/Y_int)
	f_vds = np.abs(Delta_y_prime/VDS)
	FRAC = np.sum( np.sqrt( np.sum( (np.diff(NRMvec[seg_min:seg_max+1,1:], axis = 0)**2), axis=1 ) ) ) /VDS
	gap = 1 - (sumdy/(Delta_y_prime**2))
	GAP_MAX = np.amax( np.sqrt( np.sum( np.diff(NRMvec[seg_min:seg_max+1,1:], axis = 0)**2, axis=1 ) ) ) / np.sum(np.sqrt(np.sum((np.diff(NRMvec[seg_min:seg_max+1,1:], axis = 0)**2),axis=1)))
	qual = f*gap/beta
	w = qual/np.sqrt(n-2)
	R_corr = Utilities.PearsonCorr2(X_seg, Y_seg)
	R_det = 1 - (np.sum((Y_seg - y_prime[seg])**2) / np.sum((Y_seg - ybar)**2) )

	# Carvallo's S' quality
	S1 = (np.std(X_seg, ddof = 1)*np.std(Y_seg, ddof = 1)) / (b*np.std(Y_seg, ddof = 1) + np.std(X_seg, ddof = 1) )
	S2 = np.sum( (Y_seg - b * X_seg - Y_int)**2 )
	S_qual= S1 * S2
	S_prime_qual = S_qual/(n-2)

	# Curvature
	kparams = Utilities.AraiCurvature(Xpts, Ypts)
	k = kparams[0,0]
	SSE = kparams[3,0]
	RMS = kparams[4,0]
	# curvature using only the best-fit segment
	kparams = Utilities.AraiCurvature(Xpts[seg], Ypts[seg])
	k_prime = kparams[0,0]
	SSEprime = kparams[3,0]
	RMSprime = kparams[4,0]

	## Directional stats
	Decs, Incs, Ints = Utilities.cart2dir(NRMvec[seg,1:2], NRMvec[seg,2:3], NRMvec[seg,3:4])
	TDecs, TIncs, TInts = Utilities.cart2dir(TRMvec[seg,1:2], TRMvec[seg,2:3], TRMvec[seg,3:4])

	# Rotate the NRM directions - Only used for alpha_prime and CRM_R
	if NRM_rot_flag == 1:
		Rot_Decs, Rot_Incs = Utilities.dirot(Decs, Incs, Az, Pl)
	else:
		Rot_Decs = Decs
		Rot_Incs = Incs
	if np.sum(np.isnan(Incs)) + np.sum(np.isnan(Decs)) > 0: #PmagPCA can't handle NaN or inf, so skip - only affects models with no noise
		MAD_anc = 0
		MAD_free = 0
		alpha = 0
		alpha_prime = 0
		DANG = 0
		NRM_dev = 0
		Theta = np.nan
		Dec_A = np.nan
		Inc_A = np.nan
		Dec_F = np.nan
		Inc_F = np.nan
	else:

		Dec_A, Inc_A, MAD_anc = Utilities.PmagPCA(Decs, Incs, Ints, type = 'anc')
		Dec_F, Inc_F, MAD_free = Utilities.PmagPCA(Decs, Incs, Ints, type = 'free')

		alpha = Utilities.calc_angle(np.array(([[Dec_A, Inc_A]])), np.array(([[Dec_F, Inc_F]])))

		#check for any nan values
		if ChRM == [] or np.isnan(np.sum(ChRM)) == True:
			alpha_prime = np.nan
			#		 Rot_Dec_A, Rot_Inc_A, MAD_anc
		else:
			# determine the PCA fit on the rotated directions
			Rot_Dec_A, Rot_Inc_A = Utilities.PmagPCA(Rot_Decs, Rot_Incs, Ints, type = 'anc')[0:2]
			NRM_dec, NRM_inc = Utilities.cart2dir(ChRM[0,0], ChRM[0,1], ChRM[0,2])[0:2]
			alpha_prime = Utilities.calc_angle(np.array(([[Rot_Dec_A, Rot_Inc_A]])), np.array(([[NRM_dec, NRM_inc]])))

		# Calculate DANG
		dirfit = np.empty((1,3))
		dirfit[:]=np.nan
		dirfit[0,0], dirfit[0,1], dirfit[0,2] = Utilities.dir2cart(Dec_F, Inc_F, Mag = 1)
		Centre = np.mean(NRMvec[seg, 1:],axis=0) # Centre of mass
		DANG = Utilities.rad2deg( np.arctan2(np.linalg.norm(np.cross(dirfit, Centre)), np.dot(dirfit, Centre)) )
		#	 DANG[np.where(DANG > 90)] = DANG - 90 # Take the smaller angle

		NRM_dev = (np.linalg.norm(Centre)*np.sin(Utilities.deg2rad(DANG))) / np.abs(Y_int) * 100

		# Angle between measured NRM and Blab
		NRMhat = np.empty((3,))
		NRMhat[:] = np.nan
		NRMhat[0], NRMhat[1], NRMhat[2] = Utilities.dir2cart(Dec_F, Inc_F)
		Theta = Utilities.rad2deg( np.arctan2(np.linalg.norm(np.cross(NRMhat, Blab_orient)), np.dot(NRMhat, Blab_orient[0])) )


	# Get the Fisher Mean and stats [Mdec, Minc, k, a95, R]

	a95 = Utilities.FisherMeanDir(Decs, Incs)[3]


	# Do some directional stats on the TRM data

	#PmagPCA can't handle NaN or inf, so remove - THIS IS FOR THE STOCHASTIC MODELS ONLY - only affects models with no noise
	# Add arrays together to give all nan positions in 1 array

	badvec_nan = TIncs +TDecs +TIncs +TDecs
	#find array without nans
	Good_data = np.where(~np.isnan(badvec_nan))
	#get rid of nans in arrays
	TIncs = TIncs[Good_data].reshape((len(Good_data[0]),1))
	TDecs = TDecs[Good_data].reshape((len(Good_data[0]),1))
	TInts = TInts[Good_data].reshape((len(Good_data[0]),1))

	# repeat for infs (repeat required due to change in index)
	badvec_inf = TIncs +TDecs +TIncs +TDecs

	Good_data = np.where(~np.isinf(badvec_inf))
	TIncs = TIncs[Good_data].reshape((len(Good_data[0]),1))
	TDecs = TDecs[Good_data].reshape((len(Good_data[0]),1))
	TInts = TInts[Good_data].reshape((len(Good_data[0]),1))

	if len(TIncs) < 3: #PmagPCA can't handle NaN or inf, so skip - should only affect models with no or very low noise
		alpha_TRM = np.nan
	else:
		Dec_TA, Inc_TA = Utilities.PmagPCA(np.flipud(TDecs), np.flipud(TIncs), np.flipud(TInts), type = 'anc')[0:2] # flip the data up/down (reverses order) so it behaves like demag data
		TRM_dec, TRM_inc = Utilities.cart2dir(Blab_orient[0,0], Blab_orient[0,1], Blab_orient[0,2])[0:2]

		alpha_TRM = Utilities.calc_angle(np.array(([[Dec_TA, Inc_TA]])), np.array(([[TRM_dec, TRM_inc]])))

	# AnisotroParams.y_prime check

	#seg_max - 1 so index not out of bounds
	# Blab_orient[0] in dot to give usable shape in dot product
	gamma = Utilities.rad2deg( np.arctan2(np.linalg.norm(np.cross(TRMvec[seg_max,1:], Blab_orient)), np.dot(TRMvec[seg_max,1:], Blab_orient[0])) )

	# Coe et al. (1984) CRM parameter
	if ChRM == [] or np.isnan(np.sum(ChRM)) == True:
		# Need the definition of ChRM
		CRM_R = np.nan
	else:

		if NRM_rot_flag == 1: # We need to also rotate the Blab vector into geographic coords
			tmp_D, tmp_I = Utilities.cart2dir(Blab_orient[0,0], Blab_orient[0,1], Blab_orient[0,2])[0:2]
			tmp_D, tmp_I = Utilities.dirot(tmp_D, tmp_I, Az, Pl)
			tmp_O = np.empty((3,))
			tmp_O[:] = np.nan
			tmp_O[0], tmp_O[1], tmp_O[2] = Utilities.dir2cart(tmp_D, tmp_I, Mag = 1)
			phi2=( np.arctan2(np.linalg.norm(np.cross(ChRM, tmp_O)), np.dot(ChRM[0], tmp_O)) )
		else:
			phi2 = ( np.arctan2(np.linalg.norm(np.cross(ChRM, Blab_orient)), np.dot(ChRM[0], Blab_orient[0])) )

		fit_vec = np.empty((len(Rot_Decs),3))
		fit_vec[:,0:1], fit_vec[:,1:2], fit_vec[:,2:3] = Utilities.dir2cart(Rot_Decs, Rot_Incs, Mag = 1) # Even if we don't rotate Rot_Decs/Incs contains the unrotated directions
		CRM = np.empty((fit_vec.shape[0], 1))
		CRM[:] = np.nan  # create an empty vector

		#for j in range(np.shape(fit_vec)[1]): # chnaged to allow for 2 step experiment to work, need to check if this is a bug/mistake (7/8/23)
		for j in range(np.shape(fit_vec)[0]):
			phi1 = ( np.arctan2(np.linalg.norm(np.cross(fit_vec[j,:], ChRM)), np.dot(fit_vec[j,:], ChRM[0])) )
			CRM[j,0] = Ypts[j]*np.sin(phi1)/np.sin((phi2))

		CRM_R=100*max(CRM)/(Delta_x_prime)


	## Zig-Zag
	################################
	# Ben-Yosef (2008; JGR) method #
	################################
	#r
	# Based on PmagParams.y_prime implementation
	#

	# Preallocate variable
	Frat=0
	Trat=0

	# Directions
	# Get the Fisher stats of the alternating steps
	filter = np.array([0,1,4])
	R = Utilities.FisherMeanDir(Decs[0:], Incs[0:])[4] # For all the steps
	D1, I1, R1 = np.array(Utilities.FisherMeanDir(Decs[0::2], Incs[0::2]))[filter] # For odd steps
	D2, I2, R2 = np.array(Utilities.FisherMeanDir(Decs[1::2], Incs[1::2]))[filter] # For even steps


	BY_Z= np.nan
	if len(Decs[0::2]) > 2 and len(Decs[1::2]) > 2:
		BY_Z = 0

		if Utilities.calc_angle(np.array([[D1, I1]]), np.array([[D2, I2]]))  > 3: # check that the angle between the mean directions is greater that 3 degrees
			F = (n-2) * (R1 + R2 - R) / ( n - R1 - R2 )
			fdistribution = sp.stats.f(2, 2*(n-2))
			Frat = F/fdistribution.ppf(1-0.05)

			if Frat > 1:
				BY_Z=Frat

	# Slopes
	dy = np.diff(Y_seg, axis = 0)
	dx = np.diff(X_seg, axis = 0)
	b_izzi = dy/dx
	b1 = b_izzi[0::2]
	b2 = b_izzi[1::2]

	# Suppress noise ala Tauxe
	r1 = np.sqrt( dy[0::2]**2 + dx[0::2]**2 )
	r2 = np.sqrt( dy[1::2]**2 + dx[1::2]**2 )

	b1 = np.delete(b1, np.where(r1 <= 0.1*VDS))
	b2 = np.delete(b2, np.where(r2 <= 0.1*VDS))

	if len(b1) > 2 and len(b2) > 2:

		if np.abs(np.arctan(np.mean(b1))-np.arctan(np.mean(b2))) > Utilities.deg2rad(3): # If the angle between the mean Params.slopes is greater than 3 degrees
			tstat = sp.stats.ttest_ind(b1, b2)[0]
			Trat = tstat/sp.stats.t.ppf(1-0.05, n-2)
			if Trat > 1 and Trat > Frat:
				BY_Z = Trat



	#########################
	# Yu (2012; JGR) method #
	#########################
	Z = get_Z(Xpts, Ypts, b, seg_min, seg_max)
	Z_star = get_Z_star(Xpts, Ypts, Y_int, b, seg_min, seg_max)


	#############################
	# Shaar et al. (2011, EPSL) #
	#############################
	# print("START")
	# print("Treatment")
	# print(Treatment)
	Tri_area = np.empty((len(Xpts)-2,1))
	Tri_area[:] = np.nan # The triangle areas
	Zsign = np.empty((len(Xpts)-2,1))
	Zsign[:] = np.nan # The sign of the areas
	ZI_steps = np.zeros((len(Xpts)-2,1)) # The ZI step for the ZI line length

	# create a vector of treatments and remove checks
	Zig_Treatment = Treatment[0:]  # Ignore the first point as does Shaar
	# print("Zig_Treatment")
	# print(Zig_Treatment)
	Zig_Treatment = np.delete(Zig_Treatment, np.where(Zig_Treatment==2)) # remove all checks
	Zig_Treatment = np.delete(Zig_Treatment, np.where(Zig_Treatment==3))
	Zig_Treatment = np.delete(Zig_Treatment, np.where(Zig_Treatment==4))
	Zig_Treatment = np.insert(Zig_Treatment, 0, 1)
	# print("Clean")
	# print(Zig_Treatment)
	if np.count_nonzero(NRM_in_TRM==0) !=0 and np.mod(Zig_Treatment.size, 2) !=0:
		Zig_Treatment = np.delete(Zig_Treatment, -1 )

	# Remove the extra TRM
	if Extra_TRM_Flag == 1:
		Zig_Treatment = np.delete(Zig_Treatment,-1)
	# Zig_Treatment should have an even number of elements, i.e., it must have a
	# demag and a remag
	if np.mod(Zig_Treatment.size, 2) !=0:
		raise ValueError('GetPintParams:ZigZag; Zigzag measurement must be even')
	Zig_Treatment = np.reshape(Zig_Treatment, (Zig_Treatment.size,1))
	Zig_Treatment = np.reshape(Zig_Treatment, (int(Zig_Treatment.size/2),2))

	IZZI_flag = 0
	if np.count_nonzero(Zig_Treatment[:,0]==0) > 0  and np.count_nonzero(Zig_Treatment[:,1]==0) > 0:
		# IZZI experiment
		IZZI_flag=1
	# added to allow indexing in next for loop to work correctly

	if IZZI_flag == 1:
		ZI_steps = np.zeros((len(Xpts),1))
	# Shaar uses the normalized Arai points and excludes the NRM point
	Xn = Xpts[1:]/Ypts[0]
	Yn = Ypts[1:]/Ypts[0]
	# Xn = Xpts[1:]
	# Yn = Ypts[1:]
	def line(x, A, B): # this is your 'straight line' y=f(x)
		return A*x + B

	for j in range(len(Xn)-2):

		# Determine the best-fit line of the endpoints (of the 3 used for the triangle)and use the
		# intercept of this best-fit line (EP_fit(0) and the intercept of the line with the same slope,
		# but through the middle point (Mid_int) to determine which points are higher


		EP_fit = np.polynomial.polynomial.polyfit(Xn[np.ix_([j,j+2],[0])].reshape(2,), Yn[np.ix_([j,j+2],[0])].reshape(2,), 1)

		a1 = EP_fit[0]
		a2 = Yn[j+1] - EP_fit[1]*Xn[j+1]  # Intercept when the line is passed through the midpoint

		if IZZI_flag == 1:

			# determine what step midpoint (j+1) is
			if Zig_Treatment[j+1,0] == 1 and Zig_Treatment[j+1,1] == 0:
				# Mid point is IZ step

				# Assign surrounding points as ZI steps
				ZI_steps[j] = 1
				ZI_steps[j+2] = 1

				if a1 < a2: # Midpoint is above
					Zsign[j] = -1
				else: # midpoint is above
					Zsign[j] = 1


			elif Zig_Treatment[j+1,0] == 0 and Zig_Treatment[j+1,1] == 1:
				# Midpoint is ZI step

				# Assign midpoint as a ZI step
				ZI_steps[j+1] = 1

				if a1 < a2: # Midpoint is above
					Zsign[j] = 1

				else: # midpoint is above
					Zsign[j] = -1


			#else:
				# TODO - Fix for Ben-Yosef specimen BY09_s10272a1.tdt
				# error('GetPintParams:ZigZag', 'Unexpected IZZI sequence');


		else:
			ZI_steps[j] = 1
			Zsign[j] = 1

		Tri_area[j] = Utilities.Polyarea(Xn[j:j+3], Yn[j:j+3], n = 3)

	MD_start = seg_min
	MD_end = seg_max - 1

	# print(ZI_steps)
	# print(Zig_Treatment)
	ZI_steps_Copy = ZI_steps.copy()
	ZI_steps_Copy[seg_min:seg_max+1] = 0
	ZI_steps[ZI_steps_Copy != 0] = 0
	ZI_inds = np.nonzero(ZI_steps) # find the non-zero elements

	ZI_len = np.sum( np.sqrt(np.diff(Xn[ZI_inds], axis = 0)**2 + np.diff(Yn[ZI_inds], axis = 0)**2))


	MD_area = np.sum(Tri_area)/Line_Len
	IZZI_MD_old = np.sum(Zsign[MD_start:MD_end]*Tri_area[MD_start:MD_end])/ZI_len

	# return to previous value before IZZI_MD
	Xn = Xpts[0:]/Ypts[0]
	Yn = Ypts[0:]/Ypts[0]

	if IZZI_flag==1:
		steps_raw = Treatment[np.where(np.isin(Treatment,[0,1]))]
		steps_raw = np.insert(steps_raw, 0, 0)
		steps_raw = steps_raw.reshape((int(len(steps_raw)/2),2))
		steps = ["nan"]*len(steps_raw)
		for j in range(len(steps_raw)):
			i = steps_raw[j]
			if i[0] == 0 and i[1]==1:
				steps[j] = "ZI"
			elif i[0]==1 and i[1]==0:
				steps[j]="IZ"

		steps[0] = "ZI"

		IZZI_MD = zig_params.get_IZZI_MD(Xpts,Ypts,steps,seg_min,seg_max)
	else:
		IZZI_MD = np.nan
		steps = np.nan
	## SCAT - uses the user input

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

	Check_points =  Utilities.create_array((1,1))# the x-, y-coords of all the checks within the SCAT range
	Check_points =  np.delete(Check_points, (0), axis=0)

	# find the pTRM checks
	if SCAT_check.size != 0:
		tmp_x = SCAT_check[np.where((SCAT_check[:,0:1] <= Tmax) & (SCAT_check[:,0:1] >= Tmin) & (SCAT_check[:,1:2] <= Tmax))[0], 2]  # Check performed within range, and not from above range
		tmp_T = SCAT_check[np.where((SCAT_check[:,0:1] <= Tmax) & (SCAT_check[:,0:1] >= Tmin) & (SCAT_check[:,1:2] <= Tmax))[0], 0]
		tmp_x = tmp_x.reshape(tmp_x.size,1)
		tmp_T = tmp_T.reshape(tmp_T.size,1)

		tmp_inds = np.empty((tmp_T.shape[0],))
		tmp_inds[:] = np.nan

		for j in range(len(tmp_T)):
			tmp_inds[j] = (np.nonzero(NRMvec[:,0:1] == tmp_T[j])[0])
		tmp_inds = tmp_inds.astype(int)		# set to int for indexing later
	else:
		tmp_x = np.nan
		tmp_inds = np.nan

	if np.isnan(np.sum(tmp_x)) != True and np.isnan(np.sum(tmp_inds)) != True:

		tmp_y = np.sqrt(np.sum( NRMvec[tmp_inds, 1:]**2, axis = 1))
		tmp_y = tmp_y.reshape(tmp_y.size,1)
		Check_points = np.concatenate((tmp_x, tmp_y),axis=1)

	# find the tail checks
	if tail_vec.size != 0:
		tmp_y = np.sqrt(np.sum(tail_vec[np.where((tail_vec[:,0:1] <= Tmax) & (tail_vec[:,0:1] >= Tmin))[0], 1:]**2, axis = 1))
		tmp_T = tail_vec[np.where((tail_vec[:,0:1] <= Tmax) & (tail_vec[:,0:1] >= Tmin))[0], 0]
		tmp_y = tmp_y.reshape(tmp_y.size,1)
		tmp_T = tmp_T.reshape(tmp_T.size,1)

		tmp_inds = np.empty((tmp_T.shape[0],))
		tmp_inds[:] = np.nan
		for j in range(len(tmp_T)):
			tmp_inds[j] = (np.nonzero(TRMvec[:,0:1] == tmp_T[j]))[0]
		tmp_inds = tmp_inds.astype(int)			# set to int for indexing later
	else:
		tmp_y = np.nan
		tmp_inds = np.nan

	if np.isnan(np.sum(tmp_y)) != True and np.isnan(np.sum(tmp_inds)) != True:
		tmp_x = np.sqrt(np.sum( TRMvec[tmp_inds, 1:]**2, axis = 1))
		tmp_x = tmp_x.reshape(tmp_x.size,1)
		if Check_points.size == 0:
			Check_points = np.concatenate((tmp_x, tmp_y),axis=1)
		else:
			Check_points = vstack((Check_points, np.concatenate((tmp_x, tmp_y), axis = 1)))

	# Create an array with the points to test
	if Check_points.size == 0:
		SCAT_points = (np.concatenate((X_seg, Y_seg),axis = 1)) # Add the TRM-NRM Arai plot points

	else:
		SCAT_points = vstack((np.concatenate((X_seg, Y_seg),axis = 1), Check_points)) # Add the TRM-NRM Arai plot points

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

		Check_points =  Utilities.create_array((1,1))# the x-, y-coords of all the checks within the SCAT range
		Check_points =  np.delete(Check_points, (0), axis=0)  # the x-, y-coords of all the checks within the SCAT range

		# find the pTRM checks
		if SCAT_check.size != 0:
			tmp_x = SCAT_check[np.where((SCAT_check[:,0:1] <= Tmax) & (SCAT_check[:,0:1] >= Tmin) & (SCAT_check[:,1:2] <= Tmax))[0], 2] # Check performed within range, and not from above range
			tmp_T = SCAT_check[np.where((SCAT_check[:,0:1] <= Tmax) & (SCAT_check[:,0:1] >= Tmin) & (SCAT_check[:,1:2] <= Tmax))[0], 0]
			tmp_x = tmp_x.reshape(tmp_x.size,1)
			tmp_T = tmp_T.reshape(tmp_T.size,1)

			tmp_inds = np.empty((tmp_T.shape[0],))
			tmp_inds[:] = np.nan

			for j in range(len(tmp_T)):
				tmp_inds[j] = (np.nonzero(NRMvec[:,0:1] == tmp_T[j])[0])
			tmp_inds = tmp_inds.astype(int)		# set to int for indexing later
		else:
			tmp_x = np.nan
			tmp_inds = np.nan


		if np.isnan(np.sum(tmp_x)) != True and np.isnan(np.sum(tmp_inds)) != True:
			tmp_y = np.sqrt(np.sum( NRMvec[tmp_inds, 1:]**2, axis= 1))
			tmp_y = tmp_y.reshape(tmp_y.size,1)
			if Check_points.size == 0:
				Check_points = np.concatenate((tmp_x, tmp_y),axis=1)
			else:
				Check_points = vstack((Check_points, np.concatenate((tmp_x, tmp_y), axis = 1)))
		# find the tail checks
		if tail_vec.size != 0:
			tmp_y = np.sqrt(np.sum(tail_vec[np.where((tail_vec[:,0:1] <= Tmax) & (tail_vec[:,0:1] >= Tmin))[0], 1:]**2, axis =1))
			tmp_T = tail_vec[np.where((tail_vec[:,0:1] <= Tmax) & (tail_vec[:,0:1] >= Tmin))[0], 0]
			tmp_y = tmp_y.reshape(tmp_y.size,1)
			tmp_T = tmp_T.reshape(tmp_T.size,1)

			tmp_inds = np.empty((tmp_T.shape[0],))
			tmp_inds[:] = np.nan
			for j in range(len(tmp_T)):
				tmp_inds[j] = (np.nonzero(TRMvec[:,0:1] == tmp_T[j]))[0]
			tmp_inds = tmp_inds.astype(int)			# set to int for indexing later

		else:
			tmp_y = np.nan
			tmp_inds = np.nan

		if np.isnan(np.sum(tmp_y)) != True and np.isnan(np.sum(tmp_inds)) != True:
			tmp_x = np.sqrt(np.sum( TRMvec[ tmp_inds, 1:]**2, axis = 1))
			tmp_x = tmp_x.reshape(tmp_x.size,1)
			if Check_points.size == 0:
				Check_points = np.concatenate((tmp_x, tmp_y),axis=1)
			else:
				Check_points = vstack((Check_points, np.concatenate((tmp_x, tmp_y), axis = 1)))

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

	## Common slope and common elevation tests
	# Based on the algorithms of Warton et al. (2006; Biol. Rev.)
	Check_points = Check_points
	N_check_points = Check_points.shape[0]

	check_slope = np.nan
	check_int = np.nan
	com_slope_pval = np.nan
	com_elev_pval = np.nan
	if Check_points.size != 0: # do the test
		# descriptive parameters
		cU = detrend(Check_points[:,0] , type = "constant", axis = 0)
		cV = detrend(Check_points[:,1] , type = "constant", axis = 0)
		barX = np.array([xbar, np.mean(Check_points[:,0:1])])
		barY = np.array([ybar, np.mean(Check_points[:,1:2])])
		varX = np.array([np.var(X_seg, ddof=1), np.var(Check_points[:,0], ddof=1)])
		varY = np.array([np.var(Y_seg, ddof=1), np.var(Check_points[:,1], ddof=1)])
		varXY = np.array([np.sum(U*V)/(n-1) , np.sum(cU*cV)/(N_check_points-1)])
		Ns = np.array([n, N_check_points])

		check_slope = np.sign(np.sum(cU*cV))*np.std(Check_points[:,1], ddof = 1)/np.std(Check_points[:,0], ddof = 1)
		check_int = barY[1] - check_slope *barX[1]

		# make initial guess - the average of the two slopes
		bhat = np.mean([b, check_slope])

		abs_cs = lambda x: np.abs(Utilities.common_slope(x, varX, varY, varXY, Ns))
		b_com = sp.optimize.fmin(func = abs_cs, x0= bhat, xtol = 1e-8, ftol = 1e-8, maxiter = 1e3 , disp = 0)[0]
		# determine the correlation between the fitted and residual axes
		resid_axis_1 = (Y_seg - b_com *X_seg)
		fit_axis_1 = (Y_seg + b_com *X_seg)

		resid_axis_2 = (Check_points[:,1] - b_com *Check_points[:,0]).reshape(Check_points[:,1].size,1)
		fit_axis_2 = (Check_points[:,1] + b_com *Check_points[:,0]).reshape(Check_points[:,1].size,1)

		rrf_1A = (resid_axis_1 - resid_axis_1.mean(axis=0))/resid_axis_1.std(axis=0)
		rrf_1B= (fit_axis_1 - fit_axis_1.mean(axis=0))/fit_axis_1.std(axis=0)
		rrf_1 = (np.dot(rrf_1B.T, rrf_1A)/rrf_1B.shape[0])[0]

		rrf_2A = (resid_axis_2 - resid_axis_2.mean(axis=0))/resid_axis_2.std(axis=0)
		rrf_2B= (fit_axis_2 - fit_axis_2.mean(axis=0))/fit_axis_2.std(axis=0)
		rrf_2 = (np.dot(rrf_2B.T, rrf_2A)/rrf_2B.shape[0])[0]

		rrf = np.array([ rrf_1, rrf_2])

		test_val = -np.sum((Ns-2.5) *np.log(1-rrf**2).reshape(rrf.size,))
		com_slope_pval = 1 - sp.stats.chi2.cdf(test_val, 1)  # probability that the slopes are different

		# Common elevation
		# Pearson correlations of the data used to determine the slopes

		rxy_1A = (X_seg - X_seg.mean(axis=0))/X_seg.std(axis=0)
		rxy_1B= (Y_seg - Y_seg.mean(axis=0))/Y_seg.std(axis=0)
		rxy_1 = (np.dot(rxy_1B.T, rxy_1A)/rxy_1B.shape[0])[0]

		rxy_2A = (Check_points[:,0:1] - Check_points[:,0:1].mean(axis=0))/Check_points[:,0:1].std(axis=0)
		rxy_2B = (Check_points[:,1:2] - Check_points[:,1:2].mean(axis=0))/Check_points[:,1:2].std(axis=0)
		rxy_2 = (np.dot(rxy_2B.T, rxy_2A)/rxy_2B.shape[0])[0]

		rxy = np.array([ rxy_1, rxy_2])
		rxy = rxy.reshape(rxy.size,)

		# The variances on the individual slope estimates
		varB = (1/(Ns-2)) * (varY/varX) * (1-rxy**2)

		# variance of the common slope (b_com)
		varB_com = 1/ (np.sum(1/varB))

		# variance of residuals
		varRes = (Ns-1) /(Ns-2)  * (varY - 2 *b_com *varXY + (b_com**2) *varX)

		tmp_X = vstack((barX, barX))

		var_AS = np.diag(varRes/Ns) + (varB_com *  tmp_X *np.transpose(tmp_X))

		# intercepts using the common slope
		Ahats = np.array([barY[0] - b_com * barX[0], barY[1] - b_com* barX[1]])

		L = np.array([np.ones((1,1)), -np.eye(1)])
		L=L.reshape(L.size,)

		stat = (np.transpose(L @ np.transpose(Ahats)) * (L @ np.transpose(Ahats)))/(L @ var_AS  @ np.transpose(L))
		com_elev_pval = 1-sp.stats.chi2.cdf(stat, 1)

	## pTRM checks

	n_pTRM = 0
	check = np.nan
	dCK = np.nan
	DRAT = np.nan
	maxDEV = np.nan

	CDRAT = np.nan
	CDRAT_prime = np.nan
	DRATS = np.nan
	DRATS_prime = np.nan
	mean_DRAT = np.nan
	mean_DRAT_prime = np.nan
	mean_DEV = np.nan
	mean_DEV_prime = np.nan

	dpal = np.nan

	pTRM_sign = np.nan
	CpTRM_sign = np.nan

	# The first two columns in check_pct, ALT_scalar, and ALT_vec are structured in the same way
	# (:,1) - Temperature the check is to
	# (:,2) - Temperature the check is from


	if ALT_scalar.size != 0 or np.isnan(np.sum(ALT_scalar)) != True:

		# Both temperatures must be less than the maximum
		pTRM_checks = ALT_scalar[np.where((ALT_scalar[:,0] <= Tmax) & (ALT_scalar[:,1] <= Tmax)), 2]
		n_pTRM = pTRM_checks.size
			# Catch the cases with no checks
		if n_pTRM==0:
			check = np.nan
			dCK = np.nan
			DRAT = np.nan
			maxDEV = np.nan

			CDRAT = np.nan
			CDRAT_prime = np.nan
			DRATS = np.nan
			DRATS_prime = np.nan
			mean_DRAT = np.nan
			mean_DRAT_prime = np.nan
			mean_DEV = np.nan
			mean_DEV_prime = np.nan

			dpal = np.nan
			dpal_signed = np.nan
			dpal_ratio = np.nan

			pTRM_sign = np.nan
			CpTRM_sign = np.nan


		else:
			check = 100 *np.max(np.abs(check_pct[np.where((check_pct[:,0] <= Tmax) & (check_pct[:,1] <= Tmax)), 2]))
			dCK = 100 *np.max(np.abs(pTRM_checks/X_int))
			DRAT = 100 *np.max(np.abs(pTRM_checks/Line_Len))
			maxDEV = 100 *np.max(np.abs(pTRM_checks/Delta_x_prime))

			CDRAT = np.abs( 100 *np.sum(pTRM_checks)/Line_Len )
			CDRAT_prime = np.abs( 100 *np.sum(np.abs(pTRM_checks))/Line_Len )

			DRATS = np.abs( 100 *np.sum(pTRM_checks)/ Xpts[seg[-1]] )
			DRATS_prime = np.abs( 100 *np.sum(np.abs(pTRM_checks))/Xpts[seg[-1]])

			mean_DRAT = 100 * np.abs(np.mean(pTRM_checks/Line_Len))
			mean_DRAT_prime = 100 * np.mean(np.abs(pTRM_checks/Line_Len))

			mean_DEV = 100 * np.abs(np.mean((pTRM_checks/Delta_x_prime)))
			mean_DEV_prime = 100 * np.mean(np.abs(pTRM_checks/Delta_x_prime))

			pTRM_sign = np.sign(pTRM_checks[np.where(np.abs(pTRM_checks)== np.max(np.abs(pTRM_checks)))])[0]
			CpTRM_sign = np.sign(np.sum(pTRM_checks))


		# dpal
		# the cumulative sum of the vector difference between the original TRM
		# and the repeat measurement

		# The matrix for the cumulative sum needs to be padded with zeros where
		# no pTRM checks were performed
		to_sum = np.zeros((len(TRMvec),3))  # create empty matrix of zeros
		for j in range(ALT_vec.shape[0]):
			ind = TRMvec[:,0] == ALT_vec[j,0]
			to_sum[ind,:] = ALT_vec[j,2:]



		dPal_sum = np.cumsum(to_sum,axis=0)
		corr_TRM = np.zeros((len(TRMvec),1))
		corr_TRM[0] = 0

		for j in range(1,TRMvec.shape[0]):
			corr_TRM[j] = np.sqrt(np.sum( (TRMvec[j,1:] + dPal_sum[j-1,:])**2 ) )


		Xcorr = corr_TRM[seg]
		Ucorr = detrend(Xcorr, type = "constant", axis = 0)
		corr_slope = np.sign(np.sum(Ucorr*V))*np.std(Y_seg, ddof = 1)/np.std(Xcorr, ddof = 1)
		dpal = np.abs(100*(b - corr_slope)/b)

		dpal_signed = (100*(b-corr_slope)/b)
		dpal_ratio = np.log(corr_slope/b)


	# pTRM tail checks

	n_tail = 0
	if MD_scalar.size != 0:
		# do the checks

		tail_checks = MD_scalar[np.where(MD_scalar[:,0] <= Tmax), 1]
		n_tail = tail_checks.size
		if n_tail == 0:
			dTR = np.nan
			dTRtrm = np.nan
			DRATtail = np.nan
			MDvds = np.nan
			tail_sign = np.nan
		else:
			dTR = np.max(np.abs(100*tail_checks/Y_int))
			dTRtrm = np.max(np.abs(100*tail_checks/np.sqrt(np.sum(TRMvec[-1,1:]**2))))
			DRATtail = np.max(np.abs(100*tail_checks/Line_Len))
			MDvds = np.max(np.abs(100*tail_checks/VDS))

			tail_sign = np.sign(tail_checks[np.where(np.abs(tail_checks) == np.max(np.abs(tail_checks)))])

			if len(tail_sign) > 1:
				# TODO - This is a quick fix - better solution needed
				tail_sign = np.mean(tail_sign)


		# dt*
		tstar = np.empty((1,nmax))
		tstar[:] = np.nan  # Assign the vector

		for j in range(nmax):

			if np.sum((NRMvec[j,0] == tail_vec[:,0])) > 0:

				#tind=find(tail_vec(:,1)==NRMvec(j,1)); % the index of the tail check
				tind = np.nonzero(tail_vec[:,0] == NRMvec[j,0])[0]

				MDx = tail_vec[tind,1]
				MDy = tail_vec[tind,2]
				MDz = tail_vec[tind,3]

				# This is more accurate than dot product when theta is small
				theta_dt = np.arctan2(np.linalg.norm(np.cross(NRMvec[j,1:]/Ypts[j],Blab_orient)),np.dot(NRMvec[j,1:]/ Ypts[j], Blab_orient.reshape(3,)))		# reshape to give correct shape for dot product

				# Define horizontal and vertical according to Blab_orient, such that Blab_orient is always "vertical"

				if np.abs(Blab_orient[0,0] /np.linalg.norm(Blab_orient)) == 1: # Blab is along x
					dH = np.sqrt(np.sum(NRMvec[j,1:3]**2)) - np.sqrt(MDy**2 + MDz**2)
					dZ = NRMvec[j,1] - MDx
					F_inc = Utilities.cart2dir(Blab_orient[0,1], Blab_orient[0,2], Blab_orient[0,0])[1]
					N_inc = Utilities.cart2dir(NRMvec[j,2], NRMvec[j,3], NRMvec[j,1])[1]
					inc_diff = F_inc - N_inc
				elif np.abs(Blab_orient[0,1]/np.linalg.norm(Blab_orient)) == 1: # Blab is along y
					dH = np.sqrt(np.sum(NRMvec[j,1:3]**2)) - np.sqrt(MDx**2 + MDz**2)
					dZ = NRMvec[j,2] - MDy
					F_inc = Utilities.cart2dir(Blab_orient[0,1], Blab_orient[0,0], Blab_orient[0,1])[1]
					N_inc = Utilities.cart2dir(NRMvec[j,3], NRMvec[j,1], NRMvec[j,2])[1]
					inc_diff = F_inc - N_inc
				elif np.abs(Blab_orient[0,2]/np.linalg.norm(Blab_orient)) == 1:  # Blab is along z
					dH = np.sqrt(np.sum(NRMvec[j,1:2]**2)) - np.sqrt(MDx**2 + MDy**2)
					dZ = NRMvec[j,3] - MDz
					F_inc = Utilities.cart2dir(Blab_orient[0,0], Blab_orient[0,1], Blab_orient[0,2])[1]
					N_inc = Utilities.cart2dir(NRMvec[j,1], NRMvec[j,2], NRMvec[j,3])[1]
					inc_diff = F_inc - N_inc
				else:
					raise ValueError('GetPintParams:dt_star; Blab is expected to be along either x, y, or z. If you see this contact GAP.')

					# % THIS IN NOT FULLY TESTED YET
					# % In this case Blab_orient is at an angle to x, y and z
					# % to calculate dt* we transform coordinates such that Blab_orient
					# % is along z, and we rotate the NRM and MD vector
					# %				 Rot_params=vrrotvec(Blab_orient./norm(Blab_orient), [0,0,1]); % The parameters needed to rotate the field vector to z
					# %
					# %				 % rotate the NRM
					# %				 if mod(Rot_params(4), pi/2)==0 % fields along x/y/z should be dealt with above
					# %					 % no need to rotate
					# %					 error('GetPintParams:dt_star', 'Unexpected field angle');
					# %				 else
					# %					 disp(['in the rot  ', num2str( rad2deg(Rot_params(4)) )])
					# %					 NRMrot=Rotate_Vec(NRMvec(j,2:4), Rot_params(4), Rot_params(1:3));
					# %					 MDrot=Rotate_Vec(tail_vec(tind,2:4), Rot_params(4), Rot_params(1:3));
					# %				 end
					# %
					# %				 MDx=MDrot(1);
					# %				 MDy=MDrot(2);
					# %				 MDz=MDrot(3);
					# %
					# %				 dH=sqrt(sum(NRMrot(1:2).^2,2))-sqrt(MDx^2+MDy^2);
					# %				 dZ=NRMrot(3)-MDz;
					# %				 [~, F_inc]=cart2dir(Blab_orient(1), Blab_orient(2), Blab_orient(3));
					# %				 [~, N_inc]=cart2dir(NRMrot(1), NRMrot(2), NRMrot(3));
					# %				 inc_diff=F_inc-N_inc;




				#			 dH=sqrt(sum(NRMvec(j,2:3).^2,2))-sqrt(MDx.^2+MDy.^2);
				#			 dZ=NRMvec(j,4)-MDz;
				B = dH/(np.tan(theta_dt))

				#			 [~, F_inc]=cart2dir(Blab_orient(1), Blab_orient(2), Blab_orient(3));
				#			 [~, N_inc]=cart2dir(NRMvec(j,2), NRMvec(j,3), NRMvec(j,4));
				#			 inc_diff=F_inc-N_inc; % Inclination difference in degrees

				# Roman Leonhardt's dt* implementation - as of v4.2
				if np.floor(theta_dt*1000) < 2968 and np.floor(theta_dt*1000) > 175: # TT Version 4.1
					if inc_diff > 0:
						tstar[0,j] = (-dZ + B) * np.abs(b) * 100.0/np.abs(Y_int)  # sign dependent  diff (new in Vers. 1.8)
					else:
						tstar[0,j] = (dZ - B) * np.abs(b) * 100.0/np.abs(Y_int)  # sign dependent  diff (new in Vers. 1.8)

				else:
					if np.floor(theta_dt*1000) <= 175:
						tstar[0,j] = 0
					elif np.floor(theta_dt*1000) >= 2968:  # TT Version 4.1
						tstar[0,j] = -dZ*100.0/(np.abs(X_int) + np.abs(Y_int)) # -minmax.maxmagTH/(minmax.maxmagTH+minmax.maxmagPT)*dZ * 100.0/minmax.maxmagTH;

		tstar = tstar.reshape(tstar.size,1)

		dt_star = np.nanmax(tstar[np.where(NRMvec[:,0:1] <= Tmax)])

		if dt_star < 0:
			dt_star = 0


	# Catch the case where no pTRM tail checks are used in the segment or at all
	if n_tail == 0:
		dTR = np.nan
		dTRtrm = np.nan
		DRATtail = np.nan
		MDvds = np.nan
		tail_sign = np.nan
		dt_star = np.nan


	## Additivity checks

	n_add = np.nan
	dAC = np.nan

	if ADD_vec.size !=0:
		# % The structure of the ADD_vec and ADD_vec_prime arrays
		# %  ADD_vec(:,1) - lower temperature of the remaining pTRM
		# %  ADD_vec(:,2) - upper temperature of the remaining pTRM
		# %  ADD_vec(:,3) - x component
		# %  ADD_vec(:,4) - y component
		# %  ADD_vec(:,5) - z component
		#
		# % The structure of the ADD_scalar array, which contains the scalar intensities of the estimated
		# % pTRMs (i.e., pTRM*(Ti, T0), following SPD convention)
		# %  ADD_scalar(:,1) - lower temperature of the remaining pTRM
		# %  ADD_scalar(:,2) - upper temperature of the remaining pTRM
		# %  ADD_scalar(:,3) - intensity

		tmp_N = ADD_vec.shape[0]
		AC = np.empty((tmp_N, 1))
		AC[:] = np.nan


		for j in range(tmp_N):
			# AC(j) = ADD_scalar(j,3) - Params.Xpts(Params.Temp_steps==ADD_scalar(j,1));
			AC[j] = np.sqrt(np.sum(ADD_vec[j, 2:5]**2))  - Xpts[np.where(Temp_steps == ADD_vec[j,0])]


		#Params.AC_Test=[ADD_scalar(:,1:2), 100.*AC./Params.X_int, 100.*AC_vec./Params.X_int];

		AC = AC[np.where(ADD_vec[:,0] <= Tmax and ADD_vec[:,1] <= Tmax), 0]  # Select only those within selected temperature range

		dAC = 100 * np.max(np.abs(AC))/np.abs(X_int)

		n_add = len(AC)

		if AC.size ==0:
			dAC = np.nan


	## Create an output for plotting

	pTRM_plot = np.empty((nmax,1))
	pTRM_plot[:] = np.nan
	tail_plot = np.empty((nmax,1))
	tail_plot[:] = np.nan
	Line_pts = np.empty((1,4))
	Line_pts[:] = np.nan

	# % for i=1:Params.nmax
	# %	 if ~isempty(Params.PCpoints) && ~isempty(Params.PCpoints(Params.PCpoints(:,1)==Params.Temp_steps(i),3))
	# %		 pTRM_plot(i)=Params.PCpoints(Params.PCpoints(:,1)==Params.Temp_steps(i),3);
	# %
	# % %				 pTRM_plot = [pTRM_plot;  Params.PCpoints(Params.PCpoints(:,1)==Params.Temp_steps(i),3)];
	# %
	# %		 % Get the points for the lines
	# %		 % Horizontal line
	# %		 From_temp=Params.PCpoints(Params.PCpoints(:,1)==Params.Temp_steps(i), 2); % The temp that the check was from
	# %
	# %		 Hy1=Params.Ypts(Params.Temp_steps==From_temp);
	# %		 Hy2=Hy1;
	# %
	# %		 Hx1=Params.Xpts(Params.Temp_steps==From_temp);
	# %		 Hx2=pTRM_plot(i);
	# %
	# %		 Vx1=Hx2;
	# %		 Vx2=Hx2;
	# %
	# %		 Vy1=Hy1;
	# %		 Vy2=Params.Ypts(Params.Temp_steps==Params.Temp_steps(i));
	# %
	# %		 % Vertical line
	# %
	# %		 Line_pts=[Line_pts;  num2cell( [Hx1, Hy1, Vx1, Vy1; Hx2, Hy2, Vx2, Vy2]); cell(1,4) ];
	# %
	# %	 end
	# %
	# %	 if ~isempty(Params.MDpoints) && ~isempty(Params.MDpoints(Params.MDpoints(:,1)==Params.Temp_steps(i),2))
	# %		 tail_plot(i)=Params.MDpoints(Params.MDpoints(:,1)==Params.Temp_steps(i),2);
	# %	 end
	# %
	# % end

	Plot_Arai = [Temp_steps, Xpts, Ypts, pTRM_plot, tail_plot-Ypts]
	Plot_Line = [x_prime[Seg_Ends], y_prime[Seg_Ends]]

	Plot_pTRM_Lines = Line_pts

	Plot_orth = NRMvec

	# Get the Cartesian coords of the best fits
	x_a, y_a, z_a = Utilities.dir2cart(Dec_A, Inc_A, Mag = 1)  # Get the unit vector for the anchored PCA fit
	Plot_PCA_anc = np.tile(np.array([x_a, y_a, z_a]), (n,1)) * np.tile(Ypts[seg], (1,3)) # Replicate n times and scale by the selected segment
	# Params.Plot_PCA_anc=[0, 0, 0; x_a, y_a, z_a].*max(Params.Ypts); % Replicate n times and scale by the selected segment

	x_f, y_f, z_f = Utilities.dir2cart(Dec_F, Inc_F, Mag = 1)  # Get the unit vector for the free-floating PCA fit
	Plot_PCA_free = np.tile(np.array([x_f, y_f, z_f]), (n, 1))* np.tile(Ypts[seg], (1,3))  # Replicate n times and scale by the selected segment
	# Params.Plot_PCA_anc=[0, 0, 0; x_f, y_f, z_f].*max(Params.Ypts); % Replicate n times and scale by the selected segment


	# Get the output for doing a Z-component plot

	pTRM_plot = np.empty((n,1))
	pTRM_plot[:] = np.nan
	tail_plot = np.empty((n,1))
	tail_plot[:] = np.nan
	Line_pts = np.empty((1,4))
	Line_pts[:] = np.nan
	# % for i=1:Params.nmax
	# %	 if ~isempty(Params.PCpoints) && ~isempty(Params.PCpoints(Params.PCpoints(:,1)==Params.Temp_steps(i),3))
	# %		 pTRM_plot(i)=pCheck(Params.PCpoints(:,1)==Params.Temp_steps(i),end);
	# %
	# %		 % Get the points for the lines
	# %		 % Horizontal line
	# %		 From_temp=Params.PCpoints(Params.PCpoints(:,1)==Params.Temp_steps(i), 2); % The temp that the check was from
	# %
	# %		 Hy1=NRMvec(Params.Temp_steps==From_temp,end);
	# %		 Hy2=Hy1;
	# %
	# %		 Hx1=TRMvec(Params.Temp_steps==From_temp,end);
	# %		 Hx2=pTRM_plot(i);
	# %
	# %		 Vx1=Hx2;
	# %		 Vx2=Hx2;
	# %
	# %		 Vy1=Hy1;
	# %		 Vy2=NRMvec(Params.Temp_steps==Params.Temp_steps(i),end);
	# %
	# %		 % Vertical line
	# %
	# %		 Line_pts=[Line_pts;  num2cell( [Hx1, Hy1, Vx1, Vy1; Hx2, Hy2, Vx2, Vy2]); cell(1,4) ];
	# %
	# %	 end
	# %
	# %	 if ~isempty(Params.MDpoints) && ~isempty(Params.MDpoints(Params.MDpoints(:,1)==Params.Temp_steps(i),2))
	# %		 tail_plot(i)=tail_vec(Params.MDpoints(:,1)==Params.Temp_steps(i),end);
	# %	 end
	# %
	# % end


	Plot_Z_Arai = [Temp_steps, TRMvec[:,-1], NRMvec[NRM_in_TRM,-1], pTRM_plot, tail_plot - NRMvec[NRM_in_TRM,-1]]
	# Params.Plot_Line=[Params.x_prime([Params.Seg_Ends]), Params.y_prime([Params.Seg_Ends])];

	Plot_Z_pTRM_Lines = Line_pts





	##################################
	####   New Zig-Zag parameter #####
	####	   (To be named)	 #####
	##################################
	ziggie, cum_len, arc = get_ziggie(Xpts, Ypts, seg_min, seg_max)

	## Round the stats to the recommended SPD precision

	# Arai plot
	b =  np.round(1000 * b)/1000
	sigma_b = np.round(1000 * sigma_b)/1000
	Blab = np.round(10 * Blab)/10
	Banc = np.round(10 * Banc)/10
	sigma_B = np.round(10 * sigma_B)/10
	f = np.round(1000 * f)/1000
	f_vds = np.round(1000 * f_vds)/1000
	FRAC = np.round(1000 * FRAC)/1000
	beta = np.round(1000 * beta)/1000
	gap = np.round(1000 * gap)/1000
	GAP_MAX = np.round(1000 * GAP_MAX)/1000
	qual = np.round(10 * qual)/10
	w = np.round(10 * w)/10
	k = np.round(1000 * k)/1000
	k_prime = np.round(1000 * k_prime)/1000
	SSE = np.round(1000 * SSE)/1000
	R_corr = np.round(1000 * R_corr)/1000
	R_det = np.round(1000 * R_det)/1000
	Z = np.round(10 * Z)/10
	Z_star = np.round(10 * Z_star)/10
	IZZI_MD = np.round(1000 * IZZI_MD)/1000

	# Directional
	Dec_A = np.round(10 * Dec_A)/10
	Inc_A = np.round(10 * Inc_A)/10
	Dec_F = np.round(10 * Dec_F)/10
	Inc_F = np.round(10 * Inc_F)/10

	MAD_anc = np.round(10 * MAD_anc)/10
	MAD_free = np.round(10 * MAD_free)/10
	alpha = np.round(10 * alpha)/10
	alpha_prime = np.round(10 * alpha_prime)/10

	Theta = np.round(10 * Theta)/10
	if isinstance(DANG, int) == True or isinstance(DANG, float) == True :
		DANG = np.round(10 * DANG)/10
	else:
		DANG = np.round(10 * DANG.item())/10
	if isinstance(NRM_dev, int) == True or isinstance(NRM_dev, float) == True:
			NRM_dev = np.round(10 * NRM_dev)/10
	else:
		NRM_dev = np.round(10 * NRM_dev[0])/10
	gamma = np.round(10 * gamma)/10
	if isinstance(CRM_R, int) == True or isinstance(CRM_R, float) == True:
		CRM_R = np.round(10 * CRM_R)/10
	else:
		CRM_R = np.round(10 * CRM_R.item())/10


	# pTRM checks
	check = np.round(10 * check)/10
	dCK = np.round(10 * dCK)/10
	DRAT = np.round(10 * DRAT)/10
	maxDEV = np.round(10 * maxDEV)/10

	CDRAT = np.round(10 * CDRAT)/10
	CDRAT_prime = np.round(10 * CDRAT_prime)/10
	if np.isnan(DRATS) != True:
		DRATS = np.round(10 * DRATS.item())/10
	if np.isnan(DRATS_prime) != True:
		DRATS_prime = np.round(10 * DRATS_prime.item())/10
	mean_DRAT = np.round(10 * mean_DRAT)/10
	mean_DRAT_prime = np.round(10 * mean_DRAT_prime)/10
	mean_DEV = np.round(10 * mean_DEV)/10
	mean_DEV_prime = np.round(10 * mean_DEV_prime)/10
	dpal = np.round(10 * dpal)/10


	# tail checks
	DRATtail= np.round(10 * DRATtail)/10
	dTR= np.round(10 * dTR)/10
	dTRtrm= np.round(10 * dTRtrm)/10
	MDvds= np.round(10 * MDvds)/10
	dt_star= np.round(10 * dt_star)/10


	# Additivity checks
	dAC= np.round(10 * dAC)/10


	# Anis stats
	Anis_c = np.round(1000* Anis_c)/1000
	dTRM_Anis = np.round(10 * dTRM_Anis)/10


	#s_tensor
	#eigvals
	#tau
	#Hanc
	#N_orient

	# NLT stats
	NLT_meas = np.round(100 * NLT_meas)/100
	NLT_a_est = np.round(10000 * NLT_a_est)/10000
	NLT_b_est = np.round(10000 * NLT_b_est)/10000
	dTRM_NLT = np.round(10 * dTRM_NLT)/10

	# N_Alt => n_pTRM
	# N_MD => n_tail
	# N_AC => n_add
	# d_AC => dAC
	# anis_scale => Anis_c
	# IMG_flag tmp removed


	Params = Meas_Data, Meas_Treatment, Meas_Temp,Blab_orient, Xpts, Ypts, x_prime_scaled, y_prime_scaled,nmax, Temp_steps, TRMvec, NRMvec, Blab, PCpoints, MDpoints, ADpoints, Anis_c, Hanc, s_tensor, n, Seg_Ends, Tmin, Tmax, xbar, ybar, b, Banc, sigma_b, beta, sigma_B, Y_int, X_int, x_prime, y_prime, Delta_x_prime, Delta_y_prime, Line_Len, VDS, f, f_vds, FRAC,gap, GAP_MAX, qual, w, R_corr,  R_det, S_prime_qual, k, SSE, k_prime,  SSEprime, Dec_A, Inc_A, MAD_anc, Dec_F, Inc_F, MAD_free, alpha, alpha_prime, DANG,NRM_dev, Theta, a95, alpha_TRM, gamma, CRM_R, BY_Z, Z, Z_star, IZZI_MD, steps, ziggie, Line_Len_scaled, cum_len, ziggie_new, arc, cum_len_new, SCAT_BOX, SCAT_points, SCAT, multi_SCAT, multi_SCAT_beta, Check_points, check_slope, check_int, com_slope_pval,  com_elev_pval, n_pTRM,  check, dCK, DRAT, maxDEV,  CDRAT, CDRAT_prime, DRATS, DRATS_prime, mean_DRAT, mean_DRAT_prime, mean_DEV, mean_DEV_prime, dpal, pTRM_sign, CpTRM_sign, dpal_signed, dpal_ratio, n_tail, dTR, dTRtrm, DRATtail, MDvds, tail_sign, dt_star, n_add, dAC, Plot_Arai, Plot_Line, Plot_pTRM_Lines, Plot_orth, Plot_PCA_anc, Plot_PCA_free, Plot_Z_Arai, Plot_Z_pTRM_Lines, NLT_meas, NLT_a_est, NLT_b_est, dTRM_NLT, dTRM_Anis


	stats = {"Meas_Data": Meas_Data,"Meas_Treatment" : Meas_Treatment , "Meas_Temp": Meas_Temp, "Blab_orient" : Blab_orient, "Xpts" : Xpts, "Ypts" : Ypts, "x_prime_scaled": x_prime_scaled, "y_prime_scaled": y_prime_scaled, "nmax" : nmax, "Temp_steps" : Temp_steps, "TRMvec" : TRMvec, "NRMvec" : NRMvec, "Blab" : Blab, "PCpoints" : PCpoints, "MDpoints" : MDpoints, "ADpoints" : ADpoints, "Anis_c" : Anis_c, "Hanc" : Hanc, "s_tensor" : s_tensor, "n": n, "Seg_Ends" : Seg_Ends, "Tmin" : Tmin, "Tmax" : Tmax, "xbar" : xbar, "ybar" : ybar, "b": b, "Banc" : Banc, "sigma_b" : sigma_b, "beta" : beta, "sigma_B" : sigma_B, "Y_int": Y_int, "X_int": X_int, "x_prime" : x_prime, "y_prime" : y_prime, "Delta_x_prime": Delta_x_prime, "Delta_y_prime": Delta_y_prime, "Line_Len" : Line_Len, "VDS" : VDS, "f": f, "f_vds": f_vds, "FRAC" : FRAC, "gap" : gap, "GAP_MAX" : GAP_MAX, "qual" : qual, "w": w, "R_corr" : R_corr, "R_det": R_det, "S_prime_qual" : S_prime_qual, "k": k, "SSE" : SSE, "RMS": RMS, "k_prime" : k_prime, "SSEprime" : SSEprime, "RMSprime": RMSprime, "Dec_A": Dec_A, "Inc_A": Inc_A, "MAD_anc" : MAD_anc, "Dec_F": Dec_F, "Inc_F": Inc_F, "MAD_free" : MAD_free, "alpha": alpha, "alpha_prime" : alpha_prime, "DANG" : DANG, "NRM_dev" : NRM_dev, "Theta": Theta, "a95" : a95, "alpha_TRM": alpha_TRM, "gamma": gamma, "CRM_R": CRM_R, "BY_Z" : BY_Z, "Z": Z, "Z_star" : Z_star, "IZZI_MD" : IZZI_MD, "steps": steps, "Line_Len_scaled" : Line_Len_scaled, "ziggie" : ziggie, "arc" : arc, "cum_len" : cum_len, "SCAT_BOX" : SCAT_BOX, "SCAT_points" : SCAT_points, "SCAT" : SCAT, "multi_SCAT" : multi_SCAT, "multi_SCAT_beta" : multi_SCAT_beta, "Check_points" : Check_points, "check_slope" : check_slope, "check_int": check_int, "com_slope_pval" : com_slope_pval, "com_elev_pval": com_elev_pval, "n_pTRM" : n_pTRM, "check": check, "dCK" : dCK, "DRAT" : DRAT, "maxDEV" : maxDEV, "CDRAT": CDRAT, "CDRAT_prime" : CDRAT_prime, "DRATS": DRATS, "DRATS_prime" : DRATS_prime, "mean_DRAT": mean_DRAT, "mean_DRAT_prime" : mean_DRAT_prime, "mean_DEV" : mean_DEV, "mean_DEV_prime" : mean_DEV_prime, "dpal" : dpal, "pTRM_sign": pTRM_sign, "CpTRM_sign" : CpTRM_sign, "dpal_signed" : dpal_signed, "dpal_ratio" : dpal_ratio, "n_tail" : n_tail, "dTR" : dTR, "dTRtrm": dTRtrm, "DRATtail" : DRATtail, "MDvds": MDvds, "tail_sign": tail_sign, "dt_star" : dt_star, "n_add": n_add, "dAC" : dAC, "Plot_Arai": Plot_Arai, "Plot_Line": Plot_Line, "Plot_pTRM_Lines" : Plot_pTRM_Lines, "Plot_orth": Plot_orth, "Plot_PCA_anc" : Plot_PCA_anc, "Plot_PCA_free": Plot_PCA_free, "Plot_Z_Arai" : Plot_Z_Arai, "Plot_Z_pTRM_Lines": Plot_Z_pTRM_Lines, "NLT_meas": NLT_meas, "NLT_a_est": NLT_a_est, "NLT_b_est": NLT_b_est, "dTRM_NLT": dTRM_NLT, "dTRM_Anis": dTRM_Anis, "NRM_in_TRM": NRM_in_TRM, "Line_Len": Line_Len}

	return stats


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



def intersection(a, b, radius, p2x, p2y):

	""" find the two points where a secant intersects a circle """
	dx, dy = p2x - a, p2y - b
	j = dx**2 + dy**2
	k = 2 * (dx * (p2x- a) + dy * (p2y - b))
	l = (p2x - a)**2 + (p2y -b)**2 - radius**2

	discriminant = k**2 - 4 * j * l
#	 assert (discriminant > 0), 'Not a secant!'

	t1 = (-k + discriminant**0.5) / (2 * j)
	t2 = (-k - discriminant**0.5) / (2 * j)

	return (dx * t1 + p2x, dy * t1 + p2y), (dx * t2 + p2x, dy * t2 + p2y)

def closer(a,b,p1,p2):
	dist1 = np.sqrt( (p1[0] - a)**2 + (p1[1] - b)**2 )
	dist2 = np.sqrt( (p2[0] - a)**2 + (p2[1] - b)**2 )
	if dist1 > dist2:
		return p2
	else:
		return p1

def AraiArc(k, a, b, Xn1, Yn1, Xn2,Yn2):
	r = np.abs(1/k)

	p1, p2 = intersection(a, b, r, Xn2, Yn2)
	p = closer(Xn2,Yn2,p1,p2)
	x2=p[0]
	y2=p[1]

	p1, p2 = intersection(a, b, r, Xn1, Yn1)
	p = closer(Xn1,Yn1,p1,p2)
	x1=p[0]
	y1=p[1]

	# vectors 1 and 2
	vec1 = [a-x1, b-y1]/np.linalg.norm([a-x1, b-y1])
	vec2 = [a-x2, b-y2]/np.linalg.norm([a-x2, b-y2])

	# angle between vectors
	angle = np.arctan2(vec1[0] * vec2[1] - vec1[1] * vec2[0], vec1[0] * vec2[0] + vec1[1] * vec2[1])

	if (angle < 0):
		angle += 2*np.pi
	if k<0:
		angle = 2*np.pi - angle
	# arc for angle in radians

	arc = r*angle
	return arc, (x1, y1), (x2,y2)

# Find the scaled length of the best fit line
# normalise points
def get_zigzag(Xn, Yn, seg_min, seg_max):

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
	return ziggie, cum_len, Line_Len

# Find the scaled length of the best fit line
# normalise points
def get_ziggie(Xpts, Ypts, seg_min, seg_max):

	Xn = Xpts[seg_min:seg_max+1]
	Xn = Xn/np.amax(Xn)
	Yn = Ypts[seg_min:seg_max+1]
	Yn = Yn/ np.amax(Yn)
	ymax_idx = np.where(Yn == np.max(Yn))
	xmax_idx = np.where(Xn == np.max(Xn))
	k_prime, a, b, SSE, RMS = Utilities.AraiCurvature(Xn,Yn)

	if (np.abs(k_prime) <= 1e-3) or (np.isnan(k_prime)):
		ziggie, cum_len, Line_Len = get_zigzag(Xn, Yn, seg_min, seg_max)
		return ziggie, cum_len, Line_Len
	else:
		arc, point1, point2 = AraiArc(k_prime.item(), a.item(), b.item(), Xn[0][0],Yn[0][0],Xn[-1][0], Yn[-1][0],  )
		n = len(Xn)

		# Set cumulative length to 0
		cum_len = 0.0

		# iterate through pairs of points in Arai plot
		for i in range(0, n-1):

			# find the distance between the two points
			dist = np.sqrt((Xn[i+1,0] - Xn[i,0])**2 + (Yn[i+1,0] - Yn[i,0])**2)
			# Add to the cumulative distance
			cum_len = cum_len + dist

		# calculate the log of the cumulative length over the length of the best fit arc
		ziggie = np.log(cum_len/arc)
		return ziggie, cum_len, arc
