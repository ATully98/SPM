import numpy as np
import scipy as sp
import warnings

# Import specific functions
from numpy.random import default_rng
from scipy import stats
from warnings import warn as warning


# Import model functions
import SetupErrors_v6_1
import Utilities
from Utilities import quad
from Utilities import quad2d
from Utilities import MDblocking

def MD_model_v4(Experiment, Beta_fits, lam, Banc, N_orient, Blab, T_orient, Tc, Noise, Flags, P, Rot_Treatment, Rot_angle, NLT_a, NLT_b, Bnlt):

	"""
	 Function to return output from an MD paleointensity  experiment with
	 experimental noise
	 The function is based on the beta blocking distribution modification of
	 Andy Biggin's Model P

	 v4 Includes NLT and Anisotropy effects

	 v3 adds support for the Thellier method. The Thellier method is treated
	 like an Aitken method with demag step (treatment=0) calculated as inverse
	 field steps. This can be done as the effective field (Beff) is used to
	 propagate the various field steps

	Inputs:
			Experiment - str path to experiment file
			Beta_fits - (2, ) array containing the parameters a and b
			lam -  float containing value for the lambda for the multi-domainess
			Banc - (1, ) array value for the magnitude of the ancient B field in micro-tesela
			N_orient - (1,3) array containing the vector of the ancient B field
			Blab - float value for the magnitude of the applied lab field in micro-tesela
			T_orient - (1, 3) array containing the vector direction of the applied lab field
			Tc - float value of the Curie temperature to be used
			Noise - pseudo-boolean value to set the noise on or off
			Flags - list of length 4 containing the pseudo-boolean values for flags to be used (1 for on, 0 for off)
			P - float value defining the shape of the grain (P= k1/k3)
			Rot_Treatment - (3, ) array containg the rotation axis used for the anisotropy matrix
			Rot_angle - float value for the rotation in radians used for the anisotpy matrix
			NLT_a - float value for the first scaling factor in non-linear TRM acquisition
			NLT_b - float value for the second scaling factor in non-linear TRM acquisition
			Bnlt - (6, ) array containing the NLT temperture steps, the last value is Blab
	Outputs:
			Mvec - (n, 3) array with each row containing the Magnetisation vector for each step
	 		temperatures - (n, 1) array containing the temperature for each step with noise included
		 	treatment - (n, ) array containg treatment codes for each step
			Beff - (n, 1) array containing the effective field
			NLT_vec - (len(Bnlt)+1, 3) array with each row containing the vector for each NLT check step
		 	Anis_vec - (7, 3) array containing the anis vectors for each oreintation
	"""




	## Some basic setup
	temperatures, treatment, Hx, Hy, Hz = Utilities.OpenExperiment(Experiment)
	steps = np.shape(temperatures)
	length = len(temperatures)
	Temps = (np.insert(np.unique(temperatures), (0), 0 )/100).reshape((len(np.unique(temperatures))+1,1))
	points = len(Temps)
	alph4 = 0.0 #0.001 # parameter from original Model P - Set to zero for Model B
	# open flags
	Anis_flag = Flags[0]
	NLT_flag= Flags[1]
	NRflag = Flags[2]

	# # Absolute and relative tolerances for the integration
	# Atol = 1e-8
	# Rtol = 1e-4  # 0.0001 %

	## Setup errors
	hold_time, cooling_rate, T_repeat, T_grad, T, B, res, meas, background, kappa= SetupErrors_v6_1.SetupErrors_v6(Tc, np.insert(temperatures[1:]/100, (0), 0, axis = 0), Blab)
	rng = default_rng()

	# Hold time and cooling rates variances are set to zero - NO MD THEORY!!
	hold_time = np.zeros((1, points))
	cooling_rate = np.zeros((1, points))

	# These are used to override the angular errors
	theta_scale = 1
	phi_scale = 1
	BG_scale = 1


	# set eps function to stop problems with beta function integration to 1
	eps = (np.finfo(float).eps)**(0.5)


	## Error Control
	# Uncomment each line to set individual error sources to zero - Used to
	# investigate indiviudual variances

	if Noise!=1:
		T_repeat = np.zeros(steps)
		T_grad = np.zeros(steps)

		res=0
		B=0
		theta_scale=0.0
		phi_scale=0.0
		meas=0
		BG_scale=0
		kappa = 0


	T = np.sqrt(T_repeat**2 + T_grad**2)


	## Generate random noise
	# Temperatures
	# Points heating steps and 4 heatings (NRM, TRM, tail check, pTRM
	# check)

	T_rnd = rng.normal(size = steps)*T

	# Residual fields
	# Points heating steps and 4 heatings (NRM, TRM, tail check, pTRM check)
	Bres_rnd = rng.normal(size = steps, scale = res)

	# Laboratory fields
	# Points heating steps and 2 heating steps (TRM, pTRM check)
	# use steps not points, overhead, but easier
	Blab_rnd = rng.normal(size = steps, scale = B)

	# Background noise -  Use Blab as approxiamtion
	# Points heating steps x 3 axes x 4 measurement steps
	# Scaled by Blab, which is the same as the total TRM
	BG = stats.cauchy.rvs(loc = background[0], scale = background[1], size = (length*3,1) )*Blab/100

		# truncate excessive background
	for values in BG:
		if values > 0.8/100:
			values = 0.8/100
		if values < -0.8/100:
			values = -0.8/100

	BG = np.reshape(BG, (length,3))
	BG=BG*BG_scale

	# Measurement noise
	# Random normal vector steps x 3 axes x 4 measurement steps
	RndMeas = rng.normal(size = (length,3))




	# Effective applied field orientaions
	Tmp_Angle = Utilities.FishAngleNoise(kappa, length, T_orient)


	#################################################################################################################################
	# New NLT and Anisotpy section
	# Set up rotation matrix, which represents the anisotropy axes
	if Anis_flag == 1:
		# create rotation matrix
		Anis_mat = Utilities.rotvec2mat(Rot_Treatment, Rot_angle)

		# Eigenvalues, the order does not matter
		Anis_degree = np.array(([P, 1, 1])) # Prolate
		# Anis_degree = np.array(([P, P, 1])) # Oblate
		# Anis_degree = np.array(([P, (P+1)/2, 1])) # Triaxial
		# Anis_degree = np.array(([P, (3*P+1)/4, 1])) # Prolate-Triaxial
		# Anis_degree = np.array(([P, (P+3)/4, 1])) # Oblate-Triaxial


		NO = N_orient.reshape(3,)
		tmp_NO = Anis_mat @ np.transpose(NO) # Rotate NRM to eigenvector coords

		tmp_NO = np.transpose(tmp_NO) *Anis_degree  # Scale by the eigenvakues
		N_orient = np.transpose(np.linalg.solve(Anis_mat,np.transpose(tmp_NO))) # Rotate back into sample coords
		B_Angle = Utilities.create_array(((length, 3)))

		for n in range(length):
			tmp_BA = Anis_mat @ np.transpose(Tmp_Angle[n,:])
			tmp_BA = np.transpose(tmp_BA) * Anis_degree
			B_Angle[n,:] = np.transpose(np.linalg.solve(Anis_mat,np.transpose(tmp_BA)))

	# if anisotropy off
	else:
		B_Angle = Tmp_Angle
	#################################################################################################################################

	## Define the random temps/fields etc

	# Field strength - orientation is in B_Angle
	Beff = (Blab + Blab_rnd) + Bres_rnd
	Beff[0] = Banc # this is not used, but added to keep thing cosistent
	#   set B field to zero for outfield and tail checks
	#   set to negative for reverse field
	for i in range(len(treatment)):
		if treatment[i] == 0 or treatment[i] == 3:
			Beff[i]=Bres_rnd[i]
		elif treatment[i] == 5:
			Beff[i] = -Beff[i]	  #backfield step
	Beff = Beff.reshape(length,)		#reshape to allow for clearer indexing through model

	# Beff for demag
	if np.sum(treatment[1:] ) == 0:
		Beff[0] = (0 + Blab_rnd[0]) + Bres_rnd[0]

	## Non-Linear TRM
	if NLT_flag == 1:
		Beff = NLT_a * np.tanh(NLT_b * Beff)
		Banc = NLT_a * np.tanh(NLT_b * Banc)

	# Remember the set temperatures
	Set_Temps = temperatures
	# Add noise to the temperatures
	Temps = (temperatures/100) + T_rnd

	#   Take off eps from maximum temperatures
	for i in range(len(Temps)):
		if Temps[i] >= 1:
			Temps[i] = 1- eps # eps to prevent numerical errors
	Temps[0] = 1- eps # Remove noise from the NRM acquisition step
	Tc = 1.0 # Use this as the max temperature for the integrations


	# NR not currently available
	if NRflag==1:

		warning("Non-Reciprocity Handling is Currently Under Construction and Will NOT be implemented")

	## Experimental Output

	# Experiment Axis numbers follow ThellierTool convention (modified from the
	# original Model P code)
	# 0 are NRM demag steps
	# 1 are pTRM step
	# 2 are pTRM checks
	# 3 are pTRM tail checks
	# 4 NOT USED - additivity checks
	# 5 are inverse field steps for the Thellier-Thellier protocol

	# The magnetization "landscape" is not remembered as is the case for the
	# discretized models. This has to be explicitly defined as the summation of
	# different integrals. Therefore each type of step listed above has to be
	# defined seperately. The "memory" is retained by keeping track of the
	# temperature steps and recalculating the appropriate integral for NRM
	# remainng, pTRM gained, residual pTRM, etc.


	# The model describes each step using 4 components to account for the
	# magnetizations held in different Tb1/Tb2 portions:
	#
	# 1) NRM_remain - The NRM remaining
	# 2) TRM_remain - the TRM tail remaining
	# 3) Bres_remain - Residual field remanence blocked
	# 4 )TRM_gain - TRM gained either from residual fields or from
	# direct in-field steps


	# In this section noise is be added to the temperatures. The inbuilt
	# "memory" of the system allows the over/underheat differences to
	# propagate through the calculations.

	# The exception to this is under demagnetization of residual field
	# magnetizations blocked during the NRM step.

	# The model has certain aspects hardwired in relating to this. The
	# hardwiring is with respect to the experimental procdure. The following
	# assumptions are made:

	# 1) pTRM acquisition steps always follow NRM demagnetization steps, i.e.,
	# a Coe-type experiment. This allows the unblocked esidual field
	# magneitzations to be calculated during the pTRM step. This may work for
	# IZZI experiments, but this has not been tested.

	# 2) pTRM tail steps are ALWAYS followed by a step to a MUCH HIGHER
	# TEMPERATURE. This means that residual field magnetizations can be
	# ignored.


	# Create the initail NRM
	Mvec = Utilities.create_array((length, 3))

	# number of integration points to be used
	npts = 30
	# intial integral
	blocking = lambda x1, x2: MDblocking(x1, x2, Beta_fits, lam, alph4)
	integral = quad2d(blocking, 0, Tc,  0,  Tc)
	Mvec[0,:] = Banc *N_orient *integral
	# Main Experiment

	for n in range(1,length):

		## NRM demagnetization
		if treatment[n] == 0 or treatment[n] == 5:

			if n == 1:  # start of the experiment Temps[n-1] ==0

				NRM_remain = Banc * N_orient * quad2d(blocking, Temps[n], Tc,  0,  Tc)
				TRM_remain = np.zeros((3,)) # No in-field step has been performed
				Bres_remain = np.zeros((3,)) # First heating, so no residual field remanence
				TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking, 0, Temps[n], 0, Tc)


			elif treatment[n-1] == 0 and Set_Temps[n-1] < Set_Temps[n]:
				# Demag experiment or basic IZZI

				if ((treatment[n-2] != 1) and (np.sum(treatment[1:])==0))  or  ((n == 2) and (np.sum(treatment[1:])==0)) :
					# Probably doing a demag
					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n], Tc, Temps[n-1], Tc)
					TRM_remain = Beff[n-1] * B_Angle[n-1,:] * quad2d(blocking,Temps[n], Tc, 0, Temps[n-1])
					Bres_remain = np.zeros((3,))
					TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Temps[n], 0, Tc)

					#raise ValueError('Experiment:NRM;  Sequence not Modelled')
				else:
					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n], Tc, Temps[n-2], Tc)
					TRM_remain = Beff[n-2] * B_Angle[n-2,:] * quad2d(blocking,Temps[n], Tc, 0, Temps[n-2])
					Bres_remain = np.zeros((3,))# Next higher step, so no residual field remanence from the previous demag step is remembered
					TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Temps[n], 0, Tc)

			elif treatment[n-1] == 1 and Set_Temps[n-1] == Set_Temps[n]: # I->Z steps at same temperature => Aitken/IZZI experiment or Thellier
				NRM_remain = Banc *N_orient  * quad2d(blocking,Temps[n], Tc, Temps[n-1], Tc)
				TRM_remain = Beff[n-1] * B_Angle[n-1,:] * quad2d(blocking,Temps[n], Tc, 0, Temps[n-1])
				Bres_remain = np.zeros((3,))  # Next higher demag step, so no residual field remanence from the previous demag step is remembered
				TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Temps[n], 0, Tc)


			elif treatment[n-1] == 1 and Set_Temps[n-1] < Set_Temps[n]: #Moved to the next higher temperature step in a ZI

				if treatment[n]==5:
					raise ValueError('Experiment:NRM;  This style of Thellier experiment is not modelled')


				NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n], Tc, Temps[n-1], Tc)
				TRM_remain = Beff[n-1] * B_Angle[n-1,:] * quad2d(blocking,Temps[n], Tc, 0, Temps[n-1])
				Bres_remain = np.zeros((3,)) # Next higher step, so no residual field remanence from the previous demag step is remembered
				TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Temps[n], 0, Tc)

			elif treatment[n-1] == 2: # pTRM check followed by demag => IZ steps

				# ASSUMES THAT WE ARE AT THE NEXT HIGHEST TEMPERATURE STEP
				if Set_Temps[n-2] >= Set_Temps[n]: # Test this assumption
					raise ValueError(f'Experiment:NRM; Set temperature sequence not modelled. Step {n}: {Set_Temps[n-1,0]:.2f}->{Set_Temps[n,0]:.2f}')


				# Test for expected sequence
				if treatment[n-2] !=0 or treatment[n-3] != 1:
					raise ValueError(f'Experiment:NRM; Unexpected IZZI sequence. Step {n}: {treatment[n-2]}->{treatment[n-1]}->{treatment[n]}')


				NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n], Tc, Temps[n-3], Tc)
				# TRM remaining comes in two parts. One from the pTRM check,
				# the other from the previous TRM step
				# Atol and Rtol are reduced to minimize the rounding error
				# introduced by the summation of the 2 integrals
				TRM_remain = (Beff[n-1] * B_Angle[n-1,:] * quad2d(blocking,Temps[n], Tc, 0, Temps[n-1], rounding_err=True)
					+ Beff[n-3] * B_Angle[n-3,:] * quad2d(blocking,Temps[n], Tc, Temps[n-1], Temps[n-3], rounding_err=True))

				Bres_remain = np.zeros((3,)) # Next higher step, so no residual field remanence from the previous demag step is remembered
				TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Temps[n], 0, Tc)

			elif treatment[n-1] == 3:
				# ASSUMES THAT WE ARE AT THE NEXT HIGHEST TEMPERATURE STEP
				if Set_Temps[n-1] >= Set_Temps[n]: # Test this assumption
					raise ValueError(f'Experiment:NRM;  Set temperature sequence not modelled. Step {n}: {Set_Temps[n-1,0]:.2f}->{Set_Temps[n,0]:.2f}')


				# NRM remaining along Tb1 is controlled by the current step
				# NRM remaining along Tb2 is controlled by the last in-field
				# step, which should be n-2
				# Check this, if not error!
				if treatment[n-2] != 1:
					raise ValueError(f'Experiment:NRM;  Unrecognized sequence. Step {n}: {Set_Temps[n-2,0]:.2f}->{Set_Temps[n-1,0]:.2f}->{Set_Temps[n,0]:.2f}')


				NRM_remain = Banc *N_orient * quad2d(blocking,Temps[n], Tc, Temps[n-2], Tc)
				TRM_remain = Beff[n-2] * B_Angle[n-2,:] * quad2d(blocking,Temps[n], Tc, 0, Temps[n-2])
				Bres_remain = np.zeros((3,)) # Next higher step, so no residual field remanence from the previous demag step is remembered
				TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Temps[n], 0, Tc)



			else:
				raise ValueError(f'Experiment:NRM; Experiment style not modelled. Step/treatment/previous: {n}/{treatment[n]}/{treatment[n-1]}')


			Mvec[n,:] = NRM_remain  + TRM_remain + Bres_remain + TRM_gain



		## TRM magnetization
		if treatment[n] == 1:


			if treatment[n-1] == 0 or treatment[n-1] == 5: # previous was a demag

				if Set_Temps[n-1] < Set_Temps[n]: #moved to the next highest temperature => IZ step

					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n-1], Tc, Temps[n], Tc)
					TRM_remain = np.zeros((3,)) # Overwritten by current remag
					TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Tc, 0, Temps[n])
					Bres_remain = Beff[n-1] * B_Angle[n-1,:] * quad2d(blocking,0, Temps[n-1],  Temps[n], Tc)

				elif Set_Temps[n-1] == Set_Temps[n]: # Same temperature => ZI steps

					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n-1], Tc, Temps[n], Tc)
					TRM_remain = np.zeros((3,))
					TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Tc, 0, Temps[n])
					Bres_remain = Beff[n-1] * B_Angle[n-1,:] * quad2d(blocking,0, Temps[n-1],  Temps[n], Tc)

				else:
					raise ValueError(f'Experiment:TRM;  Set temperature sequence not modelled. Step {n}: {Set_Temps[n-1,0]:.2f}->{Set_Temps[n-1,0]:.2f}')


			elif treatment[n-1] == 1:

				if n == 1: #  Aitken/IZZI starting with IZ steps

					NRM_remain = Banc * N_orient * quad2d(blocking,0, Tc, Temps[n], Tc)
					TRM_remain = np.zeros((3,))
					TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Tc, 0, Temps[n])
					Bres_remain = np.zeros((3,))

				elif treatment[n-2] == 0 and Set_Temps[n-2] == Set_Temps[n-1] and Set_Temps[n-1] < Set_Temps[n]:
					# Basic IZZI sqeuence (no checks)
					# Demag(T-1)-> Remag(T-1) -> Remag(T)
					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n-2], Tc, Temps[n], Tc)
					TRM_remain = np.zeros((3,)) #Overwritten by current remag
					TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Tc, 0, Temps[n])
					Bres_remain = Beff[n-2] * B_Angle[n-2,:] * quad2d(blocking,0, Temps[n-2],  Temps[n], Tc)

				elif treatment[n-2] == 2 and Set_Temps[n-1] < Set_Temps[n] and treatment[n-3] == 0:
					# IZZI sequnce with a check
					# This was added to Model the Shaar11 data (25/11/2013)
					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n-3], Tc, Temps[n], Tc)
					TRM_remain = np.zeros((3,)) #Overwritten by current remag
					TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Tc, 0, Temps[n])
					Bres_remain = Beff[n-3] * B_Angle[n-3,:] * quad2d(blocking,0, Temps[n-3],  Temps[n], Tc)
				else:
					raise ValueError(f'Experiment:TRM; Experimental sequence not modelled. Step {n}: {treatment[n-2]}->{treatment[n-1]}->{treatment[n]}')



			elif treatment[n-1] == 2:


				if Set_Temps[n-2] == Set_Temps[n] and treatment[n-2] == 0:
					# Coe experiment
					# Demag(T) -> check -> Remag(T)

					# NRM remaining along Tb1 is controlled by the n-2 step
					# NRM remaining along Tb2 is controlled by the current step

					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n-2], Tc, Temps[n], Tc)
					TRM_remain = np.zeros((3,))
					TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Tc, 0, Temps[n])
					Bres_remain = Beff[n-2] * B_Angle[n-2,:] * quad2d(blocking,0, Temps[n-2], Temps[n], Tc)

				elif Set_Temps[n-2] < Set_Temps[n] and treatment[n-2] == 0:
					# Aitken experiment
					# Demag(T-1) -> check -> Remag(T)

					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n-2], Tc, Temps[n], Tc)
					TRM_remain = np.zeros((3,)) # Overwritten by current remag
					TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Tc, 0, Temps[n])
					Bres_remain = Beff[n-2] * B_Angle[n-2,:] * quad2d(blocking,0, Temps[n-2], Temps[n], Tc)

				elif Set_Temps[n-2] < Set_Temps[n] and treatment[n-2] == 5:
					# Thellier experiment
					# Inverse(T-1) -> check -> Direct(T)

					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n-2], Tc, Temps[n], Tc)
					TRM_remain = np.zeros((3,)) # Overwritten by current remag
					TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Tc, 0, Temps[n])
					Bres_remain = Beff[n-2] * B_Angle[n-2,:] * quad2d(blocking,0, Temps[n-2], Temps[n], Tc)

				else:
					raise ValueError(f'Experiment:TRM;  Unrecognized sequence. Step {n}: {Set_Temps[n-2,0]:.2f}->{Set_Temps[n-1,0]:.2f}->{Set_Temps[n,0]:.2f}')



			elif treatment[n-1] == 3:

				# ASSUMES IZZI SEQUENCE
				if Set_Temps[n-1] >= Set_Temps[n] or treatment[n-2] !=1 or treatment[n-3] != 0:
					if treatment[n-2] ==5:
						dummy = 0
					else:
						raise ValueError(f'Experiment:NRM;  Unexpected IZZI sequence. Step {n}: {treatment[n-3]}->{treatment[n-2]}->{treatment[n-1]}')

				TRM_remain = np.zeros((3,)) # Overwritten by current remag
				TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Tc, 0, Temps[n])

				if Temps[n-1] < Temps[n-3]:
					# Tail underheats compared to NRM

					# NRM remaining blocked along Tb1 Temps(n-3) -> Tc
					# Along Tb2 Temps(n) -> Tc
					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n-3], Tc, Temps[n], Tc)

					# Bres_remain is in 2 parts
					# Initial NRM demag Tb1(Temps(n-1), Temps(n-3)), Tb2(Temps(n), Tc))
					# Tail check Tb1(0, Temps(n-1)), Tb2(Temps(n), Tc))
				 	# Atol and Rtol are reduced to minimize the rounding error
					# introduced by the summation of the 2 integrals
					Bres_remain = ( Beff[n-3] * B_Angle[n-3,:] * quad2d(blocking,Temps[n-1], Temps[n-3], Temps[n], Tc, rounding_err=True)
									+ Beff[n-1] * B_Angle[n-1,:] * quad2d(blocking,0, Temps[n-1], Temps[n], Tc, rounding_err=True))

				else:
					# Tail overheat or equal to NRM
					# NRM remaining is controlled by Temps[n-1]
					# Tb1( Temps(n-1), Tc ); Tb2( Temps(n), Tc )
					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n-1], Tc, Temps[n], Tc)

					# Bres_remain is controlled by Temp[s(n-1)
					# Tb1( 0, Temps(n-1)); Tb2( Temps(n), Tc )
					Bres_remain = Beff[n-1] * B_Angle[n-1,:] * quad2d(blocking,0, Temps[n-1], Temps[n], Tc)


			else:
				raise ValueError(f'Experiment:TRM;  Experiment style not modelled. Step number/type: {n}/{treatment[n-1]}')


			Mvec[n,:] = NRM_remain + TRM_remain + Bres_remain + TRM_gain



		## pTRM check

		# Since pTRM check temperature are much lower then previous temperature
		# steps we don't worry about the excess NRM/residual field
		# demagnetization due to a temperature overheat

		# NRM_remaining -
		# TRM_gained - the pTRM gained during the pTRM check
		# Bres_remain - residual field remanence
		# TRM_remain - TRM tail from previous in-field step remaining
		if treatment[n] == 2:

			# Put these here so that they can be over-ridden if needed
			TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Tc, 0, Temps[n])
			Bres_remain = Beff[n-1] * B_Angle[n-1,:] * quad2d(blocking,0, Temps[n-1], Temps[n], Tc)


			if treatment[n-1] == 0 or treatment[n-1] == 5: #previous step was a NRM demag step

				# Check experimental sequence
				if Set_Temps[n-2] == Set_Temps[n-1] and treatment[n-2] == 1:
					# In an Aitken/IZZI: Remag(T )-> Demag(T) -> pTRM_Check(T-2)

					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n-1], 1, Temps[n-2], 1)
					TRM_remain = Beff[n-2] * B_Angle[n-2,:] * quad2d(blocking,Temps[n-1], 1, Temps[n], Temps[n-2])


				elif Set_Temps[n-2] < Set_Temps[n-1] and treatment[n-2] == 3:
					# ZI steps: Tail(T-1) -> Demag(T) -> pTRM_Check(T-2)
					# NEXT HIGHER TEMPERATURE STEP
					# NRM remaining along Tb1 is controlled by the n-1 step
					# NRM remaining along Tb2 is controlled by the last
					# in-field step which is... n-3

					if treatment[n-3] != 1: # check that n-3 was an in-field step
						raise ValueError(f'Experiment:pTRM_check;  Unrecognized sequence. Step {n}: {Set_Temps[n-2,0]:.2f}->{Set_Temps[n-1,0]:.2f}->{Set_Temps[n,0]:.2f}')


					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n-1], 1, Temps[n-3], 1)


					if Set_Temps[n-2] == Set_Temps[2]: # This for checks at the start of the experiment
						TRM_remain = np.zeros((3,)) # The tail is overwritten by the check remag
					else:
						TRM_remain = Beff[n-3] *B_Angle[n-3,:] * quad2d(blocking,Temps[n-1], 1, Temps[n], Temps[n-3])



				elif Set_Temps[n-2] < Set_Temps[n-1] and treatment[n-2] == 1:
					# ZI steps: Remag(T-1) -> Demag(T) -> pTRM_Check(T-2)
					# Usually at the start of experiment or when no tail checks
					# are used

					# NEXT HIGHER TEMPERATURE STEP
					# NRM remaining along Tb1 is controlled by the n-1 step
					# NRM remaining along Tb2 is controlled by the last
					# in-field step which is... n-2
					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n-1], 1, Temps[n-2], 1)



					if Set_Temps[n-2] == Set_Temps[2]:# This for checks at the start of the experiment
						TRM_remain = np.zeros((3,)) # The tail is overwritten by the check remag
					else:
						TRM_remain = Beff[n-2] * B_Angle[n-2,:] * quad2d(blocking,Temps[n-1], 1, Temps[n], Temps[n-2])


				elif Set_Temps[n] < Set_Temps[n-2] and Set_Temps[n-2] < Set_Temps[n-1]:
					# the checks skip multiple steps


					if Set_Temps[n] == Set_Temps[n-4] and treatment[n-1] == 0 and treatment[n-2] == 0 and treatment[n-3] == 1 and treatment[n-4] == 1: # This is for the Shaar et al. (2011) experiment
						# This was added to Model the Shaar11 data (25/11/2013)
						#print('GO SHAAR!!')

						# NRM remaining Tb1 = Temps(n-1) to Tc
						# NRM remaining Tb2 = Temps(n-3) to Tc
						NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n-1], 1, Temps[n-3], 1)


						# Previous TRM Tb1 = Temps(n-1) to Tc
						# Previous TRM Tb2 = Temps(n) to Temps(n-3)
						# Field is n-3
						TRM_remain = Beff[n-3] * B_Angle[n-3,:] * quad2d(blocking,Temps[n-1], 1, Temps[n], Temps[n-3])


					else:
						raise ValueError(f'Experiment:pTRM_check;  The model does nto currently support this pTRM check sequence. Step {n}: {Set_Temps[n-2,0]:.2f}->{Set_Temps[n-1,0]:.2f}->{Set_Temps[n,0]:.2f}')


				else:
					raise ValueError(f'Experiment:pTRM_check;  Unrecognized sequence. Step {n}: {Set_Temps[n-2,0]:.2f}->{Set_Temps[n-1,0]:.2f}->{Set_Temps[n,0]:.2f}')





			elif treatment[n-1] == 3:
				# ASSUMES SQUENCE:
				# Demag(T) -> Remag(T) -> Tail(T) -> pTRM_check(T-i)

				# Check for this sequence
				if treatment[n-3] != 0 or treatment[n-2] != 1 or Set_Temps[n-1] != Set_Temps[n-3] or Set_Temps[n-1] != Set_Temps[n-2]:

					raise ValueError(f'Experiment:pTRM_check;  Unrecognized set temperature sequence. Step {n}: {Set_Temps[n-3,0]:.2f}->{Set_Temps[n-1,0]:.2f}->{Set_Temps[n,0]:.2f}')


				# NRM remaining along Tb1 is controlled by MAX([Temps(n-3), Temps(n-1)])
				# NRM remaining along Tb2 is controlled by the n-2 step

				Tb1Max = np.max([Temps[n-3], Temps[n-1]])
				NRM_remain = Banc * N_orient * quad2d(blocking,Tb1Max, Tc, Temps[n-2], Tc)
				TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Tc, 0, Temps[n])
				Bres_remain  = Beff[n-1] * B_Angle[n-1,:] * quad2d(blocking,0, Tb1Max, Temps[n], Tc)
				TRM_remain = Beff[n-2] * B_Angle[n-2,:] * quad2d(blocking,Tb1Max, Tc, Temps[n], Temps[n-2])


			elif treatment[n-1] == 1: # previous step was a TRM remag - Not allowed
				raise ValueError(f'Experiment:pTRM_check; pTRM checks cannot follow TRM remagnetization steps. Step: {n}')
			else:
				raise ValueError(f'Experiment:pTRM_check;  Experiment style not modelled. Step: {n}')



			Mvec[n,:] = NRM_remain + TRM_gain + Bres_remain + TRM_remain



		## pTRM tail check step
		if treatment[n] == 3:


			if treatment[n-1] == 0: # previous step was a NRM demag step - Not allowed
				raise ValueError(f'Experiment:tail; pTRM tail checks cannot follow NRM demagnetization steps. Step: {n}')
			elif treatment[n-1] == 1: # previous step was a TRM step - ASSUME ZI STYLE SEQUENCE
				# DEMAG(T) -> REMAG(T) -> TAIL(T)

				# Check the treatments and set temperatures
				if treatment[n-2] == 0 and Set_Temps[n-2] == Set_Temps[n]:
					Tnrm = Temps[n-2]
				else:
					raise ValueError(f'Experiment:tail;  tail check sequence error. Step: {n}')


				if treatment[n-1] == 1 and Set_Temps[n-1] == Set_Temps[n]:
					Ttrm = Temps[n-1]
				else:
					raise ValueError(f'Experiment:tail;  tail check sequence error. Step: {n}')


				# NRM along Tb1 is controled by  Tnrm
				# Along Tb2 is is controlled by Ttrm

				if Temps[n] < Tnrm:

					# NRM along Tb1 is controled by  Tnrm
					# Along Tb2 is is controlled by Ttrm
					NRM_remain = Banc * N_orient * quad2d(blocking,Tnrm, Tc, Ttrm, Tc)

					# Residual field remanence gained in Beff(n)
					# Along Tb1 controlled by 0-> Temps(n)
					# Along Tb2 controlled by 0-> Tc
					TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Temps[n], 0, Tc)

					# Residual field remanence removed from Beff(n-2)
					# Along Tb1 is controlled by Temps(n) -> Tnrm
					# Along Tb2 is controlled by 0-> Tc
					Bres_remain = Beff[n-2] * B_Angle[n-2,:] * quad2d(blocking,Temps[n], Tnrm, 0, Tc)

					# TRM tail remaining from remag step
					# Along Tb1 is controlled by Tnrm -> Tc
					# Along Tb2 is controlled by 0-> Ttrm
					TRM_remain = Beff[n-1] * B_Angle[n-1,:] * quad2d(blocking,Tnrm, Tc, 0, Ttrm)

				else: # Tnrm <= Temps(n)

					# NRM along Tb1 is controled by  Temps(n)
					# Along Tb2 is is controlled by Ttrm
					NRM_remain = Banc * N_orient * quad2d(blocking,Temps[n], Tc, Ttrm, Tc)

					# Residual field remanence gained in Beff(n)
					# Along Tb1 controlled by 0-> Temps(n)
					# Along Tb2 controlled by 0-> Tc
					TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Temps[n], 0, Tc)

					# Residual field remanence removed from Beff(n-2)
					# All residual field remanence from demag step is removed
					Bres_remain = np.zeros((3,))

					# TRM tail remaining from remag step
					# Along Tb1 is controlled by Temps(n) -> Tc
					# Along Tb2 is controlled by 0-> Ttrm
					TRM_remain = Beff[n-1] * B_Angle[n-1,:] * quad2d(blocking,Temps[n], Tc, 0, Ttrm)

			elif treatment[n-1] == 5 and Set_Temps[n-1] == Set_Temps[n]: # I->Z steps at same temperature => Aitken/IZZI experiment or Thellier
				NRM_remain = Banc *N_orient  * quad2d(blocking,Temps[n], Tc, Temps[n-1], Tc)
				TRM_remain = Beff[n-1] * B_Angle[n-1,:] * quad2d(blocking,Temps[n], Tc, 0, Temps[n-1])
				Bres_remain = np.zeros((3,))  # Next higher demag step, so no residual field remanence from the previous demag step is remembered
				TRM_gain = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, Temps[n], 0, Tc)





			elif treatment[n-1] == 2: # previous step was a pTRM check
				raise ValueError(f'Experiment:tail;  Experiment style not modelled. Step: {n}')
			else: # Not accounted for
				raise ValueError(f'Experiment:tail;  Experiment style not modelled. Step: {n}')


			Mvec[n,:]= NRM_remain + TRM_gain + Bres_remain + TRM_remain



	Beff_mvec = Beff

	## Add on measurement and background noise
	# Measurement angle

	Mvec = Utilities.FishAngleNoise(kappa, length, Mvec)
	# Add background and measurement noise


	Mvec = BG + Mvec + RndMeas * np.sqrt( (Mvec * meas)**2 )

	treatment = np.array(treatment)
	#######################################################################################################################################################################



	## Anisotropy measurements
	if Anis_flag == 1:

		# Effective applied field orientaions - call as B_Angle(n,:)
		B_axes = np.vstack((np.eye(3), -np.eye(3)))
		B_axes = np.append(B_axes ,B_axes[0:1,:], axis = 0)
		nAxes = B_axes.shape[0]
		# Setup random varaibles for measurements
		# Residual fields
		Bres_rnd = rng.normal(size = (nAxes,1)) * res

		# Laboratory fields
		Blab_rnd = rng.normal(size = (nAxes,1)) *np.transpose(B)

		# Background noise -  Use Flab as approxiamtion
		# Scaled by FLab, which is the same as the total TRM
		BG = stats.cauchy.rvs(loc = background[0], scale = background[1], size = (nAxes*3,1) )*Blab/100
		# truncate excessive background
		for values in BG:
			if values > 0.8/100:
				values = 0.8/100
			if values < -0.8/100:
				values = -0.8/100

		BG = np.reshape(BG, (nAxes,3))
		BG=BG*BG_scale


		# Measurement noise
		RndMeas = rng.normal(size = (nAxes, 3))

		# Measurement angular noise
		Tmp_Angle = Utilities.FishAngleNoise(kappa, nAxes, B_axes)

		B_Angle = Utilities.create_array((nAxes, 3))
		for n in range(nAxes):
			tmp_BA = Anis_mat @ np.transpose(Tmp_Angle[n,:])
			tmp_BA = np.transpose(tmp_BA) * Anis_degree
			B_Angle[n,:] = np.transpose(np.linalg.solve(Anis_mat,np.transpose(tmp_BA)))


		# Do the six Treatment measurements
		Anis_vec = np.empty((nAxes, 3))
		Beff = Blab + Blab_rnd + Bres_rnd

		if NLT_flag == 1:
			Beff = NLT_a * np.tanh(NLT_b * Beff) # the non-linear transform

		for n in range(nAxes):
			Anis_vec[n,:] = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, 1-eps, 0, 1-eps)


		# Add on measurement and background noise
		# Measurement angle
		Anis_vec = Utilities.FishAngleNoise(kappa, nAxes, Anis_vec)
		# Add background and measurement noise
		Anis_vec = BG + Anis_vec + RndMeas * np.sqrt( (Anis_vec * meas)**2 )

	else:
		Anis_vec = []

	## Non-linear TRM measurements
	if NLT_flag == 1:

		# Setup random varaibles for measurements

		nFields = len(Bnlt)

		# Residual fields
		Bres_rnd = rng.normal(size = (nFields,1)) * res

		# Laboratory fields
		B = (Bnlt *(0.0075/100)).reshape((nFields,1))
		Blab_rnd = rng.normal(size = (nFields,1)) * B

		# Background noise -  Use Flab as approxiamtion
		# Scaled by an assumed average magnetization of 40
		BG = stats.cauchy.rvs(loc = background[0], scale = background[1], size = (nFields*3,1) )*Blab/100
		# truncate excessive background
		for values in BG:
			if values > 0.8/100:
	   			values = 0.8/100
			if values < -0.8/100:
				values = -0.8/100

		BG = np.reshape(BG, (nFields,3))
		BG = BG * BG_scale

		# Measurement noise
		RndMeas = rng.normal(size = (nFields, 3))

		# Measurement angular noise
		Tmp_Angle = Utilities.FishAngleNoise(kappa, nFields, T_orient)

		B_Angle = Utilities.create_array((nFields, 3))
		if Anis_flag == 1:

			for n in range(nFields):
				tmp_BA = Anis_mat @ np.transpose(Tmp_Angle[n,:])
				tmp_BA = np.transpose(tmp_BA) * Anis_degree
				B_Angle[n,:] = np.transpose(np.linalg.solve(Anis_mat, np.transpose(tmp_BA)))

		else:
			B_Angle = Tmp_Angle

		# Do the measurements
		NLT_vec = Utilities.create_array((nFields, 3))

		Beff = Bnlt.reshape((Bnlt.size,1)) + Blab_rnd + Bres_rnd
		Beff = NLT_a * np.tanh(NLT_b * Beff) # the non-linear transform

		for n in range(nFields):
			NLT_vec[n,:] = Beff[n] * B_Angle[n,:] * quad2d(blocking,0, 1-eps, 0, 1-eps)

		# Add on measurement and background noise
		# Measurement angle
		NLT_vec = Utilities.FishAngleNoise(kappa, nFields, NLT_vec)
		# Add background and measurement noise
		NLT_vec = BG + NLT_vec + RndMeas * np.sqrt( (NLT_vec * meas)**2 )
	else:
		NLT_vec = []


	return Mvec, temperatures, treatment, Beff_mvec, NLT_vec, Anis_vec
