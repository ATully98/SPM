"""
Python version of SD_MD_model.m
"""

# Import Required packages
import csv
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


def SD_MD_model_v4(Banc, Blab, Tc, Experiment, N_orient, T_orient, Beta_fits, Noise, Flags, P, Rot_Treatment, Rot_angle, NLT_a, NLT_b, Bnlt):

	"""
	Function to return output from a SD paleointensity experiment subject to random noise


	Inputs:
			Banc - (1, ) array value for the magnitude of the ancient B field in micro-tesela
			Blab - float value for the magnitude of the applied lab field in micro-tesela
			Tc - float value of the Curie temperature to be used
			Experiment - str path to experiment file
			N_orient - (1,3) array containing the vector of the ancient B field
			T_orient - (1, 3) array containing the vector direction of the applied lab field
			Beta_fits - (2, ) array containing the parameters a and b
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
	 		TempsN - (n, 1) array containing the temperature for each step with noise included
		 	treatment - (n, ) array containg treatment codes for each step
			Beff - (n, 1) array containing the effective field
			NLT_vec - (len(Bnlt)+1, 3) array with each row containing the vector for each NLT check step
		 	Anis_vec - (7, 3) array containing the anis vectors for each orientation
			"""
	# Setup errors and the experiment
	#read in Experiment file
	temps, treatment, Hx, Hy, Hz = Utilities.OpenExperiment(Experiment)

	# Extract flags
	Anis_flag = Flags[0]
	NLT_flag= Flags[1]
	NRflag = Flags[2]

	# Find number of steps in experiment
	steps = np.shape(temps)
	length = len(temps)

	# Temps list to pass to Errors setup
	ErrTs = temps[1:]
	ErrTs = np.vstack([[0],ErrTs])
	ErrTs = ErrTs/100

	# Set up noise parameters
	hold_time, cooling_rate, T_repeat, T_grad, T, B, res, meas, background, kappa = SetupErrors_v6_1.SetupErrors_v6(Tc, ErrTs, Blab)
	rng = default_rng()

	# These are used to override the angular errors
	theta_scale=1
	phi_scale=1
	BG_scale=1

	#################################################################################################################

	### Error Control
	### Uncomment each line to set individual error sources to zero - Used to
	### investigate indiviudual variances

	# set eps function to stop problems with beta function integration to 1
	eps = (np.finfo(float).eps)**(0.5)

	# If noise switched off, set all noise values to zero
	if Noise!=1:

		hold_time=np.zeros(steps)
		cooling_rate=np.zeros(steps)
		T_repeat=np.zeros(steps)
		T_grad=np.zeros(steps)

		res=0
		B=0
		theta_scale=0.0
		phi_scale=0.0
		meas=0
		BG_scale=0
		kappa = 0

	T=(hold_time**2 + cooling_rate**2 + T_repeat**2 + T_grad**2)**(0.5)


	# Generate random values
	# Temperatures
	# Points heating steps and 4 heatings (NRM, TRM, tail check, pTRM check)
	T_rnd= rng.normal(size=steps, scale=T)

	# Residual fields
	# Points heating steps and 4 heatings (NRM, TRM, tail check, pTRM check)
	Bres_rnd= rng.normal(size=steps, scale= res)

	# Laboratory fields
	# Points heating steps and 2 heating steps (TRM, pTRM check)
	Blab_rnd= rng.normal(size=steps, scale= B)

	# Background noise -  Use Blab as approxiamtion
	# Points heating steps x 3 axes x 4 measurement steps
	# Scaled by BLab, which is the same as the total TRM
	BG = stats.cauchy.rvs(loc = background[0], scale = background[1], size = (len(temps)*3,1) )*Blab/100

	# truncate excessive background
	for values in BG:
		if values > 0.8/100:
			values = 0.8/100
		if values < -0.8/100:
			values = -0.8/100
	# reshape and add noise
	BG = np.reshape(BG, (len(temps),3))
	BG=BG*BG_scale

	# Measurement noise
	# Random normal vector steps x 3 axes x 4 measurement steps
	RndMeas =  rng.normal(size=(length,3))*meas

	# Effective applied field orientaions
	Tmp_Angle = Utilities.FishAngleNoise(kappa, length, T_orient)

	#################################################################################################################################
	# New NLT and Anisotpy section
	# Set up rotation matrix, which represents the anisotropy axes

	if Anis_flag == 1:

		Anis_mat = Utilities.rotvec2mat(Rot_Treatment, Rot_angle)

		# Eigenvalues, the order does not matter
		# (Un)Comment to change grain shape
		Anis_degree = np.array(([P, 1, 1])) # Prolate
		# Anis_degree = np.array(([P, P, 1])) # Oblate
		# Anis_degree = np.array(([P, (P+1)/2, 1])) # Triaxial
		# Anis_degree = np.array(([P, (3*P+1)/4, 1])) # Prolate-Triaxial
		# Anis_degree = np.array(([P, (P+3)/4, 1])) # Oblate-Triaxial

		NO = N_orient.reshape(3,)

		tmp_NO = Anis_mat @ np.transpose(NO) # Rotate NRM to eigenvector coords

		tmp_NO = np.transpose(tmp_NO) *Anis_degree  # Scale by the eigenvakues
		N_orient = np.transpose(np.linalg.solve(Anis_mat,np.transpose(tmp_NO))) # Rotate back into sample coords
		B_Angle = np.empty((length, 3))
		B_Angle[:] = np.nan
		for n in range(length):
			tmp_BA = Anis_mat @ np.transpose(Tmp_Angle[n,:])
			tmp_BA = np.transpose(tmp_BA) * Anis_degree
			B_Angle[n,:] = np.transpose(np.linalg.solve(Anis_mat,np.transpose(tmp_BA)))

	else:
		B_Angle = Tmp_Angle

	#################################################################################################################################

	## Define the random temps/fields etc

	# Field strength - orientation is in B_Angle
	Beff= Blab + Blab_rnd + Bres_rnd

	#   set B field to zero for outfield and tail checks
	#   set to negative for reverse field
	for i in range(len(treatment)):
		if treatment[i] == 0 or treatment[i] == 3:
			Beff[i]=Bres_rnd[i]
		elif treatment[i] == 5:
			Beff[i] = -Beff[i]	  #backfield step


	Beff[0]=Banc # this is not used, but added to keep thing cosistent


	# Beff for demag
	if np.sum(treatment[1:] ) == 0:
		Beff[0] = (0 + Blab_rnd[0]) + Bres_rnd[0]

	## Non-Linear TRM
	if NLT_flag == 1:
		Beff = NLT_a * np.tanh(NLT_b * Beff)
		Banc = NLT_a * np.tanh(NLT_b * Banc)

	# Remember the set temperatures
	Set_Temps=temps
	# Add noise to the temperatures
	TempsN=(temps/100) + T_rnd

	# NR to be included at future date
	if NRflag==1:

		warning("Non-Reciprocity Handling is Currently Under Construction and Will NOT be implemented")

	#   Take off eps from maximum temperatures
	for i in range(len(TempsN)):
		if TempsN[i] >= 1:
			TempsN[i] = 1- eps # eps to prevent numerical errors
	TempsN[0] = 1- eps # Remove noise from the NRM acquisition step

	#######################################################################################################################################

	### Calculate the vectors

	# Initialize measurements vectors
	Mvec = np.empty((length,3))
	Mvec[:] = np.NaN

	# Create the initail NRM
	Mvec[0,:] = Banc*N_orient* quad(Beta_fits, 0, TempsN[0])


	# Do the experiment
	# Loop through experiment steps
	for n in range(1,len(TempsN)):

		# NRM Demag steps
		if  treatment[n]==0:   #NRM Demag
			if n==1:  # If start of Experiment
				Mvec[n,:]=Mvec[n-1,:] - Banc* N_orient*  quad(Beta_fits, 0, TempsN[n])   + Beff[n] *  quad(Beta_fits, 0, TempsN[n]) * B_Angle[n,:]



			#CAUTION Alex added this bit
			#Used for basic IZZI experiment for back to back zero field steps

			elif treatment[n-1] == 0 and treatment[n]==0:

				if treatment[n-2] != 1:
					# Probably doing a demag
					raise ValueError('Experiment: NRM;  Sequence not Modelled')

				if TempsN[n] >= TempsN[n-1]:
					# TODO: Check this is correct

					# NRM_remain = Fanc.*N_orient .* quad2d(@(x1, x2) MDblocking(x1, x2, a, b, lambda, alph4), Temps(n), Tc, Temps(n-2), Tc, 'AbsTol', Atol, 'RelTol', Rtol)
					# TRM_remain = Feff(n-2).*F_Angle(n-2,:) .* quad2d(@(x1, x2) MDblocking(x1, x2, a, b, lambda, alph4), Temps(n), Tc, 0, Temps(n-2), 'AbsTol', Atol, 'RelTol', Rtol)
					# Fres_remain = [0,0,0] # Next higher step, so no residual field remanence from the previous demag step is remembered
					# TRM_gain = Feff(n).*F_Angle(n,:) .* quad2d(@(x1, x2) MDblocking(x1, x2, a, b, lambda, alph4), 0, Temps(n), 0, Tc, 'AbsTol', Atol, 'RelTol', Rtol)


					# Mvec[n,:] = Banc * N_orient * quad(Beta_fits, TempsN[n], Tc)  + Beff[n-2]  * B_Angle[n-2,:] * quad(Beta_fits, TempsN[n]) + Fres_remain + TRM_gain



					Mvec[n,:]=Banc *N_orient* quad(Beta_fits, TempsN[n], 1- eps)+ Beff[n] * quad(Beta_fits, 0, TempsN[n]) *B_Angle[n, :]
					print("CAUTION: Check Mvec calculation")



			elif treatment[n-1] == 1  and treatment[n] == 0: #IZ step

				if TempsN[n-1] <= TempsN[n]:
					#current step is higher/equal
					#All TRM removed
					TRM_demag = Beff[n-1] *  quad(Beta_fits, 0, TempsN[n-1]) * B_Angle[n-1,:]
					#excess NRM demag between Tmax and TempsN[n]



					Tmax= np.amax(TempsN[1:n])	#NRM remaining is blocked between the maximum previous temperature and Tc
					NRM_demag=Banc *N_orient * quad(Beta_fits, Tmax, TempsN[n])

				else:
					# Current step is lower
					# TRM partially removed
					TRM_demag = Beff[n-1] *  quad(Beta_fits, 0, TempsN[n]) * B_Angle[n-1,:]
					#no excess NRM Demag
					NRM_demag = [0,0,0]

				#		 previous Magnetisation vector   TRM demag'd  ExceesNRM demag   Residual fields
				Mvec[n,:]=		 Mvec[n-1,:]		 - TRM_demag -   NRM_demag +	   Beff[n] *  quad(Beta_fits, 0, TempsN[n]) * B_Angle[n,:]

			elif treatment[n-1]==2:	#for IZZI model
				#Assumes that we are at the next highest temperature steps
				#Threfore decribe Mvec(n, :) in terms of NRM remaining, NOT
				#componentes demagnetized as for other cases




				if treatment[n-2]!=0 or Set_Temps[n] < max(Set_Temps[1:n-1]):
					raise ValueError(f"Experiment:NRM; Experiment style not modelled. Step/treatment/previous: {n}/ {treatment[n]}/ {treatment[n-1]}")


				Mvec[n,:]=Banc *N_orient *  quad(Beta_fits, TempsN[n], 1- eps ) +Beff[n]* quad(Beta_fits, 0, TempsN[n])*B_Angle[n,:]


			elif treatment[n-1]==3:
				# ASSUMES THAT WE ARE AT THE NEXT HIGHEST TEMPERATURE STEP
				# Therefore decribe Mvec(n, :) in terms of NRM remaining, NOT
				# componentes demagnetized as for other cases

				if Set_Temps[n] < max(Set_Temps[1:n-1]):
					raise ValueError("NRM:Temperature; Model assumes NRM demag following a pTRM tail check is to the next highest temperature")

				Mvec[n,:]=Banc *N_orient* quad(Beta_fits, TempsN[n], 1- eps)+ Beff[n] * quad(Beta_fits, 0, TempsN[n]) *B_Angle[n, :]

			else:
				raise ValueError(f"Experiment:NRM; Experiment style not modelled. Step/treatment/previous: {n}/{treatment[n]}/{treatment[n-1]}")


		#######################################################################################################################################################################


		# TRM remag step

		if treatment[n] == 1:  #TRM remag
			if treatment[n-1] == 0:  #previous step was demag
				# WORKS FOR BOTH ZI AND IZ EXPERIMENTS

				# If overheat then excess NRM is demagnetized
				#All residual field remanence is demagnetized
				if TempsN[n-1] <= TempsN[n]:  #   This is also true for IZ Set_Temps

					if len(TempsN[1:n-1]) < 2:
						Tmax =TempsN[n-1]


					else:
						Tmax=np.amax(TempsN[1:n]) # NRM remaining is blocked between the maximum previous temperature and Tc
					NRM_demag1=Banc *N_orient * quad(Beta_fits, Tmax, TempsN[n])+ Beff[n-1]* quad(Beta_fits, 0, TempsN[n-1] )*B_Angle[n-1,:]



				else: #Underheat - No excess NRM removed
						#Only residual remanence upto TempsN[n] is removed
					NRM_demag1= Beff[n-1]* quad(Beta_fits, 0, TempsN[n]) *B_Angle[n-1,:]

							#NRM remaining																	  #Excess NRM demagnetised
				Mvec[n,:] = Mvec[n-1,:] + Beff[n]* quad(Beta_fits, 0, TempsN[n] )*B_Angle[n,:] - NRM_demag1





			elif treatment[n-1] == 2 and treatment[n-2] !=5:	# previous step was a pTRM check NOT THELLIER!


					# Here we assume previous demag to Temps(n) occurs at
					# Temps(n-2) and may be a NRM deamg (0) or tail check(3).
					# The pTRM check does not have an effect, it is at a low
					# temperature and will be complete demagnetized. The largest of
					# the NRM/TRM/tail steps will control the excess NRM demagnetized


				if treatment[n-2] == 0:
					T_old = TempsN[n-2]

					if T_old <= TempsN[n]:
						# If overheat then excess NRM is demagnetized
						# All residual field remanence is demagnetized
						NRM_demag=Banc *N_orient *quad(Beta_fits, T_old, TempsN[n]) +Beff[n-2]*quad(Beta_fits, 0, T_old) *B_Angle[n-2,:]
					else:
						# Underheat - No excess NRM removed
						# Only resdiual remanence upto Temps(n) is removed
						NRM_demag = Beff[n-2] *quad(Beta_fits, 0, TempsN[n] ) *B_Angle[n-2,:]

					# All pTRM from the check is removed
					# Must been removed as residual field will be different
					TRM_demag = Beff[n-1] * quad(Beta_fits, 0, TempsN[n-1] ) *B_Angle[n-1,:]

				elif treatment[n-2] == 3 and treatment[n-3] == 5:   # Thellier
					Tdir=TempsN[n-4]
					Tinv=TempsN[n-3]
					Ttail=TempsN[n-2]

					Tmax=max([Tdir, Tinv, Ttail])
					# NRM demag is controlled by max(Tdir, Tinv, Ttail)
					NRM_demag=Banc *N_orient *quad(Beta_fits, Tmax, TempsN[n])

					#pTRM check is completely demag'd
					check_demag=Beff[n-1] *quad(Beta_fits, 0, TempsN[n-1]) *B_Angle[n-1,:]
					 # parts of either inverse or direct TRM will remain and
					 # need to be removed
					if Tdir <= Tinv:
						# All direct TRM removed by inverse step and only
						# residual inverse needs to be removed
						# blocked between Temps(n-1) and Tinv
						dir_demag=[0,0,0]
						inv_demag = Beff[n-3] *quad(Beta_fits, TempsN[n-1], Tinv) *B_Angle[n-3, :]

					else: #Tinv < Tdir
						# direct TRM removed between Tinv and Tdir
						dir_demag=Beff[n-4] *quad(Beta_fits, Tinv, Tdir) *B_Angle[n-4, :]
						#inverse TRM is removed between the check and Tinv
						inv_demag=Beff[n-3] *quad(Beta_fits, TempsN[n-1], Tinv) *B_Angle[n-3, :]

					TRM_demag = check_demag + dir_demag + inv_demag

				else:
					raise ValueError(f"Experiment:TRM; Unrecognized previous step for TRM step. Step: {n}")



				Mvec[n,:] = (Mvec[n-1,:] +								# Previous magnetization
							Beff[n] * quad(Beta_fits, 0, TempsN[n]) *B_Angle[n,:]	# pTRM gained
							- NRM_demag										 # Excess NRM demagnetized
							- TRM_demag)										  # pTRM check demagnetized


			elif treatment[n-1] == 2 and treatment[n-2]==5:
				#  Do a Thellier step seperate - to avoid confusion
				#  ASSUMES THAT THE EXPERIMENT HAS MOVED TO THE NEXT HIGHER
				#  TEMPERATURE STEP
				#
				#  Find the temperature of the previous steps direct and inverse
				# field steps - they control the NRM demag
				#			  ind=find(treatment(2:n-1)==1, 1, 'last');
				#			  Tdir=Temps(ind);
				#			ind=find(treatment(2:n-1)==5, 1, 'last');
				#			 Tinv=Temps(ind);
				#
				#			  Tmax=max([Tdir, Tinv]); % NRM remaining is blocked between the maximum previous temperature and Tc
				# NRM demag is controlled between Tmax and Temps(n)
				#			  NRM_demag=Fanc.*quad(Pfun, Tmax, Temps(n)).*N_orient;
				#  All pTRM check is demag'd
				#			  pTRM_demag=Feff(n-1).*quad(Pfun, 0, Temps(n-1)).*B_Angle(n-1,:);
				#
				#  Inverse TRM AND/OR direct TRM are also removed - this step
				#  would be best written in terms of NRM remaining and new pTRM
				#  gained

				Mvec[n, :]=Banc *N_orient *quad(Beta_fits, TempsN[n], 1-eps) + Beff[n] *quad(Beta_fits, 0, TempsN[n]) *B_Angle[n, :]


			elif treatment[n-1]==3:
					# previous step was a pTRM check
					#ASSUMES THAT THE EXPERIMENT HAS MOVED TO THE NEXT HIGHER
					# TEMPERATURE STEP

				Tmax=np.amax(TempsN[1:n]) # NRM remaining is blocked between the maximum previous temperature and Tc

				Mvec[n,:] = (Mvec[n-1,:]								  # Previous magnetization
								- Banc *N_orient *quad(Beta_fits, Tmax, TempsN[n])	# NRM demagnetized
								+ Beff[n] *quad(Beta_fits, 0, TempsN[n]) *B_Angle[n, :])  # pTRM gained

			elif treatment[n-1] == 5: # Thellier-type
				if treatment[n-2] == 2:
					#do something

					raise ValueError(f"Experiment:TRM; This style of Thellier experiment is not yet modelled. Step number/type: {n}/{treatment[n-1]}")

				elif treatment[n-2] == 5:
					# % Find the temperature of the previous steps direct and inverse
					# % field steps - they control the NRM demag
					# %			 ind=find(treatment(2:n-1)==1, 1, 'last');
					# %			 Tdir=Temps(ind);
					# %			 ind=find(treatment(2:n-1)==5, 1, 'last');
					# %			 Tinv=Temps(ind);
					# %
					# %			 Tmax=max([Tdir, Tinv]); % NRM remaining is blocked between the maximum previous temperature and Tc

					Mvec[n, :] = Banc* N_orient *quad(Beta_fits, TempsN[n], 1-eps) + Beff[n] * quad(Beta_fits, 0, TempsN[n]) *B_Angle[n, :]

			elif treatment[n-1] == 1 and n == 1: # Aitken or IZZI experiment starting with an IZ step

				Mvec[n,:] =( Mvec[n-1,:]										  # Previous magnetization
					+ Beff[n] *(quad(Beta_fits, 0, TempsN[n])) *B_Angle[n,:]	# pTRM gained
					- Banc*quad(Beta_fits, 0, TempsN[n])*N_orient)				# NRM demagnetized


			# TODO: CHECK this is correct
			#CAUTION Alex added this bit
			elif treatment[n-1] == 1  and n != 1: # IZZI basic repeat infields


				T_old = TempsN[n-2]

				if T_old <= TempsN[n]:
					# If overheat then excess NRM is demagnetized
					# All residual field remanence is demagnetized
					NRM_demag=Banc *N_orient *quad(Beta_fits, T_old, TempsN[n]) +Beff[n-2]*quad(Beta_fits, 0, T_old) *B_Angle[n-2,:]
				else:
					# Underheat - No excess NRM removed
					# Only resdiual remanence upto Temps(n) is removed
					NRM_demag = Beff[n-2] *quad(Beta_fits, 0, TempsN[n] ) *B_Angle[n-2,:]

				# All pTRM from the check is removed
				# Must been removed as residual field will be different
				TRM_demag = Beff[n-1] * quad(Beta_fits, 0, TempsN[n-1] ) *B_Angle[n-1,:]










				Mvec[n,:] = (Mvec[n-1,:] +								# Previous magnetization
							Beff[n] * quad(Beta_fits, 0, TempsN[n]) *B_Angle[n,:]	# pTRM gained
							- NRM_demag										 # Excess NRM demagnetized
							- TRM_demag)										  # pTRM check demagnetized




			else: # Not accounted for

				raise ValueError(f"Experiment:TRM; Experiment style not modelled. Step number/type: {n}/{treatment[n-1]}")

		#######################################################################################################################################################################

		#pTRM check step

		if treatment[n] == 2:
			#		 disp(num2str(Feff(n)))
			if treatment[n-1] == 0 or   treatment[n-1] == 3: # previous step was a NRM demag step
				# Since pTRM checks are to a much lower temperature we don't
				# need to worry about excess NRM demagentization, only residual
				# field demag

				Mvec[n, :]= (Mvec[n-1,:]									   #Exisiting magnetization
							- Beff[n-1] *quad(Beta_fits, 0, TempsN[n]) *B_Angle[n-1,:] #  Residual field demag
							+ Beff[n]*quad(Beta_fits, 0, TempsN[n]) *B_Angle[n,:]  )	   # pTRM gained

			elif treatment[n-1] == 1: # previous step was a TRM step - Not allowed
				raise ValueError(f'Experiment:pTRM_check; pTRM checks cannot follow TRM remagnetization steps. Step: {n}')

			elif treatment[n-1] == 5: # Thellier-Thellier

				Mvec[n,:] = (Mvec[n-1,:]									   # Exisiting magnetization
							- Beff[n-1] *quad(Beta_fits, 0, TempsN[n]) *B_Angle[n-1, :]  # TRM demag
					+ Beff[n]*quad(Beta_fits, 0, TempsN[n]) *B_Angle[n, :] )	   # pTRM gained


				#			error('Experiment:pTRM_check', 'Thellier-Thellier experiments not yet modelled. Step: %d', n);
			else:
				# Not accounted for
				raise ValueError(f'Experiment:pTRM_check; Experiment style not modelled. Step:  {n}')

		#######################################################################################################################################################################
		## pTRM tail step
		if treatment[n] == 3:

			if treatment[n-1] == 0: # previous step was a NRM demag step - Not allowed
				raise ValueError(f'Experiment:tail; pTRM tail checks cannot follow NRM demagnetization steps. Step: {n}')
			elif treatment[n-1] == 1: # previous step was a TRM step - ASSUME COE STYLE SEQUENCE

				if treatment[n-2] == 0:
					Tnrm=TempsN[n-2]

				else:
					raise ValueError(f'Experiment:tail; tail check sequence error. Step: {n}')


				if treatment[n-1] == 1:
					Ttrm=TempsN[n-1]

				else:
					raise ValueError(f'Experiment:tail; tail check sequence error. Step: {n}')

				# Tnrm or Ttrm control the NRM remaining before this step
				if Ttrm < Tnrm:

					#				 if n==4 disp('Ttrm < Tnrm'); end
					if TempsN[n] < Tnrm:

						if TempsN[n] < Ttrm:

							# Temps(n) < Ttrm < Tnrm
							NRM_demag=[0, 0, 0] # No excess NRM demag
							#TRM partially removed
							TRM_demag = Beff[n-1] *quad(Beta_fits, 0, TempsN[n] ) *B_Angle[n-1,:]
						else: # TempsN(n) >= Ttrm

							# No excess NRM removed, but residual field
							# remanence between Ttrm and Temps is removes
							NRM_demag = Beff[n-2] *quad(Beta_fits, Ttrm, TempsN[n]) *B_Angle[n-2,:]  # resdiual fields from the NRM step
							#TRM all removed
							TRM_demag = Beff[n-1] *quad(Beta_fits, 0, Ttrm ) *B_Angle[n-1,:]



					else: # TempsN(n) >= Tnrm
						#Ttrm < Tnrm <= Temps
						# Temps controls

						NRM_demag = (Banc* N_orient* quad(Beta_fits, Tnrm, TempsN[n])   #   excess NRM
									   + Beff[n-2] *quad(Beta_fits, Ttrm, Tnrm) *B_Angle[n-2,:])  # resdiual fields from the NRM step
						#TRM all removed
						TRM_demag= Beff[n-1]*quad(Beta_fits, 0, Ttrm ) *B_Angle[n-1,:]


				else: # Ttrm >= Tnrm
					#				if n==4 disp('Ttrm >= Tnrm'); end

					if TempsN[n] < Ttrm:
						if TempsN[n] < Tnrm:

							# Temps(n) < Tnrm <=Ttrm
							# Ttrm controls nrm demag so no excess removed
							NRM_demag = [0,0,0]
							# only partial TRM removed upto Temps(n)
							TRM_demag = Beff[n-1] *quad(Beta_fits, 0, TempsN[n] ) *B_Angle[n-1,:]



						else: #Temps(n) >=Tnrm
							# Ttrm controls nrm demag so no excess removed
							NRM_demag=[0,0,0]
							# only partial TRM removed upto Temps(n)
							TRM_demag = Beff[n-1] *quad(Beta_fits, 0, TempsN[n] ) *B_Angle[n-1,:]


					else: # Temps(n) >= Ttrm

						#Tnrm <= Ttrm <= Temps
						# Temps controls excess NRM, between Ttrm and Temps
						# No residual field demag - removed by TRM step
						NRM_demag = Banc *N_orient *quad(Beta_fits, Ttrm, TempsN[n])	# excess NRM
						#TRM all removed
						TRM_demag = Beff[n-1] *quad(Beta_fits, 0, Ttrm ) *B_Angle[n-1,:]


				# end temperature flow
				Mvec[n,:] = Mvec[n-1,:] - NRM_demag - TRM_demag + Beff[n] *quad(Beta_fits, 0, TempsN[n]) *B_Angle[n,:]

			elif treatment[n-1] == 5: # Doing a Thellier experiment
				# THIS ASSUMES THAT PREVIOUS TWO STEPS WERE DIRECT THEN INVERSE
				Tdir=TempsN[n-2]
				Tinv=TempsN[n-1]


				if Tdir < Tinv:
					# All direct TRM has been removed
					# NRM remaining is controlle by max(Tinv, Temps)

					if Tinv < TempsN[n]:
							#Tdir < Tinv < Temps
							# NRM demag between Tinv and Temps
						NRM_demag = Banc* N_orient * quad(Beta_fits, Tinv, TempsN[n])
							# All inverse TRM removed
						TRM_demag = Beff[n-1] *quad(Beta_fits, 0, Tinv) *B_Angle[n-1,:]

					else: # Tinv >= Temps(n)
							# Tdir < Temps <= Tinv
							# NRM demag is controlled by Tinv - No excess removed
							NRM_demag = [0,0,0]
							# Inverse TRM is partially removed upto Temps(n)
							TRM_demag = Beff[n-1] * quad(Beta_fits, 0, TempsN[n]) *B_Angle[n-1,:]

				else: # Tdir >= Tinv OR Tinv <= Tdir

					if Tdir < TempsN[n]:

						if TempsN[n] < Tinv:
							# Temps < Tinv <= Tdir
							# NRM is controlled by Tdir - No excess demag
							NRM_demag = [0,0,0];
							# Inverse TRM is partially removed, residual direct
							# TRM remains, BUT this is already present in the
							# magnetization vector
							TRM_demag = Beff[n-1] * quad(Beta_fits, 0, TempsN[n]) *B_Angle[n-1,:]

						else: # Temps(n) >= Tinv
							#Tinv <= Temps(n) <= Tdir
							# NRM is controlled by Tdir ot TEMPS - BUT No excess demag
							NRM_demag = [0,0,0]
							# Inverse TRM is completly removed
							Inv_demag = Beff[n-1] *quad(Beta_fits, 0, Tinv) *B_Angle[n-1,:]

							# Direct TRM is partially removed between Tinv and
							# Temps(n)
							Dir_demag = Beff[n-2] *quad(Beta_fits, Tinv, TempsN[n]) *B_Angle[n-2, :]

							TRM_demag = Inv_demag + Dir_demag

					else: # Tdir >= Temps(n)
					# Tinv <= Tdir < =Temps(n)
					# Excess NRM demag between Tdir and Temps(n)
						NRM_demag = Banc *quad(Beta_fits, Tdir, TempsN[n]) *N_orient

						# All inverse TRM is remove
						Inv_demag = Beff[n-1] * quad(Beta_fits, 0, Tinv) * B_Angle[n-1, :]

						# Residual direct TRM is also removed, between Tinv and
						# Tdir
						Dir_demag = Beff[n-2] *quad(Beta_fits, Tinv, Tdir) *B_Angle[n-2,:]

						TRM_demag = Inv_demag + Dir_demag


				Mvec[n,:] = (Mvec[n-1,:]									  # Previous magnetization vector
								- TRM_demag											# TRM demagnetized
								- NRM_demag											# Excess NRM demagnetized
								+ Beff[n] *quad(Beta_fits, 0, TempsN[n]) *B_Angle[n, :])	  #Residual field remanence gained


			elif treatment[n-1] == 2: # previous step was a pTRM check
					# do something different
				raise ValueError(f"Experiment:tail; Experiment style not modelled. Step: {n}")
			else: # Not accounted for
				raise ValueError(f"Experiment:tail; Experiment style not modelled. Step: {n}")


		#######################################################################################################################################################################
		# Inverse TRM step - Thellier-Thellier
		if treatment[n] == 5:
			# THIS ASSUMES THAT THE INVERSE STEP ALWAYS FOLLOWS THE DIRECT STEP
			if treatment[n-1] == 1:
				# NRM remaining
					 #Tmax=max(Temps(2:n-1));

				if TempsN[n-1] < TempsN[n]:
					# NRM demag between Tmax and Temps(n)
					NRM_demag = Banc* N_orient * quad(Beta_fits, TempsN[n-1], TempsN[n])
					# All TRM removed
					TRM_demag = Beff[n-1] *quad(Beta_fits, 0, TempsN[n-1]) *B_Angle[n-1,:]
				else: #Temps(n) <= Temps(n-1) - underheat
					# No excess NRM demag
					NRM_demag = [0,0,0]
					# TRM partially removed up to Temps(n)
					TRM_demag = Beff[n-1]*quad(Beta_fits, 0, TempsN[n]) *B_Angle[n-1,:]


				Mvec[n,:] = (Mvec[n-1,:]									  # Previopus magnetization vector
							- TRM_demag										   # TRM demagnetized
								- NRM_demag											# Excess NRM demagnetized
								+ Beff[n]*quad(Beta_fits, 0, TempsN[n]) *B_Angle[n, :] )	 # Inverse TRM gained



				#		 elseif treatment(n-1)==2
				#			  error('Experiment:Thellier', 'not YET modelled. Step: %d', n);

				#		 elseif treatment(n-1)==3
			else:
				raise ValueError(f"Experiment:Thellier; Experiment style not modelled. Step: {n}")


		#######################################################################################################################################################################

	### Add on measurement and background noise

	#Measurement angle

	Mvec = Utilities.FishAngleNoise(kappa, length, Mvec)

	# Add background and measurement noise
	Mvec = BG + Mvec + RndMeas *( (Mvec)**2 )**(0.5)

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

		# # Measurement angular noise
		# Angle_rnd = stats.weibull_min.rvs( theta[1], loc = 0, scale = theta[0], size = (nAxes,1))*theta_scale
		# # Treatment of angular rotation
		#
		# RotTreatment_rnd = rng.normal(size = (nAxes,3))
		#
		# # Field angular noise
		# B_Angle_rnd = stats.weibull_min.rvs( phi[1], loc = 0, scale = phi[0], size = (nAxes,1))*phi_scale
		#
		# # Treatment of angular rotation
		# # Points heating steps x 3 axes x 4 steps (NRM resid, TRM, MD resid, pTRM)
		# B_RotTreatment_rnd = rng.normal(size = (nAxes,3))




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
			Anis_vec[n,:] = Beff[n] * B_Angle[n,:] * quad(Beta_fits, 0, 1-eps)

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

		# # Measurement angular noise
		# Angle_rnd = stats.weibull_min.rvs( theta[1], loc = 0, scale = theta[0], size = (nFields, 1))*theta_scale
		#
		# # Treatment of angular rotation
		# RotTreatment_rnd = rng.normal(size = (nFields,3))
		#
		# # Field angular noise
		# B_Angle_rnd = stats.weibull_min.rvs( phi[1], loc = 0, scale = phi[0], size = (nFields, 1))*phi_scale
		#
		# # Treatment of angular rotation
		# # Points heating steps x 3 axes x 4 steps (NRM resid, TRM, MD resid, pTRM)
		# B_RotTreatment_rnd = rng.normal(size = (nFields,3))

		# Effective applied field orientaions - call as B_Angle(n,:)

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
			NLT_vec[n,:] = Beff[n] * B_Angle[n,:] * quad(Beta_fits, 0, 1-eps)


		# Add on measurement and background noise
		# Measurement angle
		NLT_vec = Utilities.FishAngleNoise(kappa, nFields, NLT_vec)
		# Add background and measurement noise
		NLT_vec = BG + NLT_vec + RndMeas * np.sqrt( (NLT_vec * meas)**2 )
	else:
		NLT_vec = []

	return Mvec, TempsN, treatment, Beff, NLT_vec, Anis_vec
