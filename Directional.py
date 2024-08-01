# %MAD_anc	  -   the maximum angular deviation of the anchored PCA directional fit
# % MAD_free	 -   the maximum angular deviation of the free-floating PCA directional fit
# % alpha		-   the angle between the anchored and free-floating PCA directional fits
# % alpha_prime  -   the angle between the anchored PCA directional fit and the true NRM direction (assumed to be well known)
# % DANG		 -   the deviation angle (Tauxe & Staudigel, 2004; G-Cubed)
# % Theta		-   the angle between the applied field and the NRM direction (determined as a free-floating PCA fit to the TRM vector)
# % a95		  -   the alpha95 of the Fisher mean of the NRM direction of the best-fit segment
# % CRM_R		-   the potential CRM% as defined by Coe et al. (1984; JGR)

import numpy as np
from numpy import vstack
import Utilities



def direction_stats(Mvec, Temps, Treatment, Blab, Blab_orient, ChRM, Az, Pl, Flags, start_pt, end_pt):

	A_flag = Flags[0]
	NLT_flag = Flags[1]
	NRflag = Flags[2]
	NRM_rot_flag = Flags[3]


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

	seg_min = start_pt
	seg_max = end_pt
	seg = np.arange(seg_min, seg_max+1,1)
	Seg_Ends = np.array((seg_min, seg_max))

	Ypts=(np.sum((NRMvec[:,1:].reshape((len(NRMvec),3)))**2, axis = 1))**(0.5)







	## Directional stats
	Decs, Incs, Ints = Utilities.cart2dir(NRMvec[seg,1:2], NRMvec[seg,2:3], NRMvec[seg,3:4])

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

		# Angle between measured NRM and Blab
		NRMhat = np.empty((3,))
		NRMhat[:] = np.nan
		NRMhat[0], NRMhat[1], NRMhat[2] = Utilities.dir2cart(Dec_F, Inc_F)
		Theta = Utilities.rad2deg( np.arctan2(np.linalg.norm(np.cross(NRMhat, Blab_orient)), np.dot(NRMhat, Blab_orient[0])) )

		# Get the Fisher Mean and stats [Mdec, Minc, k, a95, R]

	a95 = Utilities.FisherMeanDir(Decs, Incs)[3]


	# # Coe et al. (1984) CRM parameter
	# if ChRM == [] or np.isnan(np.sum(ChRM)) == True:
	# 	# Need the definition of ChRM
	# 	CRM_R = np.nan
	# else:
	#
	# 	if NRM_rot_flag == 1: # We need to also rotate the Blab vector into geographic coords
	# 		tmp_D, tmp_I = Utilities.cart2dir(Blab_orient[0,0], Blab_orient[0,1], Blab_orient[0,2])[0:2]
	# 		tmp_D, tmp_I = Utilities.dirot(tmp_D, tmp_I, Az, Pl)
	# 		tmp_O = np.empty((3,))
	# 		tmp_O[:] = np.nan
	# 		tmp_O[0], tmp_O[1], tmp_O[2] = Utilities.dir2cart(tmp_D, tmp_I, Mag = 1)
	# 		phi2=( np.arctan2(np.linalg.norm(np.cross(ChRM, tmp_O)), np.dot(ChRM[0], tmp_O)) )
	# 	else:
	# 		phi2 = ( np.arctan2(np.linalg.norm(np.cross(ChRM, Blab_orient)), np.dot(ChRM[0], Blab_orient[0])) )
	#
	# 	fit_vec = np.empty((len(Rot_Decs),3))
	# 	fit_vec[:,0:1], fit_vec[:,1:2], fit_vec[:,2:3] = Utilities.dir2cart(Rot_Decs, Rot_Incs, Mag = 1) # Even if we don't rotate Rot_Decs/Incs contains the unrotated directions
	# 	CRM = np.empty((fit_vec.shape[0], 1))
	# 	CRM[:] = np.nan  # create an empty vector
	#
	# 	for j in range(np.shape(fit_vec)[1]):
	# 		phi1 = ( np.arctan2(np.linalg.norm(np.cross(fit_vec[j,:], ChRM)), np.dot(fit_vec[j,:], ChRM[0])) )
	# 		CRM[j,0] = Ypts[j]*np.sin(phi1)/np.sin((phi2))
	#
	# 	CRM_R=100*max(CRM)/(Delta_x_prime)
	CRM_R = np.nan


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
	if isinstance(CRM_R, int) == True or isinstance(CRM_R, float) == True:
		CRM_R = np.round(10 * CRM_R)/10
	else:
		CRM_R = np.round(10 * CRM_R.item())/10


	Meas_Data = Mvec
	Meas_Treatment = Treatment
	Meas_Temp = Temps
	Temp_steps = NRMvec[:,0]
	n = len(Temp_steps)
	Tmin = NRMvec[seg_min, 0]
	Tmax = NRMvec[seg_max, 0]
	Banc = 0.0

	Params = Meas_Data, Meas_Treatment, Meas_Temp, Blab_orient, Temp_steps, NRMvec, Blab, n, Seg_Ends, Tmin, Tmax, Dec_A, Inc_A, MAD_anc, Dec_F, Inc_F, MAD_free, alpha, alpha_prime, DANG,  Theta, a95,  CRM_R, Banc



	stats = {"Meas_Data": Meas_Data,"Meas_Treatment" : Meas_Treatment , "Meas_Temp": Meas_Temp, "Blab_orient" : Blab_orient, "Temp_steps" : Temp_steps, "NRMvec" : NRMvec, "Blab" : Blab, "n": n, "Seg_Ends" : Seg_Ends, "Tmin" : Tmin, "Tmax" : Tmax, "Dec_A": Dec_A, "Inc_A": Inc_A, "MAD_anc" : MAD_anc, "Dec_F": Dec_F, "Inc_F": Inc_F, "MAD_free" : MAD_free, "alpha": alpha, "alpha_prime" : alpha_prime, "DANG" : DANG,  "Theta": Theta, "a95" : a95,  "CRM_R": CRM_R, "Banc": Banc}

	return stats
