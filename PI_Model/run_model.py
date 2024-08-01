import MD_model_v4
import SD_MD_model_v4
import Utilities
import GetPintParams_v7
import numpy as np
import warnings
import os
import os.path
import pandas as pd
import time
import scipy as sp
from scipy import stats
import tdt2model
import Control_Run
import Free_Run
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
from numpy.random import default_rng





def RunModelFunc(j, Experiment, Blab, Banc, T_orient, Tc, Noise, Flags, P, NLT_a, NLT_b, Bnlt,  Az, Pl, beta_T, csv_file, Control, Free, ID, N, Experiment_Key,  field_fits, now, folder,rootfolder, Beta_fits, lam):
	rng = default_rng()


	results = []
	out = pd.DataFrame()
	times = []
	Bexp_accepted =[]
	Banc_accepted = []
	set_temps = Utilities.OpenExperiment(Experiment)[0]


	print(f"Study Iteration: {j}")
	start = time.time()


	if Noise == 1:


		N_orient = Utilities.Generate_N_orient(1)
		Rot_Treatment = Utilities.RndVec(size = (1 ,3))
		Rot_angle = [((2*np.pi - 0)* rng.random() + 0) for i in range(1)]

	else:

		N_orient = np.array(([[0,0,1]]))
		print("Default N_orient used")
		Rot_Treatment = [np.array(([0,0,1]))]
		Rot_angle = [np.radians(0)]

	fn = np.nan
	params = np.nan

	if lam < 0.01:
		print("lambda < 0.01: Using SD model")
		Mvec, temperatures, treatment, Beff, NLT_vec, Anis_vec = SD_MD_model_v4.SD_MD_model_v4(Banc, Blab, Tc, Experiment, N_orient, T_orient, Beta_fits, Noise, Flags, P, Rot_Treatment, Rot_angle, NLT_a, NLT_b, Bnlt)

	else:
		Mvec, temperatures, treatment, Beff, NLT_vec, Anis_vec = MD_model_v4.MD_model_v4(Experiment, Beta_fits, lam, Banc, N_orient, Blab, T_orient, Tc, Noise, Flags, P, Rot_Treatment, Rot_angle, NLT_a, NLT_b, Bnlt)

	if Control == 1 and Free == 0:
		stats, timer = Control_Run.Run(Mvec, set_temps, treatment, T_orient, Blab, Az, Pl, N_orient, Flags, Anis_vec,  NLT_vec, Bnlt, beta_T, Beta_fits, Banc, j, fn, params, lam = lam)
	elif Control == 0 and Free == 1:
		stats, timer = Free_Run.Run(Mvec, set_temps, treatment, T_orient, Blab, Az, Pl, N_orient, Flags, Anis_vec,  NLT_vec, Bnlt, beta_T, Beta_fits, Banc, j, fn, params, lam = lam)
	else:
		raise ValueError("Run; Invalid analysis option chosen, select Free or Control run")



	end = time.time()-start
	print("time: "+str(end))
	times.append(end)

	df = pd.DataFrame(stats)
	Bexp_accepted.append(Banc.item())
	Banc_accepted.append(df["Banc"])
	df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index = False)




	print(f"Study {j} complete")


	return times

def run(Study_iter,ID, folder, rootfolder, a, b, lam, Blab, Banc, Experiment_Key):
	warnings.simplefilter('ignore', RuntimeWarning)
	warnings.simplefilter('ignore', DeprecationWarning)
	warnings.simplefilter('ignore', FutureWarning)





	N = 1
	AnisCorr = "No"
	NLTCorr = "No"


	if a == "SAMPLE" or b == "SAMPLE":
		Beta_fits = Utilities.Beta_fits(1)[0]
	else:
		Beta_fits =np.array([a,b])


	Tc = 580.0
	Noise = 1
	Control = 1
	Free = 0

	Experiment = f"{rootfolder}/Core_Files/Experiment_{Experiment_Key}.txt"

	csv_file = f"{folder}/{ID}_Out.csv"
	if os.path.isfile(csv_file) == True:
		del_file = "y"
		if del_file == str("y"):
			## Try to delete the file ##
			try:
				os.remove(csv_file)
			except OSError as e:  ## if failed, report it back to the user ##
				print ("Error: %s - %s." % (e.filename, e.strerror))
			print(f"{csv_file} has been deleted")
		elif del_file == str("n"):
			print(f"Any new data will be appended to the existing file ({csv_file})")
		else:
			print("Invalid response input, please restart the programme")

			exit()



	now = time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime())
	print(f"Programme Initiated: {now}")
	start_time = time.time()

	print(Experiment)


	field_fits = Utilities.field_fit()

	T_orient = np.array(([[0, 0, 1]]))
	Banc = np.array([Banc])
	NRM_rot_flag = 0
	Az = []
	Pl = []
	if AnisCorr == "No":
		A_flag = 0
	elif AnisCorr == "Yes":
		A_flag = 1
	else:
		raise ValueError("Reading Error; Anisotropy Correction: input value not accepted")
	if NLTCorr == "No":
		NLT_flag = 0
	elif NLTCorr == "Yes":
		NLT_flag = 1
	else:
		raise ValueError("Reading Error; NLT Correction: input value not accepted")
	NRflag = 0
	beta_T = 0.1

	Flags = [A_flag, NLT_flag, NRflag, NRM_rot_flag]

	NLTness = 0.6
	NLT_b = np.arctanh(NLTness)/80
	NLT_a = 1/NLT_b
	P = 1
	Bnlt = np.array([10, 20, 40, 60, 80, 100, Blab])


	times = []
	for j in range(Study_iter):

		t =	RunModelFunc(j, Experiment, Blab, Banc, T_orient, Tc, Noise, Flags, P, NLT_a, NLT_b, Bnlt,  Az, Pl, beta_T, csv_file, Control, Free, ID, N, Experiment_Key,  field_fits, now, folder, rootfolder, Beta_fits, lam)
		times.append(t)


	end_time = time.time()
	mean_t = np.mean(times)
	print(f"--- {(end_time - start_time)} seconds ---")
	print("\n")
	print(f"---Average time per sample: {(mean_t)} seconds ---")
	print("\n")

	d,h,m,secs = Utilities.formattime((time.time() - start_time))
	print(f"--- Run Time:  {d:02}:{h:02}:{m:02}:{secs:02.0f}  ---")

	print("END")
	now = time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime())
	print(f"Programme Finished: {now}")
