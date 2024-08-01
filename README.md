# SPM
Stochastic Palaeointensity Model
To use SPM please use the Jupyter Notebook provided (PI Model.ipynb) once the files have been downloaded. In order to run the notebook please download the files to a local folder. The following parameters should then be set:

Study_iter - the number of simulation iterations to be performed

ID - A string of alpha-numeric characters to identify the outputfile

folder - the path to the folder to which results will be output

rootfolder - path to folder containing programme files

a, b - values for the beta distribution that describes the blocking behaviour. These can be sampled randomly from a list of real blocking behaviours. (to sample blocking behavior set a and/or b to "SAMPLE" - (a = "SAMPLE"))

lam - value for the degree of MD behaviour. 0.01 and less is considered SD, MD is capped at 0.6

Blab - the lab field to be used in uT

Banc - the ancient field to be used in uT

Experiment_Key - a key to identify which experimental protocol to be used. Experiment keys can currently be set to: "Coe", "IZZI", "Aitken", "Thellier"

The simulations can now be run. Code to produce basic information on the results is also included.
