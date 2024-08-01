"""
Script to convert .tdt files to model input
"""

# Import modules
import os.path
import numpy as np
import re
import os, shutil
import warnings
from Utilities import dir2cart
import linecache

# Set up an empty array
def Initialize_array(n):
	arr = np.empty((1,n))
	arr[:] = np.nan
	return arr

# give final array by removing first line
def Final_array(arr):
	arr = np.delete(arr, (0), axis=0)
	return arr

# add new line to array
def addline(arr, val):
	np.append(arr, val)

# read header information from input file
def readheader(header):
	line = header.split()
	Blab = float(line[0])
	return Blab

# read line and find required values
def line2input(line):


	# split line string into values
	list = line.split()

	# check correct number of values
	if len(list) != 5:
		warnings.warn("Line input incompatible: " +str(list[0])+"/" +str(list[1]))

	# find combined temp and treatment
	Temp_Treat = float(list[1])

	# Set NRM to room temperature
	if Temp_Treat == "NRM":
		Temp = 25.0
		Treatment = 1
	# Remove leading letters for demag type (e.g. TD) from temperature
	# Find temp and treatment
	else:
		format(Temp_Treat, '.1f')
		Temp_Treat = str(Temp_Treat)
		Temp, Treat = Temp_Treat.split(".")

	# set values to float
	Temp = float(Temp)
	Treat = float(Treat)
	R = float(list[2])
	Dec = float(list[3])
	Inc = float(list[4])

	return Temp, Treat, R, Dec, Inc

# find sample name from input file
def SampleName(infile):
	line = linecache.getline(infile,3)
	list = line.split()
	Sample = str(list[0])

	return Sample

# Main conversion fn
def convert(infile):

	# check if input file exists
	if os.path.isfile(infile) != True:
		print("Input file path not recogised - Conversion aborted")
		exit()

	# set up arrays
	Temperatures = Initialize_array(1)
	Treatments = Initialize_array(1)
	Mvec = Initialize_array(3)

	# iterate through folder of files
	with open(infile) as f:
		# read first line
		first_line = f.readline()
		# read header line
		header = f.readline()
		# extract Blab value from header
		Blab = readheader(header)

		#iterate through lines of the infile
		for line in f:
			# get values from the line
			Temp, Treat, R, Dec, Inc = line2input(line)   #read line and store info
			# get cart values from directional co ords
			x, y, z = dir2cart(Dec, Inc, Mag = R)

			# add values to arrays
			Temperatures = np.concatenate((Temperatures, np.array(([Temp])).reshape((1,1))), axis = 0)
			Treatments = np.append(Treatments, np.array(([Treat])).reshape(1,1))
			vec = np.array(([x, y, z])).reshape(1,3)
			Mvec = np.concatenate((Mvec, vec), axis = 0)

		#get sample name
		Sample = SampleName(infile)
	#close file
	f.close()

	# tidy up arrays
	Temperatures = Final_array(Temperatures)
	Treatments = Final_array(Treatments)
	Mvec = Final_array(Mvec)


	return Mvec, Treatments, Temperatures, Blab, Sample
