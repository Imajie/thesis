#!/usr/bin/python
from __future__ import print_function

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

def convertFile( dataFile, num_threads, num_blocks ):
	count = 0
	thread_ids = []
	sync_costs = []

	for line in dataFile.xreadlines():
		count = count + 1

		# only grab the second kernel launch
		if( count >= num_threads * num_blocks * 2 ):
			break
		elif( count >= num_threads * num_blocks ):
			data = int(line.strip().split(' ')[4])

			thread_ids.append(len(thread_ids))
			sync_costs.append(data)

	return thread_ids, sync_costs

# make sure we have a valid file to use
if( len(sys.argv) <= 1 ):
	print("usage: %s data.csv"%sys.argv[0])
	exit()

try:
	fileNameA = sys.argv[1]
	dataFileA = open( fileNameA )

	fileSplit = fileNameA[:-4].split('_')
	fileExt = fileNameA[-4:]

	file_kind   = fileSplit[0] + '_' + fileSplit[1]
	num_blocks  = int(fileSplit[2])
	num_threads = int(fileSplit[3])

	try:
		fileNameB = "%s_%i_%i%s"%(file_kind,num_blocks+1,num_threads,fileExt) 
		num_blocks_B = num_blocks+1
		dataFileB = open( fileNameB )
	except IOError:
		try:
			fileNameB = "%s_%i_%i%s"%(file_kind,num_blocks-1,num_threads,fileExt) 
			num_blocks_B = num_blocks-1
			dataFileB = open( fileNameB )
		except IOError:
			print("Error opening file %s_%i_%i%s or %s_%i_%i%s"%(file_kind,num_blocks+1,num_threads,fileExt,file_kind,num_blocks-1,num_threads,fileExt))
			exit()


except IOError:
	print("Error opening file %s"%fileNameA)
	exit()

thread_ids_A = []
sync_costs_A = []

thread_ids_B = []
sync_costs_B = []

# get data
(thread_ids_A, sync_costs_A) = convertFile( dataFileA, num_threads, num_blocks)
(thread_ids_B, sync_costs_B) = convertFile( dataFileB, num_threads, num_blocks_B)

# create a new plot to use
fig = plt.figure()
ax = fig.add_subplot( 111 )

# scatter plot
ax.scatter(thread_ids_A, sync_costs_A, c='b', label=fileNameA, s = 5, edgecolors='b' )
ax.scatter(thread_ids_B, sync_costs_B, c='r', label=fileNameB, s = 5, edgecolors='r' )

ax.set_xlabel( 'Thread ID' )
ax.set_ylabel( 'Average sync cost' )
ax.legend()

plt.show()
fig.savefig("%s_%i.png" % (file_kind,num_threads))
