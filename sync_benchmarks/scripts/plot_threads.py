#!/usr/bin/python
from __future__ import print_function

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

colors = [ 'b', 'r', 'g', 'y', 'm', 'k', 'c' ]

# make sure we have a valid file to use
if( len(sys.argv) <= 1 ):
	print("usage: %s data.csv"%sys.argv[0])
	exit()

try:
	dataFile = open( sys.argv[1] )
except IOError:
	print("Error opening file %s"%sys.argv[1])
	exit()

title = ''
threads = []
averages = []

line = dataFile.readline().strip().split(',')

title = line[0]
block_nums = map(int, line[1:-1])
thread_nums = []

for line in dataFile.xreadlines():
	data = line.strip().split(',')

	cur_thread_per_block = int(data[0])
	averages.append([])
	threads.append([])
	thread_nums.append( cur_thread_per_block )
	
	for i in range(1,len(data)-1):
		cur_threads = cur_thread_per_block*block_nums[i-1]
		val = data[i]

		averages[-1].append(float(val))
		threads[-1].append(cur_threads*4)

# create a new plot to use
fig = plt.figure()
ax = fig.add_subplot( 111 )

# scatter plot
i = 0
for pair in zip( threads, averages ):
	ax.scatter( pair[0], pair[1], c=colors[i], label=str(thread_nums[i]) )
	i = i + 1

# line at L1 cache size
ax.axvline(x=((2**10)*48)*13, label='L1 cache size (All SMs)')

# line at L2 cache size
ax.axvline(x=((2**20)*1.25), c='r', label='L2 cache size')

ax.set_title( sys.argv[1][:-4] )
ax.set_xlabel( 'Used Memory' )
ax.set_ylabel( 'Average sync cost' )
ax.legend(loc='upper left')

fig.savefig("%s.png" % (sys.argv[1][:-4]))
plt.show()
