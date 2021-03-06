#!/usr/bin/python
from __future__ import print_function

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

colors = [
	'#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#000000',
	'#800000', '#008000', '#000080', '#808000', '#800080', '#008080', '#808080',
	'#c00000', '#00c000', '#0000c0', '#c0c000', '#c000c0', '#00c0c0', '#c0c0c0',
	'#400000', '#004000', '#000040', '#404000', '#400040', '#004040', '#404040',
	'#600000', '#006000', '#000060', '#606000', '#600060', '#006060', '#606060',
	'#a00000', '#00a000', '#0000a0', '#a0a000', '#a000a0', '#00a0a0', '#a0a0a0'
]


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

skip_count = 3
skip = 0
for line in dataFile.xreadlines():
	data = line.strip().split(',')

	skip = skip + 1
	if( skip == skip_count ):
		skip = 0

		cur_thread_per_block = int(data[0])
		averages.append([])
		threads.append([])
		thread_nums.append( cur_thread_per_block )
		
		for i in range(1,len(data)-1):
			cur_threads = cur_thread_per_block*block_nums[i-1]
			val = data[i]

			averages[-1].append(float(val))
			threads[-1].append(cur_threads)

# create a new plot to use
fig = plt.figure()
ax = fig.add_subplot( 111 )
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

# scatter plot
i = 0
for pair in zip( threads, averages ):
	ax.scatter( pair[0], pair[1], c=colors[i], label=str(thread_nums[i]) )
	i = i + 1

#ax.set_title( sys.argv[1][:-4] )
ax.set_xlabel( 'Used Memory' )
ax.set_ylabel( 'Average sync cost' )
#ax.legend(loc='upper left')

ax.set_xlim( [0, 1024*2000] )
ax.set_ylim( [0, ax.get_ylim()[1]] )
fig.savefig("%s.png" % (sys.argv[1][:-4]))
plt.show()
