#!/usr/bin/python
from __future__ import print_function
from collections import defaultdict

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d

colors = [ 'b', 'r', 'g' ]

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
averages = defaultdict( list )

line = dataFile.readline().strip().split(',')

title = line[0]
thread_nums = []
cur_thread = 0

skip_count = 1
skip = 0
for line in dataFile.xreadlines():
	data = line.strip().split(',')

	skip = skip + 1
	if( skip >= skip_count ):
		skip = 0

		cur_thread = int(data[0])
		thread_nums.append(cur_thread)

		for val in data[1:-1]:
			averages[cur_thread].append(float(val))

# create a new plot to use
fig = plt.figure()
ax = fig.add_subplot( 111 )

# scatter plot
last_avgs = [ averages[key][-2] for key in sorted(averages.keys()) ]
ax.scatter( thread_nums, last_avgs )

#ax.set_title( sys.argv[1][:-4] )
ax.set_xlabel( 'Number of Threads/Block' )
ax.set_ylabel( 'Average sync cost' )

# line at different thread numbers per SM
[ax.axvline( x=x ) for x in [ 1536/y for y in range(1,10) ] ]

ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(32))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(128))

ax.set_xlim( [ 0 , 1100 ] )

# show and save
fig.savefig("%s.png" % (sys.argv[1][:-4]))
plt.show()
