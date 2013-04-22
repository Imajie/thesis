#!/usr/bin/python
from __future__ import print_function

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
blocks = []
threads = []
averages = []
pcs = []

line = dataFile.readline().strip().split(',')

title = line[0]
block_nums = map(int, line[1:-1])
thread_nums = []

skip_count = 1
skip = 0
for line in dataFile.xreadlines():
	data = line.strip().split(',')

	skip = skip + 1
	if( skip >= skip_count ):
		skip = 0

		blocks = blocks + block_nums
		thread_nums.append(int(data[0]))

		for val in data[1:-1]:
			threads.append(int(data[0]))

			if( val[:2] == '"[' ):
				entries = val[2:-3].split(';')
				for entry in entries:
					pair = entry.split('->')

					if( int(pair[0]) not in pcs ):
						pcs.append(int(pair[0]))
						averages.append( [ float(pair[1]) ] )
					else:
						idx = pcs.index(int(pair[0]))
						averages[idx].append(float(pair[1]))
			else:
				averages.append(float(val))

# create a new plot to use
fig = plt.figure()
ax = fig.add_subplot( 111, projection='3d' )

# scatter plot
if( len(pcs) > 0 ):
# if multiple syncs plot each with a legend
	for idx in xrange(len(pcs)):
		ax.scatter( blocks, threads, averages[idx], c=colors[idx], label='PC: %i'%pcs[idx] )
		ax.plot( [], [], 'o', c = colors[idx], label = 'PC: %i'%pcs[idx])
	ax.legend( )
else:
# only 1 average to plot
	ax.scatter( blocks, threads, averages )

#ax.set_title( sys.argv[1][:-4] )
ax.set_xlabel( 'Number of Blocks' )
ax.set_ylabel( 'Number of Threads/Block' )
ax.set_zlabel( 'Average sync cost' )
#ax.set_zscale( 'log' )

# line at different thread numbers per SM
ax.bar( [ 1536/x for x in range(1,16) ], [ ax.get_zlim()[1] for x in range(1,16) ], ax.get_zlim()[0], zdir='x', width=0 )

ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(32))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(128))

# surface plot
#fig = plt.figure()
#ax = fig.add_subplot( 111, projection='3d' )
#
#ax.surface( X, Y, averages )
#ax.set_title( sys.argv[1][:-4] )
#ax.set_xlabel( 'Number of Blocks' )
#ax.set_ylabel( 'Number of Threads/Block' )
#ax.set_zlabel( 'Average sync cost' )

fig.savefig("%s.png" % (sys.argv[1][:-4]))
plt.show()
