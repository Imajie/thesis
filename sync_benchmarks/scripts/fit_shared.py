#!/usr/bin/python
from __future__ import print_function

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
from pylab import polyfit,poly1d

# fit func for exp(-x)
def exp_fit_func( x, a, b, c, d ):
	return a*np.exp(-b*x+c)+d
def exp_fit_deriv( x, a, b, c, d ):
	return -b*a*np.exp(-b*x+c)

def log_fit_func( x, a, b, c, d ):
	return a*np.log(b*x+c)+d
def log_fit_deriv( x, a, b, c, d ):
	return (b*a)/(b*x+c)

# make sure we have a valid file to use
if( len(sys.argv) <= 1 ):
	print("usage: %s data.csv threads_per_block"%sys.argv[0])
	exit()

try:
	dataFile = open( sys.argv[1] )
except IOError:
	print("Error opening file %s"%sys.argv[1])
	exit()

title = ''
averages = []
pcs = []

line = dataFile.readline().strip().split(',')

title = line[0]
blocks = map(int, line[1:-1])
threads = []

for line in dataFile.xreadlines():
	data = line.strip().split(',')

	averages.append([])
	threads.append(int(data[0]))

	for val in data[1:-1]:
		averages[-1].append(np.float64(val))

fit0_m = []
fit0_b = []

fit1_m = []
fit1_b = []
fit1_c = []

fit2_m = []
fit2_b = []
fit2_c = []

for threads_per_block_to_plot in range( 64, 1024+1, 32 ):
	print("Fitting: %i"%(threads_per_block_to_plot))
	avg_idx = threads.index(threads_per_block_to_plot)

	# Linear fit for < drop 
	drop_idx = next(idx for idx,val in enumerate(blocks) if val >= (2**(14.00))/threads_per_block_to_plot)
	rise_idx = next(idx for idx,val in enumerate(blocks) if val >= (2**(16.56))/threads_per_block_to_plot)

	fit0_x = blocks[:drop_idx]
	m0,b0  = polyfit( fit0_x, averages[avg_idx][:drop_idx], 1 )
	fit0_y = poly1d( (m0,b0) )(fit0_x)

	# Log fit for drop < x < rise
	fit1_x = np.array([ float(x) for x in blocks[drop_idx:rise_idx] ])

	#generate weights for x
	weight1 = [ (x if x>1 else 1) for x in [ (blocks[i] - blocks[i-1])/10.0 for i in range(drop_idx, rise_idx) ] ]

	# fit to log
	C1 = drop_idx-1
	B1 = 1
	fit1_x_log = [ np.log(B1*(x-C1)) for x in fit1_x ]
	m1,b1  = polyfit( fit1_x_log, averages[avg_idx][drop_idx:rise_idx], 1, w=weight1 )
	fit1_y = poly1d( (m1,b1) )(fit1_x_log)

	# Log fit for > rise
	fit2_x = np.array([ float(x) for x in blocks[rise_idx:] ])

	#generate weights for x
	weight2 = [ (x if x>1 else 1) for x in [ (blocks[i] - blocks[i-1])/10.0 for i in range(rise_idx, len(blocks)) ] ]

	# fit to log
	C2 = rise_idx-1
	B2 = 1
	fit2_x_log = [ np.log(B2*(x-C2)) for x in fit2_x ]
	m2,b2  = polyfit( fit2_x_log, averages[avg_idx][rise_idx:], 1, w=weight2 )
	fit2_y = poly1d( (m2,b2) )(fit2_x_log)

	# save fit data
	fit0_m.append(m0)
	fit0_b.append(b0)

	fit1_m.append(m1)
	fit1_b.append(b1)
	fit1_c.append(C1)

	fit2_m.append(m2)
	fit2_b.append(b2)
	fit2_c.append(C2)

	# create a new plot to use
	fig = plt.figure()
	ax = fig.add_subplot( 111 )

	# line at 16*1024 threads
	ax.axvline(x=drop_idx, label='2^14 threads')
	ax.axvline(x=rise_idx, label='2^16.56 threads')

	# scatter plot
	ax.scatter( blocks, averages[avg_idx] )

	# fits
	ax.plot( fit0_x, fit0_y, label="16384 threads fit: %f*x+%f"%(m0,b0) )
	ax.plot( fit1_x, fit1_y, label="Log fit: %f*ln(x-%f)+%f"%(m1, C1, b1) )
	ax.plot( fit2_x, fit2_y, label="Log fit: %f*ln(x-%f)+%f"%(m2, C2, b2) )

	ax.set_xlabel( 'Number of Blocks' )
	ax.set_ylabel( 'Average Sync Cost' )

	ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(32))
	ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(128))

	ax.legend(loc='lower right')

	fig.savefig("%s_%i.png" % (sys.argv[1][:-4], threads_per_block_to_plot))
	#plt.show(block=False)
	#plt.pause(0.5)

# Now we have the data for each fit, so fit that

# Linear fit
fig = plt.figure()
ax = fig.add_subplot( 111 )
ax.scatter( threads, fit0_m )

ax.set_xlabel( 'Number of Threads/Blocks' )
ax.set_ylabel( 'Slope for first linear fit' )

fig.savefig("%s_%s%i.png" % (sys.argv[1][:-4], "m", 0))

fig = plt.figure()
ax = fig.add_subplot( 111 )
ax.scatter( threads, fit0_b )

ax.set_xlabel( 'Number of Threads/Blocks' )
ax.set_ylabel( 'Intercept for first linear fit' )

fig.savefig("%s_%s%i.png" % (sys.argv[1][:-4], "b", 0))

# Log fit
fig = plt.figure()
ax = fig.add_subplot( 111 )
ax.scatter( threads, fit1_m )

ax.set_xlabel( 'Number of Threads/Blocks' )
ax.set_ylabel( 'Slope for log fit' )

fig.savefig("%s_%s%i.png" % (sys.argv[1][:-4], "m", 1))

fig = plt.figure()
ax = fig.add_subplot( 111 )
ax.scatter( threads, fit1_b )

ax.set_xlabel( 'Number of Threads/Blocks' )
ax.set_ylabel( 'Intercept for log fit' )

fig.savefig("%s_%s%i.png" % (sys.argv[1][:-4], "b", 1))

fig = plt.figure()
ax = fig.add_subplot( 111 )
ax.scatter( threads, fit1_c )

ax.set_xlabel( 'Number of Threads/Blocks' )
ax.set_ylabel( 'L1size for log fit' )

# Log fit
fig = plt.figure()
ax = fig.add_subplot( 111 )
ax.scatter( threads, fit2_m )

ax.set_xlabel( 'Number of Threads/Blocks' )
ax.set_ylabel( 'Slope for log fit' )

fig.savefig("%s_%s%i.png" % (sys.argv[1][:-4], "m", 2))

fig = plt.figure()
ax = fig.add_subplot( 111 )
ax.scatter( threads, fit2_b )

ax.set_xlabel( 'Number of Threads/Blocks' )
ax.set_ylabel( 'Intercept for log fit' )

fig.savefig("%s_%s%i.png" % (sys.argv[1][:-4], "b", 2))

fig = plt.figure()
ax = fig.add_subplot( 111 )
ax.scatter( threads, fit2_c )

ax.set_xlabel( 'Number of Threads/Blocks' )
ax.set_ylabel( 'L1size for log fit' )

fig.savefig("%s_%s%i.png" % (sys.argv[1][:-4], "c", 2))

