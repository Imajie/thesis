#!/usr/bin/python
from __future__ import print_function

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from pylab import polyfit, poly1d

figure_size = (10,10)
font_size = 20
y_lim = 1200
show_eqs = True #False

def occupancy( warps_per_block ):
	return [ (float(int(48.0/warp_per_block) * warp_per_block) / 48.0) for warp_per_block in warps_per_block ]

def main():
	# make sure we have a valid file to use
	if( len(sys.argv) <= 1 ):
		print("usage: %s data.csv threads_per_block"%sys.argv[0])
		exit()

	try:
		dataFile = open( sys.argv[1] )
	except IOError:
		print("Error opening file %s"%sys.argv[1])
		exit()

	averages = []

	line = dataFile.readline().strip().split(',')

	blocks = map(int, line[1:-1])
	threads = []

	for line in dataFile.readlines():
		data = line.strip().split(',')

		averages.append([])
		threads.append(int(data[0]))

		for val in data[1:-1]:
			averages[-1].append(np.float64(val))

	fit0_m = []
	fit0_b = []

	fit1_m = []
	fit1_b = []

	fit2_m = []
	fit2_b = []
	fit2_c = []

	for threads_per_block_to_plot in range( 64, 1024+1, 32 ):
		print("Fitting: %i"%(threads_per_block_to_plot))
		avg_idx = threads.index(threads_per_block_to_plot)

		L1size = ((2**10)*16)*14 / (4*threads_per_block_to_plot)
		L2size = ((2**10)*768)	 / (4*threads_per_block_to_plot)

		# Linear fit for < 16*1024 
		drop_idx = next(idx for idx,val in enumerate(blocks) if val >= 16*1024/threads_per_block_to_plot)

		fit0_x = blocks[:drop_idx]
		m0,b0  = polyfit( fit0_x, averages[avg_idx][:drop_idx], 1 )
		fit0_y = poly1d( (m0,b0) )(fit0_x)

		# Linear fit for 16*1024 < x < L1 cache size
		L1idx = next(idx for idx,val in enumerate(blocks) if val >= L1size)

		fit1_x = blocks[drop_idx:L1idx]
		m1,b1  = polyfit( fit1_x, averages[avg_idx][drop_idx:L1idx], 1 )
		fit1_y = poly1d( (m1,b1) )(fit1_x)

		# log fit for > L1
		fit2_x = np.array([ float(x) for x in blocks[L1idx:] ])

		#generate weights for x
		weight2 = [ (x if x>1 else 1) for x in [ (blocks[i] - blocks[i-1])/10.0 for i in range(L1idx, len(blocks)) ] ]

		# fit to log
		C = L1size-1
		B = 1
		fit2_x_log = [ np.log(B*(x-C)) for x in fit2_x ]
		m2,b2  = polyfit( fit2_x_log, averages[avg_idx][L1idx:], 1, w=weight2 )
		fit2_y = poly1d( (m2,b2) )(fit2_x_log)

		# save fit data
		fit0_m.append(m0)
		fit0_b.append(b0)

		fit1_m.append(m1)
		fit1_b.append(b1)

		fit2_m.append(m2)
		fit2_b.append(b2)
		fit2_c.append(C)

		# create a new plot to use
		fig = plt.figure(figsize=figure_size)
		ax = fig.add_subplot( 111 )

		# line at 16*1024 threads
		ax.axvline(x=(2**14)/threads_per_block_to_plot, c='k', label='2^14 threads')

		# line at L1 cache size
		ax.axvline(x=L1size, label='L1 cache size (All SMs)')

		# line at L1+L2 cache size
		ax.axvline(x=L2size, c='r', label='L2 cache size')

		# scatter plot
		ax.scatter( blocks, averages[avg_idx] )

		# fits
		if show_eqs:
			ax.plot( fit0_x, fit0_y, label="16384 threads fit: %f*x+%f"%(m0,b0) )
			ax.plot( fit1_x, fit1_y, label="L1 fit: %f*x+%f"%(m1,b1) )
			ax.plot( fit2_x, fit2_y, label="L2 fit: %f*ln(x-%f)+%f"%(m2, C, b2) )

			ax.legend(loc='lower right')
		else:
			ax.plot( fit0_x, fit0_y )
			ax.plot( fit1_x, fit1_y )
			ax.plot( fit2_x, fit2_y )

			ax.xaxis.label.set_size(font_size)
			ax.yaxis.label.set_size(font_size)
			ax.legend(loc='lower right', prop={'size':font_size})
			ax.tick_params(axis='both', which='major', labelsize=font_size)
			ax.tick_params(axis='both', which='minor', labelsize=font_size)



		ax.set_xlabel( 'Number of Blocks' )
		ax.set_ylabel( 'Average Sync Cost' )

		ax.set_ylim( [0, y_lim] ) #ax.get_ylim()[1]] )
		ax.set_xlim( [0, 2000] )

		fig.savefig("%s_%i.png" % (sys.argv[1][:-4], threads_per_block_to_plot))
		#plt.show(block=False)
		#plt.pause(0.5)

	# Now we have the data for each fit, so fit that, but create equations based on number of warps
	warps = [ x/32 for x in threads ]
	max_warps_per_SM = 1536/32

	drop1_idx = next( i for i in range(len(warps)) if warps[i] > max_warps_per_SM/2 )
	drop2_idx = next( i for i in range(len(warps)) if warps[i] > max_warps_per_SM/3 )
	drop3_idx = next( i for i in range(len(warps)) if warps[i] > max_warps_per_SM/4 )
	drop4_idx = next( i for i in range(len(warps)) if warps[i] > max_warps_per_SM/5 )


	# First Linear fit
	fig = plt.figure(figsize=figure_size)
	ax = fig.add_subplot( 111 )
	ax.scatter( warps, fit0_m )

	ax.set_xlabel( 'Number of warps/Blocks' )
	ax.set_ylabel( 'Slope for first linear fit' )

	# now find and add the fits
	if( 'read' in sys.argv[1] ):
		# read = linear < 1 block/SM
		#        linear = 1

		fit_x = warps[:drop1_idx]
		m,b   = polyfit( fit_x, fit0_m[:drop1_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 1: %fx+%f'%(m,b) )

		fit_x = warps[drop1_idx:]
		m,b   = polyfit( fit_x, fit0_m[drop1_idx:], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 2: %fx+%f'%(m,b) )
	else:
		# write = linear < 2
		#         const  = 2
		#         const  = 1

		fit_x = warps[:drop2_idx]
		m,b   = polyfit( fit_x, fit0_m[:drop2_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 1: %fx+%f'%(m,b) )

		fit_x = warps[drop2_idx:drop1_idx]
		m,b   = polyfit( fit_x, fit0_m[drop2_idx:drop1_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 2: %fx+%f'%(m,b) )

		fit_x = warps[drop1_idx:]
		m,b   = polyfit( fit_x, fit0_m[drop1_idx:], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 3: %fx+%f'%(m,b) )
		
	ax.legend()

	fig.savefig("%s_%s%i.png" % (sys.argv[1][:-4], "m", 0))

	fig = plt.figure(figsize=figure_size)
	ax = fig.add_subplot( 111 )
	ax.scatter( warps, fit0_b )

	ax.set_xlabel( 'Number of warps/Blocks' )
	ax.set_ylabel( 'Intercept for first linear fit' )

	# now find and add the fits
	if( 'read' in sys.argv[1] ):
		# read = linear < 1 block/SM
		#        linear = 1

		fit_x = warps[:drop1_idx]
		m,b   = polyfit( fit_x, fit0_b[:drop1_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 1: %fx+%f'%(m,b) )

		fit_x = warps[drop1_idx:]
		m,b   = polyfit( fit_x, fit0_b[drop1_idx:], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 2: %fx+%f'%(m,b) )
	else:
		# write = linear < 2
		#         linear = 2
		#         linear = 1

		fit_x = warps[:drop2_idx]
		m,b   = polyfit( fit_x, fit0_b[:drop2_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 1: %fx+%f'%(m,b) )

		fit_x = warps[drop2_idx:drop1_idx]
		m,b   = polyfit( fit_x, fit0_b[drop2_idx:drop1_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 2: %fx+%f'%(m,b) )

		fit_x = warps[drop1_idx:]
		m,b   = polyfit( fit_x, fit0_b[drop1_idx:], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 3: %fx+%f'%(m,b) )
		
	ax.legend()
	fig.savefig("%s_%s%i.png" % (sys.argv[1][:-4], "b", 0))

	# second Linear fit
	fig = plt.figure(figsize=figure_size)
	ax = fig.add_subplot( 111 )
	ax.scatter( warps, fit1_m )

	ax.set_xlabel( 'Number of warps/Blocks' )
	ax.set_ylabel( 'Slope for second linear fit' )

	# now find and add the fits
	if( 'read' in sys.argv[1] ):
		# read = linear <  2 block/SM
		#        linear >= 2

		fit_x = warps[:drop2_idx]
		m,b   = polyfit( fit_x, fit1_m[:drop2_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 1: %fx+%f'%(m,b) )

		fit_x = warps[drop2_idx:]
		m,b   = polyfit( fit_x, fit1_m[drop2_idx:], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 2: %fx+%f'%(m,b) )
	else:
		# write = linear < 2
		#         const  = 2
		#         const  = 1
		fit_x = warps[:drop2_idx]
		m,b   = polyfit( fit_x, fit1_m[:drop2_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 1: %fx+%f'%(m,b) )

		fit_x = warps[drop2_idx:drop1_idx]
		m,b   = polyfit( fit_x, fit1_m[drop2_idx:drop1_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 2: %fx+%f'%(m,b) )

		fit_x = warps[drop1_idx:]
		m,b   = polyfit( fit_x, fit1_m[drop1_idx:], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 3: %fx+%f'%(m,b) )

	ax.legend()
	fig.savefig("%s_%s%i.png" % (sys.argv[1][:-4], "m", 1))

	fig = plt.figure(figsize=figure_size)
	ax = fig.add_subplot( 111 )
	ax.scatter( warps, fit1_b )

	ax.set_xlabel( 'Number of warps/Blocks' )
	ax.set_ylabel( 'Intercept for second linear fit' )

	# now find and add the fits
	if( 'read' in sys.argv[1] ):
		# read = linear < 2 block/SM
		#        linear = 2
		#        linear = 1
		fit_x = warps[:drop4_idx]
		m,b   = polyfit( fit_x, fit1_b[:drop4_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 1: %fx+%f'%(m,b) )

		fit_x = warps[drop4_idx:drop3_idx]
		m,b   = polyfit( fit_x, fit1_b[drop4_idx:drop3_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 2: %fx+%f'%(m,b) )

		fit_x = warps[drop3_idx:drop2_idx]
		m,b   = polyfit( fit_x, fit1_b[drop3_idx:drop2_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 3: %fx+%f'%(m,b) )

		fit_x = warps[drop2_idx:drop1_idx]
		m,b   = polyfit( fit_x, fit1_b[drop2_idx:drop1_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 4: %fx+%f'%(m,b) )

		fit_x = warps[drop1_idx:]
		m,b   = polyfit( fit_x, fit1_b[drop1_idx:], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 5: %fx+%f'%(m,b) )
	else:
		# write = linear < 2
		#         linear = 2
		#         linear = 1

		fit_x = warps[:drop2_idx]
		m,b   = polyfit( fit_x, fit1_b[:drop2_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 1: %fx+%f'%(m,b) )

		fit_x = warps[drop2_idx:drop1_idx]
		m,b   = polyfit( fit_x, fit1_b[drop2_idx:drop1_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 2: %fx+%f'%(m,b) )

		fit_x = warps[drop1_idx:]
		m,b   = polyfit( fit_x, fit1_b[drop1_idx:], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 3: %fx+%f'%(m,b) )

	ax.legend()
	fig.savefig("%s_%s%i.png" % (sys.argv[1][:-4], "b", 1))

	# Log fit
	fig = plt.figure(figsize=figure_size)
	ax = fig.add_subplot( 111 )

	fit2_m_occ = np.array(fit2_m)/occupancy(np.array(warps))

	#ax.scatter( warps, fit2_m )
	ax.plot( warps, 100*np.array(occupancy(warps)), c='r', label="% occupancy" )
	ax.scatter( warps, fit2_m_occ )
	fit2_m = fit2_m_occ

	ax.set_xlabel( 'Number of warps/Blocks' )
	ax.set_ylabel( 'Slope for log fit' )

	# now find and add the fits
	if( 'read' in sys.argv[1] ):
		# read = linear < 3 block/SM
		#        linear = 3
		#        linear = 2
		#        linear = 1
		fit_x = warps[:drop4_idx]
		m,b   = polyfit( fit_x, fit2_m[:drop4_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 1: %fx+%f'%(m,b) )

		fit_x = warps[drop4_idx:drop3_idx]
		m,b   = polyfit( fit_x, fit2_m[drop4_idx:drop3_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 2: %fx+%f'%(m,b) )

		fit_x = warps[drop3_idx:drop2_idx]
		m,b   = polyfit( fit_x, fit2_m[drop3_idx:drop2_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 3: %fx+%f'%(m,b) )

		fit_x = warps[drop2_idx:drop1_idx]
		m,b   = polyfit( fit_x, fit2_m[drop2_idx:drop1_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 4: %fx+%f'%(m,b) )

		fit_x = warps[drop1_idx:]
		m,b   = polyfit( fit_x, fit2_m[drop1_idx:], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 5: %fx+%f'%(m,b) )

	else:
		# write = linear < 2
		#         linear = 2
		#         linear = 1
		fit_x = warps[:drop4_idx]
		m,b   = polyfit( fit_x, fit2_m[:drop4_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 1: %fx+%f'%(m,b) )

		fit_x = warps[drop4_idx:drop3_idx]
		m,b   = polyfit( fit_x, fit2_m[drop4_idx:drop3_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 2: %fx+%f'%(m,b) )

		fit_x = warps[drop3_idx:drop2_idx]
		m,b   = polyfit( fit_x, fit2_m[drop3_idx:drop2_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 3: %fx+%f'%(m,b) )

		fit_x = warps[drop2_idx:drop1_idx]
		m,b   = polyfit( fit_x, fit2_m[drop2_idx:drop1_idx], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 4: %fx+%f'%(m,b) )

		fit_x = warps[drop1_idx:]
		m,b   = polyfit( fit_x, fit2_m[drop1_idx:], 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit 5: %fx+%f'%(m,b) )

	ax.legend()
	fig.savefig("%s_%s%i.png" % (sys.argv[1][:-4], "m", 2))

	fig = plt.figure(figsize=figure_size)
	ax = fig.add_subplot( 111 )

	ax.scatter( warps, fit2_b )

	ax.set_xlabel( 'Number of warps/Blocks' )
	ax.set_ylabel( 'Intercept for log fit' )

	# now find and add the fits
	if( 'read' in sys.argv[1] ):
		# read = linear 
		fit_x = warps
		m,b   = polyfit( fit_x, fit2_b, 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit: %fx+%f'%(m,b) )
	else:
		# write = linear 
		fit_x = warps
		m,b   = polyfit( fit_x, fit2_b, 1 )
		fit_y = poly1d( (m,b) )(fit_x)

		ax.plot( fit_x, fit_y, label='fit: %fx+%f'%(m,b) )

	ax.legend()
	fig.savefig("%s_%s%i.png" % (sys.argv[1][:-4], "b", 2))

	fig = plt.figure(figsize=figure_size)
	ax = fig.add_subplot( 111 )
	ax.scatter( warps, fit2_c )

	ax.set_xlabel( 'Number of warps/Blocks' )
	ax.set_ylabel( 'L1size for log fit' )

	fig.savefig("%s_%s%i.png" % (sys.argv[1][:-4], "c", 2))

if __name__ == "__main__":
	main()
