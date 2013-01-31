#!/usr/bin/python
from __future__ import print_function

import subprocess
import os
import sys

# arrays for different test parameters
blocks = range( 10, 1000+1, 10 )
threads = range(64, 256+1, 32 )
op_types = [ "read", "read_cached", "read_uncached", "write" ]
mem_types = [ "global", "shared" ]

print( "Processing data" )

if( len(sys.argv) < 1 ):
	print("Usage: %s result_dir/")
	exit()

result_dir = sys.argv[1]

# now process the files
for mem_type in mem_types:
	for op_type in op_types:

		fileName = "%s/%s_%s_averages.csv"%(result_dir,mem_type,op_type)
		numFiles = 0

		print("%s %s: "% (mem_type, op_type), end ="")

		with open(fileName, "w") as avgFile:
			# write out blocks header
			avgFile.write(op_type+",")
			for block in blocks:
				avgFile.write( str(block) + "," )

			for thread in threads:
				avgFile.write("\n" + str(thread) + ",")
				for block in blocks:
					try:
						with open( "%s/%s_%s_%i_%i.txt" % (result_dir,mem_type, op_type, block, thread) ) as dataFile:
							avg = []
							count = []
							pcs = []

							for line in dataFile.xreadlines():
								values = map(int, line.strip().split(" "))
								# values = [ CTA_ID, PC, start, end, elapsed ]

								if( values[1] not in pcs ):
									pcs.append( values[1] )
									count.append( 1 )
									avg.append( values[4] )
								else:
									idx = pcs.index( values[1] )
									count[idx] += 1
									avg[idx] += values[4]

							# average and output to file
							if( len(pcs) > 1 ):
								avgFile.write('"[')
								for idx in range( len(pcs) ):
									avgFile.write( str(pcs[idx]) + "->" + str(avg[idx]/count[idx]) + ";" )
								avgFile.write('"],')
							else:
								avgFile.write( str(avg[0]/count[0]) + ",")

							numFiles += 1
							if( numFiles % 100 == 0 ):
								print(".", end="")

					except IOError:
						# file not found
						None
			
		# make sure the average file we produced is valid
		if( numFiles == 0 ):
			os.remove( fileName )
			print("No files", end="")
		print("")
