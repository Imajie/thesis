#!/usr/bin/python
from __future__ import print_function

import subprocess
import os
import sys
import datetime

# arrays for different test parameters
blocks = range( 10, 400, 1 ) + range( 400, 1000, 10 )			# HW

threads = range(64, 256+1, 32 )
op_types = [ "read", "write" ]
mem_types = [ "global", "shared" ]

# /dev/null file pointer
devNull = open(os.devnull, 'w')

avg_count = 1
if( len(sys.argv) > 1 ):
	avg_count = int(sys.argv[1])

date = datetime.datetime.now().strftime("%b-%d-%I%M%p")

try:
	os.mkdir( "results-%s/"%date )
except IOError:
	pass

# run each type of test
for mem_type in mem_types:
	for op_type in op_types:
		for thread in threads:
			for block in blocks:
				# create stderr file
				fileName = "results-%s/%s_%s_%i_%i.txt" % (date, mem_type, op_type, block, thread)
				with open( fileName, "w" ) as errOut:
					# run the test
					print( "Running: './benchmark %i %i %s %s' %i times" % ( block, thread, op_type, mem_type, avg_count ), end="" )
					sys.stdout.flush()

					retVal = subprocess.call(["./benchmark", str(block), str(thread), str(op_type), str(mem_type), str(avg_count)], stdout=devNull, stderr=errOut);
			
					if( retVal == 0 ):
						print( " -> Successful" )
					else:
						print( " -> Failed" )

				needDelete = False

				with open( fileName, "r" ) as dataOut:
					# check if test has cache and split the file if it does
					if( dataOut.readline().strip() == "_NO_CACHE_" ):
						needDelete = True

						# new filenames to write to
						fileNameCache = fileName.replace( "read", "read_cached" )
						fileNameUncache = fileName.replace( "read", "read_uncached" )

						fileCache = open( fileNameCache, 'w' )
						fileUncache = open( fileNameUncache, 'w' )

						# start outputting uncached data
						fileOut = fileUncache

						# read all lines
						for line in dataOut.xreadlines():
							if( line.strip() == "_WITH_CACHE_" ):
								fileOut = fileCache
							else:
								fileOut.write( line )

						# flush and close
						fileCache.flush()
						fileCache.close()

						fileUncache.flush()
						fileUncache.close()

				if( needDelete ):
					# delete original file
					os.remove( fileName )

print( "Tests completed" )
