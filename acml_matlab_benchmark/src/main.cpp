/*
 * File:	main.cpp
 *
 * Author:	James Letendre
 *
 * Benchmark ACML vs MATLAB performance
 */
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "benchmark_common.h"
#include "acml_bench.h"
#include "matlab_bench.h"

/*
 * Global vars
 */
// Default benchmark type
benchmark_type_t		benchmark_type 		= ACML;
benchmark_precision_t	benchmark_precision = SINGLE;

// The sizes of the matrices to test
unsigned int benchmark_sizes[] =
{
	5, 5, 10, 20, 35,
	50, 75, 100, 150,
	200, 500, 836, 1000,
	2000, 4000, 6000, 8000
};
unsigned int benchmark_num = sizeof(benchmark_sizes)/sizeof(unsigned int);

/*
 * End Global vars
 */

/*
 * void handle_arg
 *
 * Set benchmark type or precision based on command line args
 */
void handle_arg( const char *arg )
{
	if( strcmp( "acml", arg ) == 0 )
	{
		benchmark_type = ACML;
		return;
	}
	else if( strcmp( "matlab", arg ) == 0 )
	{
		benchmark_type = MATLAB;
	}
	else if( strcmp( "single", arg ) == 0 )
	{
		benchmark_precision = SINGLE;
	}
	else if( strcmp( "double", arg ) == 0 )
	{
		benchmark_precision = DOUBLE;
	}
}

int main( int argc, char **argv )
{

	// change type based on command line input
	for( int i = 1; i < argc; i++ )
	{
		handle_arg(argv[i]);
	}

	// seed random number generator
	srand(time(NULL));

	// run benchmark
	switch( benchmark_type )
	{
		case ACML:
			run_acml();
			break;
		case MATLAB:
			run_matlab();
			break;
	}


	return 0;
}


