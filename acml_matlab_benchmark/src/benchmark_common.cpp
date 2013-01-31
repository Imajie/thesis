/*
 * File:	benchmark_common.cpp
 *
 * Author:	James Letendre
 *
 * Common types and variables for the benchmark
 */
#include "benchmark_common.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
	#include <windows.h>
	LARGE_INTEGER start = {0}, freq;
#else	// LINUX
	#include <time.h>
	static struct timespec start = { 0, 0 };
#endif

// For timing the benchmarks
void start_time(void)
{
#ifdef _WIN32
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
#else	// LINUX
	clock_gettime(CLOCK_MONOTONIC, &start);
#endif
}

double stop_time( const char *str )
{
	printf("%-60.60s", str);

#ifdef __WIN32
	if( start.QuadPart == 0 )
#else
	if( start.tv_sec == 0 && start.tv_nsec == 0 )
#endif
	{
		// not initialized
		printf("start_time() not called\n");
		return 0.0;
	}

	double elapsed = 0.0;
#ifdef _WIN32
	LARGE_INTEGER stop;
	QueryPerformanceCounter(&stop);

	elapsed = (stop.QuadPart - start.QuadPart)*1000000.0/freq.QuadPart;
#else	// LINUX
	struct timespec end;
	clock_gettime(CLOCK_MONOTONIC, &end);

	elapsed = (end.tv_sec - start.tv_sec)*1000000.0 + (end.tv_nsec - start.tv_nsec)/1000.0;
#endif

	printf("%lf\t%lf\n", elapsed, elapsed/NUM_ITER);
	fprintf(stderr, "%s\t%lf\n", str, elapsed/NUM_ITER);

	return elapsed;
}

#define GET_RAND(type, min, max)	(type)( ((type)rand())/((type)RAND_MAX)*((max)-(min)) + (min) )
// Get random matrices and vectors
void get_matrix_single( float **A, unsigned int n )
{
	*A = (float*)malloc( n*n*sizeof(float) );

	for( size_t i = 0; i < n*n; i++ )
	{
		(*A)[i] = GET_RAND(float, 0, 1);
	}
}

void get_vector_single( float **v, unsigned int n )
{
	*v = (float*)malloc( n*sizeof(float) );

	for( size_t i = 0; i < n; i++ )
	{
		(*v)[i] = GET_RAND(float, 0, 1);
	}
}

void get_matrix_double( double **A, unsigned int n )
{
	*A = (double*)malloc( n*n*sizeof(double) );

	for( size_t i = 0; i < n*n; i++ )
	{
		(*A)[i] = GET_RAND(double, 0, 1);
	}
}

void get_vector_double( double **v, unsigned int n )
{
	*v = (double*)malloc( n*n*sizeof(double) );

	for( size_t i = 0; i < n; i++ )
	{
		(*v)[i] = GET_RAND(double, 0, 1);
	}
}

