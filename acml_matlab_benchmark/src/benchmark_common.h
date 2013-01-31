/*
 * File:	benchmark_common.h
 *
 * Author:	James Letendre
 *
 * Common types and variables for the benchmark
 */
#ifndef BENCHMARK_COMMON_H
#define BENCHMARK_COMMON_H

typedef enum
{
	ACML,
	MATLAB
} benchmark_type_t;

typedef enum
{
	SINGLE,
	DOUBLE
} benchmark_precision_t;

// the benchmark type and precision
extern benchmark_type_t 		benchmark_type;
extern benchmark_precision_t	benchmark_precision;

// the sizes to run
extern unsigned int benchmark_sizes[];
extern unsigned int benchmark_num;

// number of times to run each operation
#define NUM_ITER	10

// For timing the benchmarks
void start_time(void);
double stop_time( const char *str );

// Get random matrices and vectors
void get_matrix_single( float **A, unsigned int n );
void get_vector_single( float **v, unsigned int n );
void get_matrix_double( double **A, unsigned int n );
void get_vector_double( double **v, unsigned int n );

#endif
