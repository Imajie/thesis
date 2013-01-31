/*
 * File:	acml_bench.cpp
 *
 * Author:	James Letendre
 *
 * Function prototypes for ACML benchmark run
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "benchmark_common.h"
#include "acml_bench.h"

#include <acml.h>

/*
 * run_acml
 *
 * Runs the ACML benchmark
 */
void run_acml( void )
{
	// used for computations
	double *A_d, *B_d, *u_d, *v_d;
	float  *A_f, *B_f, *u_f, *v_f;

	// timing
	double elapsed = 0.0;

	switch( benchmark_precision )
	{
		case SINGLE:
			for( size_t i = 0; i < benchmark_num; i++ )
			{
				int n = benchmark_sizes[i];
				printf("\nACML Single Precision: N = %i\n", n);

				// Generate random matrices A, B
				// and random vectors u, v
				get_matrix_single(&A_f, n);
				get_matrix_single(&B_f, n);
				get_vector_single(&u_f, n);
				get_vector_single(&v_f, n);

				// Benchmark 1: Dot product
				start_time();
				for( size_t j = 0; j < NUM_ITER; j++ )
				{
					sdot(n, u_f, 1, v_f, 1);
				}
				elapsed += stop_time("\tDot product");

				// Benchmark 2: Matrix-Vector Multiplication
				start_time();
				for( size_t j = 0; j < NUM_ITER; j++ )
				{
					sgemv('n', n, n, 1.0, A_f, n, u_f, 1, 0.0, v_f, 1);
				}
				elapsed += stop_time("\tMatrix-Vector multiplication");
				
				// Benchmark 3: Matrix-Matrix Multiplication
				start_time();
				for( size_t j = 0; j < NUM_ITER; j++ )
				{
					sgemm('n', 't', n, n, n, 1.0, A_f, n, A_f, n, 0.0, B_f, n);
				}
				elapsed += stop_time("\tMatrix-Matrix multiplication");
				
				// Benchmark 4: Cholesky Decomposition
				start_time();
				for( size_t j = 0; j < NUM_ITER; j++ )
				{
					int info;
					// from last benchmark B = A*A^T - symmetric
					memcpy( A_f, B_f, n*n*sizeof(float) );
					spotrf('L', n, A_f, n, &info);

					if( info != 0 )
					{
						printf("ERROR in Cholesky\n");
						break;
					}
				}
				elapsed += stop_time("\tCholesky Decomposition");

				/*
				// Benchmark 5: LU Decomposition
				int piv[benchmark_sizes[benchmark_num-1]];
				start_time();
				for( size_t j = 0; j < NUM_ITER; j++ )
				{
					int info;
					sgetrf(n, n, A_f, n, piv, &info);

					if( info != 0 )
					{
						printf("ERROR in LU\n");
						break;
					}
				}
				elapsed += stop_time("\tLU Decomposition");
				*/

				memcpy( B_f, A_f, n*n*sizeof(float) );
				// Benchmark 6: Matrix Inverse
				start_time();
				for( size_t j = 0; j < NUM_ITER; j++ )
				{
					int info;
					memcpy( A_f, B_f, n*n*sizeof(float) );
					spotri('L', n, A_f, n, &info);

					if( info != 0 )
					{
						printf("ERROR in inverse\n");
						break;
					}
				}
				elapsed += stop_time("\tMatrix Inverse");
				
				free( A_f );
				free( B_f );
				free( u_f );
				free( v_f );
			}

			break;

		case DOUBLE:
			for( size_t i = 0; i < benchmark_num; i++ )
			{
				int n = benchmark_sizes[i];
				printf("\nACML Double Precision: N = %i\n", n);

				// Generate random matrices A, B
				// and random vectors u, v
				get_matrix_double(&A_d, benchmark_sizes[i]);
				get_matrix_double(&B_d, benchmark_sizes[i]);
				get_vector_double(&u_d, benchmark_sizes[i]);
				get_vector_double(&v_d, benchmark_sizes[i]);

				// Benchmark 1: Dot product
				start_time();
				for( size_t j = 0; j < NUM_ITER; j++ )
				{
					ddot(n, u_d, 1, v_d, 1);
				}
				elapsed += stop_time("\tDot product");

				// Benchmark 2: Matrix-Vector Multiplication
				start_time();
				for( size_t j = 0; j < NUM_ITER; j++ )
				{
					dgemv('n', n, n, 1.0, A_d, n, u_d, 1, 0.0, v_d, 1);
				}
				elapsed += stop_time("\tMatrix-Vector multiplication");
				
				// Benchmark 3: Matrix-Matrix Multiplication
				start_time();
				for( size_t j = 0; j < NUM_ITER; j++ )
				{
					dgemm('n', 't', n, n, n, 1.0, A_d, n, A_d, n, 0.0, B_d, n);
				}
				elapsed += stop_time("\tMatrix-Matrix multiplication");
				
				// Benchmark 4: Cholesky Decomposition
				start_time();
				for( size_t j = 0; j < NUM_ITER; j++ )
				{
					int info;
					memcpy( A_d, B_d, n*n*sizeof(double) );
					dpotrf('L', n, A_d, n, &info);

					if( info != 0 )
					{
						printf("ERROR in Cholesky\n");
						break;
					}
				}
				elapsed += stop_time("\tCholesky Decomposition");

				/*
				// Benchmark 5: LU Decomposition
				int piv[benchmark_sizes[benchmark_num-1]];
				start_time();
				for( size_t j = 0; j < NUM_ITER; j++ )
				{
					int info;

					dgetrf(n, n, A_d, n, piv, &info);

					if( info != 0 )
					{
						printf("ERROR in LU\n");
						break;
					}
				}
				elapsed += stop_time("\tLU Decomposition");
				*/

				// Benchmark 6: Matrix Inverse
				start_time();
				for( size_t j = 0; j < NUM_ITER; j++ )
				{
					int info;
					dpotri('L', n, A_d, n, &info);

					if( info != 0 )
					{
						printf("ERROR in inverse\n");
						break;
					}
				}
				elapsed += stop_time("\tMatrix Inverse");

				free( A_d );
				free( B_d );
				free( u_d );
				free( v_d );
			}

			break;
	}
}

