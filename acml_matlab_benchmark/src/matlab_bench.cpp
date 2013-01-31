/*
 * File:	matlab_bench.cpp
 *
 * Author:	James Letendre
 *
 * Function prototypes for Matlab benchmark run
 */
#include "benchmark_common.h"
#include "matlab_bench.h"

#include <stdlib.h>
#include <stdio.h>

#include <engine.h>
#include <mex.h>
#include <matrix.h>

/*
 * run_matlab
 *
 * Runs the Matlab benchmark
 */
void run_matlab( void )
{
	double elapsed = 0.0;

	Engine *eng;

	// matlab variables
	mxArray *A, *B, *u, *v;

	// open Matlab
	if( !(eng = engOpen("matlab")) )
	{
		printf("Can't open Matlab\n");
		return;
	}

	for( size_t i = 0; i < benchmark_num; i++ )
	{
		int n = benchmark_sizes[i];
		printf("\nMatlab %s precision: N = %i\n", benchmark_precision == SINGLE ? "single" : "double", n);

		// allocate memory for parameters/return values
		A = mxCreateNumericMatrix(n, n, benchmark_precision == SINGLE ? mxSINGLE_CLASS : mxDOUBLE_CLASS, mxREAL);
		B = mxCreateNumericMatrix(n, n, benchmark_precision == SINGLE ? mxSINGLE_CLASS : mxDOUBLE_CLASS, mxREAL);
		u = mxCreateNumericMatrix(n, 1, benchmark_precision == SINGLE ? mxSINGLE_CLASS : mxDOUBLE_CLASS, mxREAL);
		v = mxCreateNumericMatrix(n, 1, benchmark_precision == SINGLE ? mxSINGLE_CLASS : mxDOUBLE_CLASS, mxREAL);

		switch(benchmark_precision)
		{
			case SINGLE:
				{
					float *t;
					float *m;

					get_matrix_single(&t, n);
					m = (float*)mxMalloc(n*n*sizeof(float));
					for( size_t j = 0; j < n*n; j++ )
						m[i] = t[i];
					mxSetData(A, m);
					free(t);

					get_matrix_single(&t, n);
					m = (float*)mxMalloc(n*n*sizeof(float));
					for( size_t j = 0; j < n*n; j++ )
						m[i] = t[i];
					mxSetData(B, m);
					free(t);

					get_vector_single(&t, n);
					m = (float*)mxMalloc(n*sizeof(float));
					for( size_t j = 0; j < n; j++ )
						m[i] = t[i];
					mxSetData(u, m);
					free(t);

					get_vector_single(&t, n);
					m = (float*)mxMalloc(n*sizeof(float));
					for( size_t j = 0; j < n; j++ )
						m[i] = t[i];
					mxSetData(v, m);
					free(t);
				}

				break;
			case DOUBLE:
				{
					double *t;
					double *m;

					get_matrix_double(&t, n);
					m = (double*)mxMalloc(n*n*sizeof(double));
					for( size_t j = 0; j < n*n; j++ )
						m[i] = t[i];
					mxSetData(A, m);
					free(t);

					get_matrix_double(&t, n);
					m = (double*)mxMalloc(n*n*sizeof(double));
					for( size_t j = 0; j < n*n; j++ )
						m[i] = t[i];
					mxSetData(B, m);
					free(t);

					get_vector_double(&t, n);
					m = (double*)mxMalloc(n*sizeof(double));
					for( size_t j = 0; j < n; j++ )
						m[i] = t[i];
					mxSetData(u, m);
					free(t);

					get_vector_double(&t, n);
					m = (double*)mxMalloc(n*sizeof(double));
					for( size_t j = 0; j < n; j++ )
						m[i] = t[i];
					mxSetData(v, m);
					free(t);
				}

				break;
		}

		engPutVariable( eng, "A", A );
		engPutVariable( eng, "B", A );
		engPutVariable( eng, "u", A );
		engPutVariable( eng, "v", A );

		mxDestroyArray( A );
		mxDestroyArray( B );
		mxDestroyArray( u );
		mxDestroyArray( v );

		// Benchmark 1: Dot product
		start_time();
		for( size_t i = 0; i < NUM_ITER; i++ )
		{
			if( engEvalString(eng, "d=dot(u,v);") != 0 )
			{
				printf("Error in Dot Product\n");
				break;
			}
		}
		elapsed += stop_time("\tDot Product");

		// Benchmark 2: Matrix-Vector Multiplication
		start_time();
		for( size_t i = 0; i < NUM_ITER; i++ )
		{
			if( engEvalString(eng, "b=A*u;") != 0 )
			{
				printf("Error in Matrix-Vector Multiply\n");
				break;
			}
		}
		elapsed += stop_time("\tMatrix-Vector Multiply");

		// Benchmark 3: Matrix-Matrix Multiplication
		start_time();
		for( size_t i = 0; i < NUM_ITER; i++ )
		{
			if( engEvalString(eng, "m=A*A';") != 0 )
			{
				printf("Error in Matrix-Matrix Multiply\n");
				break;
			}
		}
		elapsed += stop_time("\tMatrix-Matrix Multiply");

		// Benchmark 4: Cholesky Decomposition
		start_time();
		for( size_t i = 0; i < NUM_ITER; i++ )
		{
			if( engEvalString(eng, "[c,p]=chol(m);") != 0 )
			{
				printf("Error in Cholesky Decomposition\n");
				break;
			}
		}
		elapsed += stop_time("\tCholesky Decomposition");

		// Benchmark 5: LU Decomposition
		start_time();
		for( size_t i = 0; i < NUM_ITER; i++ )
		{
			if( engEvalString(eng, "[L,U,P]=lu(A);") != 0 )
			{
				printf("Error in LU Decomposition\n");
				break;
			}
		}
		elapsed += stop_time("\tLU Decomposition");

		// Benchmark 6: Matrix Inverse
		start_time();
		for( size_t i = 0; i < NUM_ITER; i++ )
		{
			if( engEvalString(eng, "IA=inv(A);") != 0 )
			{
				printf("Error in Matrix Inverse\n");
				break;
			}
		}
		elapsed += stop_time("\tMatrix Inverse");

	}

	// close Matlab
	engClose(eng);
}

