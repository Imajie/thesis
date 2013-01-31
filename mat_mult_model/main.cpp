#include <cstdio>
#include <cstdlib>
#include <time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Matrix sizes A = NxP B = PxM
#define MATRIX_N 100
#define MATRIX_P 100
#define MATRIX_M 100
#define IDX2C(i, j, ld) (((j)*(ld)+i))

int main( int argc, char **argv )
{
	double *A, *B, *C;
	double *cu_A, *cu_B, *cu_C;

	cudaError_t    cuError;
	cublasStatus_t cuStatus;
	cublasHandle_t cuHandle;

	// seed rand()
	srand(time(NULL));

	// allocate memory on CPU
	A = (double*)malloc(sizeof(double)*MATRIX_N*MATRIX_P);
	B = (double*)malloc(sizeof(double)*MATRIX_P*MATRIX_M);
	C = (double*)malloc(sizeof(double)*MATRIX_N*MATRIX_M);

	if( !A || !B || !C )
	{
		perror("Can't allocate CPU matrices");
		exit(EXIT_FAILURE);
	}

	// generate matrices
	for( int i = 0; i < MATRIX_N*MATRIX_P; i++ )
		A[i] = 10.0*((double)rand())/RAND_MAX;

	for( int i = 0; i < MATRIX_P*MATRIX_M; i++ )
		B[i] = 10.0*((double)rand())/RAND_MAX;

	// allocate memory on GPU
	cuError = cudaMalloc( &cu_A, sizeof(double)*MATRIX_N*MATRIX_P );

	if( cuError != cudaSuccess )
	{
		fprintf(stderr, "Can't allocate GPU matrices\n");
		exit(EXIT_FAILURE);
	}

	cuError = cudaMalloc( &cu_B, sizeof(double)*MATRIX_P*MATRIX_M );

	if( cuError != cudaSuccess )
	{
		fprintf(stderr, "Can't allocate GPU matrices\n");
		exit(EXIT_FAILURE);
	}

	cuError = cudaMalloc( &cu_C, sizeof(double)*MATRIX_N*MATRIX_M );

	if( cuError != cudaSuccess )
	{
		fprintf(stderr, "Can't allocate GPU matrices\n");
		exit(EXIT_FAILURE);
	}

	// setup cuBlas
	cuStatus = cublasCreate( &cuHandle );
	if( cuStatus != CUBLAS_STATUS_SUCCESS )
	{
		fprintf(stderr, "Error initializing cuBlas\n");
		exit(EXIT_FAILURE);
	}

	// setup matrices
	cuStatus = cublasSetMatrix( MATRIX_N, MATRIX_P, sizeof(double), A, MATRIX_N, cu_A, MATRIX_N );
	if( cuStatus != CUBLAS_STATUS_SUCCESS )
	{
		fprintf(stderr, "Error transferring matrix A\n");
		exit(EXIT_FAILURE);
	}

	cuStatus = cublasSetMatrix( MATRIX_P, MATRIX_M, sizeof(double), B, MATRIX_P, cu_B, MATRIX_P );
	if( cuStatus != CUBLAS_STATUS_SUCCESS )
	{
		fprintf(stderr, "Error transferring matrix B\n");
		exit(EXIT_FAILURE);
	}

	// multiply
	double one  = 1.0;
	double zero = 0.0;
	cuStatus = cublasDgemm( cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_N, MATRIX_M, MATRIX_P, &one, cu_A, MATRIX_N, cu_B, MATRIX_P, &zero, cu_C, MATRIX_N );

	if( cuStatus != CUBLAS_STATUS_SUCCESS )
	{
		fprintf(stderr, "Error executing matrix mult\n");
		exit(EXIT_FAILURE);
	}

	// get results
	cuStatus = cublasGetMatrix( MATRIX_N, MATRIX_M, sizeof(double), cu_C, MATRIX_N, C, MATRIX_N );
	if( cuStatus != CUBLAS_STATUS_SUCCESS )
	{
		fprintf(stderr, "Error transferring results\n");
		exit(EXIT_FAILURE);
	}
	
	// check results
	bool good = true;
	for( int i = 0; i < MATRIX_N; i++ )
	{
		for( int j = 0; j < MATRIX_M; j++ )
		{
			double sum = 0.0;
			for( int k = 0; k < MATRIX_P; k++ )
			{
				sum += A[IDX2C(i, k, MATRIX_N)]*B[IDX2C(k, j, MATRIX_P)];
			}
			// check
			if( fabs(sum - C[IDX2C(i,j,MATRIX_N)]) > 0.00001 )
			{
				good = false;
				printf("(%i, %i) sum = %f\tcu_C = %f\tMISMATCH\n", i, j, sum, C[IDX2C(i,j,MATRIX_N)]);
			}
		}
	}

	if( good )
		printf("Results Match\n");
	else
		printf("Results DO NOT Match\n");

	// cleanup
	free( A ); free( B ); free( C );
	cudaFree( cu_A ); cudaFree( cu_B ); cudaFree( cu_C );
	cublasDestroy( cuHandle );

	return 0;
}
