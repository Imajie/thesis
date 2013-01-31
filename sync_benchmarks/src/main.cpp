/*
 * Benchmarks for synchronization cost
 */
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "global_mem.h"
#include "shared_mem.h"

cudaError_t __cuda__error__;
#define CUDA_CATCH_ERROR(x)															\
( (x), ( ((__cuda__error__ = cudaGetLastError()) == cudaSuccess) ? (0) : 			\
			(fprintf(stderr, "%s:%i - Error(%d) \"%s\" in %s\n", __FILE__, __LINE__, __cuda__error__, cudaGetErrorString(__cuda__error__), #x), (1)) ) )
#define CUDA_CATCH_ERROR_EXIT(x)	( CUDA_CATCH_ERROR(x) ? (exit(1), 1) : 0 )

enum bench_op { READ, WRITE };
enum bench_mem { GLOBAL, SHARED };

int main( int argc, char** argv )
{
	// initialize the cuda device
	CUDA_CATCH_ERROR_EXIT( cudaSetDevice(0) );

	enum bench_op benchmark = READ;
	enum bench_mem mem_type = GLOBAL;
	int blocks = 10;
	int threads = 128;
	int avg_count = 1;

	int use_cache = 0;

	if( argc > 1 )
	{
		blocks = atoi(argv[1]);
	}

	if( argc > 2 )
	{
		threads = atoi(argv[2]);
	}

	if( argc > 3 )
	{
		if( strcmp( argv[3], "write" ) == 0 )
			benchmark = WRITE;
	}

	if( argc > 4 )
	{
		if( strcmp( argv[4], "shared" ) == 0 )
			mem_type = SHARED;
	}

	if( argc > 5 )
	{
		avg_count = atoi(argv[5]);
	}

	switch( mem_type )
	{
		case GLOBAL:
			switch( benchmark )
			{
				case WRITE:
					printf("Running Global Write\n");
					for(int i=0;i<avg_count;i++ ) global_mem_write( blocks, threads );
					break;
				case READ:
					printf("Running Global Read\n");
					fprintf(stderr, "_NO_CACHE_\n");
					for(int i=0;i<avg_count;i++ ) global_mem_read( blocks, threads, false);

					fprintf(stderr, "_WITH_CACHE_\n");
					for(int i=0;i<avg_count;i++ ) global_mem_read( blocks, threads, true);
					break;
			}
			break;

		case SHARED:
			switch( benchmark )
			{
				case WRITE:
					printf("Running Shared Write\n");
					for(int i=0;i<avg_count;i++ ) shared_mem_write( blocks, threads );
					break;
				case READ:
					printf("Running Shared Read\n");
					for(int i=0;i<avg_count;i++ ) shared_mem_read( blocks, threads );
					break;
			}
			break;
	}


	return 0;
}
