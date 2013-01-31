/*
 * shared memory benchmarks for sync cost
 */
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "shared_mem.h"
#include "kernels.h"

void shared_mem_write( unsigned int num_blocks, unsigned int num_threads )
{
	int block_size;
	int threads;

	// ensure minimal number of blocks and threads
	if( num_blocks == 0 )
		block_size = 1;
	else
		block_size = num_blocks;
	
	if( num_threads < 32 )
		threads = 32;
	else
		threads = num_threads;

	// initialize data to 0.0
	unsigned int *start = new unsigned int[ block_size*threads ];
	unsigned int *end = new unsigned int[ block_size*threads ];
	for( int i = 0; i < block_size*threads; i++ )
	{
		start[i] = 0.0;
		end[i] = 0.0;
	}

	// allocate space on gpu, and copy it
	unsigned int *start_gpu, *end_gpu;
	cudaMalloc( &start_gpu, block_size*threads*sizeof(unsigned int) );
	cudaMalloc( &end_gpu, block_size*threads*sizeof(unsigned int) );

	cudaMemcpy( start_gpu, start, block_size*threads*sizeof(unsigned int), cudaMemcpyHostToDevice );
	cudaMemcpy( end_gpu, end, block_size*threads*sizeof(unsigned int), cudaMemcpyHostToDevice );
	
	dim3 blocks( block_size, 1, 1 );
	shared_mem_write_kernel<<< blocks, threads, 0 >>>(start_gpu, end_gpu);

	cudaDeviceSynchronize();
	
	// copy data back, ensures kernel has finished
	cudaMemcpy( start, start_gpu, block_size*threads*sizeof(unsigned int), cudaMemcpyDeviceToHost );
	cudaMemcpy( end, start_gpu, block_size*threads*sizeof(unsigned int), cudaMemcpyDeviceToHost );

	format_data( start, end, threads, block_size );

	// now free the memory
	cudaFree( start_gpu );
	cudaFree( start_gpu );

	free( start );
	free( end );
}

void shared_mem_read( unsigned int num_blocks, unsigned int num_threads )
{
	int block_size;
	int threads;

	// ensure minimal number of blocks and threads
	if( num_blocks == 0 )
		block_size = 1;
	else
		block_size = num_blocks;
	
	if( num_threads < 32 )
		threads = 32;
	else
		threads = num_threads;

	// initialize data to 0.0
	unsigned int *start = new unsigned int[ block_size*threads ];
	unsigned int *end = new unsigned int[ block_size*threads ];
	for( int i = 0; i < block_size*threads; i++ )
	{
		start[i] = 0.0;
		end[i] = 0.0;
	}

	// allocate space on gpu, and copy it
	unsigned int *start_gpu, *end_gpu;
	cudaMalloc( &start_gpu, block_size*threads*sizeof(unsigned int) );
	cudaMalloc( &end_gpu, block_size*threads*sizeof(unsigned int) );

	cudaMemcpy( start_gpu, start, block_size*threads*sizeof(unsigned int), cudaMemcpyHostToDevice );
	cudaMemcpy( end_gpu, end, block_size*threads*sizeof(unsigned int), cudaMemcpyHostToDevice );
	
	dim3 blocks( block_size, 1, 1 );
	shared_mem_read_kernel<<< blocks, threads, 0 >>>(start_gpu, end_gpu);

	cudaDeviceSynchronize();

	// copy data back, ensures kernel has finished
	cudaMemcpy( start, start_gpu, block_size*threads*sizeof(unsigned int), cudaMemcpyDeviceToHost );
	cudaMemcpy( end, start_gpu, block_size*threads*sizeof(unsigned int), cudaMemcpyDeviceToHost );

	format_data( start, end, threads, block_size );

	// now free the memory
	cudaFree( start_gpu );
	cudaFree( start_gpu );

	free( start );
	free( end );
}
