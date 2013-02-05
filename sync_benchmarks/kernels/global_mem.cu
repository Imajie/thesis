/*
 * Global memory benchmarks for sync cost
 */
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "global_mem.h"
#include "kernels.h"

void global_mem_write( bool sim, unsigned int num_blocks, unsigned int num_threads )
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
	float *data_cpu = new float[ block_size*threads ];
	unsigned int *start = new unsigned int[ block_size*threads ];
	unsigned int *end = new unsigned int[ block_size*threads ];
	for( int i = 0; i < block_size*threads; i++ )
	{
		data_cpu[i] = 0.0;
		start[i] = 0.0;
		end[i] = 0.0;
	}

	// allocate space on gpu, and copy it
	float *data_gpu;
	unsigned int *start_gpu, *end_gpu;
	cudaMalloc( &data_gpu, block_size*threads*sizeof(float) );
	cudaMalloc( &start_gpu, block_size*threads*sizeof(unsigned int) );
	cudaMalloc( &end_gpu, block_size*threads*sizeof(unsigned int) );

	cudaMemcpy( data_gpu, data_cpu, block_size*threads*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( start_gpu, start, block_size*threads*sizeof(unsigned int), cudaMemcpyHostToDevice );
	cudaMemcpy( end_gpu, end, block_size*threads*sizeof(unsigned int), cudaMemcpyHostToDevice );
	
	dim3 blocks( block_size, 1, 1 );
	global_mem_write_kernel<<< blocks, threads, 0 >>>( data_gpu, start_gpu, end_gpu );

	cudaDeviceSynchronize();
	
	// copy data back, ensures kernel has finished
	cudaMemcpy( data_cpu, data_gpu, block_size*threads*sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( start, start_gpu, block_size*threads*sizeof(unsigned int), cudaMemcpyDeviceToHost );
	cudaMemcpy( end, end_gpu, block_size*threads*sizeof(unsigned int), cudaMemcpyDeviceToHost );

	if( !sim ) format_data( start, end, threads, block_size );

	// now free the memory
	cudaFree( data_gpu );
	cudaFree( start_gpu );
	cudaFree( end_gpu );

	free( data_cpu );
	free( start );
	free( end );
}

void global_mem_read( bool sim, unsigned int num_blocks, unsigned int num_threads, bool use_cache )
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
	float *data_cpu = new float[ block_size*threads ];
	unsigned int *start = new unsigned int[ block_size*threads ];
	unsigned int *end = new unsigned int[ block_size*threads ];
	for( int i = 0; i < block_size*threads; i++ )
	{
		data_cpu[i] = 0.0;
		start[i] = 0.0;
		end[i] = 0.0;
	}

	// allocate space on gpu, and copy it
	float *data_gpu;
	unsigned int *start_gpu, *end_gpu;
	cudaMalloc( &data_gpu, block_size*threads*sizeof(float) );
	cudaMalloc( &start_gpu, block_size*threads*sizeof(unsigned int) );
	cudaMalloc( &end_gpu, block_size*threads*sizeof(unsigned int) );

	cudaMemcpy( data_gpu, data_cpu, block_size*threads*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( start_gpu, start, block_size*threads*sizeof(unsigned int), cudaMemcpyHostToDevice );
	cudaMemcpy( end_gpu, end, block_size*threads*sizeof(unsigned int), cudaMemcpyHostToDevice );
	
	dim3 blocks( block_size, 1, 1 );
	global_mem_read_kernel<<< blocks, threads, 0 >>>( data_gpu, start_gpu, end_gpu, use_cache );

	cudaDeviceSynchronize();
	
	// copy data back, ensures kernel has finished
	cudaMemcpy( data_cpu, data_gpu, block_size*threads*sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( start, start_gpu, block_size*threads*sizeof(unsigned int), cudaMemcpyDeviceToHost );
	cudaMemcpy( end, end_gpu, block_size*threads*sizeof(unsigned int), cudaMemcpyDeviceToHost );

	if( !sim ) format_data( start, end, threads, block_size );

	// now free the memory
	cudaFree( data_gpu );
	cudaFree( start_gpu );
	cudaFree( end_gpu );

	free( data_cpu );
	free( start );
	free( end );
}
