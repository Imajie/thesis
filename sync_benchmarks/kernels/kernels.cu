/*
 * Kernels for sync cost benchmarks
 */
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "kernels.h"
#include <stdio.h>

/*
 * Write some data to global memory, then synchronize the warps
 */
__global__ void global_mem_write_kernel( float* data, unsigned int *start, unsigned int *end )
{
	// The index of the current thread
	int idx = threadIdx.x + (blockDim.x * blockIdx.x);

	unsigned int start_reg, end_reg;

	// memory write
	data[idx] = idx;

	asm volatile("mov.u32 %0, %%clock;" : "=r"(start_reg));		// time we started waiting
	__syncthreads();
	asm volatile("mov.u32 %0, %%clock;" : "=r"(end_reg));		// time we finished waiting

	// Save times to send back to the CPU
	start[idx] = start_reg;
	end[idx] = end_reg;
}


/*
 * Read some data from global memory, then synchronize the warps
 */
__global__ void global_mem_read_kernel( float* data, unsigned int *start, unsigned int *end, int use_cache )
{
	int idx = threadIdx.x + (blockDim.x * blockIdx.x);

	unsigned int start_reg, end_reg;

	if( use_cache )
	{
		// do an initial operation to cache the data
		data[idx] += 1;

		// sync here so only cached read effects second sync
		__syncthreads();
	}

	float val = data[idx];

	asm volatile("mov.u32 %0, %%clock;" : "=r"(start_reg));
	__syncthreads();
	asm volatile("mov.u32 %0, %%clock;" : "=r"(end_reg));

	start[idx] = start_reg;
	end[idx] = end_reg;

	// get rid of "never used val" warning
	if( val > 0 )
		start_reg = end_reg;
}

/*
 * Write some data to shared memory, then synchronize the warps
 */
__global__ void shared_mem_write_kernel( unsigned int *start, unsigned int *end )
{
	__shared__ float data[ 512 ];
	int idx = threadIdx.x + (blockDim.x * blockIdx.x);
	int idx2 = threadIdx.x;

	unsigned int start_reg, end_reg;
	data[idx2] = idx;

	asm volatile("mov.u32 %0, %%clock;" : "=r"(start_reg));
	__syncthreads();
	asm volatile("mov.u32 %0, %%clock;" : "=r"(end_reg));

	start[idx] = start_reg;
	end[idx] = end_reg;

	// get rid of "never used data" warning
	if( data[idx2] >= 0 )
		start_reg = end_reg;
}

/*
 * Read some data from shared memory, then synchronize the warps
 */
__global__ void shared_mem_read_kernel( unsigned int *start, unsigned int *end )
{
	__shared__ float data[ 512 ];
	int idx = threadIdx.x + (blockDim.x * blockIdx.x);
	int idx2 = threadIdx.x;
	unsigned int start_reg, end_reg;

	float val = data[idx2];

	asm volatile("mov.u32 %0, %%clock;" : "=r"(start_reg));
	__syncthreads();
	asm volatile("mov.u32 %0, %%clock;" : "=r"(end_reg));

	start[idx] = start_reg;
	end[idx] = end_reg;

	data[idx2] = val+start_reg;
}

void format_data( unsigned int *start, unsigned int *end, int block_size, int blocks )
{
	unsigned int max_elapsed = 0;
	unsigned int max_start = 0;

	for( int i = 0; i < block_size*blocks; i+=32 )
	{
		//printf("%i %i\n", start[i], end[i]);

		if( i % block_size == 0 )
		{
			if( max_elapsed > 0 )
			{
				fprintf(stderr, "0 0 %i 0 %i\n", max_start, max_elapsed);
				max_elapsed = 0;
			}
		}

		unsigned int elapsed = end[i] - start[i];

		if( elapsed > max_elapsed )
		{
			max_elapsed = elapsed;
			max_start = start[i];
		}
	}

}
