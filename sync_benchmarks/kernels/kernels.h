/*
 * Kernels for sync cost benchmarks
 */
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

/*
 * Write some data to global memory, then synchronize the warps
 */
__global__ void global_mem_write_kernel( float* data, unsigned int *start, unsigned int *end );

/*
 * Read some data from global memory, then synchronize the warps
 */
__global__ void global_mem_read_kernel( float* data, unsigned int *start, unsigned int *end, int use_cache );

/*
 * Write some data to shared memory, then synchronize the warps
 */
__global__ void shared_mem_write_kernel( unsigned int *start, unsigned int *end );

/*
 * Read some data from shared memory, then synchronize the warps
 */
__global__ void shared_mem_read_kernel( unsigned int *start, unsigned int *end );


void format_data( unsigned int *start, unsigned int *end, int threads, int blocks );
