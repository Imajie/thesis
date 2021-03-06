
// global memory write kernel launch function
void global_mem_write( bool sim, unsigned int num_blocks, unsigned int num_threads );

// global memory read kernel launch function
void global_mem_read( bool sim, unsigned int num_blocks, unsigned int num_threads, bool use_cache );
