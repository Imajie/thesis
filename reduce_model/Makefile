all: reduce

%:
	nvcc -ptx -arch=sm_13 reduction_kernel.cu -o $@_13.ptx
	nvcc -cubin -arch=sm_13 reduction_kernel.cu -o $@_13.cubin
	cuobjdump -sass $@_13.cubin > $@_13.asm
	rm $@_13.cubin
