# All directories that contain source code
DIRS=src kernels

# Name of the program to generate
TARGET=benchmark

ARCHS=13 20

###
### Shouldn't need to change below this line
###

#
# compilers
#
CXX=g++-4.4 -fPIC -ggdb
CC=gcc-4.4 -fPIC -ggdb
NVCC=nvcc

#
# source files
#
CXX_SOURCES=$(foreach dir,${DIRS},$(wildcard ${dir}/*.cpp)) 
CU_SOURCES=$(foreach dir,${DIRS},$(wildcard ${dir}/*.cu)) 

CXX_OBJS=$(subst .cpp,.o,${CXX_SOURCES})
CU_OBJS=$(subst .cu,.o,${CU_SOURCES})
OBJS=${CU_OBJS} ${CXX_OBJS}

#
# Includes
#
INCLUDE_DIRS=$(foreach dir,${DIRS},-I${dir}) 

# toolkit includes
CUDA_DIRS=-I$(CUDA_INSTALL_PATH)/include
MATLAB_DIRS=-I/usr/local/MATLAB/R2012a/extern/include
ACML_DIRS=-I/opt/acml4.4.0/gfortran32_mp/include

# all include flags
FLAGS=${INCLUDE_DIRS} ${CUDA_DIRS} 

#
# Libraries
#
LIB_DIRS=-L/usr/lib/i386-linux-gnu/

# libraries for specific toolkits
CUDA_LIBS=-L$(CUDA_INSTALL_PATH)/lib -lcudart
MATLAB_LIBS=-L/usr/local/MATLAB/R2012a/bin/glnx86 -lmat -lmex -leng -lut -lmx
BLAS_LIBS=-L/opt/acml4.4.0/gfortran32/lib -lacml -llapack -lgfortran
OCELOT_LIBS=-Wl,--no-as-needed -locelot -locelotTrace -Wl,--as-needed
BOOST_LIBS=-lboost_system-mt -lboost_filesystem-mt

# all libraries
LIBS=${CUDA_LIBS} ${BOOST_LIBS}

#
# Flags
#
CXXFLAGS=${FLAGS} -ggdb
CFLAGS=${FLAGS} -ggdb
NVCC_FLAGS=${FLAGS} $(foreach arch,$(ARCHS),-gencode arch=compute_$(arch),code=\"sm_$(arch),compute_$(arch)\")
LINK_FLAGS=${LIB_DIRS} ${LIBS} -pthread

all: ${TARGET}

${TARGET}: ${OBJS}
	${CXX} -o ${TARGET} ${OBJS} ${LINK_FLAGS}
	
%.o: %.cu
	${NVCC} -o $@ -c $< ${NVCC_FLAGS}

.PHONY: cpp_clean cu_clean clean realclean depend
clean: cpp_clean cu_clean

cpp_clean:
	@$(foreach obj,${CXX_OBJS},$(shell rm ${obj} 2> /dev/null))

cu_clean:
	@$(foreach obj,${CU_OBJS},$(shell rm ${obj} 2> /dev/null))

realclean: clean
	@-rm ${TARGET} 2> /dev/null

depend:
	@-makedepend $(CXX_SOURCES) $(CU_SOURCES) -- $(FLAGS) 2> /dev/null
	@-rm Makefile.bak

# DO NOT DELETE

src/main.o: /usr/local/cuda-4.0/cuda/include/cuda_runtime.h
src/main.o: /usr/local/cuda-4.0/cuda/include/host_config.h
src/main.o: /usr/local/cuda-4.0/cuda/include/builtin_types.h
src/main.o: /usr/local/cuda-4.0/cuda/include/device_types.h
src/main.o: /usr/local/cuda-4.0/cuda/include/driver_types.h
src/main.o: /usr/include/limits.h /usr/include/features.h
src/main.o: /usr/local/cuda-4.0/cuda/include/surface_types.h
src/main.o: /usr/local/cuda-4.0/cuda/include/texture_types.h
src/main.o: /usr/local/cuda-4.0/cuda/include/vector_types.h
src/main.o: /usr/local/cuda-4.0/cuda/include/host_defines.h
src/main.o: /usr/local/cuda-4.0/cuda/include/channel_descriptor.h
src/main.o: /usr/local/cuda-4.0/cuda/include/cuda_runtime_api.h
src/main.o: /usr/local/cuda-4.0/cuda/include/driver_functions.h
src/main.o: /usr/local/cuda-4.0/cuda/include/vector_functions.h
src/main.o: kernels/global_mem.h kernels/shared_mem.h
kernels/global_mem.o: /usr/local/cuda-4.0/cuda/include/cuda.h
kernels/global_mem.o: /usr/include/stdlib.h /usr/include/features.h
kernels/global_mem.o: /usr/include/alloca.h
kernels/global_mem.o: /usr/local/cuda-4.0/cuda/include/cuda_runtime_api.h
kernels/global_mem.o: /usr/local/cuda-4.0/cuda/include/host_defines.h
kernels/global_mem.o: /usr/local/cuda-4.0/cuda/include/builtin_types.h
kernels/global_mem.o: /usr/local/cuda-4.0/cuda/include/device_types.h
kernels/global_mem.o: /usr/local/cuda-4.0/cuda/include/driver_types.h
kernels/global_mem.o: /usr/include/limits.h
kernels/global_mem.o: /usr/local/cuda-4.0/cuda/include/surface_types.h
kernels/global_mem.o: /usr/local/cuda-4.0/cuda/include/texture_types.h
kernels/global_mem.o: /usr/local/cuda-4.0/cuda/include/vector_types.h
kernels/global_mem.o: /usr/local/cuda-4.0/cuda/include/cuda_runtime.h
kernels/global_mem.o: /usr/local/cuda-4.0/cuda/include/host_config.h
kernels/global_mem.o: /usr/local/cuda-4.0/cuda/include/channel_descriptor.h
kernels/global_mem.o: /usr/local/cuda-4.0/cuda/include/driver_functions.h
kernels/global_mem.o: /usr/local/cuda-4.0/cuda/include/vector_functions.h
kernels/global_mem.o: kernels/global_mem.h
kernels/shared_mem.o: /usr/local/cuda-4.0/cuda/include/cuda.h
kernels/shared_mem.o: /usr/include/stdlib.h /usr/include/features.h
kernels/shared_mem.o: /usr/include/alloca.h
kernels/shared_mem.o: /usr/local/cuda-4.0/cuda/include/cuda_runtime_api.h
kernels/shared_mem.o: /usr/local/cuda-4.0/cuda/include/host_defines.h
kernels/shared_mem.o: /usr/local/cuda-4.0/cuda/include/builtin_types.h
kernels/shared_mem.o: /usr/local/cuda-4.0/cuda/include/device_types.h
kernels/shared_mem.o: /usr/local/cuda-4.0/cuda/include/driver_types.h
kernels/shared_mem.o: /usr/include/limits.h
kernels/shared_mem.o: /usr/local/cuda-4.0/cuda/include/surface_types.h
kernels/shared_mem.o: /usr/local/cuda-4.0/cuda/include/texture_types.h
kernels/shared_mem.o: /usr/local/cuda-4.0/cuda/include/vector_types.h
kernels/shared_mem.o: /usr/local/cuda-4.0/cuda/include/cuda_runtime.h
kernels/shared_mem.o: /usr/local/cuda-4.0/cuda/include/host_config.h
kernels/shared_mem.o: /usr/local/cuda-4.0/cuda/include/channel_descriptor.h
kernels/shared_mem.o: /usr/local/cuda-4.0/cuda/include/driver_functions.h
kernels/shared_mem.o: /usr/local/cuda-4.0/cuda/include/vector_functions.h
kernels/shared_mem.o: kernels/shared_mem.h
