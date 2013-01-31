# All directories that contain source code
DIRS=.

# Name of the program to generate
TARGET=matMul

###
### Shouldn't need to change below this line
###

#
# compilers
#
CXX=g++ -fPIC -ggdb
CC=gcc -fPIC -ggdb
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
CUDA_DIRS=-I/usr/local/cuda/include
MATLAB_DIRS=-I/usr/local/MATLAB/R2012a/extern/include
ACML_DIRS=-I/opt/acml4.4.0/gfortran32_mp/include

# all include flags
FLAGS=${INCLUDE_DIRS} ${CUDA_DIRS} 

#
# Libraries
#
LIB_DIRS=-L/usr/lib/i386-linux-gnu/

# libraries for specific toolkits
CUDA_LIBS=-L/usr/local/cuda/lib -lcudart -lcublas
MATLAB_LIBS=-L/usr/local/MATLAB/R2012a/bin/glnx86 -lmat -lmex -leng -lut -lmx
BLAS_LIBS=-L/opt/acml4.4.0/gfortran32/lib -lacml -llapack -lgfortran
OCELOT_LIBS=-locelot -locelotTrace
BOOST_LIBS=-lboost_system-mt -lboost_filesystem-mt

# all libraries
#LIBS=${CUDA_LIBS}
LIBS=-L/usr/local/cuda/lib -lcublas ${BOOST_LIBS}

#
# Flags
#
CXXFLAGS=${FLAGS} -ggdb
CFLAGS=${FLAGS} -ggdb
NVCC_FLAGS=${FLAGS} -arch sm_13
LINK_FLAGS=${LIB_DIRS} -Wl,--no-as-needed ${OCELOT_LIBS} -Wl,--as-needed ${LIBS} -pthread

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

src/acml_bench.o: /usr/include/stdlib.h /usr/include/features.h
src/acml_bench.o: /usr/include/alloca.h /usr/include/stdio.h
src/acml_bench.o: /usr/include/libio.h /usr/include/_G_config.h
src/acml_bench.o: /usr/include/wchar.h src/benchmark_common.h
src/acml_bench.o: src/acml_bench.h
src/acml_bench.o: /opt/acml4.4.0/gfortran32_mp/include/acml.h
src/acml_bench.o: /usr/include/math.h
src/benchmark_common.o: src/benchmark_common.h /usr/include/time.h
src/benchmark_common.o: /usr/include/features.h /usr/include/xlocale.h
src/benchmark_common.o: /usr/include/stdio.h /usr/include/libio.h
src/benchmark_common.o: /usr/include/_G_config.h /usr/include/wchar.h
src/benchmark_common.o: /usr/include/stdlib.h /usr/include/alloca.h
src/main.o: /usr/include/stdlib.h /usr/include/features.h
src/main.o: /usr/include/alloca.h /usr/include/time.h /usr/include/xlocale.h
src/main.o: /usr/include/string.h src/benchmark_common.h src/acml_bench.h
src/main.o: src/matlab_bench.h
src/matlab_bench.o: src/benchmark_common.h src/matlab_bench.h
src/matlab_bench.o: /usr/include/stdlib.h /usr/include/features.h
src/matlab_bench.o: /usr/include/alloca.h /usr/include/stdio.h
src/matlab_bench.o: /usr/include/libio.h /usr/include/_G_config.h
src/matlab_bench.o: /usr/include/wchar.h
src/matlab_bench.o: /usr/local/MATLAB/R2012a/extern/include/engine.h
src/matlab_bench.o: /usr/local/MATLAB/R2012a/extern/include/matrix.h
src/matlab_bench.o: /usr/local/MATLAB/R2012a/extern/include/tmwtypes.h
src/matlab_bench.o: /usr/include/limits.h /usr/include/assert.h
src/matlab_bench.o: /usr/local/MATLAB/R2012a/extern/include/mex.h
