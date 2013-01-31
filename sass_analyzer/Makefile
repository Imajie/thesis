# All directories that contain source code
DIRS=src

# Name of the program to generate
TARGET=sass_analyze

###
### Shouldn't need to change below this line
###

#
# compilers
#
CXX=g++ -fPIC -ggdb -std=c++0x
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
FLAGS=${INCLUDE_DIRS} ${CUDA_DIRS} ${MATLAB_DIRS} ${ACML_DIRS}

#
# Libraries
#
LIB_DIRS=-L/usr/lib/i386-linux-gnu/

# libraries for specific toolkits
CUDA_LIBS=-L/usr/local/cuda/lib -lcudart -lcublas
MATLAB_LIBS=-L/usr/local/MATLAB/R2012a/bin/glnx86 -lmat -lmex -leng -lut -lmx
BLAS_LIBS=-L/opt/acml4.4.0/gfortran32/lib -lacml -llapack -lgfortran

# all libraries
#LIBS=${CUDA_LIBS} ${MATLAB_LIBS} ${BLAS_LIBS}
LIBS=

#
# Flags
#
CXXFLAGS=${FLAGS} -ggdb
CFLAGS=${FLAGS} -ggdb
NVCC_FLAGS=${FLAGS} -arch sm_13
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

