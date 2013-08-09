# Binary file
BIN = coulomb_glass

# Pathts & Includes
CUDA_INSTALL_PATH	= /usr/local/cuda
#CUDASDK_INSTALL_PATH	= /opt/cudasdk
CUDA_LIB_PATH		= $(CUDA_INSTALL_PATH)/lib64
#CUDASDK_LIB_PATH		= $(CUDASDK_INSTALL_PATH)/C/lib64
INCLUDES			+= -I. -I$(CUDA_INSTALL_PATH)/include -I../common/ -I../common/RNGcommon/ \
		#-I$(CUDASDK_INSTALL_PATH)/C/common/inc -I$(CUDASDK_INSTALL_PATH)/shared/inc 

# Flags
LIBS			= -L$(CUDA_LIB_PATH) #-L$(CUDASDK_LIB_PATH)
CFLAGS			= -O3
#CFLAGS			= -O -g
LDFLAGS			= -lrt -lm -lcufft
NVCC_FLAGS		= -O3 -arch=sm_20 --use_fast_math #--ptxas-options=-v 

# Default Values
L			= 512
SEED			= 1332277029LLU
CUTOFF			= 32
CUDA_DEVICE		= 0
OFILE			= out.dat

# Simulation Parameters & Compilation Options
PARAMETERS		= -DLL=$(L) -DCUTOFF=$(CUTOFF) -DSEED=$(SEED) -DCUDA_DEVICE=$(CUDA_DEVICE)
OPTIONS 			= -D__STDC_CONSTANT_MACROS -DMERSENNE -DUNIFORM_DISORDER -DLOGARITHMIC_SPACING \
			-DDOUBLE_PRECISION #-DMEASURE_ACCEPTANCE #-DCHECKERBOARD_INIT -DLINEAR_SPACING

# Compilers
NVCC                    = $(CUDA_INSTALL_PATH)/bin/nvcc
#VCC                    = nvcc
GCC                     = g++
LINKER                  = $(CUDA_INSTALL_PATH)/bin/nvcc

# Files
CPP_SOURCES             = mt.cpp
CU_SOURCES              = $(BIN).cu
HEADERS                 =
CPP_OBJS                = $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS                 = $(patsubst %.cu, %.o, $(CU_SOURCES))

# Rules
$(BIN): clean $(CPP_OBJS) $(CU_OBJS) $(HEADERS)
	$(LINKER) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCLUDES) $(LIBS)

$(CPP_OBJS): $(CPP_SOURCES) $(HEADERS)
	$(GCC) -c $(CPP_SOURCES) $(CFLAGS) $(INCLUDES)

$(CU_OBJS): $(CU_SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $(CU_SOURCES) $(PARAMETERS) $(OPTIONS) $(INCLUDES)

run: $(BIN)
	LD_LIBRARY_PATH=$(CUDA_LIB_PATH) ./$(BIN)
#	LD_LIBRARY_PATH=$(CUDA_LIB_PATH) valgrind --tool=memcheck --leak-check=full --track-origins=yes ./$(BIN)  

clean:
	rm -f $(BIN) *.o *.linkinfo cuda_profile_?.log

