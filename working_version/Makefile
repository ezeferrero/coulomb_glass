# Binary file
BIN = coulomb_glass

# Pathts & Includes
CUDA_INSTALL_PATH	= /usr/local/cuda
#CUDA_INSTALL_PATH	?= /usr
CUDASDK_INSTALL_PATH	= /opt/cudasdk
CUDA_LIB_PATH		= $(CUDA_INSTALL_PATH)/lib64
CUDASDK_LIB_PATH	= $(CUDASDK_INSTALL_PATH)/C/lib64
INCLUDES	       += -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDASDK_INSTALL_PATH)/C/common/inc \
			-I$(CUDASDK_INSTALL_PATH)/shared/inc -I/home/eferrero/EW_Langevin/common/

# Flags
LIBS			= -L$(CUDA_LIB_PATH) -L$(CUDASDK_LIB_PATH)
CFLAGS			= -O3
#CFLAGS			= -O -g
LDFLAGS			= -lrt -lm -lcudart -lcufft
NVCC_FLAGS		= -O3 -arch=sm_20 --use_fast_math #--ptxas-options=-v 

# Default Values
L			= 128
SEED			= 1332277027LLU
CUTOFF			= 32
CUDA_DEVICE		= 0
OFILE			= out.dat

# Simulation Parameters & Compilation Options
PARAMETERS		= -DLL=$(L) -DCUTOFF=$(CUTOFF) -DSEED=$(SEED) -DCUDA_DEVICE=$(CUDA_DEVICE)
OPTIONS 		= -DUNIFORM_DISORDER -DLOGARITHMIC_SPACING \
			-DDOUBLE_PRECISION #-DMEASURE_ACCEPTANCE #-DCHECKERBOARD_INIT -DLINEAR_SPACING

# Compilers
NVCC                    = $(CUDA_INSTALL_PATH)/bin/nvcc
#VCC                    = nvcc
GCC                     = g++-4.6
LINKER                  = g++-4.6

# Files
CPP_SOURCES             = 
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

