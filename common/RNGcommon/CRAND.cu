/*
 * CRAND.cu
 */

#include <stddef.h> // NULL, size_t
#include <curand_kernel.h> // CUDA_SAFE_CALL, CUT_CHECK_ERROR

#define RNG_NAME "CURAND"

#define SEED 1234 // random seed
#define FRAME_RNG 512	// the whole thing is framed for the RNG

// we frame the grid in FRAME_RNG*FRAME_RNG/2
#define NUM_THREADS (FRAME_RNG*FRAME_RNG/2)

#define FRAME_RNG 512	// the whole thing is framed for the RNG

// we frame the grid in FRAME_RNG*FRAME_RNG/2
#define NUM_THREADS (FRAME_RNG*FRAME_RNG/2)

typedef curandState rngState;

__device__ curandState d_rng_states[NUM_THREADS];

__device__ __inline__ float genNumber(rngState * state) {
	return curand_uniform(state);
}

__global__ void setup_kernel() {
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(SEED,tid,0,&d_rng_states[tid]);
}

static int ConfigureRandomNumbers(void) {
	dim3 dimBlock(256);
	dim3 dimGrid(NUM_THREADS/256);
	setup_kernel<<<dimGrid, dimBlock>>>();
	checkCUDAError("CURAND Setup Kernel");
	return 0;
}
