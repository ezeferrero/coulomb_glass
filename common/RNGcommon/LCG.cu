/*
 * LCG.cu
 */

#include <stddef.h> // size_t

#define RNG_NAME "Linear Congruential Generator"

#define A  1664525 // (Numerical Recipes a parameters)
#define C  1013904223 // (Numerical Recipes c parameter)
#define AA A
#define CC C
#define MULT 2.328306437080797e-10f // = 1/(2^32 - 1)

#define FRAME_RNG 512	// the whole thing is framed for the RNG

// we frame the grid in FRAME_RNG*FRAME_RNG/2
#define NUM_THREADS (FRAME_RNG*FRAME_RNG/2)

struct lcgState {
	unsigned int ran_st;
};

typedef struct lcgState rngState;

__device__ static rngState d_rng_states[NUM_THREADS];


__device__ float genNumber(rngState * state) {
	state->ran_st = AA*(state->ran_st) + CC;

	return MULT*(state->ran_st);
}

int ConfigureRandomNumbers(void) {
	struct lcgState h_lcg_states[NUM_THREADS];
	int error = 0;
	
	h_lcg_states[0].ran_st = 1;
	for(int i = 1; i < NUM_THREADS; ++i) h_lcg_states[i].ran_st = 16807*(h_lcg_states[i-1].ran_st);

	size_t size = NUM_THREADS * sizeof(struct lcgState);
	cudaMemcpyToSymbol(d_rng_states, h_lcg_states, size);
	checkCUDAError("Memcpy failed on LCG ");
	
	return error;
}
