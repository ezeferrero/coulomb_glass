/*
 * CRAND.cu
 */

#include <stddef.h> // NULL, size_t
#include "Random123/philox.h"

#define RNG_NAME "PHILOX"

#define MULT 2.328306437080797e-10f // = 1/(2^32 - 1)

#define FRAME_RNG 512	// the whole thing is framed for the RNG

// we frame the grid in FRAME_RNG*FRAME_RNG/2
#define NUM_THREADS (FRAME_RNG*FRAME_RNG/2)

#define FRAME_RNG 512	// the whole thing is framed for the RNG

// we frame the grid in FRAME_RNG*FRAME_RNG/2
#define NUM_THREADS (FRAME_RNG*FRAME_RNG/2)

struct philoxState {
	philox4x32_key_t k;
	philox4x32_ctr_t c;

	__device__ philoxState(unsigned int tid) {
		philox4x32_key_t _k = {{tid, 0xdecafbad}};
		philox4x32_ctr_t _c = {{}};
		k = _k; c = _c;
	};
};

typedef philoxState rngState;

__device__ void genNumber(rngState * state, float* a, float* b, float* c, float* d) {
	state->c.v[0]++;
	*a =  MULT * philox4x32(state->c, state->k)[0];
	*b =  MULT * philox4x32(state->c, state->k)[1];
	*c =  MULT * philox4x32(state->c, state->k)[2];
	*d =  MULT * philox4x32(state->c, state->k)[3];
}

static int ConfigureRandomNumbers(void) {
	return 0;
}
