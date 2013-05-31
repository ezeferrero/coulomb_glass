/*
 * MT.cu
 */

#include <stddef.h> // NULL, size_t

#define RNG_NAME "Mersenne Twister"

#define INITIAL_PARAMS_FILENAME "../RNGcommon/MersenneTwister.dat"
#define MT_RNG_COUNT 32768 // initial parameters in MersenneTwuister.dat
#define MT_MM     9
#define MT_NN     19
#define MT_WMASK  0xFFFFFFFFU
#define MT_UMASK  0xFFFFFFFEU
#define MT_LMASK  0x1U
#define MT_SHIFT0 12
#define MT_SHIFTB 7
#define MT_SHIFTC 15
#define MT_SHIFT1 18


#define SEED (time(NULL)) // random seed
#define FRAME_RNG 256 	// the whole thing is framed for the RNG

// we frame the grid in FRAME_RNG*FRAME_RNG/2
#define NUM_THREADS (FRAME_RNG*FRAME_RNG/2)


// Record format for MersenneTwister.dat, created by spawnTwisters.c
struct mt_struct_stripped {
	unsigned int matrix_a;
	unsigned int mask_b;
	unsigned int mask_c;
	unsigned int seed;
};

// Per-thread state object for a single twister.
struct MersenneTwisterState {
	unsigned int mt[MT_NN];
	int iState;
	unsigned int mti1;
};


typedef struct {
	struct mt_struct_stripped MT;
	struct MersenneTwisterState state;
} rngState;

// Preloaded, offline-generated seed data structure.
__device__ static mt_struct_stripped MT[NUM_THREADS];

//__device__ static MersenneTwisterState d_rng_states[NUM_THREADS];

//typedef struct MersenneTwisterState rngState;


__device__ static rngState d_rng_states[NUM_THREADS];


__device__ void MersenneTwisterInitialise(unsigned int threadID) {
	//d_rng_states[threadID].MT.matrix_a
	//d_rng_states[threadID].MT.mask_b
	//d_rng_states[threadID].MT.mask_c
	//d_rng_states[threadID].MT.seed
	d_rng_states[threadID].MT = MT[threadID];

	d_rng_states[threadID].state.mt[0] = d_rng_states[threadID].MT.seed;

	for(int i = 1; i < MT_NN; ++ i) {
		d_rng_states[threadID].state.mt[i] = (1812433253U * (d_rng_states[threadID].state.mt[i - 1] ^ 
							(d_rng_states[threadID].state.mt[i - 1] >> 30)) + i) & MT_WMASK;
	}

	d_rng_states[threadID].state.iState = 0;
	d_rng_states[threadID].state.mti1 = d_rng_states[threadID].state.mt[0];
}

__device__ unsigned int MersenneTwisterGenerate(rngState * mt_state) {
	int iState1 = (mt_state->state).iState + 1;
	int iStateM = (mt_state->state).iState + MT_MM;

	if(iState1 >= MT_NN) iState1 -= MT_NN;
	if(iStateM >= MT_NN) iStateM -= MT_NN;

	unsigned int mti = (mt_state->state).mti1;
	(mt_state->state).mti1 = (mt_state->state).mt[iState1];
	unsigned int mtiM = (mt_state->state).mt[iStateM];

	unsigned int x = (mti & MT_UMASK) | ((mt_state->state).mti1 & MT_LMASK);
	x = mtiM ^ (x >> 1) ^ ((x & 1) ? (mt_state->MT).matrix_a : 0);
	(mt_state->state).mt[(mt_state->state).iState] = x;
	(mt_state->state).iState = iState1;

	// Tempering transformation.
	x ^= (x >> MT_SHIFT0);
	x ^= (x << MT_SHIFTB) & (mt_state->MT).mask_b;
	x ^= (x << MT_SHIFTC) & (mt_state->MT).mask_c;
	x ^= (x >> MT_SHIFT1);

	return x;
}

__device__ float genNumber(rngState * state) {
 	return (((float)MersenneTwisterGenerate(state)+1.0)/4294967296.0f);
}

__global__ void warm_upCUDA () {
	const unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;

	const unsigned int tid = i*FRAME_RNG + j;

	MersenneTwisterInitialise(tid);
	// "Warm-up" the Twister to avoid initial correlation with others.
	for(unsigned int k = 0; k < 10000; ++ k) {
		MersenneTwisterGenerate(&(d_rng_states[tid]));
	}	
}

int warm_up(void) {
	assert(FRAME_RNG%2==0);
	assert((FRAME_RNG/2)%16==0);
	dim3 dimBlock(16, 16);
	dim3 dimGrid(FRAME_RNG/16, (FRAME_RNG/2)/16);


	warm_upCUDA<<<dimGrid, dimBlock>>>();
    	cudaThreadSynchronize();
  
	int error = 0;
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		error = 1;
	}

	return error;
}

int ConfigureRandomNumbers(void) {
	assert(NUM_THREADS<=MT_RNG_COUNT);

	// Read offline-generated initial configuration file.
	mt_struct_stripped *mtStripped = new mt_struct_stripped[NUM_THREADS];
	
	FILE *datFile = fopen(INITIAL_PARAMS_FILENAME, "rb");
	assert(datFile);
	assert(fread(mtStripped, sizeof(struct mt_struct_stripped) * NUM_THREADS, 1, datFile));
	fclose(datFile);

	// Seed the structure with low-quality random numbers. Twisters will need "warming up"
	// before the RNG quality improves.
	for(int i = 0; i < NUM_THREADS; ++ i) {
		mtStripped[i].seed = rand();
	}

	// Upload the initial configurations to the GPU.
	cudaMemcpyToSymbol(MT, mtStripped, sizeof(mt_struct_stripped)*NUM_THREADS, 0, cudaMemcpyHostToDevice);
	checkCUDAError("Memcpy in MT ");
	delete[] mtStripped;
	//for (int i=0; i < NUM_THREADS; i++) {
	//	d_rng_states[i].MT = mtStripped[i];
	//		cudaMemcpy(&(d_rng_states[i].MT), &(mtStripped[i]), sizeof(struct mt_struct_stripped), cudaMemcpyHostToDevice);
	//}	

	// Warming up the twisters
	return warm_up();
}
