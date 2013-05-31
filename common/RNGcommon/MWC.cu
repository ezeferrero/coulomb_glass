/*
 * MWC.cu
 */

#include <stddef.h> // NULL, size_t
#define RNG_NAME "Multiply With Carry"

#define SAFE_PRIMES_FILENAME "../RNGcommon/safeprimes_base32.txt"
#define SEED 1234 // random seed
#define FRAME_RNG 512	// the whole thing is framed for the RNG

// we frame the grid in FRAME_RNG*FRAME_RNG/2
#define NUM_THREADS (FRAME_RNG*FRAME_RNG/2)

struct mwcState {
	unsigned long long x;
	unsigned int a;
}; 

__device__ static struct mwcState d_rng_states[NUM_THREADS];


typedef struct mwcState rngState;

__device__ __inline__ float rand_MWC_co(unsigned long long* x, const unsigned int* a)
{
		//Generate a random number [0,1)
		*x=(*x&0xffffffffull)*(*a)+(*x>>32);
		return __fdividef(__uint2float_rz((unsigned int)(*x)),(float)0x100000000);// The typecast will truncate the x so that it is 0<=x<(2^32-1),__uint2float_rz ensures a round towards zero since 32-bit floating point cannot represent all integers that large. Dividing by 2^32 will hence yield [0,1)

}//end __device__ rand_MWC_co

__device__ __inline__ float rand_MWC_oc(unsigned long long* x,unsigned int* a)
{
		//Generate a random number (0,1]
		return 1.0f-rand_MWC_co(x,a);
}//end __device__ rand_MWC_oc

int init_RNG(unsigned long long *x, unsigned int *a,
            const unsigned int n_rng, const char *safeprimes_file, unsigned long long xinit)
{
	FILE *fp = NULL;
	int fscanf_result = 0;
	unsigned int begin = 0u;
	unsigned int fora,tmp1,tmp2;

	assert(x!=NULL && a!=NULL);
	assert(safeprimes_file!=NULL);

	if (strlen(safeprimes_file)==0) {
		// Try to find it in the local directory
		safeprimes_file = "safeprimes_base32.txt";
	}

	fp = fopen(safeprimes_file, "r");
	if (fp==NULL) {
		printf("Could not find the file of safeprimes (%s)! Terminating!\n", safeprimes_file);
		return 1;
	}

	fscanf_result = fscanf(fp,"%u %u %u", &begin, &tmp1, &tmp2);
	if (fscanf_result<3) {
		printf("Problem reading first safeprime from file %s\n", safeprimes_file);
		exit(1);
	}

	// Here we set up a loop, using the first multiplier in the file to generate x's and c's
	// There are some restictions to these two numbers:
	// 0<=c<a and 0<=x<b, where a is the multiplier and b is the base (2^32)
	// also [x,c]=[0,0] and [b-1,a-1] are not allowed.

	//Make sure xinit is a valid seed (using the above mentioned restrictions)
	if ((xinit == 0ull) | (((unsigned int)(xinit>>32))>=(begin-1)) | (((unsigned int)xinit)>=0xfffffffful)) {
		//xinit (probably) not a valid seed! (we have excluded a few unlikely exceptions)
		printf("%llu not a valid seed! Terminating!\n", xinit);
		return 1;
	}

	for (unsigned int i=0; i<n_rng; i++) {
		fscanf_result = fscanf(fp,"%u %u %u", &fora, &tmp1, &tmp2);
		if (fscanf_result<3) {
			printf("Problem reading safeprime %d out of %d from file %s\n", i+2, n_rng+1, safeprimes_file);
			exit(1);
		}
		a[i] = fora;
		x[i] = 0;
		while ((x[i]==0) | (((unsigned int)(x[i]>>32))>=(fora-1)) | (((unsigned int)x[i])>=0xfffffffful)) {
			//generate a random number
			xinit = (xinit&0xffffffffull)*(begin)+(xinit>>32);

			//calculate c and store in the upper 32 bits of x[i]
			x[i] = (unsigned int) floor((((double)((unsigned int)xinit))/(double)0x100000000)*fora);//Make sure 0<=c<a
			x[i] = x[i]<<32;

			//generate a random number and store in the lower 32 bits of x[i] (as the initial x of the generator)
			xinit = (xinit&0xffffffffull)*(begin)+(xinit>>32);//x will be 0<=x<b, where b is the base 2^32
			x[i] += (unsigned int) xinit;
		}
	}
	fclose(fp);

	return 0;
}

__device__ __inline__ float genNumber(rngState * state) {
	return rand_MWC_co(&(state->x), &(state->a));
}

static int ConfigureRandomNumbers(void) {
	// Allocate memory for RNG's
	unsigned long long h_x[NUM_THREADS];
	unsigned int h_a[NUM_THREADS];
	struct mwcState h_mwc_states[NUM_THREADS];
	unsigned long long seed = (unsigned long long) SEED;

	// Init RNG's
	int error = init_RNG(h_x, h_a, NUM_THREADS, SAFE_PRIMES_FILENAME, seed);

	for (unsigned int i=0; i<NUM_THREADS; i++) {
		h_mwc_states[i].x = h_x[i];
		h_mwc_states[i].a = h_a[i];
	}

	if (!error) {
		size_t size = NUM_THREADS * sizeof(struct mwcState);
		cudaMemcpyToSymbol(d_rng_states, h_mwc_states, size);
		checkCUDAError("Memcpy failed on MWC ");
	}

	return error;
}
