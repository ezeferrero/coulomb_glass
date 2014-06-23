/*
 * CUDA Coulomb Glass 
*/

/*
 * coulomb_glass.cu is a CUDA implementation of a 2D Coulomb Glass model.
 * The program simulates the Kinetic Monte Carlo dynamics of a gas of electrons hopping in a 2D square lattice 
 * with randomly distributed (quenched) energy traps.
 * This program implements the "Parallel Rejection" method described in our paper.
.*
 * Started on: Jan, 2013 by ezeferrero
 * Copyright (C) 2014 Ezequiel E. Ferrero, Alejandro B. Kolton and Matteo Palassini.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * This code was originally implemented for: "Parallel kinetic Monte Carlo simulation of Coulomb glasses",
 * E.E. Ferrero, A.B. Kolton and M. Palassini
 * http://arxiv.org/abs/XXXX
 * AIP Conference Proceedings XX, XX (2014)
 *
 * Please cite when appropriate.
 */


// RNG: PHILOX
#include <Random123/philox.h>
#include <Random123/u01.h>

//Define Philox Seed
#ifdef DOUBLE_PRECISION
	typedef r123::Philox2x64 RNG2;
	typedef r123::Philox4x64 RNG4;
#else
	typedef r123::Philox2x32 RNG2;
	typedef r123::Philox4x32 RNG4;
#endif

#include <iostream>
#include <fstream> 	/* print stuff*/
#include <vector>
#include <cstdlib>
#include <cmath>
#include <sys/times.h>
#include <assert.h>	/* assertions */
//#include <omp.h>
//#include <cuda.h>
//#include <cuda_runtime.h>
#include <cufft.h>

#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/find.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/inner_product.h>

#include "cutil.h"	/* CUDA_SAFE_CALL, CUT_CHECK_ERROR */
#include "../common/histo.h"
#include "../common/timers_cpu.hpp"

using namespace std;

#ifdef DOUBLE_PRECISION
typedef double FLOAT;
#else
typedef float  FLOAT;
#endif

#ifndef SEED
#define SEED 1332277027LLU
#endif

#ifndef LL
#define LL 1024
#endif
#define NN (LL*LL)
#define SMALL_SIZE (LL<=32)

#ifndef T_DATA
#define T_DATA 1000
#endif

#ifndef T_DATA2
#define T_DATA2 100000
#endif

#ifndef ELECTRIC_FIELD
#define ELECTRIC_FIELD 0.1
#endif

#define NBINS 500
#define LEFTLIMIT -10.
#define RIGHTLIMIT 10.

#ifndef W
#define W 1.
#endif

#ifndef LOC_LENGTH
#define LOC_LENGTH 1
#endif

#ifndef CUTOFF
#define CUTOFF (32*LOC_LENGTH)
#endif

// Hardware parameters for GTX 470/480 (GF100)
#define SHARED_PER_BLOCK 49152
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_GRID 65535

#define TILE 256
#define TILE_X 8
#define TILE_Y 32

// Some predefined Functions (just in case)
#define MAX(a,b) (((a)<(b))?(b):(a))	/* maximum */
#define MIN(a,b) (((a)<(b))?(a):(b))	/* minimum */ 
#define ABS(a) (((a)< 0)?(-a):(a))	/* absolute value */
#define DISTANCE(a,b) ((a>b)?(a-b):(b-a))

/////////////////// DEVICE FUNCTIONS ///////////////////

/*--------GLOBAL VARIABLES--------*/

__device__ unsigned int d_pair[2];
__device__ unsigned int d_jump_coordinates[4];
__device__ int d_dipolar_moment;
__device__ unsigned int d_global_winner;
__device__ unsigned int d_grid_winner;
__device__ unsigned int d_accepted;

extern void init_genrand(unsigned long s);
extern double genrand_real3(void);
extern double gasdev_real3(void);

/*--------BINARY SEARCH FUNCTION FOR ONE THREAD--------*/
__device__ 
unsigned int LowerBound(const FLOAT *tower, const FLOAT key, unsigned int imin, unsigned int imax)
{
	// continue searching while [imin,imax] is not empty
	while (imax >= imin)
	{
	/* calculate the midpoint for roughly equal partition */
	unsigned int imid = imin + (imax-imin)/2;

	// determine which subarray to search
	if(tower[imid] <  key)
	// change min index to search upper subarray
	imin = imid + 1;
	else if(tower[imid] > key )
	// change max index to search lower subarray
	imax = imid - 1;
	else
	// key found at index imid
	return imid ;
	}
	// key not found
	return imax;
}

/*--------RNG ROUTINES--------*/
__device__
FLOAT DeviceBoxMuller(const FLOAT u1, const FLOAT u2)
{
#ifdef DOUBLE_PRECISION
	FLOAT r = sqrt( -2.0*log(u1) );
	FLOAT theta = 2.0*M_PI*u2;
	return r*sin(theta);
#else
	FLOAT r = sqrtf( -2.0*logf(u1) );
	FLOAT theta = 2.0*M_PI*u2;
	return r*sinf(theta);
#endif
}


__device__
void PhiloxRandomQuartet(const unsigned int index, const unsigned int time, FLOAT *r1, FLOAT *r2, FLOAT *r3, FLOAT *r4)
{
	RNG4 rng;
	RNG4::ctr_type c_pair={{}};
	RNG4::key_type k_pair={{}};
	RNG4::ctr_type r_quartet;

		// keys = threadid
		k_pair[0]= index;
		// time counter
		c_pair[0]= 2*time+1;
		c_pair[1]= SEED; // eventually do: PHILOX_SEED + sample;
		// random number generation
		r_quartet = rng(c_pair, k_pair);

	#ifdef DOUBLE_PRECISION
		*r1= u01_open_closed_64_53(r_quartet[0]);
		*r2= u01_open_closed_64_53(r_quartet[1]);
		*r3= u01_open_closed_64_53(r_quartet[2]);
		*r4= u01_open_closed_64_53(r_quartet[3]);
	#else
		*r1= u01_open_closed_32_53(r_quartet[0]);
		*r2= u01_open_closed_32_53(r_quartet[1]);
		*r3= u01_open_closed_32_53(r_quartet[2]);
		*r4= u01_open_closed_32_53(r_quartet[3]);
	#endif
}

__device__
void PhiloxRandomPair(const unsigned int index, const unsigned int time, FLOAT *r1, FLOAT *r2)
{
	RNG2 rng;
	RNG2::ctr_type c_pair={{}};
	RNG2::key_type k_pair={{}};
	RNG2::ctr_type r_pair;

		// keys = threadid
		k_pair[0]= index;
		// time counter
		c_pair[0]= 2*time;
		c_pair[1]= SEED; // eventually do: PHILOX_SEED + sample;
		// random number generation
		r_pair = rng(c_pair, k_pair);

	#ifdef DOUBLE_PRECISION
		*r1= u01_open_closed_64_53(r_pair[0]);
		*r2= u01_open_closed_64_53(r_pair[1]);
	#else
		*r1= u01_open_closed_32_53(r_pair[0]);
		*r2= u01_open_closed_32_53(r_pair[1]);
	#endif
}

/*--------UPDATE ROUTINES--------*/

__global__
void Kernel_ProposeJumpsAndEvaluateMetropolis(const unsigned int M, const FLOAT beta, const unsigned int *d_occ_number,\
					const FLOAT *local_energy, unsigned int *d_block_winners, const int *dx, \
					const int *dy, const FLOAT *tower, unsigned int *prop_pairs, \
					const FLOAT electric_field, const unsigned int tower_size, \
					const unsigned int rng_counter)
{
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx<M){

		FLOAT r1, r2, r3, r4;
		//RNG CALL
		//PhiloxRandomPair(idx, rng_counter, &r1, &r2);
		PhiloxRandomQuartet(idx, rng_counter, &r1, &r2, &r3, &r4);

		// choose origin at random using r1
		unsigned int site1 = r1*NN;
		// Identify the spatial position of site1 (origin)
		int y1 = site1/LL;
		int x1 = site1 - y1 * LL;

		// choose a jump from the tower at random using r4

		// const unsigned int index = LowerBound(tower, r4, 0, tower_size-1)+1; <--- corrected 8/9/2013
		const unsigned int index = LowerBound(tower-1, r4, 0, tower_size);

		int deltax=dx[index];
		int deltay=dy[index];

		// Identify the spatial position of site2 (target)
		unsigned int x2 = (x1+deltax) - LL*((x1+deltax)>=LL) + LL*((x1+deltax)<0);
		unsigned int y2 = (y1+deltay) - LL*((y1+deltay)>=LL) + LL*((y1+deltay)<0);
		unsigned int site2 = x2 + y2*LL;

		// Compute the distance between the two sites
#ifdef DOUBLE_PRECISION
		const FLOAT r = sqrt((FLOAT)(deltax*deltax + deltay*deltay));
#else
		const FLOAT r = sqrtf((FLOAT)(deltax*deltax + deltay*deltay));
#endif

		// Compute the difference in energy (electron is proposed to hop from site1 to site2)
		FLOAT delta_energy = local_energy[site2] - local_energy[site1] - 1./r - electric_field*deltax; 
		//if(delta_energy < 0) delta_energy = 0;

		// Shared arrays to choose a "winner" within a block
		__shared__ bool s_acceptance[TILE];
		__shared__ unsigned int s_winner;

		// evaluate acceptance avoiding moves that would take an electron from an empty site or put it in an \
		occupied site
		bool success = ( (d_occ_number[site1]>0) && (d_occ_number[site2]<1) && (r2 < exp(-beta*delta_energy)) );
		s_acceptance[threadIdx.x] = success;

		// synchonize and count accepted
		unsigned int block_success = __syncthreads_count(success);

		// one thread reduction to choose the first winner of the block
		if (threadIdx.x==0){
			if(block_success){
				unsigned int winner_thread=0;
				while((s_acceptance[winner_thread])==0){winner_thread++;}
				d_block_winners[blockIdx.x] = winner_thread;
				s_winner = winner_thread;
			}else{
				d_block_winners[blockIdx.x] = TILE;
				s_winner = TILE;
			}
		}

		// synchonize everybody waits for tread0 to finish
		__syncthreads();

		// winner_thread writes its jump to the global array
		if (threadIdx.x==s_winner){
			prop_pairs[2*blockIdx.x    ] = site1;
			prop_pairs[2*blockIdx.x +1 ] = site2;
#ifdef MEASURE_ACCEPTANCE
			atomicAdd(&d_accepted, block_success);
#endif
		}
	}
}

__global__
void Kernel_FindGlobalWinner(const unsigned int M, const unsigned int *d_block_winners)
{
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx==0){
		unsigned int grid_winner=0;
		while( ((d_block_winners[grid_winner])==TILE) && (grid_winner<(M/TILE)) ){grid_winner++;}
		if (grid_winner < (M/TILE)){
			d_global_winner = grid_winner*TILE + d_block_winners[grid_winner];
			d_grid_winner   = grid_winner;
		}else{
			d_global_winner = M;
		}
	}
}

__global__
void Kernel_PerformHopping(const unsigned int winner, const unsigned int *prop_pairs, unsigned int *d_occ_number)
{
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx==0){
		const unsigned int site1  = prop_pairs[2*d_grid_winner    ];
		const unsigned int site2  = prop_pairs[2*d_grid_winner + 1];

		// Update the occupation numbers
		assert(d_occ_number[site1]>0);
		d_occ_number[site1] -= 1;
		d_occ_number[site2] += 1;

		// Identify the spatial position of site1 (origin)
		d_jump_coordinates[0] = site1/LL;				//y1
		d_jump_coordinates[1] = site1 - d_jump_coordinates[0]*LL;	//x1
		int x1 = d_jump_coordinates[1];

		// Identify the spatial position of site2 (target)
		d_jump_coordinates[2] = site2/LL;				//y2
		d_jump_coordinates[3] = site2 - d_jump_coordinates[2]*LL;	//x2
		int x2 = d_jump_coordinates[3];

		d_dipolar_moment += x2 - x1 + ((x2-x1>LL/2)*(-1)+(x2-x1<-LL/2))*LL;

		#ifdef MEASURE_ACCEPTANCE
		d_accepted=0;
		#endif
	}
}

__global__
void Kernel_UpdateLocalEnergies(FLOAT *d_local_energy)
{
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if (idx<LL && idy<LL){

		const unsigned int y1 = d_jump_coordinates[0];
		const unsigned int x1 = d_jump_coordinates[1];

		const unsigned int y2 = d_jump_coordinates[2];
		const unsigned int x2 = d_jump_coordinates[3];

		unsigned int dx1 = (idx > x1)?idx-x1:x1-idx; if(dx1 > (LL>>1)) dx1 = LL - dx1; 
		unsigned int dx2 = (idx > x2)?idx-x2:x2-idx; if(dx2 > (LL>>1)) dx2 = LL - dx2;

		unsigned int dy1 = (idy > y1)?idy-y1:y1-idy; if(dy1 > (LL>>1)) dy1 = LL - dy1; 
		unsigned int dy2 = (idy > y2)?idy-y2:y2-idy; if(dy2 > (LL>>1)) dy2 = LL - dy2; 

#ifdef DOUBLE_PRECISION
		const FLOAT r1 = sqrt((FLOAT)(dx1*dx1 + dy1*dy1));
		const FLOAT r2 = sqrt((FLOAT)(dx2*dx2 + dy2*dy2));
#else
		const FLOAT r1 = sqrtf((FLOAT)(dx1*dx1 + dy1*dy1));
		const FLOAT r2 = sqrtf((FLOAT)(dx2*dx2 + dy2*dy2));
#endif

		//d_local_energy[idy*LL+idx] += ((r2>0)?1./r2:0) - ((r1>0)?1./r1:0);
		//This is the same as above with better performance
		if ((idx==x1)&&(idy==y1)){d_local_energy[idy*LL+idx] += (1./r2);
		}else if((idx==x2)&&(idy==y2)){d_local_energy[idy*LL+idx] -= (1./r1);
		}else{d_local_energy[idy*LL+idx] += (1./r2) - (1./r1);}

	}
}


/*--------INITIALIZATION ROUTINES--------*/

__global__
void Kernel_InitializeOccNumber(const unsigned int rng_counter, unsigned int *d_occ_number)
{
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx<NN){
#ifndef CHECKERBOARD_INIT
			//Random Initialization
			FLOAT r1, r2;
			//RNG CALL
			PhiloxRandomPair(idx, rng_counter, &r1, &r2);
			d_occ_number[idx] = (r1 < 0.5);
			d_occ_number[idx + NN/2] = (r2 < 0.5);
#else
			//Checkerboard Initialization
			d_occ_number[idx] = (idx+(idx/LL)%2)%2;
			d_occ_number[idx + NN/2] = (idx+(idx/LL)%2)%2;
#endif
	}
}

__global__
void Kernel_CastCufftRealToFLOATAndNormalize(const cufftReal *d_local_ene, FLOAT *d_local_energy)
{
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if (idx<LL && idy<LL){
		d_local_energy[idx + idy*LL] = d_local_ene[idx + idy*LL]/(1.*NN);
	}
}

__global__
void Kernel_ComplexProduct(const unsigned int size, const cufftComplex *d_half_magn_c, cufftComplex *d_local_ene_c)
{
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx<size){
		const FLOAT a1 = d_local_ene_c[idx].x;
		const FLOAT b1 = d_local_ene_c[idx].y;
		const FLOAT a2 = d_half_magn_c[idx].x;
		const FLOAT b2 = d_half_magn_c[idx].y;

		d_local_ene_c[idx].x = a1*a2-b1*b2;
		d_local_ene_c[idx].y = a1*b2+b1*a2;
	}
}

__global__
void Kernel_InitLocalEneFillHalfMagn(const unsigned int *d_occ_number, cufftReal *d_half_magn, cufftReal *d_local_ene)
{
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if (idx<LL && idy<LL){
		const int dx = (idx > (LL/2))?LL-idx:idx;
		const int dy = (idy > (LL/2))?LL-idy:idy;

#ifdef DOUBLE_PRECISION
		const FLOAT r = sqrt((FLOAT)(dx*dx + dy*dy));
#else
		const FLOAT r = sqrtf((FLOAT)(dx*dx + dy*dy));
#endif

		d_half_magn[idx + idy * LL] = (d_occ_number[idx + idy * LL] - 0.5);
		d_local_ene[idx + idy * LL] = (r > 0.)?1./r:0.;
	}
}

__global__ void Kernel_InitializeRandomFields(const unsigned int rng_counter, FLOAT *d_random_field)
{
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx<NN){
		FLOAT r1, r2;
		//RNG CALL
		PhiloxRandomPair(idx, rng_counter, &r1, &r2);
#ifdef UNIFORM_DISORDER
		d_random_field[idx] = W*(2.*r1-1.);
#else //Gaussian
		d_random_field[idx] = W*DeviceBoxMuller(r1,r2);
#endif
	}
}

/////////////////// HOST FUNCTIONS ///////////////////

/*--------RNG ROUTINES--------*/

FLOAT BoxMuller()
{
	static bool empty = true;
	static FLOAT x1;
	static FLOAT x2;
	if(empty) {

		const FLOAT u1 = genrand_real3();
		const FLOAT u2 = genrand_real3();

		x1 = sqrt(-2. * log(u1)) * cos(2. * M_PI * u2);
		x2 = sqrt(-2. * log(u1)) * sin(2. * M_PI * u2);
		empty = false;
		return x1;
	} else {
		empty = true;
		return x2;
	}
}

/*--------INITIALIZATION ROUTINES--------*/

///////////////////////////////////////////////////////////////////////////////
// Fills transition tower
// h == probabilities tower : P(transicion i) \propto h[i+1]-h[i]
// (dx[i],dy[i]) == hop displacement transition "i" 
// int cutoff == LL/2
// RETURN actual number of transitions
unsigned int FillTransitionMatrix(thrust::device_vector<int> &dx, thrust::device_vector<int> &dy, \
				thrust::device_vector<FLOAT> &h, FLOAT &gamma)
{
	thrust::fill(h.begin(),h.end(),0.0);
	int cutoff=CUTOFF;
	if (LL<CUTOFF*2) cutoff=LL/2;
	// inicializacion 
	unsigned int k=0;
	float p,r;
	for(int i=-cutoff;i<cutoff;i++){
		for(int j=-cutoff;j<cutoff;j++){
			dx[k]=i; dy[k]=j;
			r=sqrt(i*i+j*j);
			if(r>0 && r<=cutoff){
#ifndef FLAT_HOPPING
				p = exp(-r*2./LOC_LENGTH);
#else
				p = 1; // flat hopping
#endif
			h[k]=(k>0)?(h[k-1]+p):(p);
			k++;
			}
		}
	}
	unsigned int tower_size = k-1; // actual number of transitions
	gamma = h[tower_size];
	// tower normalization: h[0] <= h[1] <= ... <= h[tower_size]=1
	thrust::transform(h.begin(),h.end(),thrust::make_constant_iterator(gamma),h.begin(),thrust::divides<FLOAT>());

	return tower_size;
}

// Recall that e_i = phi_i + \sum_j 1/r_ij (n_j - 1/2), we're doing only the second sum here
void CalculateLocalEnergies_onGPU(const unsigned int *d_occ_number, FLOAT *d_local_energy)
{
	//Prepare cuFFT transforms
	cufftHandle Sr2c;
	cufftHandle Sc2r;
	CUFFT_SAFE_CALL(cufftPlan2d(&Sr2c,LL,LL,CUFFT_R2C));
	CUFFT_SAFE_CALL(cufftPlan2d(&Sc2r,LL,LL,CUFFT_C2R));

	//cuFFT Real arrays
	cufftReal *d_half_magn, *d_local_ene;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_half_magn, sizeof(cufftReal)*LL*LL));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_local_ene, sizeof(cufftReal)*LL*LL));

	//cuFFT Complex arrays
	cufftComplex *d_half_magn_c, *d_local_ene_c;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_half_magn_c, sizeof(cufftComplex)*LL*(LL/2+1)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_local_ene_c, sizeof(cufftComplex)*LL*(LL/2+1)));

	// Fill Real arrays
	assert(LL%TILE_X==0);
	assert(LL%TILE_Y==0);
	dim3 dimBlock2D(TILE_X, TILE_Y);
	dim3 dimGrid2D(LL/TILE_X, LL/TILE_Y);
	assert(dimBlock2D.x*dimBlock2D.y<=THREADS_PER_BLOCK);
	assert(dimGrid2D.x<=BLOCKS_PER_GRID && dimGrid2D.y<=BLOCKS_PER_GRID);

	Kernel_InitLocalEneFillHalfMagn<<<dimGrid2D, dimBlock2D>>>(d_occ_number, d_half_magn, d_local_ene);

	// cuFFT execute RealToComplex
	CUFFT_SAFE_CALL(cufftExecR2C(Sr2c, d_half_magn, d_half_magn_c));
	CUFFT_SAFE_CALL(cufftExecR2C(Sr2c, d_local_ene, d_local_ene_c));

	// Perform complex product
	dim3 dimBlock; dim3 dimGrid;
	if(SMALL_SIZE){
	assert(LL%16==0);
	dimBlock.x=16;
	dimGrid.x=((LL*(LL/2+1)+16-1)/16);
	}else{
	dimBlock.x=TILE;
	dimGrid.x=((LL*(LL/2+1)+TILE-1)/TILE);
	}
	assert(dimBlock.x<=THREADS_PER_BLOCK);
	assert(dimGrid.x<=BLOCKS_PER_GRID);

	Kernel_ComplexProduct<<<dimGrid, dimBlock>>>(LL*(LL/2+1), d_half_magn_c, d_local_ene_c);

	// cuFFT execute ComplexToReal and Normalize
	CUFFT_SAFE_CALL(cufftExecC2R(Sc2r, d_local_ene_c, d_local_ene));

	Kernel_CastCufftRealToFLOATAndNormalize<<<dimGrid2D, dimBlock2D>>>(d_local_ene, d_local_energy);

	// free memory
	cudaFree(d_half_magn); cudaFree(d_half_magn_c); cudaFree(d_local_ene); cudaFree(d_local_ene_c);
	cufftDestroy(Sr2c); cufftDestroy(Sc2r);
}

void InitializeOccNumber_onGPU(const unsigned int counter, unsigned int *d_occ_number)
{
	dim3 dimBlock(TILE);
	dim3 dimGrid(NN/2/TILE);
	assert(NN/2%TILE==0);
	assert(dimBlock.x<=THREADS_PER_BLOCK);
	assert(dimGrid.x<=BLOCKS_PER_GRID);

	Kernel_InitializeOccNumber<<<dimGrid,dimBlock>>>(counter, d_occ_number);

// TODO: check for half/half balance and fix if not
//	thrust::device_ptr<unsigned int> occ_number_ptr (d_occ_number);
//	int numberofones = thrust::reduce(occ_number_ptr, occ_number_ptr+NN);

}

void InitializeOccNumber_onCPU(unsigned int *occ_number, unsigned int *d_occ_number)
{
	unsigned int n_filled = 0;
	unsigned int n_empty  = 0;

	for(unsigned int i = NN; i--; ) {

		occ_number[i] = 1*(genrand_real3()<0.5);

		if (occ_number[i]==1){ n_filled++; }else{n_empty++;} 
	}

	// Here we correct the balance permuting 0s and 1s at random
	int excessofones = (n_filled - n_empty);
	unsigned int N =NN;
	while (excessofones!=0){
		unsigned int i = (int)(genrand_real3()*N);
		if (excessofones>0 && occ_number[i]==1) {occ_number[i]=0; excessofones-=2;}
		if (excessofones<0 && occ_number[i]==0) {occ_number[i]=1; excessofones+=2;}
	}

	CUDA_SAFE_CALL(cudaMemcpy(d_occ_number, &occ_number[0], sizeof(unsigned int)*NN, cudaMemcpyHostToDevice));
}

void InitializeOccNumberFromFile(unsigned int *occ_number, unsigned int *d_occ_number, ifstream &stored_configuration)
{
	if(stored_configuration.good()){
		for(unsigned int i = NN; i--; ) stored_configuration >> occ_number[i];
	}else{
	cout << "Problems with stored_configurations " << endl;
	}

	CUDA_SAFE_CALL(cudaMemcpy(d_occ_number, &occ_number[0], sizeof(unsigned int)*NN, cudaMemcpyHostToDevice));
}

void InitializeRandomFields(const unsigned int counter, FLOAT *d_random_field){
	dim3 dimBlock(TILE);
	dim3 dimGrid(NN/TILE);
	assert(NN%TILE==0);
	assert(dimBlock.x<=THREADS_PER_BLOCK);
	assert(dimGrid.x<=BLOCKS_PER_GRID);

	Kernel_InitializeRandomFields<<<dimGrid,dimBlock>>>(counter, d_random_field);
}

void InitializeRandomFieldsFromFile(FLOAT *random_field, FLOAT *d_random_field, ifstream &stored_random_fields)
{
	if(stored_random_fields.good()){
		for(unsigned int i = NN; i--; ){ stored_random_fields >> random_field[i];
		//printf("i %d rf %f", i ,random_field[i]);
		}
	}else{
	cout << "Problems with stored_random_field " << endl;
	}

	CUDA_SAFE_CALL(cudaMemcpy(d_random_field, &random_field[0], sizeof(FLOAT)*NN, cudaMemcpyHostToDevice));
}

/*--------CALCULATION ROUTINES--------*/
/*
void CalculateAccumulateEnergyHistogram(unsigned int index, thrust::device_vector<FLOAT> &histogram_accum, \
					FLOAT leftlimit, FLOAT rightlimit, int numberofbins, FLOAT *local_energy){

	thrust::device_vector<FLOAT> input;
	input.resize(NN);
	thrust::copy(local_energy, local_energy + NN, input.begin());

	thrust::device_vector<FLOAT> histogram(numberofbins);
	dense_histogram_data_on_device(input, histogram, leftlimit, rightlimit);

	thrust::transform(histogram.begin(), histogram.end(), histogram_accum.begin()+index*numberofbins, \
			histogram_accum.begin()+index*numberofbins, thrust::plus<FLOAT>());
}
*/

void ComputeGlobalEnergyDipolarMomentAndPrint(thrust::device_ptr<unsigned int> &occ_number_ptr, \
					thrust::device_ptr<FLOAT> &local_energy_ptr, \
					thrust::device_ptr<FLOAT> &random_field_ptr, int dipolar_moment, \
					FLOAT kmc_time, unsigned long long kmc_steps, const FLOAT sum_random_fields, \
					const FLOAT acceptance)
{
	// Compute the total energy
	FLOAT t1 = thrust::reduce(local_energy_ptr, local_energy_ptr + NN, 0.0, thrust::plus<FLOAT>());
	FLOAT t2 = thrust::inner_product(occ_number_ptr, occ_number_ptr + NN, local_energy_ptr, 0.0);
	FLOAT t3 = thrust::inner_product(occ_number_ptr, occ_number_ptr + NN, random_field_ptr, 0.0);
	const FLOAT total_energy = 0.5 * (t2 + t3) - 0.25 * (t1 - sum_random_fields);
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&dipolar_moment, d_dipolar_moment, sizeof(unsigned int), 0, \
			cudaMemcpyDeviceToHost));

	#warning OUTPUT HERE
	#ifdef MEASURE_ACCEPTANCE
	if (kmc_steps==0) cout << "# kmc_time	energy_per_site	dipolar_moment	kmc_steps	acceptance" << endl;
	// On 5/8/2013 we added normalization to kmc_time, kmc_steps and dipolar_moment, that were missing.
	cout << std::setprecision (15) << kmc_time/(1.*NN) << " " << total_energy/(1.*NN) << " " << \
			dipolar_moment/(1.*NN) << " "<< kmc_steps/(1.*NN) << " "<< acceptance << " " << endl;
	#else
	if (kmc_steps==0) cout << "# kmc_time	energy_per_site	dipolar_moment	kmc_steps acceptance" << endl;
	// On 5/8/2013 we added normalization to kmc_time, kmc_steps and dipolar_moment, that were missing.
	cout << std::setprecision (15) << kmc_time/(1.*NN) << " " << total_energy/(1.*NN) << " " << \
			dipolar_moment/(1.*NN) << " " << kmc_steps/(1.*NN) << " " << acceptance << endl;
	#endif
}

void CheckSelfConsistency(const unsigned int *d_occ_number, const FLOAT *d_local_energy, const FLOAT *d_random_field, \
		vector<unsigned int> &occ_number, vector<FLOAT> &local_energy, vector<FLOAT> &random_field)
{
	cout << "### Check if the result is self-consistent .. ";

	FLOAT *d_test;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_test, sizeof(FLOAT)*NN));
	CalculateLocalEnergies_onGPU(d_occ_number, d_test);
	vector<FLOAT> test(NN);
	CUDA_SAFE_CALL(cudaMemcpyAsync(&test[0], d_test, sizeof(FLOAT)*NN, cudaMemcpyDeviceToHost, 0));
	CUDA_SAFE_CALL(cudaMemcpyAsync(&local_energy[0], d_local_energy, sizeof(FLOAT)*NN, cudaMemcpyDeviceToHost,0));
	CUDA_SAFE_CALL(cudaMemcpyAsync(&random_field[0], d_random_field, sizeof(FLOAT)*NN, cudaMemcpyDeviceToHost,0));
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	#ifdef DOUBLE_PRECISION
	FLOAT tolerance = 1.e-5;
	#else
	FLOAT tolerance = 1.e-3;
	#endif
	for(unsigned int i = NN; i--; ) 
		if(fabs(local_energy[i] - (test[i] + random_field[i])) > tolerance) {
			cout << "ERROR " << i << " " << (test[i]+random_field[i]) << " " << local_energy[i] << endl;
			exit(1);
		}

	cout << "OK!" << endl;
}

/*--------PRINT ROUTINES--------*/
/*
void PrintEnergyHistogramAccum(ofstream &fstr, int maxindex,  int s, FLOAT leftlimit, FLOAT rightlimit, \
				thrust::device_vector<FLOAT> &histogram_accum)
{
	//thrust::host_vector<int> H = histogram_accum;
	FLOAT increment= (rightlimit-leftlimit)/NBINS;

	fstr << "#Velocity histogram for" << s << "samples " << endl;
	for(int index=0; index<maxindex; index++){
		fstr << "#Velocity histogram for two to the" << index << endl;
		for(int i=0;i<NBINS;i++)
		{
		fstr << leftlimit+(i)*increment << " " << histogram_accum[index*NBINS+i] / FLOAT(s) << endl;
		}
		fstr << endl << endl;
	}
	fstr << endl;
};
*/

void PrintLocalEnergies(vector<FLOAT> &local_energy, FLOAT *d_local_energy, const unsigned long long kmc_steps, \
			unsigned int iterations, const unsigned int M, const unsigned int ITEMAX, \
			const FLOAT TEMPERATURE)
{
	//TODO: Implement histogram calculation
	//CalculateAccumulateEnergyHistogram(LL, index1, histogram_accum, LEFTLIMIT, RIGHTLIMIT, NBINS, d_local_energy);
	//index1++;

	char filename [128];
	CUDA_SAFE_CALL(cudaMemcpy(&local_energy[0], d_local_energy, sizeof(FLOAT)*NN, \
	cudaMemcpyDeviceToHost));
	sprintf(filename, "LocalEnergies-L%iM%iT%.4fite%i.dat",LL,M,TEMPERATURE,1000000000+iterations);
	ofstream file1(filename);
	file1 << "#Local energies after" << iterations << "hops, corresponding to time (normalized with system size)" \
	<< kmc_steps/(1.*NN) << endl;
	for(unsigned int i = NN; i--; ) file1 << local_energy[i] << endl;
}

void PrintRandomFields(const FLOAT *d_random_field, vector<FLOAT> random_field, const unsigned int M, \
			const FLOAT TEMPERATURE)
{
	//TODO: Make this copy and print Async
	char filename [128];
	CUDA_SAFE_CALL(cudaMemcpy(&random_field[0], d_random_field, sizeof(FLOAT)*NN, cudaMemcpyDeviceToHost));
	sprintf(filename, "RandomField-L%iM%iT%.4f.dat",LL,M,TEMPERATURE);
	ofstream file1(filename);
//	file1 << "#Quenched Random Fields for a sample of size" << LL << "times" << LL << endl;
	for(unsigned int i = NN; i--; ) file1 << random_field[i] << endl;	// Print the quenched random field
}

void PrintOccNumbers(const unsigned int *d_occ_number, vector<unsigned int> occ_number, const unsigned int M, \
			const FLOAT TEMPERATURE)
{
	//TODO: Make this copy and print Async
	char filename [128];
	CUDA_SAFE_CALL(cudaMemcpy(&occ_number[0], d_occ_number, sizeof(unsigned int)*NN, cudaMemcpyDeviceToHost));
	sprintf(filename, "OccNumber-L%iM%iT%.4f.dat",LL,M,TEMPERATURE);
	ofstream file1(filename);
//	file1 << "#Occupation Numbers for a sample of size" << LL << "times" << LL << endl;
	for(unsigned int i = NN; i--; ) file1 << occ_number[i] << endl;	// Print the quenched random field
}

/////////////////// MAIN ///////////////////
//TODO: Clean main()
int main(int argc, char **argv) {

	/*------------INPUT----------------*/

	if(argc < 4) {
		cerr << "\n\n\t\tUsage: " << argv[0] << " TEMPERATURE NUMBER_OF_ITERATIONS log2(M) [SEED_CPU_RNG] \
[stored_random_fields (name with quotes)] [stored_configuration (name with quotes)] \n\n";
		return 1;
	}

	// TODO: Free M, let it be multiples of TILE, not necessarily power of 2.
	const FLOAT TEMPERATURE = atof(argv[1]);
	const FLOAT beta = 1./TEMPERATURE;
	const unsigned int ITEMAX = atoi(argv[2]);
	const unsigned int M = 0x1u << atoi(argv[3]);
	assert(M%TILE==0);
	unsigned int seed_random = times(NULL);
	if(argc > 4) seed_random = atoi(argv[4]);
	ifstream stored_random_fields;
	if(argc > 5) stored_random_fields.open(argv[5]);
	ifstream stored_configuration;
	if(argc > 6) stored_configuration.open(argv[6]);

	//print simulation parameters
	cout << "### SEED genrand3()....................: " << seed_random << endl;
	cout << "### SEED Philox .......................: " << SEED << endl;
	cout << "### L .................................: " << LL    << endl;
	cout << "### TEMPERATURE .......................: " << TEMPERATURE << endl;
	cout << "### Disorder Streght ..................: " << W    << endl;
#ifdef UNIFORM_DISORDER
	cout << "### Disorder Distribution .............: UNIFORM " << endl;
#else
	cout << "### Disorder Distribution .............: GAUSSIAN " << endl;
#endif
	cout << "### Localization Length ...............: " << LOC_LENGTH << endl;
	cout << "### Max Update steps (#hops) ..........: " << ITEMAX << endl; 
	cout << "### Threads for parallel rejection ....: " << M << endl;

	// Set CUDA Device
#ifdef CUDA_DEVICE
	CUDA_SAFE_CALL(cudaSetDevice(CUDA_DEVICE));
#endif

	// Initialize Mersenne
	init_genrand(seed_random);

	/*------- DECLARE ARRAYS ON HOST --------*/
	vector<unsigned int> occ_number(NN);	//System
	vector<FLOAT>  local_energy(NN);	//Local energies
	vector<FLOAT>  random_field(NN);	//Quenched random field

	// Arrays for tower sampling
	thrust::device_vector<FLOAT> h(NN);
	thrust::device_vector<int> dx(NN);
	thrust::device_vector<int> dy(NN);

	/*------- CONSTRUCT TOWER FOR HOPS SAMPLING -------*/
	FLOAT gamma;
	unsigned int tower_size = FillTransitionMatrix(dx, dy, h, gamma);
	cout << "## number of posible transitions = " << tower_size << ", Gamma " << gamma << endl;

	// readjust size according to resulting number of transitions
	h.resize(tower_size);
	dx.resize(tower_size);
	dy.resize(tower_size);

	FLOAT *raw_ptr_h = thrust::raw_pointer_cast(h.data());
	int *raw_ptr_dx = thrust::raw_pointer_cast(dx.data());
	int *raw_ptr_dy = thrust::raw_pointer_cast(dy.data());

	/*------- INITIALIZE OCCUPATION NUMBER -------*/
	//allocating memory for "occ_number" on the device
	unsigned int *d_occ_number;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_occ_number , sizeof(unsigned int)*NN));
	if(argc > 6){
		InitializeOccNumberFromFile(&occ_number[0], d_occ_number, stored_configuration);
	}else{
		InitializeOccNumber_onCPU(&occ_number[0], d_occ_number);
		//InitializeOccNumber_onGPU(ITEMAX + 2, d_occ_number);
	}

	/*------- INITIALIZE RANDOM FIELDS -------*/
	//allocating memory for "random_field" on the device
	FLOAT *d_random_field;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_random_field, sizeof(FLOAT)*NN));
	if(argc > 5){
		InitializeRandomFieldsFromFile(&random_field[0], d_random_field, stored_random_fields);
	}else{
		InitializeRandomFields(ITEMAX + 3, d_random_field);
	}

	//Print Random Fields
	PrintRandomFields(d_random_field, random_field, M, TEMPERATURE);

	/*------- INITIALIZE SPATIAL LOCAL ENERGIES -------*/
	//allocating memory for "local_energy" on the device
	FLOAT *d_local_energy;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_local_energy, sizeof(FLOAT)*NN));
	CalculateLocalEnergies_onGPU(d_occ_number, d_local_energy);

	/*------- ADD RANDOM FIELDS TO LOCAL ENERGIES -------*/
	thrust::device_ptr<FLOAT> local_energy_ptr(d_local_energy);
	thrust::device_ptr<FLOAT> random_field_ptr(d_random_field);
	thrust::transform(random_field_ptr, random_field_ptr+NN, local_energy_ptr, local_energy_ptr, \
			thrust::plus<FLOAT>());
	const FLOAT sum_random_fields = thrust::reduce(random_field_ptr, random_field_ptr + NN);

	/*------- INITIALIZE HISTOGRAMS -------*/
	/*thrust::device_vector<FLOAT> histogram_accum;
	histogram_accum.resize(NBINS*(ITEMAX/T_DATA+1));
	thrust::fill(histogram_accum.begin(),histogram_accum.end(),0);
	*/

	/*------- INITIALIZE FURTHER VARIABLES ------*/
	unsigned long long kmc_steps = 0;
	FLOAT kmc_time = 0.;
	unsigned int total_acc_moves = 0;
	unsigned int rng_counter=0;
	unsigned int attemps_counter=0;
	unsigned int logincrement=10;
	unsigned int logincrement2=1;
	unsigned int global_winner;
	#ifdef MEASURE_ACCEPTANCE
	unsigned int accepted;
	#endif
	FLOAT acceptance;
	int dipolar_moment;
	//FLOAT electric_field = ELECTRIC_FIELD;
	FLOAT electric_field = TEMPERATURE/10.;

	unsigned int *d_prop_pairs;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_prop_pairs, sizeof(unsigned int)*(M/TILE)*2));

	unsigned int *d_block_winners;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_block_winners, sizeof(unsigned int)*(M/TILE)));

	thrust::device_ptr<unsigned int> occ_number_ptr  (d_occ_number);

	// Turn on cronometer
	cpu_timer clock;
	clock.tic();

	/*------- UPDATES MAIN LOOP ------*/
	acceptance=1;
	for (unsigned int iterations=0; iterations<ITEMAX+1; iterations++) {

		// Print some data if some condition occurs
		#ifdef LINEAR_SPACING
		if((iterations % T_DATA) == 0) {
		#elif LOGARITHMIC_SPACING
		if( (iterations % logincrement == 0) || ((iterations) % (logincrement/2) == 0 ) || \
			((iterations) % (logincrement/3) == 0) || ((iterations) % (logincrement/4) == 0 ) || \
			((iterations) % (logincrement/6) == 0 ) || ((iterations) % (logincrement/8) == 0 )){
		#else
		if((iterations == ITEMAX) == 0) {
		#endif

			ComputeGlobalEnergyDipolarMomentAndPrint(occ_number_ptr, local_energy_ptr, random_field_ptr, \
								dipolar_moment, kmc_time, kmc_steps, sum_random_fields,\
								acceptance);

		}
		if (iterations == logincrement*10) logincrement*=10;

		// Print some other data if some other condition occurs
		#ifdef LINEAR_SPACING
		if((iterations % T_DATA2) == 0) {
		#elif LOGARITHMIC_SPACING
		if((iterations == logincrement2)) {
		#else
		if((iterations == ITEMAX) == 0) {
		#endif

			PrintLocalEnergies(local_energy, d_local_energy, kmc_steps, iterations, M, ITEMAX, TEMPERATURE);
		}
		if (iterations == logincrement2) logincrement2*=10; 

		// Propose jumps, Find winner, Update time
		attemps_counter=0;
		do {
			assert(M%TILE==0);
			Kernel_ProposeJumpsAndEvaluateMetropolis<<<M/TILE, TILE>>>(M, beta, d_occ_number, \
						d_local_energy, d_block_winners, raw_ptr_dx, raw_ptr_dy, raw_ptr_h, \
						d_prop_pairs, electric_field, tower_size, rng_counter);

			Kernel_FindGlobalWinner<<<1, 1>>>(M, d_block_winners);

			CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&global_winner, d_global_winner, sizeof(unsigned int), 0, \
			cudaMemcpyDeviceToHost));

			kmc_steps += (global_winner == M)?M:global_winner + 1;

		rng_counter ++;
		attemps_counter ++;
		}while(global_winner == M);
		total_acc_moves++;

		FLOAT rand_time = gasdev_real3();

		FLOAT ntrials = ((attemps_counter-1)*M + global_winner);
		kmc_time += ntrials/gamma + sqrt(ntrials)*rand_time/gamma;


		#ifdef MEASURE_ACCEPTANCE
		CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&accepted, d_accepted, sizeof(unsigned int), 0, \
		cudaMemcpyDeviceToHost));
		acceptance = accepted/(FLOAT)(attemps_counter*M);
		#endif
		// A more simple acceptance per hop is: 
		acceptance=1.0/(FLOAT)((attemps_counter-1)*M + global_winner + 1);
		

		// Perform a hop
		Kernel_PerformHopping<<<1, 1>>>(global_winner, d_prop_pairs, d_occ_number);

		// Update energies
		assert(LL%TILE_X==0);
		assert(LL%TILE_Y==0);
		dim3 dimBlock2D(TILE_X, TILE_Y);
		dim3 dimGrid2D(LL/TILE_X, LL/TILE_Y);
		Kernel_UpdateLocalEnergies<<<dimGrid2D, dimBlock2D>>>(d_local_energy);

	}
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	cout << "### ------ CUDA COULOMB GLASS ------ " << endl;
	// Turn off cronometer
	clock.tac();
	clock.print();
	FLOAT total_time = clock.cpu_elapsed();
	cout << "### KMC steps     : " << kmc_steps/(float)NN << endl;
	cout << "### Acc. moves    : " << total_acc_moves  << endl;
	cout << "### Trial moves   : " << kmc_steps  << endl;
	cout << "### Total time    : " << total_time  << " ms" << endl;
	cout << "### Time per jump : " << total_time/total_acc_moves << " ms/jump" << endl;
	cout << "### Time per MCs  : " << total_time/(kmc_steps/NN) << " ms/MCS" << endl;

	//// DtoH COPY, CHECK AND PRINT RESULTS ////
	CheckSelfConsistency(d_occ_number, d_local_energy, d_random_field, occ_number, local_energy, random_field);

	//// STORE ARRAYS TO RESTART SIMULATION AT THIS POINT ////
	PrintOccNumbers(d_occ_number, occ_number, M, TEMPERATURE);
	//RandomFields already stored

/*	#warning OUTPUT HERE
	cout << "   " << endl;
	cout << "### Local energies" << endl;
	//CUDA_SAFE_CALL(cudaMemcpy(&local_energy[0], d_local_energy, sizeof(FLOAT)*NN, cudaMemcpyDeviceToHost));
	for(unsigned int i = NN; i--; ) cout << local_energy[i] << endl;	// Print the local energy

	char filename [128];
	sprintf(filename, "EnergyDensityHistogram%iM%ib%.2f.dat",LL,M,beta);
	ofstream file5(filename);
//	PrintEnergyHistogramAccum(file5, ITEMAX/T_DATA+1, 1, LEFTLIMIT, RIGHTLIMIT, histogram_accum);
*/

	return 0;
};

