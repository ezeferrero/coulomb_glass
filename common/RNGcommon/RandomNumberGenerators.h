/*
 * RandomNUmberGenerators.h
 *
 *  Created on: May 16, 2012
 *      Author: larry
 */

#ifndef RANDOMNUMBERGENERATORS_H_
#define RANDOMNUMBERGENERATORS_H_

/* LCG: Linear Congruential Generator */
#ifdef RNG_LCG
#include "LCG.cu"
#endif

/* MT: Mersenne Twister */
#ifdef RNG_MT
#include "MT.cu"
#endif

/* MWC: Multiply With Carry */
#ifdef RNG_MWC
#include "MWC.cu"
#endif

/* CRAND: CURand (nVidia random number generator) */
#ifdef RNG_CRAND
#include "CRAND.cu"
#endif

/* PHILOX: Counter-based random number generator */
#ifdef RNG_PHILOX
#include "PHILOX.cu"
#endif


#endif /* RANDOMNUMBERGENERATORS_H_ */
