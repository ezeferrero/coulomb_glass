#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

/*
 *
 *      From CUDA By Example An Introduction to General-Purpose GPU Programming”
 *  by Jason Sanders and Edward Kandrot, Addison-Wesley, Upper Saddle River, NJ, 2011
 *
 */

// Macro for handle errors
__host__ static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


/* Function to check for CUDA runtime errors */
static void checkCUDAError(const char* msg) {
	/* Get last error */
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
    	/* Print the error */
        printf("Cuda error: %s %s\n",msg, cudaGetErrorString( err));
        /* Abort the program */
        exit(EXIT_FAILURE);
    }
}






//
// un timer para procesos en la GPU
// 
// USO: gpu_timer T; T.gpu_tic();...kernels...;T.gpu_tac();T.print_gpu_elapsed();
class gpu_timer{
	private:
        cudaEvent_t start, stop;
        float elapsedTime;
        float av_elapsedTime;
        float dev_elapsedTime;
        unsigned long count;

        public:
        gpu_timer(){
        	reset();
        };
        void reset(){
        	av_elapsedTime=dev_elapsedTime=0.0;
        	count=0;
        };
        void tic(){
            HANDLE_ERROR( cudaEventCreate( &start ) );
            HANDLE_ERROR( cudaEventCreate( &stop ) );
        	HANDLE_ERROR( cudaEventRecord( start, 0 ) );
        }
        void tac(){
            HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
            HANDLE_ERROR( cudaEventSynchronize( stop ) );
            HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,start, stop ) );
            HANDLE_ERROR( cudaEventDestroy( start ) );
            HANDLE_ERROR( cudaEventDestroy( stop ) );
            av_elapsedTime+=elapsedTime;
            dev_elapsedTime+=elapsedTime*elapsedTime;
            count++;
        } 
        void print_stat()
        {
        	if(count==1)
            printf( "###GPU Time elapsed %.8f ms \n", elapsedTime);
        	else
            printf( "###GPU Time elapsed %.8f +- %.8f ms (%ld samples)\n",
            		average_elapsed_time(), dev_elapsed_time(), count);
        }
        float print()
        {
            printf( "###GPU Time elapsed %.8f ms \n", elapsedTime);
	    return elapsedTime;
        }
        float elapsed_time()
        {
            return elapsedTime;
        }
        float average_elapsed_time()
        {
            return av_elapsedTime/count;
        }
        float dev_elapsed_time()
        {
        	float av=average_elapsed_time();
            return sqrt(dev_elapsedTime/count - av*av)/sqrt(count);
        }

        ~gpu_timer()
	    {};
};



#if  (defined(__MACH__))
#include "mac_cpu_timer.h"
#else
//
//	cpu_timer defined here
//
#include "linux_cpu_timer.h"
#endif
