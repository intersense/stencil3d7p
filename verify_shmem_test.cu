//jacobi7.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_call.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "getopt.h"
#include "jacobi7_cuda_shared.h"
#include "include/jacobi7.h"

//#ifndef TIME_TILE_SIZE
//#warning TIME_TILE_SIZE is not set, defaulting to 1
//#define TIME_TILE_SIZE 2
//#endif

// Timer function
double rtclock(){
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}


int main(int argc, char* *argv){
    if(argc != 8) {
        printf("USAGE: %s <NX> <NY> <NZ> <TX> <TY> <TZ> <TIME STEPS>\n", argv[0]);
        return 1;
    }
    // program parameters trans
    const int nx = atoi(argv[1]);
    const int ny = atoi(argv[2]);
    const int nz = atoi(argv[3]);
    const int tx = atoi(argv[4]);
    const int ty = atoi(argv[5]);
    const int tz = atoi(argv[6]);
    const int timesteps = atoi(argv[7]);
    
    
    // Allocate host buffers
    /*float *h_dA = (float*) malloc(xyz_byetes);
    float *h_dB = (float*) malloc(xyz_byetes);
    float *h_dA1 = (float*) malloc(xyz_byetes);
    float *h_dB1 = (float*) malloc(xyz_byetes);
    */
    
    // Always use device 0
    cudaSetDevice(0);

    /* set the ratio of cache/shared memory
    cudaFuncCachePreferNone: Default function cache configuration, no preference
    cudaFuncCachePreferShared: Prefer larger shared memory and smaller L1 cache
    cudaFuncCachePreferL1: Prefer larger L1 cache and smaller shared memory
    */
    CHECK_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    // Allocate device buffers
    
    dim3 grid(nx/tx, ny/ty, nz/tz);
    dim3 block(tx, ty, tz);

    
    float *tmp;
    float *tmp1;
    //float fac = 6.0/(h_dA[0] * h_dA[0]);
    const int sharedMemSize = nx * ny * nz * sizeof(float);
    printf("sharedMemSize:%d\n",sharedMemSize);
    double startTime = rtclock();
    // Run the GPU kernel
    for(int t = 0; t < timesteps; t += TIME_TILE_SIZE) {         
        jacobi3d_7p_shmem_test<<<grid, block, sharedMemSize>>>();
        // swap input and output
        //tmp = input;
        //input =  output;
        //output = tmp;
    }
    
    // Free buffers
    /*free(h_dA);
    free(h_dB);
    free(h_dA1);
    free(h_dB1);*/
    //CHECK_CALL(cudaFree(d_dA));
    //CHECK_CALL(cudaFree(d_dB));

}