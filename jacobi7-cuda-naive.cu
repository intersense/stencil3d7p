//jacobi7.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_call.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "getopt.h"
#include "include/jacobi7_cuda.h"

// Timer function
double rtclock(){
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}


int main(int argc, char* *argv){
    if(argc != 7) {
        printf("USAGE: %s <NX> <NY> <NZ> <TX> <TY> <TIME STEPS>\n", argv[0]);
        return 1;
    }
    // program parameters trans
    const int nx = atoi(argv[1]);
    const int ny = atoi(argv[2]);
    const int nz = atoi(argv[3]);
    const int tx = atoi(argv[4]);
    const int ty = atoi(argv[5]);
    const int timesteps = atoi(argv[6]);
    
    const int xyz = nx * ny * nz;
    const int xyz_byetes = xyz * sizeof(float);

    float *h_dA;
    float *h_dB;
    float *d_dA;
    float *d_dB;

    
    // Allocate host buffers
    h_dA = (float*) malloc(xyz_byetes);
    h_dB = (float*) malloc(xyz_byetes);

    // grid data iniatialization   
    // randomly generaed test data
    srand(time(NULL));
    int i = 0;
    for(; i < xyz; i++) {
        h_dA[i] = (float)rand() / (float)RAND_MAX;
        h_dB[i] = 0.0;
    }
    printf("h_dB[%d]:%f\n", 2+32*(3+32*4), h_dB[2+32*(3+32*4)]);
    printf("h_dA[%d]:%f\n", 2+32*(3+32*4), h_dA[2+32*(3+32*4)]);

    // Always use device 0
    cudaSetDevice(0);

    /* set the ratio of cache/shared memory
    cudaFuncCachePreferNone: Default function cache configuration, no preference
    cudaFuncCachePreferShared: Prefer larger shared memory and smaller L1 cache
    cudaFuncCachePreferL1: Prefer larger L1 cache and smaller shared memory
    */
    CHECK_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    // Allocate device buffers
    CHECK_CALL(cudaMalloc((void**)&d_dA, xyz_byetes));
    CHECK_CALL(cudaMalloc((void**)&d_dB, xyz_byetes));
    
    // Copy to device
    CHECK_CALL(cudaMemcpy(d_dA, h_dA, xyz_byetes, cudaMemcpyHostToDevice));
    CHECK_CALL(cudaMemcpy(d_dB, h_dB, xyz_byetes, cudaMemcpyHostToDevice));
    
    // Setup the kernel
    float* input = d_dA;
    float* output = d_dB;
    dim3 grid(nx/tx, ny/ty);
    dim3 block(tx, ty);

    // Run the kernel
    double startTime = rtclock();
    float *tmp;
    float fac = 6.0/(h_dA[0]*h_dA[0]);
    for(int t = 0; t < timesteps; t += 1) {
        //const int sharedMemSize = block.x * block.y * sizeof(float);
        //jacobi3d_7p_naive<<<grid, block>>>(input, output, nx, ny, nz, fac);
        //jacobi3d_7p_glmem<<<grid, block>>>(input, output, nx, ny, nz, fac);
        const int sharedMemSize = block.x * block.y * sizeof(float); // local layer, top layer and bottom layer
        jacobi3d_7p_shmem_test<<<grid, block, sharedMemSize>>>(input, output, nx, ny, nz, fac);
        // swap input and output
        tmp = input;
        input =  output;
        output = tmp;
    }
    SYNC_DEVICE();
    ASSERT_STATE("Kernel");
    double endTime = rtclock();
    double elapsedTime = endTime - startTime;
  
    printf("Elapsed Time:%lf\n", elapsedTime);
    double flops = xyz * 7.0 * timesteps;
    double gflops = flops / elapsedTime / 1e9;

    printf("%lf GFlop/s\n", gflops);
    
    CHECK_CALL(cudaMemcpy(h_dB, input, xyz_byetes, cudaMemcpyDeviceToHost));

    printf("h_dB[%d]:%f\n", 2+32*(3+32*4), h_dB[2+32*(3+32*4)]);
    printf("h_dA[%d]:%f\n", 2+32*(3+32*4), h_dA[2+32*(3+32*4)]);

    // Free buffers
    free(h_dA);
    free(h_dB);
    CHECK_CALL(cudaFree(d_dA));
    CHECK_CALL(cudaFree(d_dB));

}