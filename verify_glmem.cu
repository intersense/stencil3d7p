//jacobi7.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_call.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "getopt.h"
#include "include/jacobi7_cuda.h"
#include "include/jacobi7.h"

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

    float *h_dA1;
    float *h_dB1;
    
    // Allocate host buffers
    h_dA = (float*) malloc(xyz_byetes);
    h_dB = (float*) malloc(xyz_byetes);
    h_dA1 = (float*) malloc(xyz_byetes);
    h_dB1 = (float*) malloc(xyz_byetes);

    // grid data iniatialization   
    // randomly generaed test data
    srand(time(NULL));
    int i = 0;
    for(; i < xyz; i++) {
        h_dA[i] = 1 + (float)rand() / (float)RAND_MAX;
        h_dB[i] =  h_dA[i];
        h_dA1[i] = h_dA[i];
        h_dB1[i] = h_dA[i];
    }
    printf("Start computing...");
    printf("h_dB[%d]:%f\n", 2+32*(3+32*4), h_dB[2+32*(3+32*4)]);
    printf("h_dA[%d]:%f\n", 2+32*(3+32*4), h_dA[2+32*(3+32*4)]);

    float *B = 0;
    const int ldb = 0;
    const int ldc = 0;
    

    // Always use device 0
    cudaSetDevice(0);

    /* set the ratio of cache/shared memory
    cudaFuncCachePreferNone: Default function cache configuration, no preference
    cudaFuncCachePreferShared: Prefer larger shared memory and smaller L1 cache
    cudaFuncCachePreferL1: Prefer larger L1 cache and smaller shared memory
    */
    //CHECK_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    // Allocate device buffers
    CHECK_CALL(cudaMalloc((void**)&d_dA, xyz_byetes));
    CHECK_CALL(cudaMalloc((void**)&d_dB, xyz_byetes));
    
    // Copy to device
    CHECK_CALL(cudaMemcpy(d_dA, h_dA, xyz_byetes, cudaMemcpyHostToDevice));
    //CHECK_CALL(cudaMemcpy(d_dB, h_dB, xyz_byetes, cudaMemcpyHostToDevice));
    CHECK_CALL(cudaMemcpy(d_dB, d_dA, xyz_byetes, cudaMemcpyDeviceToDevice));
    
    // Setup the kernel
    float* input = d_dA;
    float* output = d_dB;
    dim3 grid(nx/tx, ny/ty);
    dim3 block(tx, ty);

    // Run the kernel
    
    float *tmp;
    float *tmp1;
    float fac = 6.0/(h_dA[0] * h_dA[0]);

    double startTime = rtclock();
    // Run the GPU kernel
    for(int t = 0; t < timesteps; t += 1) {
        jacobi3d_7p_glmem<<<grid, block>>>(input, output, nx, ny, nz, fac);
        // swap input and output
        tmp = input;
        input =  output;
        output = tmp;
    }
    
    SYNC_DEVICE();
    ASSERT_STATE("Kernel");
    double endTime = rtclock();
    double elapsedTimeG = endTime - startTime;
  
    printf("Elapsed Time:%lf\n", elapsedTimeG);
    double flops = xyz * 7.0 * timesteps;
    double gflops = flops / elapsedTimeG / 1e9;
    printf("(GPU) %lf GFlop/s\n", gflops);
    
    // Copy the result to main memory
    CHECK_CALL(cudaMemcpy(h_dB, input, xyz_byetes, cudaMemcpyDeviceToHost));
    
    // Run the CPU version
    startTime = rtclock();
    for(int t = 0; t < timesteps; t += 1) {
        jacobi7(nx, ny, nz, h_dA1, B, ldb, h_dB1, ldc);
        tmp1 = h_dA1;
        h_dA1 = h_dB1;
        h_dB1 = tmp1;
    }
    endTime = rtclock();
    double elapsedTimeC = endTime - startTime;

    printf("Elapsed Time:%lf\n", elapsedTimeC);
    flops = xyz * 7.0 * timesteps;
    gflops = flops / elapsedTimeC / 1e9;
    printf("(CPU) %lf GFlop/s\n", gflops);


    // compare the results btw CPU and GPU version
    double errorNorm, refNorm, diff;
    errorNorm = 0.0;
    refNorm = 0.0;
    for (; i < xyz; ++i){
        diff = h_dA1[i] - h_dB[i];
        errorNorm += diff * diff;
        refNorm += h_dA1[i] * h_dA1[i];
        /*if (h_dB[i+nx*(j+ny*k)] != h_dA1[i+nx*(j+ny*k)])
                   diff = 1;*/
    }
    errorNorm = sqrt(errorNorm);
    refNorm   = sqrt(refNorm);

    printf("Error Norm:%lf\n", errorNorm);
    printf("Ref Norm:%lf\n", refNorm);
  
    if(abs(refNorm) < 1e-7) {
      printf("Correctness, FAILED\n");
    }
    else if((errorNorm / refNorm) > 1e-2) {
      printf("Correct  ness, FAILED\n");
    }
    else {
      printf("Correctness, PASSED\n");
    }

    printf("h_dB[%d]:%f\n", 2+ny*(3+nz*4), h_dB[2+ny*(3+nz*4)]);
    printf("h_dA[%d]:%f\n", 2+ny*(3+nz*4), h_dA[2+ny*(3+nz*4)]);
    printf("h_dB1[%d]:%f\n", 2+ny*(3+nz*4), h_dB1[2+ny*(3+nz*4)]);
    printf("h_dA1[%d]:%f\n", 2+ny*(3+nz*4), h_dA1[2+ny*(3+nz*4)]);
    printf("-----------------------------------\n");
    printf("h_dB[%d]:%f\n", 3+ny*(4+nz*5), h_dB[3+ny*(4+nz*5)]);
    printf("h_dA[%d]:%f\n", 3+ny*(4+nz*5), h_dA[3+ny*(4+nz*5)]);
    printf("h_dB1[%d]:%f\n", 3+ny*(4+nz*5), h_dB1[3+ny*(4+nz*5)]);
    printf("h_dA1[%d]:%f\n", 3+ny*(4+nz*5), h_dA1[3+ny*(4+nz*5)]);
    // Free buffers
    free(h_dA);
    free(h_dB);
    free(h_dA1);
    free(h_dB1);
    CHECK_CALL(cudaFree(d_dA));
    CHECK_CALL(cudaFree(d_dB));

}