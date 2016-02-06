//jacobi7.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "getopt.h"
#include <cuda_call.h>
#include <jacobi7_cuda.h>
#include <jacobi7.h>

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
    const int xyz_bytes = xyz * sizeof(float);

    float *h_dA;
    float *h_dB;
    float *d_dA;
    float *d_dB;

    float *h_dA1;
    float *h_dB1;
    
    // Allocate host buffers
    h_dA = (float*) malloc(xyz_bytes);
    h_dB = (float*) malloc(xyz_bytes);
    h_dA1 = (float*) malloc(xyz_bytes);
    h_dB1 = (float*) malloc(xyz_bytes);

    // grid data iniatialization   
    // randomly generaed test data
    srand(time(NULL));
    
    for(int i = 0; i < xyz; i++) {
        h_dA[i] = (float)rand() / (float)RAND_MAX;
        h_dB[i] =  h_dA[i];
        h_dA1[i] = h_dA[i];
        h_dB1[i] = h_dA[i];
    }
    printf("Start computing...\n");
    printf("h_dB[%d]:%f\n", 2+32*(3+32*4), h_dB[2+32*(3+32*4)]);
    printf("h_dA[%d]:%f\n", 2+32*(3+32*4), h_dA[2+32*(3+32*4)]);
    printf("-----------------------------------\n");

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
    CHECK_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    // Allocate device buffers
    CHECK_CALL(cudaMalloc((void**)&d_dA, xyz_bytes));
    CHECK_CALL(cudaMalloc((void**)&d_dB, xyz_bytes));
    
    // Copy to device
    CHECK_CALL(cudaMemcpy(d_dA, h_dA, xyz_bytes, cudaMemcpyHostToDevice));
    //CHECK_CALL(cudaMemcpy(d_dB, h_dB, xyz_bytes, cudaMemcpyHostToDevice));
    //CHECK_CALL(cudaMemcpy(d_dB, d_dA, xyz_bytes, cudaMemcpyDeviceToDevice));
    
    // Setup the kernel

    dim3 grid(nx/tx, ny/ty);
    dim3 block(tx, ty);

    
    float *tmp;
    const float fac = 6.0/(h_dA[0] * h_dA[0]);
    //const int sharedMemSize = (block.x + 2) * (block.y + 2) * sizeof(float); 

    double startTime = rtclock();
    // Run the GPU kernel
    for(int t = 0; t < timesteps; t += 1) {
        
        jacobi3d_7p_shmem_adam_reg<<<grid, block>>>(d_dA, d_dB, nx, ny, nz, fac);
        // swap input and output
        tmp = d_dA;
        d_dA =  d_dB;
        d_dB = tmp;
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
    float *resultG;
    if(timesteps % 2){
        CHECK_CALL(cudaMemcpy(h_dA, d_dA, xyz_bytes, cudaMemcpyDeviceToHost));
    }
    else{
        CHECK_CALL(cudaMemcpy(h_dA, d_dB, xyz_bytes, cudaMemcpyDeviceToHost));
    }
    resultG = h_dA;
    
    // Run the CPU version
    startTime = rtclock();
    for(int t = 0; t < timesteps; t += 1) {
        jacobi7(nx, ny, nz, h_dA1, B, ldb, h_dB1, ldc, fac);
        tmp = h_dA1;
        h_dA1 = h_dB1;
        h_dB1 = tmp;
    }

    endTime = rtclock();
    double elapsedTimeC = endTime - startTime;

    float *resultC;
    if (timesteps % 2)
        resultC = h_dA1;
    else
        resultC = h_dB1;

    printf("Elapsed Time:%lf\n", elapsedTimeC);
    flops = xyz * 7.0 * timesteps;
    gflops = flops / elapsedTimeC / 1e9;
    printf("(CPU) %lf GFlop/s\n", gflops);


    // compare the results btw CPU and GPU version
    double errorNorm, refNorm, diff;
    errorNorm = 0.0;
    refNorm = 0.0;
    for (int i=0; i < xyz; ++i){
        diff = resultC[i] - resultG[i];
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
      printf("Correctness, FAILED\n");
    }
    else {
      printf("Correctness, PASSED\n");
    }

    printf("resultC[%d]:%f\n", 2+ny*(3+nz*4), resultC[2+ny*(3+nz*4)]);
    printf("resultG[%d]:%f\n", 2+ny*(3+nz*4), resultG[2+ny*(3+nz*4)]);
    printf("-----------------------------------\n");
    printf("resultC[%d]:%f\n", 3+ny*(4+nz*5), resultC[3+ny*(4+nz*5)]);
    printf("resultG[%d]:%f\n", 3+ny*(4+nz*5), resultG[3+ny*(4+nz*5)]);


    // Free buffers
    free(h_dA);
    free(h_dB);
    free(h_dA1);
    free(h_dB1);
    CHECK_CALL(cudaFree(d_dA));
    CHECK_CALL(cudaFree(d_dB));

}