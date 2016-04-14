//jacobi7.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "getopt.h"

#include <cuda_call.h>
#include <jacobi7_cuda_35d.h>
#include <jacobi7.h>

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
        h_dA[i] = 2;//(float)rand() / (float)RAND_MAX;
        h_dB[i] =  h_dA[i];
        h_dA1[i] = h_dA[i];
        h_dB1[i] = h_dA[i];
    }

    float *share_out_d = (float*) malloc((tx+2) * (ty+2) * sizeof(float));
    float *share_out_h = (float*) malloc((tx+2) * (ty+2) * sizeof(float));


    printf("Start computing...\n");
    printf("h_dB[%d]:%f\n", 1+nx*(1+ny*1), h_dB[1+nx*(1+ny*1)]);
    printf("h_dA[%d]:%f\n", 3+nx*(4+ny*5), h_dA[3+nx*(4+ny*5)]);    

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
    CHECK_CALL(cudaMalloc((void**)&share_out_d, (tx+2) * (ty+2) * sizeof(float)));
    
    // Copy to device
    CHECK_CALL(cudaMemcpy(d_dA, h_dA, xyz_bytes, cudaMemcpyHostToDevice));
    CHECK_CALL(cudaMemcpy(d_dB, h_dB, xyz_bytes, cudaMemcpyHostToDevice));
    CHECK_CALL(cudaMemcpy(d_dB, d_dA, xyz_bytes, cudaMemcpyDeviceToDevice));
    
    // Setup the kernel

    dim3 grid(nx/(tx-2), ny/(ty-2), 1);
    dim3 block(tx, ty);

    if (nx % (tx- 2)) ++grid.x;
    if (ny % (ty - 2)) ++grid.y;
    float *tmp;
    const float fac = 6.0/(h_dA[0] * h_dA[0]);
    const int sharedMemSize = 9 * (block.x+2) * (block.y+2) * sizeof(float);
    printf("sharedmemeory Size:%dk\n",sharedMemSize/1024);
    double startTime = rtclock();
    // Run the GPU kernel

    for(int t = 0; t < timesteps; t += 2) {         
        jacobi7_35d<<<grid, block, sharedMemSize>>>(d_dA, d_dB, nx, ny, nz, fac);
        //  printf("t:%d\n",t);
        // swap input and output
        tmp  = d_dA;
        d_dA = d_dB;
        d_dB = tmp;
    }

    SYNC_DEVICE();
    ASSERT_STATE("jacobi7_35d");
    double endTime = rtclock();
    double elapsedTimeG = endTime - startTime;

    printf("Elapsed Time:%lf\n", elapsedTimeG);
    double flops = xyz * 7.0 * timesteps;
    double gflops = flops / elapsedTimeG / 1e9;
    printf("(GPU) %lf GFlop/s\n", gflops);
    
    // Copy the result to main memory
    float *resultG;
    //if((timesteps/2) % 2){
        CHECK_CALL(cudaMemcpy(h_dA, d_dA, xyz_bytes, cudaMemcpyDeviceToHost));
    /*}
    else{
        CHECK_CALL(cudaMemcpy(h_dA, d_dB, xyz_bytes, cudaMemcpyDeviceToHost));
    }*/
    resultG = h_dA;

    
    // Run the CPU version
    startTime = rtclock();
    for(int t = 0; t < timesteps; t += 1) {
        jacobi7(nx, ny, nz, h_dA1, h_dB1, fac);
        tmp = h_dA1;
        h_dA1 = h_dB1;
        h_dB1 = tmp;
    }
    float *resultC;
    if (timesteps % 2)
        resultC = h_dB1;
    else
        resultC = h_dA1;

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
    for (int i = 0; i < xyz; ++i){
        diff = resultC[i] - resultG[i];
        errorNorm += diff * diff;
        refNorm += resultC[i] * resultG[i];
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
    int i;
    int j;
    int k;

    i = 1;
    j = 1;
    k = ny-2;
    for (k=nz-2;k>nz/2;k--)
    {
        printf("(%d,%d,%d)\n",i,j,k);
        printf("resultC[%d]:%f\n", i+nx*(j+ny*k), resultC[i+nx*(j+ny*k)]);
        printf("resultG[%d]:%f\n", i+nx*(j+ny*k), resultG[i+nx*(j+ny*k)]);
        printf("-----------------------------------\n");
    }

    // Free buffers
    free(h_dA);
    free(h_dB);
    free(h_dA1);
    free(h_dB1);
    CHECK_CALL(cudaFree(d_dA));
    CHECK_CALL(cudaFree(d_dB));

}