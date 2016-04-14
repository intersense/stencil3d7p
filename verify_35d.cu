#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "getopt.h"
#include <jacobi7_cuda_35d.h>
#include <jacobi7.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

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

    float *h_A;
    float *h_B;
    float *d_A;
    float *d_B;

    float *h_A1;
    float *h_B1;
    
    int devId = 0;
    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, devId));
    printf("Device : %s\n", prop.name);
    checkCuda( cudaSetDevice(devId));
    
    // Allocate host buffers
    checkCuda(cudaMallocHost((void**)&h_A, xyz_bytes)); // host pinned
    checkCuda(cudaMallocHost((void**)&h_B, xyz_bytes));
    
    // for comparison btw CPU and GPU version
    checkCuda(cudaMallocHost((void**)&h_A1, xyz_bytes));
    checkCuda(cudaMallocHost((void**)&h_B1, xyz_bytes));

    // grid data iniatialization   
    // randomly generaed test data
    srand(time(NULL));
    
    for(int i = 0; i < xyz; i++) {
        h_A[i] = 1 + (float)rand() / (float)RAND_MAX;
        h_B[i] =  h_A[i];
        h_A1[i] = h_A[i];
        h_B1[i] = h_A[i];
    }

    float *share_out_d = (float*) malloc((tx+2) * (ty+2) * sizeof(float));
    float *share_out_h = (float*) malloc((tx+2) * (ty+2) * sizeof(float));


    printf("Start computing...\n");
    printf("h_B[%d]:%f\n", 1+nx*(1+ny*1), h_B[1+nx*(1+ny*1)]);
    printf("h_A[%d]:%f\n", 3+nx*(4+ny*5), h_A[3+nx*(4+ny*5)]);    

    // Always use device 0
    cudaSetDevice(0);

    /* set the ratio of cache/shared memory
    cudaFuncCachePreferNone: Default function cache configuration, no preference
    cudaFuncCachePreferShared: Prefer larger shared memory and smaller L1 cache
    cudaFuncCachePreferL1: Prefer larger L1 cache and smaller shared memory
    
    checkCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    */

    // Allocate device buffers
    checkCuda(cudaMalloc((void**)&d_A, xyz_bytes));
    checkCuda(cudaMalloc((void**)&d_B, xyz_bytes));
    checkCuda(cudaMalloc((void**)&share_out_d, (tx+2) * (ty+2) * sizeof(float)));
    
    // Copy to device
    checkCuda(cudaMemcpy(d_A, h_A, xyz_bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, xyz_bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, d_A, xyz_bytes, cudaMemcpyDeviceToDevice));
    
    // Setup the kernel

    dim3 grid(nx/(tx-2), ny/(ty-2), 1);
    dim3 block(tx, ty);

    if (nx % (tx- 2)) ++grid.x;
    if (ny % (ty - 2)) ++grid.y;
    float *tmp;
    const float fac = 6.0/(h_A[0] * h_A[0]);
    const int sharedMemSize = 9 * (block.x+2) * (block.y+2) * sizeof(float);
    printf("sharedmemeory Size:%dk\n",sharedMemSize/1024);
    
    // create events and streams
    cudaEvent_t startEvent, stopEvent;
    
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));
    // Run the GPU kernel
    // timing start include data transfer and memory allocation
    checkCuda(cudaEventRecord(startEvent,0));
    for(int t = 0; t < timesteps; t += 2) {         
        jacobi7_35d<<<grid, block, sharedMemSize>>>(d_A, d_B, nx, ny, nz, fac);
        //  printf("t:%d\n",t);
        // swap input and output
        tmp  = d_A;
        d_A = d_B;
        d_B = tmp;
    }

    // timing end pure gpu computing
    checkCuda( cudaEventRecord(stopEvent, 0));
    checkCuda( cudaEventSynchronize(stopEvent));
    float ms;
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent));

    printf("Time of shared memory version (pure GPU) (ms): %f\n", ms);
  
    double gflop = (xyz * 1e-9) * 7.0 * timesteps;
    double gflop_per_sec = gflop * 1e3 / ms;
    printf("(GPU) %lf GFlop/s\n", gflop_per_sec);
    double mupdate_per_sec = ((xyz >> 20) * timesteps) * 1e3 / ms;
    printf("(GPU) %lf M updates/s\n", mupdate_per_sec);
    
    // Copy the result to main memory
    float *resultG;
    if((timesteps/2) % 2){
        checkCuda(cudaMemcpy(h_A, d_B, xyz_bytes, cudaMemcpyDeviceToHost));
    }
    else{
        checkCuda(cudaMemcpy(h_A, d_A, xyz_bytes, cudaMemcpyDeviceToHost));
    }
    resultG = h_A;

    
    // Run the CPU version
    double startTime, endTime;
    startTime = rtclock();
    for(int t = 0; t < timesteps; t += 1) {
        jacobi7(nx, ny, nz, h_A1, h_B1, fac);
        tmp = h_A1;
        h_A1 = h_B1;
        h_B1 = tmp;
    }
    float *resultC;
    if (timesteps % 2)
        resultC = h_A1;
    else
        resultC = h_B1;

    endTime = rtclock();
    double elapsedTimeC = endTime - startTime;

    printf("Elapsed Time:%lf\n", elapsedTimeC);
    double flops = xyz * 7.0 * timesteps;
    double gflops = flops / elapsedTimeC / 1e9;
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
    
    // simple comparison
    /*
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
    */
    // Free buffers
    checkCuda( cudaEventDestroy(startEvent));
    checkCuda( cudaEventDestroy(stopEvent));
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_A1);
    cudaFreeHost(h_B1);
    cudaFree(d_A);
    cudaFree(d_B);
}