//jacobi7.cu
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "getopt.h"
#include "jacobi7_cuda_3d.h"
#include "jacobi7.h"

//#ifndef TIME_TILE_SIZE
//#warning TIME_TILE_SIZE is not set, defaulting to 1
//#define TIME_TILE_SIZE 2
//#endif

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
    
    const int xyz = nx * ny * nz;
    const int xyz_bytes = xyz * sizeof(float);

    float *h_A;
    float *h_B;
    float *d_A;
    float *d_B;

    float *h_A1;
    float *h_B1;
    
    // Allocate host buffers
    checkCuda(cudaMallocHost((void**)&h_A, xyz_bytes)); // host pinned
    checkCuda(cudaMallocHost((void**)&h_B, xyz_bytes));
    
    // for comparison btw CPU and GPU version
    checkCuda(cudaMallocHost((void**)&h_A1, xyz_bytes));
    checkCuda(cudaMallocHost((void**)&h_B1, xyz_bytes));

    // grid data iniatialization   
    // randomly generaed test data
    srand(time(NULL));
    int i = 0;
    for(; i < xyz; i++) {
        h_A[i] = 1 + (float)rand() / (float)RAND_MAX;
        h_B[i] =  h_A[i];
        h_A1[i] = h_A[i];
        h_B1[i] = h_A[i];
    }

    // Always use device 0
    cudaSetDevice(0);
    printf("Start computing...");
    /* set the ratio of cache/shared memory
    cudaFuncCachePreferNone: Default function cache configuration, no preference
    cudaFuncCachePreferShared: Prefer larger shared memory and smaller L1 cache
    cudaFuncCachePreferL1: Prefer larger L1 cache and smaller shared memory
    */
    //checkCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    // Allocate device buffers
    checkCuda(cudaMalloc((void**)&d_A, xyz_bytes));
    checkCuda(cudaMalloc((void**)&d_B, xyz_bytes));
    
    // Copy to device
    checkCuda(cudaMemcpy(d_A, h_A, xyz_bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, d_A, xyz_bytes, cudaMemcpyDeviceToDevice));
    
    // Setup the kernel
    float* input = d_A;
    float* output = d_B;
    dim3 grid(nx/tx, ny/ty, nz/tz);
    dim3 block(tx, ty, tz);

    printf("grid:(%d, %d, %d)\n", grid.x, grid.y, grid.z);
    printf("block:(%d, %d, %d)\n", tx, ty, tz);
    
    float *tmp;
    float *tmp1;
    float fac = 6.0/(h_A[0] * h_A[0]);
    const int sharedMemSize = (block.x + 2) * (block.y + 2) * (block.z + 2) * sizeof(float);
    printf("sharedMemSize:%d\n",sharedMemSize);
    float ms;
    cudaEvent_t startEvent, stopEvent;
    checkCuda( cudaEventCreate(&startEvent));
    checkCuda( cudaEventCreate(&stopEvent));

    checkCuda( cudaEventRecord(startEvent,0) );
    // Run the GPU kernel
    for(int t = 0; t < timesteps; t += 1) {         
        jacobi3d_7p_shmem_3d<<<grid, block, sharedMemSize>>>(input, output, nx, ny, nz, fac);
        // swap input and output
        tmp = input;
        input =  output;
        output = tmp;
    }
    checkCuda( cudaEventRecord(stopEvent, 0));
    checkCuda( cudaEventSynchronize(stopEvent));
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time of shared memory version (pure GPU) (ms): %f\n", ms);
  
    double gflop = (xyz * 1e-9) * 7.0 * timesteps;
    double gflop_per_sec = gflop * 1e3 / ms;
    printf("(GPU) %lf GFlop/s\n", gflop_per_sec);
    double mupdate_per_sec = ((xyz >> 20) * timesteps) * 1e3 / ms;
    printf("(GPU) %lf M updates/s\n", mupdate_per_sec);

    
    // Copy the result to main memory
    if(timesteps%2==0)
        checkCuda( cudaMemcpy(h_A, output, xyz_bytes, cudaMemcpyDeviceToHost));
    else
        checkCuda( cudaMemcpy(h_A, input, xyz_bytes, cudaMemcpyDeviceToHost));
    float *gpuResult = h_A;
    
    // Run the CPU version
    //double startTime = rtclock();
    for(int t = 0; t < timesteps; t += 1) {
        jacobi7(nx, ny, nz, h_A1, h_B1, fac);
        tmp1 = h_A1;
        h_A1 = h_B1;
        h_B1 = tmp1;
    }
    //double endTime = rtclock();
    //double elapsedTimeC = endTime - startTime;
    float *cpuResult;
    if ((timesteps%2) == 0)
        cpuResult = h_B1;
    else
        cpuResult = h_A1;

    /*printf("Elapsed Time:%lf\n", elapsedTimeC);
    double flops = xyz * 7.0 * timesteps;
    double gflops = flops / elapsedTimeC / 1e9;
    printf("(CPU) %lf GFlop/s\n", gflops);*/
    

    // compare the results btw CPU and GPU version
    double errorNorm, refNorm, diff;
    errorNorm = 0.0;
    refNorm = 0.0;
    i = 0;
    for (; i < xyz; ++i){
        diff = cpuResult[i] - gpuResult[i];
        errorNorm += diff * diff;
        refNorm += cpuResult[i] * cpuResult[i];
        if (abs(diff)> 1e-4)
        {
            printf("GPU[%d]=%f\n", i, gpuResult[i]);
            printf("CPU[%d]=%f\n", i, cpuResult[i]);
        }

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

    // cleanup
    checkCuda( cudaEventDestroy(startEvent));
    checkCuda( cudaEventDestroy(stopEvent));
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_A1);
    cudaFreeHost(h_B1);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}