//jacobi7.cu
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "getopt.h"
#include "jacobi7_cuda.h"
#include "jacobi7.h"

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
        printf("USAGE: %s <0row_or_1col_first> <NX> <NY> <NZ> <TX> <TY> <TIME STEPS>\n", argv[0]);
        return 1;
    }
    // program parameters trans
    const int row_or_col = atoi(argv[1]);
    const int nx = atoi(argv[2]);
    const int ny = atoi(argv[3]);
    const int nz = atoi(argv[4]);
    const int tx = atoi(argv[5]);
    const int ty = atoi(argv[6]);
    const int timesteps = atoi(argv[7]);
    
    void (*kernel)(float *, float *, const int , const int , const int , float );
    // the first arg determins the row first or column first
    // 0: row first; 1: column first
    if (row_or_col == 0)
        kernel = &jacobi3d_7p_glmem;
    if (row_or_col == 1)
        kernel = &jacobi3d_7p_glmem_col; 
    if (row_or_col == 2)
        kernel = &jacobi3d_7p_glmem_col_row; 

    const int xyz = nx * ny * nz;
    const int xyz_bytes = xyz * sizeof(float);

    float *h_dA, *h_dA1;
    float *h_dB, *h_dB1;
    float *d_dA;
    float *d_dB;
    
    // Allocate host buffers
    checkCuda(cudaMallocHost((void**)&h_dA, xyz_bytes)); // host pinned
    checkCuda(cudaMallocHost((void**)&h_dB, xyz_bytes));
    // for comparison btw CPU and GPU version
    checkCuda(cudaMallocHost((void**)&h_dA1, xyz_bytes));
    checkCuda(cudaMallocHost((void**)&h_dB1, xyz_bytes));

    // grid data iniatialization   
    // randomly generaed test data
    srand(time(NULL));
    int i = 0;
    for(; i < xyz; i++) {
        h_dA[i] = 1 + (float)rand() / (float)RAND_MAX;
        h_dA1[i] = h_dB1[i] = h_dB[i] =  h_dA[i];
    }
    printf("Start computing... \n");   

    // Always use device 0
    checkCuda(cudaSetDevice(0));

    /* set the ratio of cache/shared memory
    cudaFuncCachePreferNone: Default function cache configuration, no preference
    cudaFuncCachePreferShared: Prefer larger shared memory and smaller L1 cache
    cudaFuncCachePreferL1: Prefer larger L1 cache and smaller shared memory
    */
    //checkCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    // Allocate device buffers
    checkCuda(cudaMalloc((void**)&d_dA, xyz_bytes));
    checkCuda(cudaMalloc((void**)&d_dB, xyz_bytes));
    
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    float milliseconds = 0;

    checkCuda(cudaEventRecord(start));
    
    // Copy to device
    checkCuda(cudaMemcpy(d_dA, h_dA, xyz_bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Data %dMB transferred H2D time:%f ms\n", xyz_bytes >> 20, milliseconds);
    printf("Bandwidth H2D:%f MB/s\n", (float)(xyz_bytes >> 20)/(milliseconds/1000));

    checkCuda(cudaMemcpy(d_dB, d_dA, xyz_bytes, cudaMemcpyDeviceToDevice));
    
    // Setup the kernel
    float* input = d_dA;
    float* output = d_dB;
    // modify nx/tx and ny/ty to (nx+tx-1)/tx and (ny+ty-1)/ty
    // inorder to avoid wrong configuration
    dim3 grid((nx+tx-1)/tx, (ny+ty-1)/ty);
    dim3 block(tx, ty);
    printf("grid:(%d, %d)\n", grid.x, grid.y);
    printf("block:(%d, %d)\n", tx, ty);

    float *tmp;
    float fac = 6.0/(h_dA[0] * h_dA[0]);

    checkCuda(cudaEventRecord(start));
    for(int t = 0; t < timesteps; t += 1) {
        kernel<<<grid, block>>>(input, output, nx, ny, nz, fac);
        tmp = input;
        input =  output;
        output = tmp;
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("GPU kernel Elapsed Time (pure GPU):%f ms\n", milliseconds);
    double gflop = (xyz * 1e-9) * 7.0 * timesteps;
    double gflop_per_sec = gflop * 1e3 / milliseconds;
    printf("(GPU) %lf GFlop/s\n", gflop_per_sec);
    double mupdate_per_sec = ((xyz >> 20) * timesteps) * 1e3 / milliseconds;
    printf("(GPU) %lf M updates/s\n", mupdate_per_sec);
    
    float *gpuResult;
    // Copy the result to main memory
    if(timesteps%2==0){
        checkCuda(cudaEventRecord(start));
        checkCuda(cudaMemcpy(h_dB, output, xyz_bytes, cudaMemcpyDeviceToHost));
        checkCuda(cudaEventRecord(stop));
        checkCuda(cudaEventSynchronize(stop));
        checkCuda(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("Data %dMB transferred D2H time:%f ms\n", xyz_bytes >> 20, milliseconds);
        printf("Bandwidth D2H:%f MB/s\n", (float)(xyz_bytes >> 20)/(milliseconds/1000));
        gpuResult =  h_dB;
    }
    else{
        checkCuda(cudaEventRecord(start));
        checkCuda(cudaMemcpy(h_dA, input, xyz_bytes, cudaMemcpyDeviceToHost));
        checkCuda(cudaEventRecord(stop));
        checkCuda(cudaEventSynchronize(stop));
        checkCuda(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("Data %dMB transferred D2H time:%f ms\n", xyz_bytes >> 20, milliseconds);
        printf("Bandwidth D2H:%f MB/s\n", (float)(xyz_bytes >> 20)/(milliseconds/1000));
        gpuResult = h_dA;
    }
    
    

    // Run the CPU version
    //float startTime = rtclock();
    float *tmp1;
    for(int t = 0; t < timesteps; t += 1) {
        jacobi7(nx, ny, nz, h_dA1, h_dB1, fac);
        tmp1 = h_dA1;
        h_dA1 = h_dB1;
        h_dB1 = tmp1;
    }
    float *cpuResult;
    if (timesteps%2 == 0)
        cpuResult = h_dB1;
    else
        cpuResult = h_dA1;
    /*float endTime = rtclock();
    double elapsedTimeC = endTime - startTime;

    printf("Elapsed Time:%lf\n", elapsedTimeC);
    flops = xyz * 7.0 * timesteps;
    gflops = flops / elapsedTimeC / 1e9;
    printf("(CPU) %lf GFlop/s\n", gflops);
    */

    // compare the results btw CPU and GPU version
    double errorNorm, refNorm, diff;
    errorNorm = 0.0;
    refNorm = 0.0;
    i = 0;
    for (; i < xyz; ++i){
        diff = cpuResult[i] - gpuResult[i];
        errorNorm += diff * diff;
        refNorm += cpuResult[i] * cpuResult[i];
        if (gpuResult[i] != gpuResult[i])
                   diff = 1;
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

    // Free buffers
    cudaFreeHost(h_dA);
    cudaFreeHost(h_dB);
    cudaFreeHost(h_dA1);
    cudaFreeHost(h_dB1);
    cudaFree(d_dA);
    cudaFree(d_dB);
    return 0;
}