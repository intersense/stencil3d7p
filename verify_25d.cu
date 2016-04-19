//jacobi7.cu
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <jacobi7_cuda_shared.h>
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

int main(int argc, char* *argv){
    if(argc != 7) {
        printf("USAGE: %s <Store_Cached><NX> <NY> <NZ> <TX> <TY> <TIME STEPS>\n", argv[0]);
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

    float *h_A, *h_A1;
    float *h_B, *h_B1;
    float *d_A;
    float *d_B;
    
    int devId = 0;
    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, devId));
    //printf("Device : %s\n", prop.name);
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
    int i = 0;
    for(; i < xyz; i++) {
        h_A[i] = 1 + (float)rand() / (float)RAND_MAX;
        h_A1[i] = h_B1[i] = h_B[i] =  h_A[i];
    }
    
    // A simple comparison of the result
    /*int testIndex = 3 + 3*nx+ 3*nx*ny;
    printf("Iniatialized data[%d]=%f\n", testIndex , h_A[testIndex]);
    printf("h_A[%d]=%f\n", testIndex, h_A[testIndex]);
    printf("h_B[%d]=%f\n", testIndex, h_B[testIndex]);
    printf("h_A1[%d]=%f\n", testIndex, h_A1[testIndex]);
    printf("h_B1[%d]=%f\n", testIndex, h_B1[testIndex]);
    */
    const float fac = 6.0/(h_A[0] * h_A[0]);
    float *tmp;

    // modify nx/tx and ny/ty to (nx+tx-1)/tx and (ny+ty-1)/ty
    // inorder to avoid wrong configuration
    dim3 grid((nx+tx-1)/tx, (ny+ty-1)/ty);
    dim3 block(tx, ty);

    printf("grid:(%d, %d)\n", grid.x, grid.y);
    printf("block:(%d, %d)\n", tx, ty);
    float ms; // elapsed time in milliseconds
    //printf("Start computing...\n");   

    /* set the ratio of cache/shared memory
    cudaFuncCachePreferNone: Default function cache configuration, no preference
    cudaFuncCachePreferShared: Prefer larger shared memory and smaller L1 cache
    cudaFuncCachePreferL1: Prefer larger L1 cache and smaller shared memory
     
    checkCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
   */
    // set the shared memory bank size to eight bytes
    //checkCuda(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    
    // k-1, k, and k+1 planes are stored in shmem
    const int sharedMemSize = 3 * (block.x + 2) * (block.y + 2) * sizeof(float); 
    printf("Shared Memory Size: %dKB\n", sharedMemSize>>10);
    // create events and streams
    cudaEvent_t startEvent, stopEvent, startEvent1, stopEvent1;
    
    checkCuda( cudaEventCreate(&startEvent));
    checkCuda( cudaEventCreate(&stopEvent));
    checkCuda( cudaEventCreate(&startEvent1));
    checkCuda( cudaEventCreate(&stopEvent1));

    // timing start include data transfer and memory allocation
    checkCuda( cudaEventRecord(startEvent,0));
    
    // Allocate device buffers
    checkCuda(cudaMalloc((void**)&d_A, xyz_bytes)); // device
    checkCuda(cudaMalloc((void**)&d_B, xyz_bytes));

    float* input = d_A;
    float* output = d_B;
    
    // copy data to device
    checkCuda( cudaMemcpy(d_A, h_A, xyz_bytes, cudaMemcpyHostToDevice));
    checkCuda( cudaMemcpy(d_B, d_A, xyz_bytes, cudaMemcpyDeviceToDevice));
    
    // timing start pure gpu computing
    checkCuda( cudaEventRecord(startEvent1, 0));

    // Run the GPU kernel
    for(int t = 0; t < timesteps; t += 1) {
        jacobi3d_7p_25d<<<grid, block, sharedMemSize>>>(input, output, nx, ny, nz, fac);
        tmp = input;
        input =  output;
        output = tmp;
    }
    // timing end pure gpu computing
    checkCuda( cudaEventRecord(stopEvent1, 0));
    checkCuda( cudaEventSynchronize(stopEvent1));
    checkCuda( cudaEventElapsedTime(&ms1, startEvent1, stopEvent1));

    printf("Time of shared memory version (pure GPU) (ms): %f\n", ms1);
  
    double gflop = (xyz * 1e-9) * 7.0 * timesteps;
    double gflop_per_sec = gflop * 1e3 / ms1;
    printf("(GPU) %lf GFlop/s\n", gflop_per_sec);
    double mupdate_per_sec = ((xyz >> 20) * timesteps) * 1e3 / ms1;
    printf("(GPU) %lf M updates/s\n", mupdate_per_sec);

    checkCuda( cudaMemcpy(h_A, input, xyz_bytes, cudaMemcpyDeviceToHost));

    checkCuda( cudaEventRecord(stopEvent, 0));
    checkCuda( cudaEventSynchronize(stopEvent));
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent));
    float *gpuResult = h_A;

    printf("Time of shared memory version (ms): %f\n", ms);
    printf("(including data transfer and memory allocation in GPU.)\n");
  
    gflop = (xyz * 1e-9) * 7.0 * timesteps;
    gflop_per_sec = gflop * 1e3 / ms;
    printf("(GPU) %lf GFlop/s\n", gflop_per_sec);
    mupdate_per_sec = ((xyz >> 20) * timesteps) * 1e3 / ms;
    printf("(GPU) %lf M updates/s\n", mupdate_per_sec);

    // Run the CPU version
    //float startTime = rtclock();
    float *tmp1;
    for(int t = 0; t < timesteps; t += 1) {
        jacobi7(nx, ny, nz, h_A1, h_B1, fac);
        tmp1 = h_A1;
        h_A1 = h_B1;
        h_B1 = tmp1;
    }
    float *cpuResult;
    cpuResult = h_A1;
    
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
    /*printf("GPU[%d]=%f\n", testIndex, gpuResult[testIndex]);
    printf("CPU[%d]=%f\n", testIndex, cpuResult[testIndex]);
    printf("h_A[%d]=%f\n", testIndex, h_A[testIndex]);
    printf("h_B[%d]=%f\n", testIndex, h_B[testIndex]);
    printf("h_A1[%d]=%f\n", testIndex, h_A1[testIndex]);
    printf("h_B1[%d]=%f\n", testIndex, h_B1[testIndex]);
    
    testIndex = 2 + 2*nx+ 2*nx*ny;
    printf("GPU[%d]=%f\n", testIndex, gpuResult[testIndex]);
    printf("CPU[%d]=%f\n", testIndex, cpuResult[testIndex]);
    printf("h_A[%d]=%f\n", testIndex, h_A[testIndex]);
    printf("h_B[%d]=%f\n", testIndex, h_B[testIndex]);
    printf("h_A1[%d]=%f\n", testIndex, h_A1[testIndex]);
    printf("h_B1[%d]=%f\n", testIndex, h_B1[testIndex]);
    */
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