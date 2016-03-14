//jacobi7.cu
#include <cuda.h>
#include <stdio.h>
#include <jacobi7_cuda.h>

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

void initial_data(float *h_A, float *h_B, const int xyz){
    // randomly generaed test data
    srand(time(NULL));
    int i = 0;
    for(; i < xyz; i++) {
        h_A[i] = 1 + (float)rand() / (float)RAND_MAX;
        h_B[i] =  h_A[i];
    }
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
    
    int devId = 0;
    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, devId));
    printf("Device : %s\n", prop.name);
    checkCuda( cudaSetDevice(devId));
    
    // Allocate host buffers
    checkCuda(cudaMallocHost((void**)&h_A, xyz_bytes)); // host pinned
    checkCuda(cudaMallocHost((void**)&h_B, xyz_bytes));
    



    // grid data iniatialization   
    // randomly generaed test data
    initial_data(h_A, h_B, xyz);

    const float fac = 6.0/(h_A[0] * h_A[0]);
    float *tmp;

    dim3 grid(nx/tx, ny/ty);
    dim3 block(tx, ty);

    float ms, ms1; // elapsed time in milliseconds
    printf("Start computing...\n");   

    /* set the ratio of cache/shared memory
    cudaFuncCachePreferNone: Default function cache configuration, no preference
    cudaFuncCachePreferShared: Prefer larger shared memory and smaller L1 cache
    cudaFuncCachePreferL1: Prefer larger L1 cache and smaller shared memory
    
    checkCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    */

    

    const int sharedMemSize = (block.x + 2) * (block.y + 2) * sizeof(float); 

    // create events and streams
    cudaEvent_t startEvent, stopEvent, startEvent1, stopEvent1;
    
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    checkCuda( cudaEventCreate(&startEvent1));
    checkCuda( cudaEventCreate(&stopEvent1));

    // timing start include data transfer and memory allocation
    checkCuda( cudaEventRecord(startEvent,0) );
    
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
        jacobi3d_7p_shmem_adam<<<grid, block, sharedMemSize>>>(input, output, nx, ny, nz, fac);
        // swap input and output
        tmp = input;
        input =  output;
        output = tmp;
    }
    // timing end pure gpu computing
    checkCuda( cudaEventRecord(stopEvent1, 0));
    checkCuda( cudaEventSynchronize(stopEvent1));
    checkCuda( cudaEventElapsedTime(&ms1, startEvent1, stopEvent1));

    printf("Time of shared memory version (pure GPU) (ms): %f\n", ms1);
  
    double gflop = (xyz >> 30) * 7.0 * timesteps;
    double gflop_per_sec = gflop * 1e3 / ms1;
    printf("(GPU) %lf GFlop/s\n", gflop_per_sec);
    double mupdate_per_sec = ((xyz >> 20) * timesteps) * 1e3 / ms1;
    printf("(GPU) %lf M updates/s\n", mupdate_per_sec);



    if(timesteps%2==0)
        checkCuda( cudaMemcpy(h_A, output, xyz_bytes, cudaMemcpyDeviceToHost));
    else
        checkCuda( cudaMemcpy(h_A, input, xyz_bytes, cudaMemcpyDeviceToHost));
    checkCuda( cudaEventRecord(stopEvent, 0));
    checkCuda( cudaEventSynchronize(stopEvent));
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent));
    
    printf("Time of shared memory version (ms): %f\n", ms);
    printf("(including data transfer and memory allocation in GPU.)\n");
  
    gflop = (xyz >> 30) * 7.0 * timesteps;
    gflop_per_sec = gflop * 1e3 / ms;
    printf("(GPU) %lf GFlop/s\n", gflop_per_sec);
    mupdate_per_sec = ((xyz >> 20) * timesteps) * 1e3 / ms;
    printf("(GPU) %lf M updates/s\n", mupdate_per_sec);


    // cleanup
    checkCuda( cudaEventDestroy(startEvent));
    checkCuda( cudaEventDestroy(stopEvent));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    return 0;

}