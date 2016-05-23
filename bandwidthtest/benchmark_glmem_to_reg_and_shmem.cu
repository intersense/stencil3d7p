#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define NUM 200

__global__ void glmem2reg(float * in, const int num)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int x_s = threadIdx.x;
    float a[NUM];
    if(x < num){
        a[x_s] = in[x];  
        a[x_s] = a[x_s] + 1.0;
        in[x] = a[x_s];       
    }
 
}

__global__ void glmem2shmem(float * in, const int num)
{
    extern __shared__ float a[];
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int x_s = threadIdx.x;
    if(x < num){
        a[x_s] = in[x];  
        a[x_s] = a[x_s] + 1.0;
        in[x] = a[x_s];       
    }

}

__global__ void glmem2shmem_reg(float * in, const int num)
{

}

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
    if(argc != 4) {
        printf("USAGE: %s <NX>K <TX> <reg_0_shared_1>\n", argv[0]);
        return 1;
    }
    // program parameters trans
    // st_cached is for whether the computed results cached in shared memory
    // 0: no; 1: yes
    const int nx = atoi(argv[1])<<10;
    const int tx = atoi(argv[2]);
    const int reg_or_shared = atoi(argv[3]);

    /*void (*kernel)(float *, int);
    if (reg_or_shared == 0)
        kernel = glmem2reg;
    else kernel = glmem2shmem;
    */
    const int bytes = nx * sizeof(float); 

    float *h_A, *d_A;

    int devId = 0;
    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, devId));
    //printf("Device : %s\n", prop.name);
    checkCuda( cudaSetDevice(devId));
    
    // Allocate host buffers
    checkCuda(cudaMallocHost((void**)&h_A, bytes)); // host pinned

    // grid data iniatialization   
    // randomly generaed test data
    srand(time(NULL));
    int i = 0;
    for(; i < nx; i++) {
        h_A[i] = 1 + (float)rand() / (float)RAND_MAX;
    }
    dim3 grid((nx+tx-1)/tx,1,1);
    dim3 block(tx, 1);

    printf("grid:(%d, %d)\n", grid.x, grid.y);
    printf("block:(%d, %d)\n", block.x, block.y);
    float ms, ms1; // elapsed time in milliseconds
    // create events and streams
    cudaEvent_t startEvent, stopEvent, startEvent1, stopEvent1;
    
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));
    checkCuda(cudaEventCreate(&startEvent1));
    checkCuda(cudaEventCreate(&stopEvent1));

    // timing start include data transfer and memory allocation
    checkCuda(cudaEventRecord(startEvent,0));
    
    // Allocate device buffers
    checkCuda(cudaMalloc((void**)&d_A, bytes)); // device
    // copy data to device
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    if (reg_or_shared == 0)//reg
        glmem2reg<<<grid, block>>>(h_A, nx);
    if (reg_or_shared == 1)//shmem
        glmem2shmem<<<grid, block, NUM*sizeof(float)>>>(h_A, nx);

    

    // timing end pure gpu computing
    checkCuda(cudaEventRecord(stopEvent1, 0));
    checkCuda(cudaEventSynchronize(stopEvent1));
    checkCuda(cudaEventElapsedTime(&ms1, startEvent, stopEvent1));
    printf("Time (pure GPU) (ms): %f\n", ms1);

    checkCuda( cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost));

    checkCuda( cudaEventRecord(stopEvent, 0));
    checkCuda( cudaEventSynchronize(stopEvent));
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time (ms): %f\n", ms);
    printf("(including data transfer and memory allocation in GPU.)\n");

    // cleanup
    checkCuda( cudaEventDestroy(startEvent));
    checkCuda( cudaEventDestroy(stopEvent));
    checkCuda( cudaEventDestroy(startEvent1));
    checkCuda( cudaEventDestroy(stopEvent1));
    cudaFreeHost(h_A);
    cudaFree(d_A);

    return 0;
}
