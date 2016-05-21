#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include <math.h>

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

// copy from src to dst
__global__ void stride_copy0(float *src, float *dst, int len, int stride){
    int i = ((blockDim.x * blockIdx.x + threadIdx.x) * stride) % len;
    dst[i] = src[i];
}
// increase the values of src
__global__ void stride_copy1(float *src, float *dst, int len, int stride){
    int i = ((blockDim.x * blockIdx.x + threadIdx.x) * stride) % len;
    src[i] = src[i]+1;
}
int main(int argc, char* *argv)
{
    if(argc != 5) {
        printf("USAGE: %s <len in MB> <stride> <block_x> <0 or 1>\n", argv[0]);
        return 1;
    }
    // program parameters trans

    int len = atoi(argv[1]);
    len = len << 20;
    const int stride = atoi(argv[2]);
    // blockx: blockDim.x
    const int blockx = atoi(argv[3]);
    const int data_bytes = len * sizeof(float);
    const int choice = atoi(argv[4]);

    void (*kernel)(float *, float *, int, int);
    
    float *src_d, *src;
    float *dst_d;

    // Allocate host buffers
    checkCuda(cudaMallocHost((void**)&src, data_bytes)); // host pinned


    // for comparison btw CPU and GPU version
    if(choice == 0){
        checkCuda(cudaMalloc((void**)&dst_d, data_bytes));
        checkCuda(cudaMalloc((void**)&src_d, data_bytes));
        kernel = stride_copy0;
    }
    if(choice == 1){
        checkCuda(cudaMalloc((void**)&src_d, data_bytes));
        kernel = stride_copy1;
    }

    // randomly generaed test data
    srand(time(NULL));
    int i = 0;
    for(; i < len; i++) {
        src[i] = 1 + (float)rand() / (float)RAND_MAX;
    }

    // Always use device 0
    checkCuda(cudaSetDevice(0));
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    float milliseconds = 0;

    printf("Data size:%dM floats\n", len>>20);   
    printf("Start testing coalescing... \n");   

    // Copy to device
    checkCuda(cudaMemcpy(src_d, src, data_bytes, cudaMemcpyHostToDevice));

    checkCuda(cudaEventRecord(start));
    dim3 grid(len/blockx);
    dim3 block(blockx);
    kernel<<<grid,block>>>(src_d, dst_d, len, stride);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time:%fms\n", milliseconds);
    float bandwidth = (data_bytes>>20)/milliseconds/(10^3);
    printf("Bandwidth:%f MB/s\n", bandwidth);
    checkCuda( cudaEventDestroy(start));
    checkCuda( cudaEventDestroy(stop));

    cudaFreeHost(src);
    cudaFree(src_d);
    cudaFree(dst_d);
    return 0;
}
