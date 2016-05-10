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

__global__ void stride_copy(float *src, float *dst, int len, int stride){
    int i = ((blockDim.x * blockIdx.x + threadIdx.x) * stride) % len;
    dst[i] = src[i];
}

int main(int argc, char* *argv)
{
    if(argc != 4) {
        printf("USAGE: %s <len in MB> <stride> <block_x>\n", argv[0]);
        return 1;
    }
    // program parameters trans

    int len = atoi(argv[1]);
    len = len << 20;
    const int stride = atoi(argv[2]);
    // blockx: blockDim.x
    const int blockx = atoi(argv[3]);
    const int data_bytes = len * sizeof(float);

    float *src_d, *src;
    float *dst_d;

    // Allocate host buffers
    checkCuda(cudaMallocHost((void**)&src, data_bytes)); // host pinned

    // for comparison btw CPU and GPU version
    checkCuda(cudaMalloc((void**)&src_d, data_bytes));
    checkCuda(cudaMalloc((void**)&dst_d, data_bytes));

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

    printf("Data size:%dMB\n", len);   
    printf("Start testing coalescing... \n");   

    // Copy to device
    checkCuda(cudaMemcpy(src_d, src, data_bytes, cudaMemcpyHostToDevice));

    checkCuda(cudaEventRecord(start));
    dim3 grid(len/blockx);
    dim3 block(blockx);
    stride_copy<<<grid,block>>>(src_d, dst_d,len,stride);
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
