#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define ITERATIONS 8192
#define NX 512
#define NY 512
#define SIZE (NX*NY)

typedef int TYPE;

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void readBenchmark_no_PC(TYPE *d_arr)
{
    // assign unique partititions to blocks,
    // where number of partitions is 8
    int curPartition = blockIdx.x % 8;
    // size of eache partition: 256 bytes
    int elemsInPartition = 256 / sizeof(TYPE);
    // jumpto unique partition
    int startIndex = elemsInPartition * curPartition;

    TYPE readVal = 0;
    /*ITERATIONS = 8192, but it can be any fixed and known number. Loop counter 'x' ensures coalescing */
    for (int x=0; x<ITERATIONS; x +=16)
    {
        /* offset guarantees to restrict the index to the same partition */
        int offset = ((threadIdx.x + x) % elemsInPartition);
        int index = startIndex + offset;
        // Read from global memory location
        readVal = d_arr[index];
    }
    /* Write once to memory to prevent the above code from being optimized out */
    d_arr[0] = readVal;
}
__global__ void readBenchmark_PC(TYPE *d_arr)
{
    // size of each partition: 256 bytes
    int elemsInPartition = 256 / sizeof(TYPE);
    TYPE readVal = 0;
    for (int x = 0; x < ITERATIONS; x+=16)
    {
        int index = ((threadIdx.x + x) % elemsInPartition);
        readVal = d_arr[index];
    }
    d_arr[0] = readVal;
}

int main(int argc, char* *argv)
{
        if(argc != 5) {
        printf("USAGE: %s <kernel_0_noPC_1_PC> <bx> <by> <times>\n", argv[0]);
        return 1;
    }
    const int PC = atoi(argv[1]);
    const int bx = atoi(argv[2]);
    const int by = atoi(argv[3]);
    const int times = atoi(argv[4]);

    const int data_size = SIZE * sizeof(TYPE);

    TYPE *a;
    TYPE *d_a; 

    void (*kernel)(TYPE *);

    if (PC == 0) kernel = &readBenchmark_no_PC;
    if (PC == 1) kernel = &readBenchmark_PC;

    // Allocate host buffers
    checkCuda(cudaMallocHost((void**)&a, data_size)); // host pinned

    // Use device 0
    checkCuda(cudaSetDevice(0));

    // Allocate device buffers
    checkCuda(cudaMalloc((void**)&d_a, data_size));
    
    int i = 0;
    for(; i < SIZE; i++) {
        a[i] = 1 + rand() / RAND_MAX;
    }


    float milliseconds = 0;
    
    // Copy to device
    checkCuda(cudaMemcpy(d_a, a, data_size, cudaMemcpyHostToDevice));

    dim3 grid((NX+bx-1)/bx, (NY+by-1)/by);
    dim3 block(bx, by);
    printf("grid:(%d, %d)\n", grid.x, grid.y);
    printf("block:(%d, %d)\n", bx, by);
    
    // Timing
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    
    // The kernel
    int t = 0;
    for(; t < Times; t++)
    {
        kernel<<<grid, block>>>(d_a);
    }
    

    checkCuda(cudaEventCreate(&stop));
    checkCuda(cudaEventSynchronize(stop));
    
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("GPU kernel Elapsed Time (pure GPU):%f ms\n", milliseconds);
    
    cudaFreeHost(a);
    cudaFree(d_a);
    return 0;
}