#include <cuda.h>
#include <stdio.h>
#include <math.h>

__global__ glmem2reg(float * in, const int num)
{
    int x = threadIdx.x;

}

__global__ glmem2shmem(float * in, const int num)
{

}

__global__ glmem2shmem_reg(float * in, const int num)
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
    if(argc != 3) {
        printf("USAGE: %s <NX> <TX>\n", argv[0]);
        return 1;
    }
    // program parameters trans
    // st_cached is for whether the computed results cached in shared memory
    // 0: no; 1: yes
    const int nx = atoi(argv[1]);
    const int tx = atoi(argv[2]);

    const int bytes = nx * sizeof(float); 

    float *h_A;

    int devId = 0;
    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, devId));
    //printf("Device : %s\n", prop.name);
    checkCuda( cudaSetDevice(devId));
    
    // Allocate host buffers
    checkCuda(cudaMallocHost((void**)&h_A, xyz_bytes)); // host pinned
    checkCuda(cudaMallocHost((void**)&h_B, xyz_bytes));

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
    printf("block:(%d, %d)\n", tx, ty);

}
