//test_global_l1_supported.cu
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

int main(){
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop,0));
    if (prop.globalL1CacheSupported)
        printf("Global L1 Cache Supported\n");
    else
        printf("Global L1 Cache NOT Supported\n");

    if (prop.localL1CacheSupported)
        printf("Local L1 Cache Supported\n");
    else
        printf("Local L1 Cache NOT Supported\n");
    return 0;
    }