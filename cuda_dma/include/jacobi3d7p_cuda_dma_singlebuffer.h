#include "stencil_math.h"
#include "cudaDMAv2.h"
#define PARAM_ALIGNMENT 8 
#define PARAM_DMA_WARPS 1
#define PARAM_BYTES_PER_THREAD (12*16) 
#define PARAM_RADIUS 1

__global__ 
void jacobi3d7p_dma_single_buffer(const float *src_buffer, float *dst_buffer, const int nx, const int ny, const int nz, const int tx, const int ty)
{
    int row_stride = tx+2;
    int slice_stride = row_stride * (ty+2);

    // Declare our shared memory buffer, base tile + halo region 
    __shared__ float buffer[3*(tx+2)*(ty+2)];
    CudaDMAStrided<true/*warp specialized*/, PARAM_ALIGNMENT, PARAM_BYTES_PER_THREAD,
                  (tx+2)*sizeof(float)/*elmt size*/, PARAM_RADIUS, (ty+2*PARAM_RADIUS)/*num elements*/>
    dma_ld(0/*dmaID*/, (tx*ty)/*num compute threads*/, (tx*ty)/*dma_threadIdx_start*/, 
            row_stride*sizeof(float)/*src stride*/, (tx+2*PARAM_RADIUS)*sizeof(float)/*dst stride*/);

    // Offset for this threadblock
    const unsigned block_offset = blockIdx.x*(tx+2) + blockIdx.y*(ty+2)*row_stride;
    if (dma_ld.owns_this_thread())
    {
      // DMA warps
      // Don't forget to backup to the upper left corner of the slice including the halo region
      const float *src_ptr = src_buffer + block_offset;
      for (int iz = 0; iz < nz; iz++)
      {
        dma_ld.execute_dma<true>(src_ptr, buffer);
        src_ptr += slice_stride;
      }
    }
    else
    {
        // Compute warps
        // Start the DMA threads going
        dma_ld.start_async_dma();
        const unsigned tx = threadIdx.x % tx;
        const unsigned ty = threadIdx.x / ty;
        float *dst_ptr = dst_buffer + block_offset + ty*row_stride + tx;
        // Note we unroll the loop one time to facilitate
        // the right number of synchronizations with the DMA threads
        for (int iz = 0; iz < nz-1; iz++)
        {
            // Wait for the DMA threads to finish loading
            dma_ld.wait_for_dma_finish();
            // Process the buffer;
            perform_stencil(buffer, tx, ty, dst_ptr);
            // Start the next transfer
            dma_ld.start_async_dma();
            // Update our out index
            dst_ptr += slice_stride;
        }
        // Handle the last loop iteration
        dma_ld.wait_for_dma_finish();
        perform_stencil(buffer, tx, ty, dst_ptr);
    }
}