
__device__ __forceinline__ void perform_stencil(float *src_ptr, const int nx, const int ny, const int tx, const int ty, float *dst_ptr, float fac)
{
    // src_ptr: a XY plane of the stencil grid, including halo and the center region
    // it is 3*(tx+2)*(ty+2) --> the shared memory size
    // the thread block is tx*ty
    // src_ptr[0:(tx+2)*(ty+2))                 -> bottom layer
    // src_ptr[(tx+2)*(ty+2):2*(tx+2)*(ty+2))   -> curent layer
    // src_ptr[2*(tx+2)*(ty+2):3*(tx+2)*(ty+2)) -> up layer

    // global coordinates
    const int gx = threadIdx.x + blockDim.x * blockIdx.x;
    const int gy = threadIdx.y + blockDim.y * blockIdy.y;
    const int g = gx + gy * nx;

    // local coordinates
    const int lx = threadIdx.x + 1;
    const int ly = threadIdx.y + 1;
    const int l  = lx + ly*(tx+2) + (tx+2)*(ty+2); // current thread deal with the data is in the middle layer in shared memory array

    float c, w, e, n, s, b, u;
    // boundary condition
    bool isboundary = (gx == 0) || (gx == nx-1) || (gy == 0) || (gy == ny-1);
    if (!isboundary){
        c = src_ptr[l];
        w = src_ptr[l-1];
        e = src_ptr[l+1];
        n = src_ptr[l+(tx+2)];
        s = src_ptr[l-(tx+2)];
        u = src_ptr[l+(tx+2)*(ty+2)];
        b = src_ptr[l-(tx+2)*(ty+2)];
        dst_ptr[l] = w + e + n + s + u + b - c*fac;
    }
}