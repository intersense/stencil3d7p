
void perform_stencil(float *src_ptr, const unsigned tx, const unsigned ty, float *dst_ptr)
{
    // src_ptr: a XY plane of the stencil grid, including halo and the center region
    // it is (tx+2)*(ty+2) --> the shared memory size
    // the thread block is tx*ty
    
}