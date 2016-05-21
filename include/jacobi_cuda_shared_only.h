__global__ void jacobi3d_7p_shmem_only(float * d_in, float * d_out, const int nx, const int ny, const int nz, const float fac)
{
  const int bx = blockDim.x;
  const int by = blockDim.y;
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  const int tx = threadIdx.x + 1;
  const int ty = threadIdx.y + 1;

  // the size XY-dimension of shared memory
  const int x_s  = bx + 2;
  const int y_s  = by + 2;
  const int xy_s = x_s * y_s;
 
  int CURRENT_G = ix + iy*nx + nx*ny;
  int CURRENT_S = tx + ty*x_s;
  extern __shared__ float s_data[];

  float curr;
  float east, west;
  float north, south;
  float top, down;

  float temp;

  // Load current, top, and down nodes into shared memory
  //down
  s_data[CURRENT_S - xy_s] = d_in[CURRENT_G - nx*ny];
  //curr
  s_data[CURRENT_S]       = d_in[CURRENT_G];
  //top
  s_data[CURRENT_S + xy_s] = d_in[CURRENT_G + nx*ny]; 

  // Load halo region into shared memory (curr)
  if(tx == 1 && ix > 0)       s_data[CURRENT_S - 1]   = d_in[CURRENT_G - 1];
  if(tx == bx && ix < nx-1)   s_data[CURRENT_S + 1]   = d_in[CURRENT_G + 1];
  if(ty == 1 && iy > 0)       s_data[CURRENT_S - x_s] = d_in[CURRENT_G - nx];
  if(ty == by && iy < ny-1)   s_data[CURRENT_S + x_s] = d_in[CURRENT_G + nx];
  /*
  // Load halo region into shared memory (top)
  if(tx == 1 && ix > 0)       s_data[CURRENT_S  + xy_s - 1]   = d_in[CURRENT_G + nx*ny - 1];
  if(tx == bx && ix < nx-1)   s_data[CURRENT_S  + xy_s + 1]   = d_in[CURRENT_G + nx*ny + 1];
  if(ty == 1 && iy > 0)       s_data[CURRENT_S  + xy_s - x_s] = d_in[CURRENT_G + nx*ny - nx];
  if(ty == by && iy < ny-1)   s_data[CURRENT_S  + xy_s + x_s] = d_in[CURRENT_G + nx*ny + nx];
  // Load halo region into shared memory (down)
  if(tx == 1 && ix > 0)       s_data[CURRENT_S  - xy_s - 1]   = d_in[CURRENT_G - nx*ny - 1];
  if(tx == bx && ix < nx-1)   s_data[CURRENT_S  - xy_s + 1]   = d_in[CURRENT_G - nx*ny + 1];
  if(ty == 1 && iy > 0)       s_data[CURRENT_S  - xy_s - x_s] = d_in[CURRENT_G - nx*ny - nx];
  if(ty == by && iy < ny-1)   s_data[CURRENT_S  - xy_s + x_s] = d_in[CURRENT_G - nx*ny + nx];
  */
  __syncthreads();

  if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1)
  {
    d_out[CURRENT_G] = s_data[CURRENT_S+1] + s_data[CURRENT_S-1] + s_data[CURRENT_S-x_s] + s_data[CURRENT_S+x_s] + s_data[CURRENT_S-xy_s] + s_data[CURRENT_S+xy_s] + fac * s_data[CURRENT_S];
  }
  #pragma unroll 
  for(int k=1; k<nz-2; k++)
  {
    CURRENT_G += nx*ny;
    __syncthreads();
    // Re-use data already in shared mem to move forward
    /*down  = curr;
    curr  = s_data[CURRENT_S] = top;
    top = d_in[CURRENT_G + nx*ny];
    */
    //down
    s_data[CURRENT_S-xy_s] = s_data[CURRENT_S];
    //curr
    s_data[CURRENT_S] = s_data[CURRENT_S+xy_s];
    
    //top
    s_data[CURRENT_S+xy_s] = d_in[CURRENT_G + nx*ny];
    // Load halo region into shared memory
    if(tx == 1 && ix > 0)     s_data[CURRENT_S - 1]  = d_in[CURRENT_G - 1];
    if(tx == bx && ix < nx-1) s_data[CURRENT_S + 1]  = d_in[CURRENT_G + 1];
    if(ty == 1 && iy > 0)     s_data[CURRENT_S - x_s] = d_in[CURRENT_G - nx];
    if(ty == by && iy < ny-1) s_data[CURRENT_S + x_s] = d_in[CURRENT_G + nx];
    __syncthreads();
    
    // Perform computation and write to output grid (excluding edge nodes)
    if(ix > 0 && ix < nx-1 && iy > 0 && iy < ny-1)
    {
      d_out[CURRENT_G] = s_data[CURRENT_S+1] + s_data[CURRENT_S-1] + s_data[CURRENT_S-x_s] + s_data[CURRENT_S+x_s] + s_data[CURRENT_S-xy_s] + s_data[CURRENT_S+xy_s] + fac * s_data[CURRENT_S];
    }
  }
}
