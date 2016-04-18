// the share memory is 3d grid and 3d block, the same with 3d stencil
__global__ void jacobi3d_7p_shmem_3d(float *d_in, float *d_out, const int nx, const int ny, const int nz, float fac){
  // block dimension
  const int bx = blockDim.x;
  const int by = blockDim.y;
  const int bz = blockDim.z;

  //shared memory dimension
  const int share_x = bx + 2;
  const int share_y = by + 2;
  const int share_z = bz + 2;

  // thread global dimension
  const int ix = threadIdx.x + blockIdx.x * bx;
  const int iy = threadIdx.y + blockIdx.y * by;
  const int iz = threadIdx.z + blockIdx.z * bz;

  // grid points to be computed, dimensions
  const int s_x = threadIdx.x + 1;
  const int s_y = threadIdx.y + 1;
  const int s_z = threadIdx.z + 1;

  int CURRENT_G = ix + nx * (iy + ny * iz);
  int CURRENT_S = s_x + share_x * (s_y + share_y * s_z);


  extern __shared__ float s_data[];

  float curr;
  float right, left;
  float up, down;
  float front, back;

  float temp;

  // Load current, front, and back nodes into shared and register memory
  curr  = s_data[CURRENT_S] = d_in[CURRENT_G];
 

  // Load halo region into shared memory
  if(s_x == 1 && ix > 0)              s_data[CURRENT_S - 1]  = d_in[CURRENT_G - 1];
  if(s_x == share_x-2 && ix < nx-1)   s_data[CURRENT_S + 1]  = d_in[CURRENT_G + 1];
  if(s_y == 1 && iy > 0)              s_data[CURRENT_S - share_x] = d_in[CURRENT_G - nx];
  if(s_y == share_y-2 && iy < ny-1)   s_data[CURRENT_S + share_x] = d_in[CURRENT_G + nx];
  if(s_z == 1 && iz >0)               s_data[CURRENT_S - share_x * share_y] = d_in[CURRENT_G - nx * ny];
  if(s_z == share_z-2 && iz < nz-1)   s_data[CURRENT_S + share_x * share_y] = d_in[CURRENT_G + nx * ny];
  __syncthreads();

  // Load shared memory into registers
  curr  = s_data[CURRENT_S];
  right = s_data[CURRENT_S + 1];
  left  = s_data[CURRENT_S - 1];
  up    = s_data[CURRENT_S - share_x];
  down  = s_data[CURRENT_S + share_x];
  front = s_data[CURRENT_S - share_x * share_y];
  back  = s_data[CURRENT_S + share_x * share_y];

  if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1 && iz > 0 && iz < nz -1)
  {
    temp = right + left + up + down + front + back - curr * fac;
    d_out[CURRENT_G] = temp;
  }
}

// 3D + temporal blocking
// 1: not only loading the data to shared memory, but computing them
// 2: compute like non-temporal version
__global__ void jacobi3d_7p_shmem_3d_temporal(float *d_in, float *d_out, const int nx, const int ny, const int nz, float fac){
  // block dimension
  const int bx = blockDim.x;
  const int by = blockDim.y;
  const int bz = blockDim.z;

  //shared memory dimension
  const int share_x = bx + 2;
  const int share_y = by + 2;
  const int share_z = bz + 2;

  // thread global dimension
  const int ix = threadIdx.x + blockIdx.x * bx;
  const int iy = threadIdx.y + blockIdx.y * by;
  const int iz = threadIdx.z + blockIdx.z * bz;

  // grid points to be computed, dimensions
  const int s_x = threadIdx.x + 1;
  const int s_y = threadIdx.y + 1;
  const int s_z = threadIdx.z + 1;

  int C_G = ix + nx * (iy + ny * iz);
  int C_S = s_x + share_x * (s_y + share_y * s_z);


  extern __shared__ float s_data[];

  float curr;
  float right, left;
  float up, down;
  float front, back;

  float temp;

  bool boundary1 = ix > 1 && ix < nx-2 && iy > 1 && iy < ny - 2 && iz > 1 && iz < nz - 2;
  bool boundary2 = ix > 0 && ix < nx-1 && iy > 0 && iy < ny - 1 && iz > 0 && iz < nz - 1;
  // Load c, front, and back nodes into shared and register memory
  if (boundary1){
    curr  = s_data[C_S] = d_in[C_G];
  }  
  
  // Load halo region into shared memory
  if(s_x == 1 && ix > 0) {
    // condition: C_G - 1 - 1, C_G - 1 + 1, C_G - 1 - nx, C_G - 1 + nx, C_G - 1 - nx * ny, C_G - 1 + nx * ny and C_G - 1
    if (boundary1){
      s_data[C_S - 1] = d_in[C_G - 1 - 1] + d_in[C_G - 1 + 1] + d_in[C_G - 1 - nx] + d_in[C_G - 1 + nx] + d_in[C_G - 1 - nx * ny] + d_in[C_G - 1 + nx * ny] - fac * d_in[C_G - 1];
    }
  }
  if(s_x == share_x-2 && ix < nx-1){
    if(boundary1){
      s_data[C_S + 1] = d_in[C_G + 1 - 1] + d_in[C_G + 1 + 1] + d_in[C_G + 1 - nx] + d_in[C_G + 1 + nx] + d_in[C_G + 1 - nx * ny] + d_in[C_G + 1 + nx * ny] - fac * d_in[C_G + 1];
    }
  }
  if(s_y == 1 && iy > 0){
    if (boundary1){
      s_data[C_S - share_x] = d_in[C_G - nx - 1] + d_in[C_G - nx + 1] + d_in[C_G - nx - nx] + d_in[C_G - nx + nx] + d_in[C_G - nx - nx * ny] + d_in[C_G - nx + nx * ny] - fac * d_in[C_G - nx];
    } 
  }
  if(s_y == share_y-2 && iy < ny-1){
    if (boundary1){
      s_data[C_S + share_x] = d_in[C_G + nx - 1] + d_in[C_G + nx + 1] + d_in[C_G + nx - nx] + d_in[C_G + nx + nx] + d_in[C_G + nx - nx * ny] + d_in[C_G + nx + nx * ny] - fac * d_in[C_G + nx];
    }
  }
  if(s_z == 1 && iz >0){
    if (boundary1){
      s_data[C_S - share_x * share_y] = d_in[C_G - nx * ny - 1] + d_in[C_G - nx * ny + 1] + d_in[C_G - nx * ny - nx] + d_in[C_G - nx * ny + nx] + d_in[C_G - nx * ny - nx * ny] + d_in[C_G - nx * ny + nx * ny] - fac * d_in[C_G - nx * ny];
    }
  }
  if(s_z == share_z-2 && iz < nz-1){
    if (boundary1){
      s_data[C_S + share_x * share_y] = d_in[C_G + nx * ny - 1] + d_in[C_G + nx * ny + 1] + d_in[C_G + nx * ny - nx] + d_in[C_G + nx * ny + nx] + d_in[C_G + nx * ny - nx * ny] + d_in[C_G + nx * ny + nx * ny] - fac * d_in[C_G + nx * ny];
    }
  }
  __syncthreads();

  if(boundary2)
  {
      // Load shared memory into registers
    curr  = s_data[C_S];
    right = s_data[C_S + 1];
    left  = s_data[C_S - 1];
    up    = s_data[C_S - share_x];
    down  = s_data[C_S + share_x];
    front = s_data[C_S - share_x * share_y];
    back  = s_data[C_S + share_x * share_y];
    temp = right + left + up + down + front + back - curr * fac;
    d_out[C_G] = temp;
  }
}