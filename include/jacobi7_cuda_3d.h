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
  const int s_x = threadIdx.x;
  const int s_y = threadIdx.y;
  const int s_z = threadIdx.z;

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
 
  // 1
  // Load halo region into shared memory
  if(s_x == 0 && ix > 0)         s_data[CURRENT_S - 1]  = d_in[CURRENT_G - 1];
  if(s_x == bx+1 && ix < nx-1)   s_data[CURRENT_S + 1]  = d_in[CURRENT_G + 1];
  if(s_y == 0 && iy > 0)         s_data[CURRENT_S - share_x] = d_in[CURRENT_G - nx];
  if(s_y == by+1 && iy < ny-1)   s_data[CURRENT_S + share_x] = d_in[CURRENT_G + nx];
  if(s_z == 0 && iz >0)          s_data[CURRENT_S - share_x * share_y] = d_in[CURRENT_G - nx * ny];
  if(s_z == bz+1 && iz < nz-1)   s_data[CURRENT_S + share_x * share_y] = d_in[CURRENT_G + nx * ny];
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
    s_data[CURRENT_S] = right + left + up + down + front + back - curr * fac;
  }
  __syncthreads();

  // 2
  if (s_x >= 1 && s_x <= bx && s_y >= 1 && s_y <= by && s_z >= 1 && s_z <= bz && ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1 && iz > 0 && iz < nz -1)
  {
    curr  = s_data[CURRENT_S];
    right = s_data[CURRENT_S + 1];
    left  = s_data[CURRENT_S - 1];
    up    = s_data[CURRENT_S - share_x];
    down  = s_data[CURRENT_S + share_x];
    front = s_data[CURRENT_S - share_x * share_y];
    back  = s_data[CURRENT_S + share_x * share_y]; 
    temp = right + left + up + down + front + back - curr * fac;
    d_out[CURRENT_G] = temp;
  }


  /*if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1 && iz > 0 && iz < nz -1)
  {
    temp = right + left + up + down + front + back - curr * fac;
    d_out[CURRENT_G] = temp;
  }*/
}