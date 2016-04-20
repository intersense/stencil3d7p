__global__ void jacobi3d_7p_shmem_adam(double * d_in, double * d_out, const int nx, const int ny, const int nz, const double fac)
{
  const int bx = blockDim.x;
  const int by = blockDim.y;
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  const int tx = threadIdx.x + 1;
  const int ty = threadIdx.y + 1;

  // the size X-dimension of shared memory
  const int x_s = bx + 2;

  int CURRENT_G = ix + iy*nx + nx*ny;
  int CURRENT_S = tx + ty*(bx+2);
  extern __shared__ double s_data[];

  double curr;
  double right, left;
  double up, down;
  double front, back;

  double temp;

  // Load current, front, and back nodes into shared and register memory
  back  = d_in[CURRENT_G - nx*ny];
  curr  = s_data[CURRENT_S] = d_in[CURRENT_G];
  front = d_in[CURRENT_G + nx*ny]; 

  // Load halo region into shared memory
  if(tx == 1 && ix > 0)       s_data[CURRENT_S - 1]  = d_in[CURRENT_G - 1];
  if(tx == bx && ix < nx-1)   s_data[CURRENT_S + 1]  = d_in[CURRENT_G + 1];
  if(ty == 1 && iy > 0)       s_data[CURRENT_S - x_s] = d_in[CURRENT_G - nx];
  if(ty == by && iy < ny-1)   s_data[CURRENT_S + x_s] = d_in[CURRENT_G + nx];
  __syncthreads();

  // Load shared memory into registers
  curr  = s_data[CURRENT_S];
  right = s_data[CURRENT_S + 1];
  left  = s_data[CURRENT_S - 1];
  up    = s_data[CURRENT_S - x_s];
  down  = s_data[CURRENT_S + x_s];

  if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1)
  {
    temp = right + left + up + down + front + back - curr * fac;
    d_out[CURRENT_G] = temp;
  }
  //#pragma unroll 
  for(int k=1; k<nz-2; k++)
  {
    CURRENT_G += nx*ny;
    __syncthreads();
    // Re-use data already in shared mem and registers to move forward
    back  = curr;
    curr  = s_data[CURRENT_S] = front;
    front = d_in[CURRENT_G + nx*ny];

    // Load halo region into shared memory
    if(tx == 1 && ix > 0)     s_data[CURRENT_S - 1]  = d_in[CURRENT_G - 1];
    if(tx == bx && ix < nx-1) s_data[CURRENT_S + 1]  = d_in[CURRENT_G + 1];
    if(ty == 1 && iy > 0)     s_data[CURRENT_S - x_s] = d_in[CURRENT_G - nx];
    if(ty == by && iy < ny-1) s_data[CURRENT_S + x_s] = d_in[CURRENT_G + nx];
    __syncthreads();

    // Load shared memory into registers
    right = s_data[CURRENT_S + 1];
    left  = s_data[CURRENT_S - 1];
    up    = s_data[CURRENT_S - x_s];
    down  = s_data[CURRENT_S + x_s];
    
    // Perform computation and write to output grid (excluding edge nodes)
    if(ix > 0 && ix < nx-1 && iy > 0 && iy < ny-1)
    {
      temp = right + left + up + down + front + back - curr * fac;
      d_out[CURRENT_G] = temp;
    }
  }
}

// compute then store the result to shared memory, after all the data in shared memory updated, store to glmem
// worse than no store cached in shared memory
__global__ void jacobi3d_7p_shmem_adam_store_shmem(double * d_in, double * d_out, const int nx, const int ny, const int nz, const double fac)
{
  const int bx = blockDim.x;
  const int by = blockDim.y;
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  const int tx = threadIdx.x + 1;
  const int ty = threadIdx.y + 1;

  int CURRENT_G = ix + iy*nx + nx*ny;
  int CURRENT_S = tx + ty * (bx+2);

  extern __shared__ double s_data[];
  // the size X-dimension of shared memory
  const int x_s = bx + 2;
  double curr;
  double right, left;
  double up, down;
  double front, back;

  double temp;

  // Load current, front, and back nodes into shared and register memory
  back  = d_in[CURRENT_G - nx*ny];
  curr  = s_data[CURRENT_S] = d_in[CURRENT_G];
  front = d_in[CURRENT_G + nx*ny]; 

  // Load halo region into shared memory
  if(tx == 1 && ix > 0)       s_data[CURRENT_S - 1]  = d_in[CURRENT_G - 1];
  if(tx == bx && ix < nx-1) s_data[CURRENT_S + 1]  = d_in[CURRENT_G + 1];
  if(ty == 1 && iy > 0)       s_data[CURRENT_S - x_s] = d_in[CURRENT_G - nx];
  if(ty == by && iy < ny-1) s_data[CURRENT_S + x_s] = d_in[CURRENT_G + nx];
  __syncthreads();

  // Load shared memory into registers
  curr  = s_data[CURRENT_S];
  right = s_data[CURRENT_S + 1];
  left  = s_data[CURRENT_S - 1];
  up    = s_data[CURRENT_S - x_s];
  down  = s_data[CURRENT_S + x_s];
  
  // for store cache
  __syncthreads();

  if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1)
  {
    
    temp = right + left + up + down + front + back - curr * fac;
    s_data[CURRENT_S] = temp;
    //d_out[CURRENT_G] = s_data[CURRENT_S+1] + s_data[CURRENT_S-1] +s_data[CURRENT_S-bx] + s_data[CURRENT_S+bx] +front + back - s_data[CURRENT_S] * fac;
    __syncthreads();
    d_out[CURRENT_G] = s_data[CURRENT_S];
  }
  //#pragma unroll 
  for(int k=1; k<nz-2; k++)
  {
    CURRENT_G += nx*ny;

    __syncthreads();
    // Re-use data already in shared mem and registers to move forward
    back  = curr;
    curr  = s_data[CURRENT_S] = front;
    front = d_in[CURRENT_G + nx*ny];

    // Load halo region into shared memory
    if(tx == 1 && ix > 0)       s_data[CURRENT_S - 1]  = d_in[CURRENT_G - 1];
    if(tx == bx && ix < nx-1) s_data[CURRENT_S + 1]  = d_in[CURRENT_G + 1];
    if(ty == 1 && iy > 0)       s_data[CURRENT_S - x_s] = d_in[CURRENT_G - nx];
    if(ty == by && iy < ny-1) s_data[CURRENT_S + x_s] = d_in[CURRENT_G + nx];
    __syncthreads();

    // Load shared memory into registers
    right = s_data[CURRENT_S + 1];
    left  = s_data[CURRENT_S - 1];
    up    = s_data[CURRENT_S - x_s];
    down  = s_data[CURRENT_S + x_s];
    // for store cache
    __syncthreads();
    
    // Perform computation and write to output grid (excluding edge nodes)
    if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1)
    {
      temp = right + left + up + down + front + back - curr * fac;
      s_data[CURRENT_S] = temp;
      
      __syncthreads();

      d_out[CURRENT_G] = s_data[CURRENT_S];
    }
  }
}
// shared memory only store current, west and east in shared memory
__global__ void jacobi3d_7p_shmem_adam_cwe_shmem(double * d_in, double * d_out, const int nx, const int ny, const int nz, const double fac)
{
  const int bx = blockDim.x;
  //const int by = blockDim.y;
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  const int tx = threadIdx.x + 1;
  //const int ty = threadIdx.y;// + 1;

  int CURRENT_G = ix + iy*nx + nx*ny;
  int CURRENT_S = tx; //+ ty*bx;

  extern __shared__ double s_data[];

  double curr;
  double right, left;
  double up, down;
  double front, back;

  double temp;

  // Load current, front, and back nodes into shared and register memory
  back  = d_in[CURRENT_G - nx*ny];
  curr  = s_data[CURRENT_S] = d_in[CURRENT_G];
  front = d_in[CURRENT_G + nx*ny]; 

  // Load halo region into shared memory
  if(tx == 1 && ix > 0)       s_data[CURRENT_S - 1]  = d_in[CURRENT_G - 1];
  if(tx == bx-2 && ix < nx-1) s_data[CURRENT_S + 1]  = d_in[CURRENT_G + 1];
  //if(ty == 1 && iy > 0)       s_data[CURRENT_S - bx] = d_in[CURRENT_G - nx];
  //if(ty == by-2 && iy < ny-1) s_data[CURRENT_S + bx] = d_in[CURRENT_G + nx];
  __syncthreads();

  // Load shared memory into registers
  curr  = s_data[CURRENT_S];
  right = s_data[CURRENT_S + 1];
  left  = s_data[CURRENT_S - 1];
  
  if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1)
  {
    up    = d_in[CURRENT_G - nx];
    down  = d_in[CURRENT_G + nx];
    temp = right + left + up + down + front + back - curr * fac;
    d_out[CURRENT_G] = temp;
  }

  for(int k=1; k<nz-2; k++)
  {
    CURRENT_G += nx*ny;

    __syncthreads();
    // Re-use data already in shared mem and registers to move forward
    back  = curr;
    curr  = s_data[CURRENT_S] = front;
    front = d_in[CURRENT_G + nx*ny];

    // Load halo region into shared memory
    if(tx == 1 && ix > 0)       s_data[CURRENT_S - 1]  = d_in[CURRENT_G - 1];
    if(tx == bx-2 && ix < nx-1) s_data[CURRENT_S + 1]  = d_in[CURRENT_G + 1];
    //if(ty == 1 && iy > 0)       s_data[CURRENT_S - bx] = d_in[CURRENT_G - nx];
    //if(ty == by-2 && iy < ny-1) s_data[CURRENT_S + bx] = d_in[CURRENT_G + nx];
    __syncthreads();

    // Load shared memory into registers
    right = s_data[CURRENT_S + 1];
    left  = s_data[CURRENT_S - 1];

    // Perform computation and write to output grid (excluding edge nodes)
    if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1)
    {
      up    = d_in[CURRENT_G - nx];
      down  = d_in[CURRENT_G + nx];
      temp = right + left + up + down + front + back - curr * fac;
      d_out[CURRENT_G] = temp;
    }
  }
}


__global__ void jacobi3d_7p_shmem_adam_reg(double * d_in, double * d_out, const int nx, const int ny, const int nz, const double fac)
{
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  int CURRENT_G = ix + iy*nx + nx*ny;
  //__syncthreads();


  double curr;
  double right, left;
  double up, down;
  double front, back;

  double temp;

  if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1)
  {  

    // Load current, front, and back nodes into shared and register memory
    back  = d_in[CURRENT_G - nx*ny];
    curr  = d_in[CURRENT_G];
    front = d_in[CURRENT_G + nx*ny]; 
    // Load halo region into rigister
    /*if(tx == 1 && ix > 0)*/       left  = d_in[CURRENT_G - 1];
    /*if(tx == bx-2 && ix < nx-1)*/ right = d_in[CURRENT_G + 1];
    /*if(ty == 1 && iy > 0) */      up    = d_in[CURRENT_G - nx];
    /*if(ty == by-2 && iy < ny-1)*/ down  = d_in[CURRENT_G + nx];
    //__syncthreads();
    temp = right + left + up + down + front + back - curr * fac;
    d_out[CURRENT_G] = temp;
  }

  for(int k=1; k<nz-2; k++)
  {
    CURRENT_G += nx*ny;

    //__syncthreads();
    // Perform computation and write to output grid (excluding edge nodes)
    if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1)
    {
      // Re-use data already in registers to move forward
      back  = curr;
      curr  = front;
      front = d_in[CURRENT_G + nx*ny];
      //__syncthreads();
      // Load halo region into register
      /*if(tx == 1 && ix > 0)*/       left = d_in[CURRENT_G - 1];
      /*if(tx == bx-2 && ix < nx-1)*/ right = d_in[CURRENT_G + 1];
      /*if(ty == 1 && iy > 0)*/       up    = d_in[CURRENT_G - nx];
      /*if(ty == by-2 && iy < ny-1)*/ down  = d_in[CURRENT_G + nx];
      //__syncthreads();
      temp = right + left + up + down + front + back - curr * fac;
      d_out[CURRENT_G] = temp;
    }
    //__syncthreads();
  }
}
