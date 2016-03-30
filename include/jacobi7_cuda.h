
// jacobi7 3d navie implementation
// input: input, output: output, nx, ny, nz: 3 dimensions of input and output
__global__ void jacobi3d_7p_naive(float *input, float *output, const int nx, const int ny, const int nz, float fac){
    int i, j, k;   

    for (k = 1; k < nz - 1; k++) {
      for (j = 1; j < ny - 1; j++) {
        for (i = 1; i < nx - 1; i++) {
            output[i+nx*(j+ ny*k)] = 
                input[i+nx*(j+ ny*( k + 1))] +
                input[i+nx*(j+ ny*( k - 1))] +
                input[i+nx*((j + 1)+ ny*k)] +
                input[i+nx*((j - 1)+ ny*k)] +
                input[i+1+nx*(j+ ny*k)] +
                input[(i - 1)+nx*(j+ ny*k)]
                -  input[i+nx*(j+ ny*k)] *fac ;                
        }
      }
    }
}

// jacobi7 3d implementation using global memory
__global__ void jacobi3d_7p_glmem(float *input, float *output, const int nx, const int ny, const int nz, float fac){
  // thread coordination in the whole grid
  const int tx_b = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty_b = blockIdx.y * blockDim.y + threadIdx.y;
  int xy   = nx * ny;
  int k = 1;
  int c_g, e, w, n, s, t, b;
  float c_d, e_d, w_d, n_d, s_d, t_d, b_d;
  int isboundary = (tx_b == 0 || tx_b == nx-1 || ty_b == 0 || ty_b == ny-1) ? 1 : 0;
  if (!isboundary){
    //#pragma unroll 4 
    for (; k < nz - 1; k++){    
      c_g = tx_b + ty_b * nx + k * xy;
      c_d = input[c_g];
      e = c_g + 1;
      w = c_g - 1;
      n = c_g + nx;
      s = c_g - nx;
      t = c_g + xy;
      b = c_g - xy;
      e_d = input[e];
      w_d = input[w];
      n_d = input[n];
      s_d = input[s];
      t_d = input[t];
      b_d = input[b];
      output[c_g] = e_d + w_d + n_d + s_d + t_d + b_d - c_d * fac;
    }
  }
}

__global__ void jacobi3d_7p_shmem_adam(float * d_in, float * d_out, const int nx, const int ny, const int nz, const float fac)
{
  const int bx = blockDim.x;
  const int by = blockDim.y;
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  const int tx = threadIdx.x + 1;
  const int ty = threadIdx.y + 1;

  int CURRENT_G = ix + iy*nx + nx*ny;
  int CURRENT_S = tx + ty*bx;

  extern __shared__ float s_data[];
  /*const float C0 = 1.0f;
  const float C1 = 7.0f;*/

  float curr;
  float right, left;
  float up, down;
  float front, back;

  float temp;

  // Load current, front, and back nodes into shared and register memory
  back  = d_in[CURRENT_G - nx*ny];
  curr  = s_data[CURRENT_S] = d_in[CURRENT_G];
  front = d_in[CURRENT_G + nx*ny]; 

  // Load halo region into shared memory
  if(tx == 1 && ix > 0)       s_data[CURRENT_S - 1]  = d_in[CURRENT_G - 1];
  if(tx == bx-2 && ix < nx-1) s_data[CURRENT_S + 1]  = d_in[CURRENT_G + 1];
  if(ty == 1 && iy > 0)       s_data[CURRENT_S - bx] = d_in[CURRENT_G - nx];
  if(ty == by-2 && iy < ny-1) s_data[CURRENT_S + bx] = d_in[CURRENT_G + nx];
  __syncthreads();

  // Load shared memory into registers
  curr  = s_data[CURRENT_S];
  right = s_data[CURRENT_S + 1];
  left  = s_data[CURRENT_S - 1];
  up    = s_data[CURRENT_S - bx];
  down  = s_data[CURRENT_S + bx];

  if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1)
  {
    temp = right + left + up + down + front + back - curr * fac;
    d_out[CURRENT_G] = temp;
    //d_out[CURRENT_G] = s_data[CURRENT_S+1] + s_data[CURRENT_S-1] +s_data[CURRENT_S-bx] + s_data[CURRENT_S+bx] +front + back - s_data[CURRENT_S] * fac;
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
    if(ty == 1 && iy > 0)       s_data[CURRENT_S - bx] = d_in[CURRENT_G - nx];
    if(ty == by-2 && iy < ny-1) s_data[CURRENT_S + bx] = d_in[CURRENT_G + nx];
    __syncthreads();

    // Load shared memory into registers
    right = s_data[CURRENT_S + 1];
    left  = s_data[CURRENT_S - 1];
    up    = s_data[CURRENT_S - bx];
    down  = s_data[CURRENT_S + bx];

    // Perform computation and write to output grid (excluding edge nodes)
    if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1)
    {
      temp = right + left + up + down + front + back - curr * fac;
      d_out[CURRENT_G] = temp;
      //d_out[CURRENT_G] = s_data[CURRENT_S+1] + s_data[CURRENT_S-1] +s_data[CURRENT_S-bx] + s_data[CURRENT_S+bx] +front + back - s_data[CURRENT_S] * fac;
    }
  }
}

__global__ void jacobi3d_7p_shmem_adam_reg(float * d_in, float * d_out, const int nx, const int ny, const int nz, const float fac)
{
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  int CURRENT_G = ix + iy*nx + nx*ny;
  //__syncthreads();


  float curr;
  float right, left;
  float up, down;
  float front, back;

  float temp;

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

  if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1 && iz >0 && iz < nz -1)
  {
    temp = right + left + up + down + front + back - curr * fac;
    d_out[CURRENT_G] = temp;
  }
}



__global__ void jacobi3d_7p_shmem_test(){
  extern __shared__ float sh_buffer[];
  //sh_buffer[1] = 0.0;
  int a = 1;
  int b = 2;
  a = a+b;
  }