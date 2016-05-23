
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
// loading data row first
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
  #pragma unroll 
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

// jacobi7 3d implementation using global memory
// loading data column first
__global__ void jacobi3d_7p_glmem_col(float *input, float *output, const int nx, const int ny, const int nz, float fac){
  // thread coordination in the whole grid
  const int tx_b = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty_b = blockIdx.y * blockDim.y + threadIdx.y;
  int xy   = nx * ny;
  int k = 1;
  int c_g, e, w, n, s, t, b;
  float c_d, e_d, w_d, n_d, s_d, t_d, b_d;
  int isboundary = (tx_b == 0 || tx_b == nx-1 || ty_b == 0 || ty_b == ny-1) ? 1 : 0;
  if (!isboundary){
  //#pragma unroll 
    for (; k < nz - 1; k++){    
      c_g = ty_b + tx_b * ny + k * xy;
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

// jacobi7 3d implementation using global memory
// loading data column first; store row first
__global__ void jacobi3d_7p_glmem_col_row(float *input, float *output, const int nx, const int ny, const int nz, float fac){
  // thread coordination in the whole grid
  const int tx_b = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty_b = blockIdx.y * blockDim.y + threadIdx.y;
  int xy   = nx * ny;
  int k = 1;
  int c_g, e, w, n, s, t, b;
  int c_g_cr;
  float c_d, e_d, w_d, n_d, s_d, t_d, b_d;
  int isboundary = (tx_b == 0 || tx_b == nx-1 || ty_b == 0 || ty_b == ny-1) ? 1 : 0;
  if (!isboundary){
  //#pragma unroll 
    for (; k < nz - 1; k++){    
      c_g = ty_b + tx_b * ny + k * xy;
      c_g_cr = tx_b + ty_b * nx + k * xy;
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
      
      output[c_g_cr] = e_d + w_d + n_d + s_d + t_d + b_d - c_d * fac;
    }
  }
}

