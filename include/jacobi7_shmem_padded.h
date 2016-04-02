// padded for avoiding bank conflicts
// make the layout of shared memory as follows:
/*
_n__n__n__n__
__c__c__c__c_
s__s__s__s__s
*/
__global__ void jacobi3d_7p_shmem_padded(float * d_in, float * d_out, const int nx, const int ny, const int nz, const float fac)
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
    }
  }
}
