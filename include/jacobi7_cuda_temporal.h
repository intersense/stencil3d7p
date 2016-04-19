// temporal versions

//#ifndef TIME_TILE_SIZE
//#warning TIME_TILE_SIZE is not set, defaulting to 1
#define TIME_TILE_SIZE 2
// refer benchmarks
// time tile is 2

__global__ void jacobi3d_7p_25d(float * d_in, float * d_out, const int nx, const int ny, const int nz, const float fac)
{
  const int bx = blockDim.x;
  const int by = blockDim.y;
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  const int tx = threadIdx.x + 1;
  const int ty = threadIdx.y + 1;

  // the size X-dimension of shared memory
  const int x_s = bx + 2;
  const int y_s = by + 2;

  int k = 1;
  int C_G = ix + iy*nx + nx*ny;
  int C_S = tx + ty * x_s;
  extern __shared__ float s_data[];
  
  float *k_0 = s_data; // back plane
  float *k_1 = s_data + x_s * y_s; // current plane
  float *k_2 = s_data + 2 * x_s * y_s; // front plane

  float curr;
  float right, left;
  float up, down;
  float front, back;

  float temp;
  
  // boundary for the core
  bool boundary_c = (ix > 0 && ix < nx - 1 && iy > 0 && iy < ny - 1);
  // boundary for the halo
  bool boundary_h = (ix > 1 && ix < nx - 2 && iy > 1 && iy < ny - 2);
  // boundary for k
  bool boundary_k = (k >=1 && k <= nz - 2);

  // 1 
  // k_0 and k_2 need no halo
  // compute k_0 and store in k_0
  if (boundary_c){
    int C_G_0 = C_G - nx * ny;
    k_0[C_S] = d_in[C_G_0];

    // compute k_1 and store in k_1
    k_1[C_S] = d_in[C_G - 1] + d_in[C_G + 1] + d_in[C_G - nx] + d_in[C_G + nx] + d_in[C_G - nx * ny] + d_in[C_G + nx * ny] - fac * d_in[C_G];

    // halo of k_1
    if(tx == 1 && ix > 1) {
      if (boundary_h){
        k_1[C_S - 1] = d_in[C_G - 1 - 1] + d_in[C_G - 1 + 1] + d_in[C_G - 1 - nx] + d_in[C_G - 1 + nx] + d_in[C_G - 1 - nx * ny] + d_in[C_G - 1 + nx * ny] - fac * d_in[C_G - 1];
      }
    } else{
      k_1[C_S - 1] = d_in[C_G - 1];
    }
    if(tx == bx && ix < nx - 2){
      if(boundary_h){
        k_1[C_S + 1] = d_in[C_G + 1 - 1] + d_in[C_G + 1 + 1] + d_in[C_G + 1 - nx] + d_in[C_G + 1 + nx] + d_in[C_G + 1 - nx * ny] + d_in[C_G + 1 + nx * ny] - fac * d_in[C_G + 1];
      }
    } else{
      k_1[C_S + 1] = d_in[C_G + 1];
    }
    if(ty == 1 && iy > 1){
      if (boundary_h){
        k_1[C_S - x_s] = d_in[C_G - nx - 1] + d_in[C_G - nx + 1] + d_in[C_G - nx - nx] + d_in[C_G - nx + nx] + d_in[C_G - nx - nx * ny] + d_in[C_G - nx + nx * ny] - fac * d_in[C_G - nx];
      } 
    }else{
      k_1[C_S - x_s] = d_in[C_G - nx];
    }
    if(ty == by && iy < ny - 2){
      if (boundary_h){
        k_1[C_S + x_s] = d_in[C_G + nx - 1] + d_in[C_G + nx + 1] + d_in[C_G + nx - nx] + d_in[C_G + nx + nx] + d_in[C_G + nx - nx * ny] + d_in[C_G + nx + nx * ny] - fac * d_in[C_G + nx];
      }
    }else{
      k_1[C_S + x_s] = d_in[C_G + nx];
    }
  
    // compute k_2 and store in k_2
    int C_G_2 = C_G + nx * ny;
    k_2[C_S] = d_in[C_G_2 - 1] + d_in[C_G_2 + 1] + d_in[C_G_2 - nx] + d_in[C_G_2 + nx] + d_in[C_G_2 - nx * ny] + d_in[C_G_2 + nx * ny] - fac * d_in[C_G_2];
    
    // halo of k_2
    if(tx == 1 && ix > 1) {
      if (boundary_h){
        k_2[C_S - 1] = d_in[C_G_2 - 1 - 1] + d_in[C_G_2 - 1 + 1] + d_in[C_G_2 - 1 - nx] + d_in[C_G_2 - 1 + nx] + d_in[C_G_2 - 1 - nx * ny] + d_in[C_G_2 - 1 + nx * ny] - fac * d_in[C_G_2 - 1];
      } else{
        k_2[C_S - 1] = d_in[C_G_2 - 1];
      }
    }
    if(tx == bx && ix < nx - 2){
      if(boundary_h){
        k_2[C_S + 1] = d_in[C_G_2 + 1 - 1] + d_in[C_G_2 + 1 + 1] + d_in[C_G_2 + 1 - nx] + d_in[C_G_2 + 1 + nx] + d_in[C_G_2 + 1 - nx * ny] + d_in[C_G_2 + 1 + nx * ny] - fac * d_in[C_G_2 + 1];
      }
    } else{
      k_2[C_S + 1] = d_in[C_G_2 + 1];
    }
    if(ty == 1 && iy > 1){
      if (boundary_h){
        k_2[C_S - x_s] = d_in[C_G_2 - nx - 1] + d_in[C_G_2 - nx + 1] + d_in[C_G_2 - nx - nx] + d_in[C_G_2 - nx + nx] + d_in[C_G_2 - nx - nx * ny] + d_in[C_G_2 - nx + nx * ny] - fac * d_in[C_G_2 - nx];
      } 
    }else{
      k_2[C_S - x_s] = d_in[C_G_2 - nx];
    }
    if(ty == by && iy < ny - 2){
      if (boundary_h){
        k_2[C_S + x_s] = d_in[C_G_2 + nx - 1] + d_in[C_G_2 + nx + 1] + d_in[C_G_2 + nx - nx] + d_in[C_G_2 + nx + nx] + d_in[C_G_2 + nx - nx * ny] + d_in[C_G_2 + nx + nx * ny] - fac * d_in[C_G_2 + nx];
      }
    }else{
      k_2[C_S + x_s] = d_in[C_G_2 + nx];
    }
    __syncthreads();
    // 2
    d_out[C_G] = k_1[C_S - 1] + k_1[C_S + 1] + k_1[C_S - x_s] + k_1[C_S + x_s] + k_0[C_S] + k_2[C_S] - fac * k_1[C_S];
  }

/*****************************************************************/
/* 04/18
k_2 will be new k_1, k_1 will be the new k_0, only new k_2 need to be computed
just like the data reuse in shared version (non-temporal)
*/
/****************************************************************/

  //#pragma unroll 
  for(k=2; k<=nz-2; k++)
  {
    C_G += nx*ny;
    __syncthreads();
    // Re-use data already in shared mem and registers to move forward
    k_0 = k_1;
    k_1 = k_2;
    // 1: compute k2 and its halo
    if (boundary_c){
      int C_G_2 = C_G + nx * ny;
      k_2[C_S] = d_in[C_G_2 - 1] + d_in[C_G_2 + 1] + d_in[C_G_2 - nx] + d_in[C_G_2 + nx] + d_in[C_G_2 - nx * ny] + d_in[C_G_2 + nx * ny] - fac * d_in[C_G_2];
      
      // halo of k_2
      if(tx == 1 && ix > 1) {
        if (boundary_h){
          k_2[C_S - 1] = d_in[C_G_2 - 1 - 1] + d_in[C_G_2 - 1 + 1] + d_in[C_G_2 - 1 - nx] + d_in[C_G_2 - 1 + nx] + d_in[C_G_2 - 1 - nx * ny] + d_in[C_G_2 - 1 + nx * ny] - fac * d_in[C_G_2 - 1];
        } else{
          k_2[C_S - 1] = d_in[C_G_2 - 1];
        }
      }
      if(tx == bx && ix < nx - 2){
        if(boundary_h){
          k_2[C_S + 1] = d_in[C_G_2 + 1 - 1] + d_in[C_G_2 + 1 + 1] + d_in[C_G_2 + 1 - nx] + d_in[C_G_2 + 1 + nx] + d_in[C_G_2 + 1 - nx * ny] + d_in[C_G_2 + 1 + nx * ny] - fac * d_in[C_G_2 + 1];
        }
      } else{
        k_2[C_S + 1] = d_in[C_G_2 + 1];
      }
      if(ty == 1 && iy > 1){
        if (boundary_h){
          k_2[C_S - x_s] = d_in[C_G_2 - nx - 1] + d_in[C_G_2 - nx + 1] + d_in[C_G_2 - nx - nx] + d_in[C_G_2 - nx + nx] + d_in[C_G_2 - nx - nx * ny] + d_in[C_G_2 - nx + nx * ny] - fac * d_in[C_G_2 - nx];
        } 
      }else{
        k_2[C_S - x_s] = d_in[C_G_2 - nx];
      }
      if(ty == by && iy < ny - 2){
        if (boundary_h){
          k_2[C_S + x_s] = d_in[C_G_2 + nx - 1] + d_in[C_G_2 + nx + 1] + d_in[C_G_2 + nx - nx] + d_in[C_G_2 + nx + nx] + d_in[C_G_2 + nx - nx * ny] + d_in[C_G_2 + nx + nx * ny] - fac * d_in[C_G_2 + nx];
        }
      }else{
        k_2[C_S + x_s] = d_in[C_G_2 + nx];
      }
      __syncthreads();

      // 2
      d_out[C_G] = k_1[C_S - 1] + k_1[C_S + 1] + k_1[C_S - x_s] + k_1[C_S + x_s] + k_0[C_S] + k_2[C_S] - fac * k_1[C_S];
  } // end for (k)
}


__global__ void jacobi3d_7p_temporal_reg(const float * f1, float * f2, int nx, int ny, int nz, float fac)
{
  const int bx = blockDim.x;
  const int by = blockDim.y;
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;


}



__global__ void jacobi3d_7p_temporal_old(const float * f1, float * f2, int nx, int ny, int nz, float fac)
{
  int c, sc;
  int c2, sc2;
  extern __shared__ float sb[];
  float *sb1 = sb;
  float *sb2 = sb + blockDim.x * blockDim.y;
  float *sb3 = sb + blockDim.x * blockDim.y * 2;
  //float *sb_tmp; // swap sb1, sb2, and sb3
  float v;
  const int i = min(nx-1, max(0,(blockDim.x -2) * blockIdx.x + threadIdx.x -1));
  const int j = min(ny-1, max(0,(blockDim.y -2) * blockIdx.y + threadIdx.y -1));
  
  const int xy = nx * ny;
  
  c = i + j * nx + xy; // k=1
  sc = threadIdx.x + threadIdx.y * blockDim.x;
  

  int w = /*(i == 0)      ? c :*/ c - 1;
  int e = /*(i == nx-1)   ? c :*/ c + 1;
  int n = /*(j == 0)      ? c :*/ c - nx;
  int s = /*(j == ny-1)   ? c :*/ c + nx;
  int b = c - xy;
  int t = c + xy;
  
  if(i > 0 && i < nx-1 & j > 0 && j < ny-1){
    v = f1[w] + f1[e] + f1[n] + f1[s] + f1[b] + f1[t] - fac * f1[c];
    sb2[sc] = v;
  }

  sb1[sc] = f1[b]; // load k=0 layer into share memory

  c += xy;
  w += xy;
  e += xy;
  n += xy;
  s += xy;
  t += xy;
  b += xy;

  const int i2 = min(nx-1, (blockDim.x -2) * blockIdx.x + min(threadIdx.x, blockDim.x - 3));
  const int j2 = min(ny-1, (blockDim.y -2) * blockIdx.y + min(threadIdx.y, blockDim.y - 3));
  c2 = i2 + j2 * nx + xy;
  sc2 = (i2 % (blockDim.x-2)) + 1 + ((j2 % (blockDim.y-2)) + 1) * blockDim.x;

  int w2 = (i2 == 0)    ? sc2 : sc2 - 1;
  int e2 = (i2 == nx-1) ? sc2 : sc2 + 1;
  int n2 = (j2 == 0)    ? sc2 : sc2 - blockDim.x;
  int s2 = (j2 == ny-1) ? sc2 : sc2 + blockDim.x;

  {
    //int k = 1;
    if(i > 0 && i < nx-1 & j > 0 && j < ny-1){
      v = f1[w] + f1[e] + f1[n] + f1[s] + f1[b] + f1[t] - fac * f1[c];
      sb3[sc] = v;
    }
    c += xy;
    w += xy;
    e += xy;
    n += xy;
    s += xy;
    b += xy;
    t += xy;

    __syncthreads();
    
    if(i > 0 && i < nx-1 & j > 0 && j < ny-1){
      f2[c2] = sb2[w2] + sb2[e2] + sb2[s2] + sb2[n2] + sb1[sc2] + sb3[sc2] - fac * sb2[sc2];
    }
    c2 += xy;
    //sb_tmp = sb1;
    sb1 = sb2;
    sb2 = sb3;
    //sb3 = sb_tmp;//could delete?
   }

   for (int k=2; k < nz - 2; ++k){
    if(i > 0 && i < nx-1 & j > 0 && j < ny-1){
      v = f1[w] + f1[e] + f1[n] + f1[s] + f1[b] + f1[t] - fac * f1[c];
      sb3[sc] = v;
    }
    c += xy; 
    w += xy;
    e += xy;
    n += xy;
    s += xy;
    b += xy;
    t += xy;

    __syncthreads();

    if(i > 0 && i < nx-1 & j > 0 && j < ny-1){
      f2[c2] = sb2[w2] + sb2[e2] + sb2[s2] + sb2[n2] + sb1[sc2] + sb3[sc2] - fac * sb2[sc2];
    }
    c2 += xy;

    //__syncthreads();

    //sb_tmp = sb1;
    sb1 = sb2;
    sb2 = sb3;
    //sb3 = sb_tmp;
   }
   sb3[sc] = f1[t];

  __syncthreads();

  if(i > 0 && i < nx-1 & j > 0 && j < ny-1){
    f2[c2] = sb2[w2] + sb2[e2] + sb2[s2] + sb2[n2] + sb1[sc2] + sb3[sc2] - fac * sb2[sc2];
  }


  // 像剥洋葱一样，从外面到里面一层一层的计算，模式都是一样的
  // 从第二层开始就无需读取global memory
  // 根据这种模式进行自动化处理
}


//#endif
