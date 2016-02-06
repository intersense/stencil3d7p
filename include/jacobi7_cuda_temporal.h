// temporal versions

//#ifndef TIME_TILE_SIZE
//#warning TIME_TILE_SIZE is not set, defaulting to 1
#define TIME_TILE_SIZE 2
// refer benchmarks
// time tile is 2

__global__ void jacobi3d_7p_temporal(const float * f1, float * f2, int nx, int ny, int nz, float fac)
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
