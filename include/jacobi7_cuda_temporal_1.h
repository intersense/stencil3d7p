// temporal versions

//#ifndef TIME_TILE_SIZE
//#warning TIME_TILE_SIZE is not set, defaulting to 1
#define TIME_TILE_SIZE 2
// refer benchmarks
// time tile is 2

__global__ void jacobi3d_7p_temporal(float * f1, float * f2, const int nx, const int ny, const int nz, const float fac)
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
  

  int w = (i == 0)      ? c : c - 1;
  int e = (i == nx-1)   ? c : c + 1;
  int n = (j == 0)      ? c : c - nx;
  int s = (j == ny-1)   ? c : c + nx;
  int b = c - xy;
  int t = c + xy;
  
  int sw = sc - 1;
  int se = sc + 1;
  int sn = sc - blockDim.x;
  int ss = sc + blockDim.x;

  
  //sb2[sc] = f1[c]; // load k=1 layer into share memory sb2
  //sb1[sc] = f1[b]; // load k=0 layer into share memory sb1
  //sb3[sc] = f1[t]; // load k=2 layer into share memory sb3
  
  __syncthreads();

  // on the boundary, there are two line is the same with the original boundary, so the 0 and 1 , and the last one and two line need not compute

  if(i > 0 && i < nx-1 & j > 0 && j < ny-1){
    v = f1[w] + f1[e] + f1[n] + f1[s] + f1[b] + f1[t] - fac * f1[c];
    // compute the k=1 layer
    //v = sb2[sw] + sb2[se] + sb2[sn] + sb2[ss] + sb1[sc] + sb3[sc] - sb2[sc] * fac;
    sb2[sc] = v; // points not on boundaries are computed, the boundary points remain the same.
  }

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
    if(i > 0 && i < nx-1 & j > 0 && j < ny-1){// compute the top layer
      v = f1[w] + f1[e] + f1[n] + f1[s] + f1[b] + f1[t] - fac * f1[c];
      sb3[sc] = v;
    }
    else{
      sb3[sc] = f1[c];// load the boundary to share memory sb3
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
    if(i > 0 && i < nx-1 & j > 0 && j < ny-1){ // load the top layer
      v = f1[w] + f1[e] + f1[n] + f1[s] + f1[b] + f1[t] - fac * f1[c];
      sb3[sc] = v;
    }
    else{
      sb3[sc] = f1[c];// load the boundary to share memory sb3
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
   //sb3[sc] = f1[t];

  __syncthreads();

  if(i > 0 && i < nx-1 & j > 0 && j < ny-1){
    //f2[c2] = sb2[w2] + sb2[e2] + sb2[s2] + sb2[n2] + sb1[sc2] + sb3[sc2] - fac * sb2[sc2];
    f2[c2] = sb2[w2] + sb2[e2] + sb2[s2] + sb2[n2] + sb1[sc2] + f1[t] - fac * sb2[sc2];
  }
}