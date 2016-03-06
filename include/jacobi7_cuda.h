
// jacobi7 3d navie implementation
// input: input, output: output, nx, ny, nz: 3 dimensions of input and output
__global__ void jacobi3d_7p_naive(float *input, float *output, const int nx, const int ny, const int nz, float fac){
    int i, j, k, t;   

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
    #pragma unroll 8 
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


// jacobi7 3d implementation using shared memory
// shared memory: 3 times of a block
__global__ void jacobi3d_7p_shmem(float *input, float *output, const int nx, const int ny, const int nz, float fac){
  extern __shared__ float sh_buffer[];
  // thread coordination in the whole grid
  const int tx_b = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty_b = blockIdx.y * blockDim.y + threadIdx.y;
  const int xy   = nx * ny;
  const int txy  = blockDim.x * blockDim.y;
  int k = 1;
  int c_l, e_l, w_l, n_l, s_l, t_l, b_l;
  int c_g, e_g, w_g, n_g, s_g, t_g, b_g;
  float c_d, e_d, w_d, n_d, s_d, t_d, b_d;
  
  // extended shared memory with outer boundary data
  //const int tx_tran = threadIdx.x + 1;
  //const int ty_tran = threadIdx.y + 1;

  // c_g: index of current point in global memory, c_l: index of current point in shared memory
  c_g = tx_b + ty_b * nx + k * xy;
  c_l = threadIdx.x + blockDim.x * threadIdx.y;
  //c_l = tx_tran + blockDim.x * threadIdx.y;

  c_d = sh_buffer[c_l] = input[c_g];
  t_g = c_g + xy;
  t_l = c_l + txy;
  t_d = sh_buffer[t_l] = input[t_g];
  b_g = c_g - xy;
  b_l = c_l + 2 * txy;
  b_d = sh_buffer[b_l] = input[b_g];

  // loac outer boundary data into share memory


  __syncthreads();  
  // load data in to shared memory
  // every thread is responsible to load current point, top point and bottom point in to shared memory：currentlayer, toplayer, bottomlayer
 

  // when the shared memory was filled by every thread in the block, caomputing the result

  int isboundary = ((tx_b == 0) || (tx_b == nx-1) || (ty_b == 0) || (ty_b == ny-1)) ? 1 : 0;
  if (!isboundary){
    
    /* dele the judgement conditions, load the ...
    */
    for (; k < nz - 1; ++k){

      if (threadIdx.x == blockDim.x - 1) {
        e_g = c_g + 1;
        e_d = input[e_g]; 
      }
      else{
        e_l = c_l + 1; 
        e_d = sh_buffer[e_l];
      }   
      // this could remove divergance
      /*e_l = c_l + 1;
      e_g = c_g + 1;
      if (e_l % blockDim.x !=0)
        e_d = sh_buffer[e_l];
      else
        e_d  = input[e_g];*/ 


      if (threadIdx.x == 0) {
        w_g = c_g - 1;
        w_d = input[w_g];
      }
      else {
        w_l = c_l - 1; 
        w_d = sh_buffer[w_l];
      } 

      if (threadIdx.y == blockDim.y - 1) {
        n_g = c_g + nx; 
        n_d = input[n_g];
      }
      else {
        n_l = c_l + blockDim.x; 
        n_d = sh_buffer[n_l];
      } 


      if (threadIdx.y == 0) {
        s_g = c_g - nx;   
        s_d = input[s_g];
      }
      else {
        s_l = c_l - blockDim.x; 
        s_d = sh_buffer[s_l];
      } 

      __syncthreads();  
      //c_d = e_d + w_d + n_d + s_d + t_d + b_d - c_d * fac;  
      output[c_g] = e_d + w_d + n_d + s_d + sh_buffer[t_l] + sh_buffer[b_l] - sh_buffer[c_l] * fac;
      
      //__syncthreads();

      c_g += xy;
      t_g = c_g + xy;
      // reuse data in shared memory
      c_d = sh_buffer[c_l] = t_d;
      t_d = sh_buffer[t_l] = input[t_g];
      b_d = sh_buffer[b_l] = c_d; 
      __syncthreads();
    }
  }
  

}

// jacobi7 3d implementation using shared memory
__global__ void jacobi3d_7p_shmem1(float *input, float *output, const int nx, const int ny, const int nz, float fac){
  // thread coordination in the whole grid
  extern __shared__ float sh_buffer[];
  const int tx_b = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty_b = blockIdx.y * blockDim.y + threadIdx.y;
  const int xy   = nx * ny;
  const int txy  = blockDim.x * blockDim.y;
  //int k = 1;
  int c_l, e_l, w_l, n_l, s_l, t_l, b_l;
  int c_g, e_g, w_g, n_g, s_g, t_g, b_g;
  float c_d, e_d, w_d, n_d, s_d, t_d, b_d;

  // for shared memory 
  const int tx = threadIdx.x + 1;
  const int ty = threadIdx.y + 1;

  int isboundary = ((tx_b == 0) || (tx_b == nx-1) || (ty_b == 0) || (ty_b == ny-1)) ? 1 : 0;


   c_g = tx_b + ty_b * nx + xy;
   c_l = tx + ty * blockDim.x; 
   c_d = sh_buffer[c_l] = input[c_g];
   t_g = c_g + xy;
   t_d = input[t_g];
   b_g = c_g - xy;
   b_d = input[b_g]; 
   // halo
   if (tx==1 && tx_b > 0)             sh_buffer[t_l-1] = input[c_g-1];//w
   if (tx==blockDim.x && tx_b < nx-1) sh_buffer[t_l+1] = input[c_g+1];//e
   if (ty==1 && ty_b >0 )             sh_buffer[t_l + blockDim.x] = input[c_g + nx];//n
   if (ty==blockDim.y && ty_b < ny-1) sh_buffer[t_l - blockDim.x] = input[c_g - nx];//s
   
   __syncthreads();

    c_d = sh_buffer[c_l];
    e_d = sh_buffer[e_l];
    w_d = sh_buffer[w_l];
    n_d = sh_buffer[n_l];
    s_d = sh_buffer[s_l];
   if(!isboundary){
    //output[c_g] = sh_buffer[t_l-1] + sh_buffer[t_l+1] + sh_buffer[t_l+blockDim.x] + sh_buffer[t_l-blockDim.x] + t_d + b_d - c_d * fac; 
    output[c_g] = e_d + w_d + n_d + s_d +t_d + b_d - c_d * fac;
   }

  for(int k=2; k < nz-1; k++){
    c_g += xy;
    __syncthreads();
    t_g = c_g + xy;
    b_d = c_d;
    c_d = sh_buffer[c_l] = t_d;
    //b_g = c_g - xy;
    t_d = input[t_g];
    // halo
   if (tx==1 && tx_b > 0)             sh_buffer[t_l-1] = input[c_g-1];//w
   if (tx==blockDim.x && tx_b < nx-1) sh_buffer[t_l+1] = input[c_g+1];//e
   if (ty==1 && ty_b >0 )             sh_buffer[t_l + blockDim.x] = input[c_g + nx];//n
   if (ty==blockDim.y && ty_b < ny-1) sh_buffer[t_l - blockDim.x] = input[c_g - nx];//s
    
    __syncthreads(); 
    c_d = sh_buffer[c_l];
    e_d = sh_buffer[e_l];
    w_d = sh_buffer[w_l];
    n_d = sh_buffer[n_l];
    s_d = sh_buffer[s_l];
    if(!isboundary){

    //output[c_g] = sh_buffer[t_l-1] + sh_buffer[t_l+1] + sh_buffer[t_l+blockDim.x] + sh_buffer[t_l-blockDim.x] + t_d + b_d - c_d * fac; 
     output[c_g] = e_d + w_d + n_d + s_d +t_d + b_d - c_d * fac;
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
  const int bx = blockDim.x;
  const int by = blockDim.y;
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  const int tx = threadIdx.x + 1;
  const int ty = threadIdx.y + 1;

  int CURRENT_G = ix + iy*nx + nx*ny;
  //__syncthreads();
  //int CURRENT_S = tx + ty*bx;

  //extern __shared__ float s_data[];

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

// jacobi7 3d implementation using shared memory
// shared memory: equal to block size
__global__ void jacobi3d_7p_shmem2(float *input, float *output, const int nx, const int ny, const int nz, float fac){
  extern __shared__ float sh_buffer[];
  // thread coordination in the whole grid
  const int tx_b = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty_b = blockIdx.y * blockDim.y + threadIdx.y;
  const int xy   = nx * ny;
  const int txy  = blockDim.x * blockDim.y;
  int k = 1;
  int c_l, e_l, w_l, n_l, s_l, t_l, b_l;
  int c_g, e_g, w_g, n_g, s_g, t_g, b_g;
  float c_d, e_d, w_d, n_d, s_d, t_d, b_d;


  int isboundary = ((tx_b == 0) || (tx_b == nx-1) || (ty_b == 0) || (ty_b == ny-1)) ? 1 : 0;
  const int ix = threadIdx.x;
  const int iy = threadIdx.y;

  c_g = tx_b + ty_b * nx + k * xy;
  c_l = ix + iy * blockDim.x; 
  c_d = sh_buffer[c_l] = input[c_g];
  t_g = c_g + xy;
  t_d = input[t_g];
  b_g = c_g - xy;
  b_d = input[b_g]; 

  __syncthreads();   


     // halo
    if (ix==0)
      w_d = input[c_g-1];//w
    else
      w_d = sh_buffer[c_l-1]; 

    if (ix==blockDim.x - 1)
      e_d = input[c_g + 1];//e
    else
      e_d = sh_buffer[c_l + 1]; 

    if (iy==0) 
      n_d = input[c_g - nx];//n
    else
      n_d = sh_buffer[c_l - blockDim.x];  

    if (iy==blockDim.y -1)
      s_d = input[c_g+nx];//s
    else
      s_d = sh_buffer[c_l + blockDim.x];
  if(!isboundary){ 
    output[c_g] = e_d + w_d + s_d + n_d + t_d + b_d - c_d * fac; 
  }

  for(k=2; k < nz - 1; k++){
    c_g += xy;
    __syncthreads();
    t_g = c_g + xy;
    //b_g = c_g - xy;
    b_d = c_d;
    c_d = sh_buffer[c_l] = t_d;
    t_d = input[t_g];
    __syncthreads(); 

    
      // halo
      if (ix==0)
        w_d = input[c_g-1];//w
      else
        w_d = sh_buffer[c_l-1];   

      if (ix==blockDim.x-1)
        e_d = input[c_g+1];//e
      else
        e_d = sh_buffer[c_l+1];   

      if (iy==0) 
        n_d = input[c_g - nx];//n
      else
        n_d = sh_buffer[c_l - blockDim.x];    

      if (iy==blockDim.y -1)
        s_d = input[c_g+nx];//s
      else
        s_d = sh_buffer[c_l + blockDim.x];
    if(!isboundary){ 
      output[c_g] = e_d + w_d + s_d + n_d + t_d + b_d - c_d * fac; 
      }
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


__global__ void jacobi3d_7p_shmem_adam_overlap(const float * d_in, float * d_out, int nx, int ny, int nz, float fac)
{}
  /*const int bx = blockDim.x;
  const int by = blockDim.y;
  const int ix = threadIdx.x;
  const int iy = threadIdx.y;
  const int i  =  threadIdx.x + blockIdx.x * blockDim.x;
  const int j  = threadIdx.y + blockIdx.y * blockDim.y;



  int c_g = i + j*nx + nx*ny;
  int c_s = (ix+2) + (iy+2)*(bx+4);

  extern __shared__ float sb[];
  float *sb1 = sb;
  float *sb2 = sb + (bx+4)*(by+4);
  float *sb3 = sb + (bx+4)*(by+4)*2;

  float curr;
  float right, left;
  float up, down;
  float front, back;

  float temp;



  // Load current, front, and back nodes into shared and register memory
  back  = d_in[c_g - nx*ny];
  curr  = sb2[c_s] = d_in[c_g];
  front = d_in[c_g + nx*ny]; 

  // Load halo region into shared memory
  if(ix == 0 && i > 0){
      sb2[c_s - 1]  = d_in[c_g - 1];
      sb2[c_s - 2]  = d_in[c_g - 2];
    }
  if(ix == bx-1 && i < nx-1){
    sb2[c_s + 1]  = d_in[c_g + 1];
    sb2[c_s + 2]  = d_in[c_g + 2];
  }
  if(iy == 1 && j > 0){
    sb2[c_s - bx]      = d_in[c_g - nx];
    sb2[c_s - 2 * bx]  = d_in[c_g - 2*nx];
  }
  if(iy == by-1 && j < ny-1){
    sb2[c_s + bx] = d_in[c_g + nx];
    sb2[c_s + 2*bx] = d_in[c_g + 2*nx];
  }
  // top-left
  if(ix == 0 && iy == 0 && i< nx-1 && j<ny-1){
    sb2[c_s-bx-1] = d_in[c_g-nx-1];
    sb2[c_s-bx-2] = d_in[c_g-nx-2];
    sb2[c_s-2*bx-1] = d_in[c_g-2*nx-1];
    sb2[c_s-2*bx-2] = d_in[c_g-2*nx-2];
  }

  // down-right
  if(ix == bx-1 && iy == by-1 && i< nx-1 && j<ny-1){
    sb2[c_s+bx+1] = d_in[c_g+nx+1];
    sb2[c_s+bx+2] = d_in[c_g+nx+2];
    sb2[c_s+2*bx+1] = d_in[c_g+2*nx+1];
    sb2[c_s+2*bx+2] = d_in[c_g+2*nx+2];
  }
  // down-left
  if(ix==bx-1 && iy==0 && i<nx-1 && j<ny-1){
    sb2[c_s+bx -1] = d_in[c_g+nx-1];
    sb2[c_s+bx -2] = d_in[c_g+nx-2];
    sb2[c_s+2*bx -1] = d_in[c_g+2*nx -1];
    sb2[c_s+2*bx -2] = d_in[c_g+2*nx -2];
  }
  //top-right
  if(ix==0 && iy==ny-1 && i<nx-1 && j<ny-1){
    sb2[c_s-bx+1] = d_in[c_g-nx+1];
    sb2[c_s-bx +2] = d_in[c_g-nx+2];
    sb2[c_s-2*bx +1] = d_in[c_g-2*nx+1];
    sb2[c_s-2*bx +2] = d_in[c_g-2*nx+2];
  }
  __syncthreads();

  // Load shared memory into registers
  curr  = sb[CURRENT_S];
  right = sb[CURRENT_S + 1];
  left  = sb[CURRENT_S - 1];
  up    = sb[CURRENT_S - bx];
  down  = sb[CURRENT_S + bx];

  if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1)
  {
    temp = right + left + up + down + front + back - curr * fac;

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
}*/

  ///old implamentation
  /*// block position
  const int bx = blockDim.x;
  const int by = blockDim.y;

  // thread position in global memory
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  //shared memory dimension
  const int share_x = bx + 2 * TIME_TILE_SIZE;
  const int share_y = by + 2 * TIME_TILE_SIZE;


  // stencil grid points position in shared memory
  const int s_x = threadIdx.x + TIME_TILE_SIZE;
  const int s_y = threadIdx.y + TIME_TILE_SIZE;

  // top left position of shared memory: every layer : global memory
  int g_x1 = blockIdx.x * blockDim.x - TIME_TILE_SIZE;
  int g_y1 = blockIdx.y * blockDim.y - TIME_TILE_SIZE;
  int pos1 = g_x1 + g_y1 * nx;

  // down right position of shared memory: every layer : global memory
  int g_x2 = blockIdx.x * blockDim.x + TIME_TILE_SIZE;
  int g_y2 = blockIdx.y * blockDim.y + TIME_TILE_SIZE;
  int pos2 = g_x2 + g_y2 * nx;

  // and the length of every line is blockDim.x

  int k = 1;
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
    d_out[CURRENT_G] = s_data[CURRENT_S+1] + s_data[CURRENT_S-1] +s_data[CURRENT_S-bx] + s_data[CURRENT_S+bx] +front + back - s_data[CURRENT_S] * fac;
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
}*/


// jacobi7 3d overlapped implementation, origin
/*__global__ void jacobi3d_7p_overlap0(float *d_in, float *d_out, const int nx, const int ny, const int nz, 
                                     const int tx, const int ty, const int tz, float fac){
  // block dimension
  const int bx = blockDim.x;
  const int by = blockDim.y;
  const int bz = blockDim.z;

  //shared memory dimension
  const int share_x = bx;
  const int share_y = by;
  const int share_z = bz;

  // thread global dimension
  const int ix = threadIdx.x + tx * bx;
  const int iy = threadIdx.y + ty * by;
  const int iz = threadIdx.z + tz * bz;

  // grid points to be computed, dimensions
  const int s_x = threadIdx.x + TIME_TILE_SIZE;
  const int s_y = threadIdx.y + TIME_TILE_SIZE;
  const int s_z = threadIdx.z + TIME_TILE_SIZE;

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
  for (int h = 1; h<=TIME_TILE_SIZE; h++){
    //left
    if(s_x == TIME_TILE_SIZE && ix > 0)              
      s_data[CURRENT_S - h]  = d_in[CURRENT_G - h];
    // right
    if(s_x == tx+TIME_TILE_SIZE-1 && ix < nx-1)   
      s_data[CURRENT_S + h]  = d_in[CURRENT_G + h];
    // up
    if(s_y == TIME_TILE_SIZE && iy > 0)              
      s_data[CURRENT_S - h*share_x] = d_in[CURRENT_G - h*nx];
    // down
    if(s_y == ty+TIME_TILE_SIZE-1 && iy < ny-1)   
      s_data[CURRENT_S + h*share_x] = d_in[CURRENT_G + h*nx];
    //front
    if(s_z == TIME_TILE_SIZE && iz >0)               
      s_data[CURRENT_S - h*share_x * share_y] = d_in[CURRENT_G - h * nx * ny];
    // back
    if(s_z == tz+TIME_TILE_SIZE-1 && iz < nz-1)   
      s_data[CURRENT_S + h* share_x * share_y] = d_in[CURRENT_G + h * nx * ny];

    // up-left
    if (s_x==TIME_TILE_SIZE && sy = TIME_TILE_SIZE && sz = TIME_TILE_SIZE){
          
    }
        }
      }
    }
  }

  

  __syncthreads();

  for (int tt = 0; tt<TIME_TILE_SIZE; tt++){
    // Load shared memory into registers
    curr  = s_data[CURRENT_S];
    right = s_data[CURRENT_S + 1];
    left  = s_data[CURRENT_S - 1];
    up    = s_data[CURRENT_S - share_x];
    down  = s_data[CURRENT_S + share_x];
    front = s_data[CURRENT_S - share_x * share_y];
    back  = s_data[CURRENT_S + share_x * share_y];
    __syncthreads();  

    if(ix > 0 && ix < nx-1 & iy > 0 && iy < ny-1 && iz >0 && iz < nz -1)
    {
      temp = right + left + up + down + front + back - curr * fac;
      s_data[CURRENT_S] = temp;
    __syncthreads();
    }
 
  if(s_x >= (TIME_TILE_SIZE) &&
      s_x <= (tx +(TIME_TILE_SIZE-1)) &&
      s_y >= (TIME_TILE_SIZE) &&
      s_y <= (ty+(TIME_TILE_SIZE-1)) &&
      tz >= (TIME_TILE_SIZE) &&
      tz <= (tz+(TIME_TILE_SIZE-1))) {
    d_out[CURRENT_G] = s_data[CURRENT_S];
  
  }
  }
}

*/


// jacobi7 3d implementation: overlapped referred ics12
/*__global__ void jacobi3d_7p_overlap(float *input, float *output, const int nx, const int ny, const int nz, float fac){
  const int bx = blockDim.x;
  const int by = blockDim.y;
  const int bz = blockDim.z;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tz = threadIdx.z;

  int ix = threadIdx.x + (blockDim.x - 2 * (TIME_TILE_SIZE-1)) * blockIdx.x;
  int iy = threadIdx.y + (blockDim.y - 2 * (TIME_TILE_SIZE-1)) * blockIdx.y;
  int iz = threadIdx.z + (blockDim.z - 2 * (TIME_TILE_SIZE-1)) * blockIdx.z;
  ix -= TIME_TILE_SIZE-1;
  iy -= TIME_TILE_SIZE-1;
  iz -= TIME_TILE_SIZE-1;

  int c_g = ix + nx * (iy + ny * iz);
  int c_l = tx + bx * (ty + by * tz);
  int r_l = c_l + 1;
  int l_l = c_l - 1;
  int t_l = c_l + bx;
  int d_l = c_l - bx;
  int f_l = c_l - bx*by;
  int b_l = c_l + bx*by;

  extern __shared__ float s_data[];

  // every thread load its grid point into shared memory
  float c = ((ix >= 0) && (ix <= nx - 1) && (iy >= 0) && (iy <= ny-1) && (iz >= 0) && (iz <= nz -1)) ? input[c_g] : 0.0;
  /*float r = ((ix+1 >= 0) && (ix+1 <= nx -1) && (iy >= 0) && (iy <= ny-1) && (iz >= 0) && (iz <= nz -1)) ? input[c_g+1] : 0.0;
  float l = ((ix-1 >= 0) && (ix-1 <= nx -1) && (iy >=0) && (iy <= ny-1) && (iz >= 0) && (iz <= nz -1)) ? input[c_g-1] : 0.0;
  float t = ((ix >= 0) && (ix <= nx-1) && (iy+1 >= 0) && (iy+1 <= ny-1) && (iz >= 0) && (iz <= nz -1)) ? input[c_g+nx] : 0.0;
  float d = ((ix >= 0) && (ix <= nx-1) && (iy-1 >= 0) && (iy-1 <= ny-1) && (iz >= 0) && (iz <= nz -1)) ? input[c_g-nx] : 0.0;
  float f = ((ix >= 0) && (ix <= nx - 1) && (iy >= 0) && (iy <= ny-1) && (iz - 1 >= 0) && (iz -1 <= nz - 1)) ? input[c_g-nx*ny] : 0.0;
  float b = ((ix >= 0) && (ix <= nx - 1) && (iy >= 0) && (iy <= ny-1) && (iz + 1>= 0) && (iz + 1 <= nz - 1)) ? input[c_g+nx*ny] : 0.0;
  */
  /*s_data[c_l] = c;
  //__syncthreads();
  //
  float r = ((r_l >= 0) && (r_l<=bx-1)) ? s_data[r_l] : 0.0;
  float l = ((l_l >= 0) && (l_l<=bx-1)) ? s_data[l_l] : 0.0;
  float t = ((t_l >= 0) && (t_l<=by-1)) ? s_data[t_l] : 0.0;
  float d = ((d_l >= 0) && (d_l<=by-1)) ? s_data[d_l] : 0.0;
  float f = ((f_l >= 0) && (f_l<=bz-1)) ? s_data[f_l] : 0.0;
  float b = ((b_l >= 0) && (b_l<=bz-1)) ? s_data[b_l] : 0.0;
  float tmp = r + l + t + d + f + b - c * fac;
  s_data[c_l] = ((ix > 0) && (ix < nx - 1) && (iy > 0) && (iy < ny-1) && (iz > 0) && (iz < nz -1)) ? tmp : s_data[c_l];
  __syncthreads();
  
  for(int tt = 0; tt < TIME_TILE_SIZE-1; ++tt) {
    c = s_data[c_l];
    /*r = ((ix+1 >= 0) && (ix+1 <= nx -1) && (iy >= 0) && (iy <= ny-1) && (iz >= 0) && (iz <= nz -1)) ? s_data[r_l] : 0.0;
    l = ((ix-1 >= 0) && (ix-1 <= nx -1) && (iy >=0) && (iy <= ny-1) && (iz >= 0) && (iz <= nz -1)) ? s_data[l_l] : 0.0;
    t = ((ix >= 0) && (ix <= nx-1) && (iy+1 >= 0) && (iy+1 <= ny-1) && (iz >= 0) && (iz <= nz -1)) ? s_data[t_l] : 0.0;
    d = ((ix >= 0) && (ix <= nx-1) && (iy-1 >= 0) && (iy-1 <= ny-1) && (iz >= 0) && (iz <= nz -1)) ? s_data[d_l] : 0.0;
    f = ((ix >= 0) && (ix <= nx - 1) && (iy >= 0) && (iy <= ny-1) && (iz-1 >= 0) && (iz-1 <= nz - 1)) ? s_data[f_l] : 0.0;
    b = ((ix >= 0) && (ix <= nx - 1) && (iy >= 0) && (iy <= ny-1) && (iz+1 >= 0) && (iz+1 <= nz - 1)) ? s_data[b_l] : 0.0;*/
    /*r = ((r_l >= 0) && (r_l<=bx-1)) ? s_data[r_l] : 0.0;
    l = ((l_l >= 0) && (l_l<=bx-1)) ? s_data[l_l] : 0.0;
    t = ((t_l >= 0) && (t_l<=by-1)) ? s_data[t_l] : 0.0;
    d = ((d_l >= 0) && (d_l<=by-1)) ? s_data[d_l] : 0.0;
    f = ((f_l >= 0) && (f_l<=bz-1)) ? s_data[f_l] : 0.0;
    b = ((b_l >= 0) && (b_l<=bz-1)) ? s_data[b_l] : 0.0;
    tmp = r + l + t + d + f + b - c * fac;
    s_data[c_l] = ((ix > 0) && (ix < nx - 1) && (iy > 0) && (iy < ny-1) && (iz > 0) && (iz < nz -1)) ? tmp : s_data[c_l];
    // Sync before re-reading shared
    __syncthreads();
  }
  if(tx >= (TIME_TILE_SIZE-1) &&
      tx <= (blockDim.x-1-(TIME_TILE_SIZE-1)) &&
      ty >= (TIME_TILE_SIZE-1) &&
      ty <= (blockDim.y-1-(TIME_TILE_SIZE-1)) &&
      tz >= (TIME_TILE_SIZE-1) &&
      tz <= (blockDim.z-1-(TIME_TILE_SIZE-1))) {
     output[c_g] = s_data[c_l];
   
  }
}
*/
__global__ void jacobi3d_7p_shmem_test(){
  extern __shared__ float sh_buffer[];
  //sh_buffer[1] = 0.0;
  int a = 1;
  int b = 2;
  a = a+b;
  }