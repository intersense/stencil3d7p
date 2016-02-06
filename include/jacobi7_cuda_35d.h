

__device__ void readBlockAndHalo(float *d_in, int k, const int nx, const int ny, float *shm){
    int bx = (blockDim.x + 2);

    int i = blockIdx.x * (blockDim.x-2) + threadIdx.x - 1;
    int j = blockIdx.y * (blockDim.y-2) + threadIdx.y - 1;

    // shared memory position by threadIdx
    int sb_x = threadIdx.x + 1;
    int sb_y = threadIdx.y + 1;
    shm[sb_x+0+(sb_y+0)*bx] = d_in[i + j*nx + k*nx*ny];
    shm[sb_x-1+(sb_y+0)*bx] = d_in[i-1 + (j+0)*nx + k*nx*ny];
    shm[sb_x+1+(sb_y+0)*bx] = d_in[i+1 + (j+0)*nx + k*nx*ny];
    shm[sb_x+0+(sb_y+1)*bx] = d_in[i+0 + (j+1)*nx + k*nx*ny];
    shm[sb_x+0+(sb_y-1)*bx] = d_in[i+0 + (j-1)*nx + k*nx*ny];
    // the boundary of stencil grid
    if(i == -1){ //top
        shm[sb_x+0+(sb_y+0)*bx] = -1.0;
    }
    else if(i == nx){// right boundary
        shm[sb_x+0+(sb_y+0)*bx] = -1.0;
    }
    else if(j == -1){// left boundary
        shm[sb_x+0+(sb_y+0)*bx] = -1.0;
    }
    else if(j==ny){//bottom boundary
        shm[sb_x+0+(sb_y+0)*bx] = -1.0;
    }
    else{
        shm[sb_x+0+(sb_y+0)*bx] = d_in[i + j*nx + k*nx*ny];
    }

    // load the halo into shared memory, the boundary of sharedmemroy
    if(sb_x==1){
        if(i==-1 || j==ny) shm[sb_x-1+(sb_y+0)*bx] = -1.0; // top halo
        else  shm[sb_x-1+(sb_y+0)*bx] = d_in[i-1 + (j+0)*nx + k*nx*ny];

        if(sb_y==1){
            if(j==-1||i==-1) shm[sb_x-1+(sb_y-1)*bx] = -1.0; // top left halo
            else shm[sb_x-1+(sb_y-1)*bx] = d_in[i-1 + (j-1)*nx + k*nx*ny];
        }

        if(sb_y==blockDim.y) {// top right
            if(i==-1 || j==ny) shm[sb_x-1+(sb_y+1)*bx] = -1.0;
            else shm[sb_x-1+(sb_y+1)*bx] = d_in[i-1 + (j+1)*nx + k*nx*ny];
        }
    }
    if(sb_x==blockDim.x){ //down
        if(i==nx || j==ny) shm[sb_x+1+(sb_y+0)*bx] = -1.0;
        else shm[sb_x+1+(sb_y+0)*bx] = d_in[i+1 + (j+0)*nx + k*nx*ny];

        if(sb_y==1){ // down left
            if(i==nx) shm[sb_x+1+(sb_y-1)*bx] = -1.0;
            else shm[sb_x+1+(sb_y-1)*bx] = d_in[i+1 + (j-1)*nx + k*nx*ny];
        }
        if(sb_y==blockDim.y){// down right
            if(i==nx || j==ny) shm[sb_x+1+(sb_y+1)*bx] = -1.0;
            else shm[sb_x+1+(sb_y+1)*bx] = d_in[i+1 + (j+1)*nx + k*nx*ny];
        }  
    }
    if (sb_y==1) { //left halo
        if(j==-1||i==-1||i==nx) shm[sb_x+0+(sb_y-1)*bx] = -1.0;
        else   shm[sb_x+0+(sb_y-1)*bx] = d_in[i+0 + (j-1)*nx + k*nx*ny];
    }
    if (sb_y==blockDim.y){//right halo
        if (j==ny||i==-1||i==nx) shm[sb_x+0+(sb_y+1)*bx] = -1.0;
        else shm[sb_x+0+(sb_y+1)*bx] = d_in[i+0 + (j+1)*nx + k*nx*ny];
    }
}

__global__ void jacobi7_35d_halo_test(float *d_in, float *d_out, const int nx, const int ny, const int nz, const float fac, int bx, int by){
    /************************* Test the load of block and halo region ******************/
    /*test parameter 60 60 60 12 12 4 **/
    if (blockIdx.x==bx && blockIdx.y==by)// down right
    //if (blockIdx.x==5 && blockIdx.y==0)// down left
    //if (blockIdx.x==0 && blockIdx.y==0)//top left
    //if (blockIdx.x== 2 && blockIdx.y==5) // right
    //if (blockIdx.x== 0 && blockIdx.y==5) // top right
    //if (blockIdx.x== 1 && blockIdx.y==5) //right
    //if (blockIdx.x== 1 && blockIdx.y==0) //left
    //if (blockIdx.x== 0 && blockIdx.y==1) //top
    //if (blockIdx.x==5 && blockIdx.y==2)//down
        readBlockAndHalo(d_in, 0, nx, ny, d_out);
}

__global__ void jacobi7_35d(float *d_in, float *d_out, const int nx, const int ny, const int nz, const float fac){
    extern __shared__ float sh[];
    const int xy_share_size = (blockDim.x+2) * (blockDim.y+2);

    // shared memory used by k=0~3 layers
    float *k00 = sh;
    float *k10 = sh + xy_share_size;
    float *k20 = k10 + xy_share_size;

    
    // shared memory used by layers after 4
    float *sn1  = k00;
    float *sn2  = k10;
    float *sn3  = k20;
    float *s0   = sn3 + xy_share_size;
    float *s1   = s0 + xy_share_size;
    float *s2   = s1 + xy_share_size;
    float *s3   = s2 + xy_share_size;
    float *s4   = s3 + xy_share_size;
    float *s5   = s4 + xy_share_size;
    
    int i = blockIdx.x * (blockDim.x-2) + threadIdx.x - 1;
    int j = blockIdx.y * (blockDim.y-2) + threadIdx.y - 1;

    //float t11, t21;

    readBlockAndHalo(d_in, 0, nx, ny, k00); // load layer 0
    readBlockAndHalo(d_in, 1, nx, ny, k10); // load layer 1
    readBlockAndHalo(d_in, 2, nx, ny, k20); // load layer 2
    __syncthreads();

    //compute1
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float w,e,n,s,t,b,c;
    int c_s = tx+1 + (blockDim.x+2) * (ty+1);
    int sbx = blockDim.x +2;
    int w_s = c_s - 1;
    int e_s = c_s + 1;
    int n_s = c_s + sbx;
    int s_s = c_s - sbx;
    w = k10[w_s];
    e = k10[e_s];
    n = k10[n_s];
    s = k10[s_s];
    t = k20[c_s];
    b = k00[c_s];
    c = k10[c_s];
    //if(s3[c_s]!=0){
    if(w!=-1.0 && e!=-1.0 && n!=-1.0 && s!=-1.0 && c!=-1.0){
        float t11 = w + e + n + s + t + b - fac * c;
        s3[c_s] = t11;
    }
    else s3[c_s] = c;

    readBlockAndHalo(d_in, 3, nx, ny, s0); // loda layer 3
    __syncthreads();
    w = k20[w_s];
    e = k20[e_s];
    n = k20[n_s];
    s = k20[s_s];
    t = s0[c_s];
    b = k10[c_s];
    c = k20[c_s];
    if(w!=-1.0 && e!=-1.0 && n!=-1.0 && s!=-1.0 && c!=-1.0){
        float t21 = w + e + n + s + t + b - fac * c;
        s4[c_s] = t21;
    }
    else
        s4[c_s] = c;

    readBlockAndHalo(d_in, 4, nx, ny, s1); //load layer 4
    
    __syncthreads();
    //compute 31
    w = s0[w_s];
    e = s0[e_s];
    n = s0[n_s];
    s = s0[s_s];
    t = s1[c_s];
    b = k20[c_s];
    c = s0[c_s];
    if(w!=-1.0 && e!=-1.0 && n!=-1.0 && s!=-1.0 && c!=-1.0){
        float t31 = w + e + n + s + t + b - fac * c;
        s5[c_s] = t31;
    }
    else 
        s5[c_s] = c;
    readBlockAndHalo(d_in, 5, nx, ny, s2); // load layer 5

    // compute 12
    w = s3[w_s];
    e = s3[e_s];
    n = s3[n_s];
    s = s3[s_s];
    t = s4[c_s];
    b = k00[c_s];
    c = s3[c_s];
    if(threadIdx.x > 0 && threadIdx.x < blockDim.x-1 && threadIdx.y>0 && threadIdx.y<blockDim.y-1){
        if(w!=-1.0 && e!=-1.0 && n!=-1.0 && s!=-1.0 && c!=-1.0){     
            float t12 = w + e + n + s + t + b - fac * c;
            d_out[i+nx*j+nx*ny] = t12;    // output to global memory
        }
    }

    //s5 = k31;
    //s4 = k21;
    //s3 = k11;
    // iterate through z dimension
    int k=2;
    for(; k<nz-4; k++){
        //compute1
        __syncthreads();
        w = s1[w_s];
        e = s1[e_s];
        n = s1[n_s];
        s = s1[s_s];
        t = s2[c_s];
        b = s0[c_s];
        c = s1[c_s];
        if(w!=-1.0 && e!=-1.0 && n!=-1.0 && s!=-1.0 && c!=-1.0){
            float tk1 = w + e + n + s + t + b - fac * c;
            sn2[c_s] = tk1;
        }
        else
            sn2[c_s] = c;
        //compute2
        w = s4[w_s];
        e = s4[e_s];
        n = s4[n_s];
        s = s4[s_s];
        t = s5[c_s];
        b = s3[c_s];
        c = s4[c_s];
        if(threadIdx.x > 0 && threadIdx.x < blockDim.x-1 && threadIdx.y>0 && threadIdx.y<blockDim.y-1){
            if(w!=-1.0 && e!=-1.0 && n!=-1.0 && s!=-1.0 && c!=-1.0){
                float t2 = w + e + n + s + t + b - fac * c;
                d_out[i+nx*j+k*nx*ny] = t2;    // output to global memory
            }
        }
        readBlockAndHalo(d_in, k+4, nx, ny, sn1);
        s0 = s1;
        s1 = s2;
        s2 = sn1;
        s3 = s4;
        s4 = s5;
        s5 = sn2;

    }
    k += 1;
    //the last 3 layers
    //1
    //compute1
    w = s1[w_s];
    e = s1[e_s];
    n = s1[n_s];
    s = s1[s_s];
    t = s2[c_s];
    b = s0[c_s];
    c = s1[c_s];
    if(w!=-1.0 && e!=-1.0 && n!=-1.0 && s!=-1.0 && c!=-1.0){
        float tk1 = w + e + n + s + t + b - fac * c;
        sn2[c_s] = tk1;
    }
    else sn2[c_s] = c;
    //compute2
    w = s4[w_s];
    e = s4[e_s];
    n = s4[n_s];
    s = s4[s_s];
    t = s5[c_s];
    b = s3[c_s];
    c = s4[c_s];
    if(threadIdx.x > 0 && threadIdx.x < blockDim.x-1 && threadIdx.y>0 && threadIdx.y<blockDim.y-1){
        if(w!=-1.0 && e!=-1.0 && n!=-1.0 && s!=-1.0 && c!=-1.0){
            float t2 = w + e + n + s + t + b - fac * c;
            d_out[i+nx*j+k*nx*ny] = t2;    // output to global memory
        }
    }

    s0 = s1;
    s1 = s2;
    s2 = sn1;
    s3 = s4;
    s4 = s5;
    s5 = sn2;
    //__syncthreads();
    //2
    //compute2
    k += 1;
    w = s4[w_s];
    e = s4[e_s];
    n = s4[n_s];
    s = s4[s_s];
    t = s5[c_s];
    b = s3[c_s];
    c = s4[c_s];
    if(threadIdx.x > 0 && threadIdx.x < blockDim.x-1 && threadIdx.y>0 && threadIdx.y<blockDim.y-1){
        if(w!=-1.0 && e!=-1.0 && n!=-1.0 && s!=-1.0 && c!=-1.0){
            float t2 = w + e + n + s + t + b - fac * c;
            d_out[i+nx*j+k*nx*ny] = t2;    // output to global memory
        }
    }
    s3 = s4;
    s4 = s5;
    s5 = sn1;
    //__syncthreads();
    //3
    k += 1;
    w = s4[w_s];
    e = s4[e_s];
    n = s4[n_s];
    s = s4[s_s];
    t = s5[c_s];
    b = s3[c_s];
    c = s4[c_s];
    if(threadIdx.x > 0 && threadIdx.x < blockDim.x-1 && threadIdx.y>0 && threadIdx.y<blockDim.y-1){
        if(w!=-1.0 && e!=-1.0 && n!=-1.0 && s!=-1.0 && c!=-1.0){
            float t2 = w + e + n + s + t + b - fac * c;
            d_out[i+nx*j+k*nx*ny] = t2;    // output to global memory
        }
    }

}

