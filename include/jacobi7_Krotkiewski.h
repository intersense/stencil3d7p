__global__ void jacobi7_Krotkiewski(float *d_in, float *d_out, const int nx, const int ny, const int nz, const float fac){
    extern __shared__ float shm[];//shared memory holds the data block and the halo
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    float w,e,n,s,t,b,c;// values in neighouring grid points are explilcitly read from shm to the registers before computations

    float t1, t2; // intermediate stencil results
    readBlockAndHalo(d_in, 0, tx, ty, shm); //read block of data from 0th XY-subplane
    
    shm2regs(shm, tx, ty); //copy required thread data from shm to registers

    readBlockAndHalo(d_in, 1, tx, ty, shm); //initiate reading of the next block of data into shm
    t1 = computeStencil_plane0();   // compute stencil using data available in register

    shm2regs(shm, tx, ty);
    readBlockAndHalo(d_in, 2, tx, ty, shm);
    t2 = computeStencil_plane0();
    t1 += computeStencil_plane1();

    d_out += ix + iy*nx;
    for(int k=3; k<nz-1; k++){
        shm2regs(shm, tx, ty);
        readBlockAndHalo(d_in, k, tx, ty, shm);

        d_out = d_out + nx*ny;
        d_out[0] = t1 + computeStencil_plane2();
        t1 = t2 + computeStencil_plane1();
        t2 = computeStencil_plane0();
    }
    shm2regs(shm, tx, ty);
    d_out[0] = t1 + computeStencil_plane2();
}
__device__ void shm2regs(float *shm, int tx, int ty){

}
__device__ void readBlockAndHalo(float *d_in, int k, int tx, int ty, float *shm){
    int bx = (blockDim.x + 2);
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    shm[tx+0+(ty+0)*bx] = d_in[ix + iy*nx + k*nx*ny];
    if(tx==1){
                            shm[tx-1+(ty+0)*bx] = d_in[ix-1 + (iy+0)*nx + k*nx*ny];
        if(ty==1)           shm[tx-1+(ty-1)*bx] = d_in[ix-1 + (iy-1)*nx + k*nx*ny];
        if(ty==blockDim.y)  shm[tx-1+(ty+1)*bx] = d_in[ix-1 + (iy+1)*nx + k*nx*ny];
    }
    if(tx==blockDim.x){
                            shm[tx+1+(ty+0)*bx] = d_in[ix+1 + (iy+0)*nx + k*nx*ny];
        if(ty==1)           shm[tx+1+(ty-1)*bx] = d_in[ix+1 + (iy-1)*nx + k*nx*ny];
        if(ty==blockDim.y)  shm[tx+1+(ty+1)*bx] = d_in[ix+1 + (iy+1)*nx + k*nx*ny];
    }
    if (ty==1)              shm[tx+0+(ty-1)*bx] = d_in[ix+0 + (iy-1)*nx + k*nx*ny];
    if (ty==blockDim.y)     shm[tx+0+(ty+1)*bx] = d_in[ix+0 + (iy+1)*nx + k*nx*ny];
}