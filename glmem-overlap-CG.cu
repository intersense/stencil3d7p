
#include <stdio.h>
#include <jacobi7_cuda.h>
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void initial_data(float *h_A, float *h_B, const int xyz){
    // randomly generaed test data
    srand(time(NULL));
    int i = 0;
    for(; i < xyz; i++) {
        h_A[i] = 1 + (float)rand() / (float)RAND_MAX;
        h_B[i] =  h_A[i];
    }
}

int main(int argc, char* *argv){
     if(argc != 7) {
        printf("USAGE: %s <NX> <NY> <NZ> <TX> <TY> <TIME STEPS>\n", argv[0]);
        return 1;
    }
    // program parameters trans
    const int nx = atoi(argv[1]);
    const int ny = atoi(argv[2]);
    const int nz = atoi(argv[3]);
    const int tx = atoi(argv[4]);
    const int ty = atoi(argv[5]);
    const int timesteps = atoi(argv[6]);
    
    int devId = 0;
    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, devId));
    printf("Device : %s\n", prop.name);
    checkCuda( cudaSetDevice(devId));

    
    const int xyz = nx * ny * nz;
    const int xyz_bytes = xyz * sizeof(float);
    
    const int nstreams = 4;
    const int streamSize = nz/nstreams; // sliced along z dimension
    const int streamBytes = (streamSize+2) * nx * ny * sizeof(float);

    float *h_A;
    float *h_B;
    float *d_A;
    float *d_B;
    
    // Allocate host buffers
    checkCuda(cudaMallocHost((void**)&h_A, xyz_bytes)); // host pinned
    checkCuda(cudaMallocHost((void**)&h_B, xyz_bytes)); 
    checkCuda(cudaMalloc((void**)&d_A, xyz_bytes)); // device
    checkCuda(cudaMalloc((void**)&d_B, xyz_bytes));

    // grid data iniatialization   
    initial_data(h_A, h_B, xyz);

    float fac = 6.0/(h_A[0] * h_A[0]);

    dim3 grid(nx/tx, ny/ty);
    dim3 block(tx, ty);
    float* input = d_A;
    float* output = d_B;

    float ms; // elapsed time in milliseconds

    // create events and streams
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    cudaStream_t stream[nstreams];
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    checkCuda( cudaEventCreate(&dummyEvent) );
    for (int i = 0; i < nstreams; ++i)
      checkCuda( cudaStreamCreate(&stream[i]) );
    float *tmp;

    // baseline case - sequential transfer and execute
    /*checkCuda( cudaEventRecord(startEvent,0) );
    checkCuda( cudaMemcpy(d_A, h_A, xyz_bytes, cudaMemcpyHostToDevice));
    checkCuda( cudaMemcpy(d_B, d_A, xyz_bytes, cudaMemcpyDeviceToDevice));

    // Run the GPU kernel
    for(int t = 0; t < timesteps; t += 1) {
        jacobi3d_7p_glmem<<<grid, block>>>(input, output, nx, ny, nz, fac);
        // swap input and output
        tmp = input;
        input =  output;
        output = tmp;
    }
    if(timesteps%2==0)
        checkCuda( cudaMemcpy(h_A, output, xyz_bytes, cudaMemcpyDeviceToHost));
    else
        checkCuda( cudaMemcpy(h_A, input, xyz_bytes, cudaMemcpyDeviceToHost));
    checkCuda( cudaEventRecord(stopEvent, 0));
    checkCuda( cudaEventSynchronize(stopEvent));
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent));

    printf("Time for sequential transfer and execute (ms): %f\n", ms);
    */
    // aysnchronous version 1: loop over {copy, kernel, copy}
    initial_data(h_A, h_B, xyz);

    checkCuda( cudaEventRecord(startEvent,0));
    // the first stream
    const int stream0_in_bytes = (streamSize+1) * nx * ny * sizeof(float);
    const int stream0_out_bytes = streamSize * nx * ny * sizeof(float);
    checkCuda( cudaMemcpyAsync(d_A, h_A, stream0_in_bytes, cudaMemcpyHostToDevice, stream[0]));
        // Run the GPU kernel
    for(int t = 0; t < timesteps; t += 1) {
        jacobi3d_7p_glmem<<<grid, block, 0, stream[0]>>>(d_A, d_B, nx, ny, streamSize+1, fac);
        // swap input and output
        tmp = d_A;
        d_A =  d_B;
        d_B = tmp;
    }
    if(timesteps%2==0)
        checkCuda( cudaMemcpyAsync(h_A, d_B, stream0_out_bytes, cudaMemcpyDeviceToHost));
    else
        checkCuda( cudaMemcpyAsync(h_A, d_A, stream0_out_bytes, cudaMemcpyDeviceToHost));

    // the last stream
    const int streaml_in_bytes = (streamSize+1) * nx * ny * sizeof(float);
    const int streaml_out_bytes = streamSize * nx * ny * sizeof(float);
    const int offset_l = (nstreams-1) * streamSize - nx * ny; 
    checkCuda( cudaMemcpyAsync(&d_A[offset_l], &h_A[offset_l], streaml_in_bytes, cudaMemcpyHostToDevice, stream[nstreams-1]));
        // Run the GPU kernel
    input = &d_A[offset_l];
    output = &d_B[offset_l];
    for(int t = 0; t < timesteps; t += 1) {
        jacobi3d_7p_glmem<<<grid, block, 0, stream[nstreams-1]>>>(input, output, nx, ny, streamSize+1, fac);
        // swap input and output
        tmp = input;
        input =  output;
        output = tmp;
    }
    if(timesteps%2==0)
        checkCuda( cudaMemcpyAsync(&h_A[offset_l], output, streaml_out_bytes, cudaMemcpyDeviceToHost, stream[nstreams-1]));
    else
        checkCuda( cudaMemcpyAsync(&h_A[offset_l], input, streaml_out_bytes, cudaMemcpyDeviceToHost, stream[nstreams-1]));

    // the middle stream
    for (int i = 1; i < nstreams-1; ++i){ 
        int offset = (i * streamSize - 1) * nx * ny;
        checkCuda( cudaMemcpyAsync(&d_A[offset], &h_A[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]));
        // Run the GPU kernel
        input = &d_A[offset];
        output = &d_B[offset];
        for(int t = 0; t < timesteps; t += 1) {
            jacobi3d_7p_glmem<<<grid, block, 0, stream[i]>>>(input, output, nx, ny, streamSize+2, fac);
            // swap input and output
            tmp = input;
            input =  output;
            output = tmp;
        }
        if(timesteps%2==0)
            checkCuda( cudaMemcpyAsync(&h_A[offset + nx*ny], output+nx*ny, streamSize*nx*ny*sizeof(float), cudaMemcpyDeviceToHost, stream[i]));
        else
            checkCuda(  cudaMemcpyAsync(&h_A[offset + nx*ny], input + nx*ny, streamSize*nx*ny*sizeof(float), cudaMemcpyDeviceToHost, stream[i]));
    }
    checkCuda( cudaEventRecord(stopEvent, 0));
    checkCuda( cudaEventSynchronize(stopEvent));
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time for aysnchronous V1 transfer and execute (ms): %f\n", ms);

    double gflop = (xyz * 1e-9) * 7.0 * timesteps;
    double gflop_per_sec = gflop * 1e3 / ms;
    printf("(GPU) %lf GFlop/s\n", gflop_per_sec);
    double mupdate_per_sec = ((xyz >> 20) * timesteps) * 1e3 / ms;
    printf("(GPU) %lf M updates/s\n", mupdate_per_sec);
    // asynchronous version 2:

    // cleanup
    checkCuda( cudaEventDestroy(startEvent));
    checkCuda( cudaEventDestroy(stopEvent));
    checkCuda(cudaEventDestroy(dummyEvent));
    for (int i=0; i < nstreams; ++i)
        checkCuda( cudaStreamDestroy(stream[i]));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    return 0;
}
