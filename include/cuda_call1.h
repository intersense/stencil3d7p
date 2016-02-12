#define CHECK_CALL(err)   checkCudaCall(err, __FILE__, __LINE__)
//#define SYNC_DEVICE()     syncDevice(__FILE__, __LINE__)
#define ASSERT_STATE(msg) checkCudaState(msg, __FILE__, __LINE__)

inline void checkCudaCall(cudaError   err,
                          const char *file,
                          const int   line) {
  /*if(cudaSuccess != err) {
    std::cerr << file << "(" << line << ") :  checkCudaCall failed - " <<
      cudaGetErrorString(err) << std::endl;
    //assert(false && "Cuda error");
    */
    if(cudaSuccess != err) {
        printf("(<< %d << ): checkCudaCall failed - %s\n", line, cudaGetErrorString(err));
        exit(-1);
    }
  }

/*inline void syncDevice(const char *file,
                       const int   line) {
  //cudaError err = cudaDeviceSynchronize();
  //CUresult err  = cuCtxSynchronize();
  //checkCudaCall(err, file, line);

  //cudaThreadSynchronize();
    cudaDeviceSynchronize();
}*/

inline void checkCudaState(const char *errorMessage,
                           const char *file,
                           const int   line) {
  cudaError_t err = cudaGetLastError();
  /*if(cudaSuccess != err) {
    std::cerr << errorMessage << std::endl;
    checkCudaCall(err, file, line);
  }*/
    if(cudaSuccess != err){
        printf("%s\n",errorMessage);
        checkCudaCall(err,file,line);
    }
}