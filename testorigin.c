//testorigin.c
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "getopt.h"
#include "include/jacobi7.h"
//void jacobi7(const int nx,const int ny, int nz, float* A0,const int timesteps,const float* B,const int ldb, float* Anext,const int ldc);

// Timer function
double rtclock(){
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}

int main(int argc, char* *argv)
{
    if(argc != 5) {
        printf("USAGE: %s <NX> <NY> <NZ> <TIME STEPS>\n", argv[0]);
        return 1;
    }
    const int nx = atoi(argv[1]);
    const int ny = atoi(argv[2]);
    const int nz = atoi(argv[3]);
    const int timesteps = atoi(argv[4]);
    float * a ;
    float * anext;
    float * B  = 0;
    const int ldb = 0;
    const int ldc = 0;
    const int xyz = nx * ny * nz;
    const int xyz_byetes = xyz * sizeof(float);
    a = (float*) malloc(xyz_byetes);
    anext = (float*) malloc(xyz_byetes);
    //int randomSeed = time(NULL);
    srand(time(NULL));
    int i = 0;
    for(; i < xyz; i++) 
    {
        a[i] = (float)rand() / (float)RAND_MAX;
        anext[i] = a[i];
    }
    printf("A[%d]:%f\n", 2+32*(3+32*4), a[2+32*(3+32*4)]);
    printf("Anext[%d]:%f\n", 2+32*(3+32*4),anext[2+32*(3+32*4)]);
    double start = rtclock();
    int t=0;
    for (; t<timesteps; ++t){
        jacobi7(nx, ny, nz,  a, B, ldb, anext, ldc);
    }
    
    double end = rtclock();
    double timespan = end - start;
    printf("Jacobi7 running time:%lf\n", timespan);

    printf("A[%d]:%f\n", 2+32*(3+32*4),a[2+32*(3+32*4)]);
    printf("Anext[%d]:%f\n", 2+32*(3+32*4),anext[2+32*(3+32*4)]);
    free(a);
    free(anext);
    return 0;
}

