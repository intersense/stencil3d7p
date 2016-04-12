void jacobi7(const int nx,const int ny, const int nz, float* A0, float* Anext, float fac) 
{
  //float *temp_ptr;
  int i, j, k;   
  //float *l0, *lnext;
  //float fac = 6.0/(A0[0]*A0[0]);
  //float *tmp;
//  for (t = 0; t < timesteps; t++) {
    /*if (t%2 == 0) { l0 = A0; lnext = Anext; }
    else {lnext = A0; l0 = Anext; }*/
    for (k = 1; k < nz - 1; k++) {
      for (j = 1; j < ny - 1; j++) {
        for (i = 1; i < nx - 1; i++) {
            Anext[i+nx*(j+ ny*k)] = 
                A0[i+nx*(j+ ny*( k + 1))] +
                A0[i+nx*(j+ ny*( k - 1))] +
                A0[i+nx*((j + 1)+ ny*k)] +
                A0[i+nx*((j - 1)+ ny*k)] +
                A0[(i+1)+nx*(j+ ny*k)] +
                A0[(i - 1)+nx*(j+ ny*k)]
                -  A0[i+nx*(j+ ny*k)] *fac ;                
        }
      }
    }
//    tmp = A0;
//    A0 = Anext;
//   Anext = tmp;
//  }
}