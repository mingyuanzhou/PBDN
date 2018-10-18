
#include <mex.h>
/*#include "cokus.c"
#define RAND_MAX_32 4294967295.0
//#define MAX(a,b) ((a) > (b) ? a : b)*/



void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) {
    mwIndex i, j;
    double *L, *r, *x;
    mwIndex Msize, Nsize; 
    
    
    x = mxGetPr(prhs[0]);
    r = mxGetPr(prhs[1]);
    
    Msize = mxGetM(prhs[0]);
    Nsize = mxGetN(prhs[0]);
    
    
    plhs[0] = mxCreateDoubleMatrix(Msize, Nsize, mxREAL);
    L = mxGetPr(plhs[0]);
    
    for (j=0;j<Msize*Nsize;j++) {
        for(i=0;i< (mwIndex) x[j];i++) {
            /*if  ((double) randomMT() <= (r[j]/(r[j]+i) *RAND_MAX_32))*/
            if  (((double) rand() / RAND_MAX) <= (r[j]/(r[j]+ i)))
                L[j]++;
        }
    }
}

