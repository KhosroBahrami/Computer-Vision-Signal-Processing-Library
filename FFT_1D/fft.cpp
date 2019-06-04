//----------------------------------------------------------------------------
// Fast Fourier Transform implementation in C++
//
// FFT
// fft_CPU(N, xt, xf);  
// inputs:             
//     N: number of points in FFT (must equal 2^n for some integer n >= 1)
//     xt: N time-domain samples given in rectangular form (Re x, Im x)
// output: 
//     xf: N frequency-domain samples calculated in rectangular form (Re X, Im X) 
//
// IFFT
// ifft_CPU(N, xt, xf);  
// inputs:             
//     N: number of points in FFT (must equal 2^n for some integer n >= 1)
//     xf: N frequency-domain samples given in rectangular form (Re X, Im X) 
// output: 
//     xt: N time-domain samples calculated in rectangular form (Re x, Im x)
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359 

using namespace std;

void fft(int N, int offset, int delta, double xt[][2], double xf[][2], double x[][2]);





// FFT 
extern "C" void fft_CPU(int N, double xt[][2], double xf[][2])
{
  // Declare array of result values
  double x[N][2]; 
  // Calculate FFT 
  fft(N, 0, 1, xt, xf, x);
}



// IFFT 
extern "C" void ifft_CPU(int N, double xt[][2], double xf[][2])
{
  int N2 = N/2;      // half number of points in IFFT 
  int i;              
  double tmp0, tmp1;  
  // Calculate IFFT via reciprocity property of DFT
  fft_CPU(N, xf, xt);
  xt[0][0] = xt[0][0]/N; // Re part 
  xt[0][1] = xt[0][1]/N; // Im part
  xt[N2][0] = xt[N2][0]/N;  
  xt[N2][1] = xt[N2][1]/N;
  for(i=1; i<N2; i++)
    {
      tmp0 = xt[i][0]/N;       
      tmp1 = xt[i][1]/N;
      xt[i][0] = xt[N-i][0]/N;  
      xt[i][1] = xt[N-i][1]/N;
      xt[N-i][0] = tmp0;       
      xt[N-i][1] = tmp1;
    }
}



// FFT calculation via recursion
void fft(int N, int offset, int delta, double xt[][2], double xf[][2], double x[][2])
{
  int N2 = N/2;    // half number of points in FFT 
  int k;                    
  double cs, sn;   // cosine and sine 
  int k00, k01, k10, k11;   
  double tmp0, tmp1;       

  if(N != 2)  // Perform recursive step
    {
      // Calculate two (N/2)-point DFT's
      fft(N2, offset, 2*delta, xt, x, xf);
      fft(N2, offset+delta, 2*delta, xt, x, xf);

      // Combine the two (N/2)-point DFT's into one N-point DFT
      for(k=0; k<N2; k++)
        {
          k00 = offset + k*delta; 
          k01 = k00 + N2*delta;
          k10 = offset + 2*k*delta;
          k11 = k10 + delta;
          cs = cos(2*PI*k/(double)N); 
          sn = sin(2*PI*k/(double)N);
          tmp0 = cs * x[k11][0] + sn * x[k11][1];
          tmp1 = cs * x[k11][1] - sn * x[k11][0];
          xf[k01][0] = x[k10][0] - tmp0;
          xf[k01][1] = x[k10][1] - tmp1;
          xf[k00][0] = x[k10][0] + tmp0;
          xf[k00][1] = x[k10][1] + tmp1;
        }
    }
  else  // Perform 2-point DFT 
    {
      k00 = offset; 
      k01 = k00 + delta;
      xf[k01][0] = xt[k00][0] - xt[k01][0];
      xf[k01][1] = xt[k00][1] - xt[k01][1];
      xf[k00][0] = xt[k00][0] + xt[k01][0];
      xf[k00][1] = xt[k00][1] + xt[k01][1];
    }
}





