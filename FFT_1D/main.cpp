//----------------------------------------------------------------------------
// Main file to run FFT on GPU & CPU 
//

// Number of FFT points
#define NUM_POINTS 16
/* 1 - forward FFT, -1 - inverse FFT */
#define DIRECTION 1

#include <iostream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>

using namespace std;


extern "C" void fft_CPU(int N, double xt[][2], double xf[][2]);
extern "C" void ifft_CPU(int N, double xt[][2], double xf[][2]);
extern "C" void fft_GPU(int N, double xt[][2], double xf[][2]);


void display(double x[][2])
{
   for(int i=0 ; i<NUM_POINTS ; ++i)
      cout << x[i][0] << " ";
   cout << endl;
}





int main() {

      // Data and buffer 
   int direction;

   double data_t[NUM_POINTS][2]; // data in time domain
   double data_f[NUM_POINTS][2]; // data in freq domain

   // Initialize data in time domain
   srand(time(NULL));
   for(int i=0; i<NUM_POINTS; i++) {
      data_t[i][0] = 1;//rand();
      data_t[i][1] = 0;//rand();
   }
   cout<<"\ndata_t:\n";
   display(data_t);

   // calculate fft on GPU  
   fft_GPU(NUM_POINTS, data_t, data_f);

   cout << "\ndata_f (GPU):\n"; 
   display(data_f);

   // calculate fft on CPU
   fft_CPU(NUM_POINTS, data_t, data_f);
   
   cout << "\ndata_f (CPU):\n"; 
   display(data_f);

   // calculate ifft on CPU
   //ifft_CPU(NUM_POINTS, data_f, data_t);

   
   return 0;
}


