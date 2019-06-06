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


// transpose matrix
void transpose(double x[][NUM_POINTS][2])
{
   double temp;
   for(int i=0; i<NUM_POINTS; i++)
      for(int j=0; j<NUM_POINTS; j++)
      {
         temp = x[i][j][0]; 
         x[i][j][0] = x[j][i][0];
         x[j][i][0] = temp;
         temp = x[i][j][1];
         x[i][j][1] = x[j][i][1];
         x[j][i][1] = temp;
      }
}




int main() {

   // Data and buffer 
   int direction;

   double data_t[NUM_POINTS][NUM_POINTS][2]; // data in time domain
   double data_tt[NUM_POINTS][NUM_POINTS][2]; // data in time domain
   double data_f[NUM_POINTS][NUM_POINTS][2]; // data in freq domain

   // Initialize data in time domain
   srand(time(NULL));
   for(int i=0; i<NUM_POINTS; i++) 
     for(int j=0; j<NUM_POINTS; j++) 
     {
        data_t[i][j][0] = 1;//rand();
        data_t[i][j][1] = 0;//rand();
     }
   cout<<"\ndata_t:\n";
   //display(data_t);


   // 2D FFT on GPU 
   // Apply 1D FFT on each row
   for(int i=0; i<NUM_POINTS; i++)  
      fft_GPU(NUM_POINTS, data_t[i], data_tt[i]);
   transpose(data_tt);
   // Apply 1D FFT on each column
   for(int i=0; i<NUM_POINTS; i++)  
      fft_GPU(NUM_POINTS, data_tt[i], data_f[i]);

   cout << "\ndata_f (GPU):\n"; 
   //display(data_f);


   // 2D FFT on CPU 
   // Apply 1D FFT on each row
   for(int i=0; i<NUM_POINTS; i++)  
      fft_CPU(NUM_POINTS, data_t[i], data_tt[i]);
   transpose(data_tt);
   // Apply 1D FFT on each column
   for(int i=0; i<NUM_POINTS; i++)  
      fft_CPU(NUM_POINTS, data_tt[i], data_f[i]);


   cout << "\ndata_f (CPU):\n"; 
   //display(data_f);


   // 2D FFT on GPU 
   // Apply 1D FFT on each row
   for(int i=0; i<NUM_POINTS; i++)  
      ifft_CPU(NUM_POINTS, data_tt[i], data_t[i]);
   transpose(data_tt);
   // Apply 1D FFT on each column
   for(int i=0; i<NUM_POINTS; i++)  
      ifft_CPU(NUM_POINTS, data_f[i], data_tt[i]);


   
   return 0;
}


