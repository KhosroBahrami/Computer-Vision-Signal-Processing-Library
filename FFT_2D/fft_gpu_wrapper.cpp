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



// Find a GPU or CPU associated with the first available platform 
cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   // Identify a platform 
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   // Access a device 
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   return dev;
}



// Create program from a file and compile it 
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   // Read program file and place content into buffer 
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   // Create program from file 
   program = clCreateProgramWithSource(ctx, 1, 
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   // Build program 
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      // Find size of log and print to std output 
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}




extern "C" void fft_GPU(int N, double xt[][2], double xf[][2])
{
  
   // Host/device data structures 
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel init_kernel, stage_kernel, scale_kernel;
   cl_int err, i;
   size_t global_size, local_size;
   cl_ulong local_mem_size;
   

   // Data and buffer 
   int direction;
   unsigned int num_points, points_per_group, stage;
   //float data[NUM_POINTS*2];
   double error;
   double data_t[NUM_POINTS][2]; // data in time domain
   double data_f[NUM_POINTS][2]; // data in freq domain
   cl_mem data_buffer;

   // Initialize data in time domain
   srand(time(NULL));
   for(int i=0; i<NUM_POINTS; i++) {
      data_t[i][0] = xt[i][0];//rand();
      data_t[i][1] = xt[i][1];//rand();
   }
   
   // Create a device and context 
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   // Build the program 
   program = build_program(context, device, "fft.cl");

   // Create kernels for the FFT 
   init_kernel = clCreateKernel(program, "fft_init", &err);
   if(err < 0) {
      printf("Couldn't create the initial kernel: %d", err);
      exit(1);
   };
   stage_kernel = clCreateKernel(program, "fft_stage", &err);
   if(err < 0) {
      printf("Couldn't create the stage kernel: %d", err);
      exit(1);
   };
   scale_kernel = clCreateKernel(program, "fft_scale", &err);
   if(err < 0) {
      printf("Couldn't create the scale kernel: %d", err);
      exit(1);
   };

   // Create buffer 
   data_buffer = clCreateBuffer(context, 
         CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 2*NUM_POINTS*sizeof(float), data_t, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);
   };

   // Determine maximum work-group size 
   err = clGetKernelWorkGroupInfo(init_kernel, device, 
      CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_size), &local_size, NULL);
   if(err < 0) {
      perror("Couldn't find the maximum work-group size");
      exit(1);   
   };
   local_size = (int)pow(2, trunc(log2((float)local_size)));

   // Determine local memory size
   err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, 
      sizeof(local_mem_size), &local_mem_size, NULL);
   if(err < 0) {
      perror("Couldn't determine the local memory size");
      exit(1);   
   };

   // Initialize kernel arguments 
   direction = DIRECTION;
   num_points = NUM_POINTS;
   points_per_group = local_mem_size/(2*sizeof(float));
   if(points_per_group > num_points)
      points_per_group = num_points;

   // Set kernel arguments
   err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &data_buffer);
   err |= clSetKernelArg(init_kernel, 1, local_mem_size, NULL);
   err |= clSetKernelArg(init_kernel, 2, sizeof(points_per_group), &points_per_group);
   err |= clSetKernelArg(init_kernel, 3, sizeof(num_points), &num_points);
   err |= clSetKernelArg(init_kernel, 4, sizeof(direction), &direction);
   if(err < 0) {
      printf("Couldn't set a kernel argument");
      exit(1);   
   };

   // Create a command queue 
   queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   // Enqueue initial kernel 
   global_size = (num_points/points_per_group)*local_size;
   err = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, &global_size, 
                                &local_size, 0, NULL, NULL); 
   if(err < 0) {
      perror("Couldn't enqueue the initial kernel");
      exit(1);
   }

   // Enqueue further stages of the FFT 
   if(num_points > points_per_group) {

      err = clSetKernelArg(stage_kernel, 0, sizeof(cl_mem), &data_buffer);
      err |= clSetKernelArg(stage_kernel, 2, sizeof(points_per_group), &points_per_group);
      err |= clSetKernelArg(stage_kernel, 3, sizeof(direction), &direction);
      if(err < 0) {
         printf("Couldn't set a kernel argument");
         exit(1);   
      };
      for(stage = 2; stage <= num_points/points_per_group; stage <<= 1) {
         clSetKernelArg(stage_kernel, 1, sizeof(stage), &stage);
         err = clEnqueueNDRangeKernel(queue, stage_kernel, 1, NULL, &global_size, 
                                      &local_size, 0, NULL, NULL); 
         if(err < 0) {
            perror("Couldn't enqueue the stage kernel");
            exit(1);
         }
      }
   }

   // Scale values if performing the inverse FFT 
   if(direction < 0) {
      err = clSetKernelArg(scale_kernel, 0, sizeof(cl_mem), &data_buffer);
      err |= clSetKernelArg(scale_kernel, 1, sizeof(points_per_group), &points_per_group);
      err |= clSetKernelArg(scale_kernel, 2, sizeof(num_points), &num_points);
      if(err < 0) {
         printf("Couldn't set a kernel argument");
         exit(1);   
      };
      err = clEnqueueNDRangeKernel(queue, scale_kernel, 1, NULL, &global_size, 
                                   &local_size, 0, NULL, NULL); 
      if(err < 0) {
         perror("Couldn't enqueue the initial kernel");
         exit(1);
      }
   }

   // Read the results 
   err = clEnqueueReadBuffer(queue, data_buffer, CL_TRUE, 0, 
         2*NUM_POINTS*sizeof(float), data_f, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);   
   }
   

   // Deallocate resources 
   clReleaseMemObject(data_buffer);
   clReleaseKernel(init_kernel);
   clReleaseKernel(stage_kernel);
   clReleaseKernel(scale_kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   for(int i=0; i<NUM_POINTS; i++) {
      xf[i][0] = data_f[i][0];
      xf[i][1] = data_f[i][1];
   }   
   

}


