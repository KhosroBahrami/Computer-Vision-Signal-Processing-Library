#!/bin/bash
  
g++ -o main fft.cpp fft_gpu_wrapper.cpp main.cpp -lOpenCL -lm `pkg-config --cflags --libs opencv`

#g++ -o main fft.cpp main.cpp `pkg-config --cflags --libs opencv`





