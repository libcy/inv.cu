#!/bin/bash
nvcc src/inv.cu --std=c++11 -arch=sm_50 -lcublas -lcusolver -lcufft -o inv.out
