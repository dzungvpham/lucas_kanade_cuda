# A C CUDA implementation of the Lucas Kanade Optical Flow Algorithm

This repo contains CPU and GPU CUDA C code for calculating optical flow using the Lucas-Kanade method. The code was compiled and tested with CUDA 10.0 and `jpeg-9c` on an NVIDIA GeForce GTX 1070 CPU (compute capability 6.1). This is the final project for CS 338: Parallel Processing - Williams College.

## Compilation

To compile, you will need CUDA 10.0 and the jpeg library, which can be downloaded from http://www.ijg.org/files/. Once you have the libraries in place, edit the `Makefile` so that the proper `inc` and `lib` paths to the jpeg library are included. Also, change the target architecture argument depending on the compute capability of your GPU. If you are on Unix, use `Makefile.unix` instead, which should hopefully take care of everything.

## Run

To run, type:

```lucas_kanade <WINDOW_SIZE> <BLOCK_SIZE> <PATH_TO_FRAME_1> <PATH_TO_FRAME_2> <PATH_TO_FLOW_VISUALIZATION>```

Example:

```lucas_kanade 25 32 input/sphere_big_odd/frame_0.jpg input/sphere_big_odd/frame_1.jpg out.jpg```

By default, the program will run the horizontal tiled GPU kernel with no pitch only once. You can configure which kernel to run, whether or not to use pitch, whether or not to test against CPU, the number of runs, etc., by modifying the appropriate constants in the source code and recompiling. Note that this hassle is done intentionally since the other options only serve to slow the program down. Therefore, there is no neat command-line arguments for these configurations. The output is HSV representation of the flow.

## Profiler

Use `nvprof` to profile the code. See the Makefile for window for the commands. It's recommended to change the number of run in the source code to get more stable average time.

## Report

See `final_report.pdf` for detailed descriptions of the algorithm and the program.

## Bugs?

Feel free to open GitHub issues.
