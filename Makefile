.PHONY: all test

all: lucas_kanade.cu
	nvcc -O3 -I common/inc -I "C:\Program Files\jpeg\jpeg-9c" -ljpeg -L "C:\Program Files\jpeg\jpeg-9c\x64\Release" -o lucas_kanade.exe -arch=compute_61 lucas_kanade.cu

test:
	nvprof --log-file nvprof_log.txt lucas_kanade 25 input/sphere_big/frame_0.jpg input/sphere_big/frame_1.jpg test.jpg
