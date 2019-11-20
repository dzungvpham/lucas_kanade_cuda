all: lucas_kanade.cu
	nvcc -O3 -I common/inc -I "C:\Program Files\jpeg\jpeg-9c" -ljpeg -L "C:\Program Files\jpeg\jpeg-9c\x64\Release" -o lucas_kanade.exe -arch=compute_61 lucas_kanade.cu
