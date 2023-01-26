normal:
	gcc PerlinFlux.c open-simplex-noise-in-c/open-simplex-noise.c Utils.c MergeSort.c -I open-simplex-noise-in-c/ -ltiff -Wall -lm -lpthread -Ofast -o perlinflux_c

cuda:
	nvcc PerlinFluxCUDA.cu open-simplex-noise-in-c/open-simplex-noise.c Utils.cpp CUDAUtils.cu MergeSort.cpp -I open-simplex-noise-in-c/ -ltiff -lm -arch=sm_61 -O2 -o perlinflux_cuda

testtiff:
	g++ test.cpp Utils.cpp open-simplex-noise-in-c/open-simplex-noise.c MergeSort.cpp -I open-simplex-noise-in-c/ -ltiff -lm -o test

testtiff_debug:
	g++ test.cpp Utils.cpp open-simplex-noise-in-c/open-simplex-noise.c MergeSort.cpp -I open-simplex-noise-in-c/ -ltiff -lm -g -o test
