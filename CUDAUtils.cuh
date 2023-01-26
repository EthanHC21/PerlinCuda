#include "Utils.hpp"

__host__ void cudaStructarrcpy(void **d_arr, void *h_arr, size_t arrBytes, void **structLocPtr);

__device__ int ipowCuda(int base, int exp);

__device__ void linterp2(uint i, CudaData *cudaData, volatile double *f_x, volatile double *f_y);

__device__ void getAcceleration(uint i, CudaData *cudaData, volatile double *a_x, volatile double *a_y);

__global__ void verlet(CudaData *cudaData);