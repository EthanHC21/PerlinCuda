#include "CUDAUtils.cuh"
#include "Utils.hpp"

__host__ void cudaStructarrcpy(void **d_arr, void *h_arr, size_t arrBytes, void **structLocPtr) {
    // malloc array on device
    cudaMalloc(d_arr, arrBytes);

    // copy the contents of the array to the device
    cudaMemcpy(*d_arr, h_arr, arrBytes, cudaMemcpyHostToDevice);

    // set the pointer of the device struct to that of the copied array
    cudaMemcpy(structLocPtr, d_arr, sizeof(void *), cudaMemcpyHostToDevice);
}

__device__ int ipowCuda(int base, int exp) {
    int result = 1;
    for (;;) {
        if (exp & 1) result *= base;
        exp >>= 1;
        if (!exp) break;
        base *= base;
    }

    return result;
}

__device__ void linterp2(uint i, CudaData *cudaData, volatile double *f_x, volatile double *f_y) {
    // extract coords
    double x = cudaData->particles->xPos[i];
    double y = cudaData->particles->yPos[i];

    // printf("i: %d, x: %.6f, y: %.6f\n", i, x, y);

    // get the row and col of the particle
    double row = y / cudaData->field->scl;
    double col = x / cudaData->field->scl;
    double nextRow = fmod(row + 1, (double)cudaData->simInfo->rows);
    double nextCol = fmod(col + 1, (double)cudaData->simInfo->cols);

    // indices
    uint idx1 = cudaData->simInfo->cols * ((uint)row) + ((uint)col);
    uint idx2 = cudaData->simInfo->cols * ((uint)nextRow) + ((uint)col);
    uint idx3 = cudaData->simInfo->cols * ((uint)row) + ((uint)nextCol);
    uint idx4 = cudaData->simInfo->cols * ((uint)nextRow) + ((uint)nextCol);
    // get angles
    double angle1 = cudaData->field->field[idx1];
    double angle2 = cudaData->field->field[idx2];
    double angle3 = cudaData->field->field[idx3];
    double angle4 = cudaData->field->field[idx4];

    // printf("angle1: %.6f, angle2: %.6f, angle3: %.6f, angle4: %.6f\n", angle1, angle2, angle3, angle4);

    // avoid doing triginometric calculations multiple times
    double v1x = cos(angle1);
    double v1y = sin(angle1);
    double v2x = cos(angle2);
    double v2y = sin(angle2);
    double v3x = cos(angle3);
    double v3y = sin(angle3);
    double v4x = cos(angle4);
    double v4y = sin(angle4);

    // inerp along the row for the left column
    double v5x = (v2x - v1x) * (row - (uint)row) + v1x;
    double v5y = (v2y - v1y) * (row - (uint)row) + v1y;

    // interp along the row for the right column
    double v6x = (v4x - v3x) * (row - (uint)row) + v3x;
    double v6y = (v4y - v3y) * (row - (uint)row) + v3y;

    // interp along the column of the interped vectors
    *f_x = (v6x - v5x) * (col - (uint)col) + v5x;
    *f_y = (v6y - v5y) * (col - (uint)col) + v5y;
}

__device__ void getAcceleration(uint i, CudaData *cudaData, volatile double *a_x, volatile double *a_y) {
    // forces
    volatile double f_x, f_y;

    // get force from the field
    linterp2(i, cudaData, &f_x, &f_y);

    // apply field strength and convert from grids to pixels
    f_x *= cudaData->field->fieldStrength * cudaData->field->scl;
    f_y *= cudaData->field->fieldStrength * cudaData->field->scl;

    // printf("i: %d, f_x: %.6f, f_y: %.6f\n", i, f_x, f_y);

    // drag force
    // printf("x_vel: %.6f, y_vel: %.6f\n", cudaData->particles->xVel[i], cudaData->particles->yVel[i]);
    double mag = sqrt(pow(cudaData->particles->xVel[i], 2) + pow(cudaData->particles->yVel[i], 2));
    double f_drag_x = cudaData->particles->xVel[i] * mag * -1 * cudaData->particles->C_d;
    double f_drag_y = cudaData->particles->yVel[i] * mag * -1 * cudaData->particles->C_d;
    // printf("mag: %.6f, f_drag_x: %.6f, f_drag_y: %.6f\n", mag, f_drag_x, f_drag_y);

    // calculate the acceleration
    *a_x = (f_x + f_drag_x) / cudaData->particles->m;
    *a_y = (f_y + f_drag_y) / cudaData->particles->m;
}

__global__ void verlet(CudaData *cudaData) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;

    // check we're in range of the arrays
    if (i < cudaData->particles->numParticles) {
        // acceleration variables
        volatile double a_x, a_y;

        // get the acceleration
        getAcceleration(i, cudaData, &a_x, &a_y);

        // printf("i: %d, a_x: %.6f, a_y: %.6f\n", i, a_x, a_y);

        // calculate v_(t/2)
        cudaData->particles->xVel[i] += a_x * .5;
        cudaData->particles->yVel[i] += a_y * .5;

        // apply boundary conditions
        cudaData->particles->xPos[i] = fmod(cudaData->particles->xPos[i] + cudaData->particles->xVel[i] + (double)cudaData->simInfo->width, (double)cudaData->simInfo->width);
        cudaData->particles->yPos[i] = fmod(cudaData->particles->yPos[i] + cudaData->particles->yVel[i] + (double)cudaData->simInfo->height, (double)cudaData->simInfo->height);

        // INCREMENT THE ACCUMULATOR
        uint x = cudaData->particles->xPos[i];
        uint y = cudaData->particles->yPos[i];
        uint idx = x + y * cudaData->simInfo->width;
        // printf("idx for add: %d\n", idx);
        // ↓ actually fine, so long as we compile with -arch=sm_86 (which I did)
        atomicAdd((double *)&cudaData->imageData->count[idx], 1.0);
        atomicAdd((double *)&cudaData->imageData->countSorted[idx], 1.0);

        // get the acceleration
        getAcceleration(i, cudaData, &a_x, &a_y);

        // calculate v_(t/2)
        cudaData->particles->xVel[i] += a_x * .5;
        cudaData->particles->yVel[i] += a_y * .5;

        // printf("verlet iteration complete\n");
    }
}