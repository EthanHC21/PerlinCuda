#include <stdio.h>

__global__ void exampleCuda() {
    printf("Hello from GPU!\n");
}

int main() {
    printf("Hello from CPU!\n");

    exampleCuda<<<1, 1>>>();

    cudaDeviceSynchronize();

    printf("Program complete\n");

    return EXIT_SUCCESS;
}