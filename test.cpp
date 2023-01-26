#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <unistd.h>

#include "MergeSort.hpp"
#include "Utils.hpp"

void setStackSize(int sizeMB) {
    const rlim_t kStackSize = sizeMB * 1024 * 1024;  // min stack size = 16 MB
    struct rlimit rl;
    int result;

    result = getrlimit(RLIMIT_STACK, &rl);
    if (result == 0) {
        if (rl.rlim_cur < kStackSize) {
            rl.rlim_cur = kStackSize;
            result = setrlimit(RLIMIT_STACK, &rl);
            if (result != 0) {
                fprintf(stderr, "setrlimit returned result = %d\n", result);
                exit(EXIT_FAILURE);
            }
        }
    }
}

int run() {
    if (system("rm -rf out/test.tiff") != 0) {
        printf("failed to remove old file\n");
        exit(EXIT_FAILURE);
    }

    uint32_t width = 1080;
    uint32_t height = 1350;
    int samplesPerPixel = 3;
    uint num = width * height;

    uint16_t* data = (uint16_t*)calloc(width * height, sizeof(uint16_t));
    for (uint64_t i = 0; i < height * width; i++) {
        data[i] = (uint16_t)i;
    }

    uint16_t* image = (uint16_t*)calloc(width * height * 3, sizeof(uint16_t));
    for (uint64_t i = 0; i < height * width; i++) {
        image[3 * i] = 65535;
        image[3 * i + 1] = 65535;
        image[3 * i + 2] = 65535;
    }

    printf("Allocating array\n");

    double* count = (double*)calloc(num, sizeof(double));

    printf("count[num-1]: %.6f\n", count[num - 1]);

    printf("Beginning mergeSort\n");

    mergeSort(count, 0, num - 1);

    printf("image[0]: %u, image[1]: %u, image[2]: %u\n", image[0], image[1], image[2]);

    writeTiff16RGB(image, (char*)"out/test.tiff", width, height);

    return EXIT_SUCCESS;
}

int main() {
    setStackSize(512);

    return run();
}