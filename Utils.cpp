#include "Utils.hpp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <tiffio.h>
#include <time.h>
#include <unistd.h>

#include "MergeSort.hpp"
#include "open-simplex-noise-in-c/open-simplex-noise.h"

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

void writeTiff16BW(volatile uint16_t *img, char *name, uint32_t width, uint32_t height) {
    uint16_t *image = (uint16_t *)img;

    TIFF *out = TIFFOpen(name, "w");

    int samplesPerPixel = 1;

    TIFFSetField(out, TIFFTAG_IMAGEWIDTH,
                 width);  // set the width of the image
    TIFFSetField(out, TIFFTAG_IMAGELENGTH,
                 height);  // set the height of the image
    TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL,
                 samplesPerPixel);  // set number of channels per pixel
    TIFFSetField(out, TIFFTAG_BITSPERSAMPLE,
                 16);  // set the size of the channels
    TIFFSetField(out, TIFFTAG_ORIENTATION,
                 ORIENTATION_TOPLEFT);  // set the origin of the image.
    //   Some other essential fields to set that you do not have to
    //   understand for now.
    TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

    tsize_t linebytes = width * samplesPerPixel;

    uint16_t *buf = (uint16_t *)_TIFFmalloc(linebytes * sizeof(uint16_t));

    for (uint row = 0; row < height; row++) {
        memcpy(buf, &(image[row * linebytes]), linebytes * sizeof(uint16_t));
        TIFFWriteScanline(out, buf, row, 0);
    }

    TIFFClose(out);

    if (buf) _TIFFfree(buf);
}

void writeTiff16RGB(volatile uint16_t *img, char *name, uint32_t width, uint32_t height) {
    uint16_t *image = (uint16_t *)img;

    TIFF *out = TIFFOpen(name, "w");

    int samplesPerPixel = 3;

    // uint32_t *image = calloc(width * height * samplesPerPixel,
    // sizeof(uint16_t)); for (uint32_t i = 0; i < height * width *
    // samplesPerPixel; i++) {
    //     image[i] = i;
    // }

    TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);                 // set the width of the image
    TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);               // set the height of the image
    TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);  // set number of channels per pixel
    TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 16);                 // set the size of the channels
    TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);  // set the origin of the image.
    //   Some other essential fields to set that you do not have to understand for now.
    TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

    tsize_t linebytes = width * samplesPerPixel;

    uint16_t *buf = (uint16_t *)_TIFFmalloc(linebytes * sizeof(uint16_t));

    for (uint row = 0; row < height; row++) {
        memcpy(buf, &(image[row * linebytes]), linebytes * sizeof(uint16_t));
        TIFFWriteScanline(out, buf, row, 0);
    }

    TIFFClose(out);

    if (buf) _TIFFfree(buf);
}

double randDouble(double min, double max) {
    return (double)rand() / RAND_MAX * (max - min) + min;
}

void *initializeParticles(void *ptr) {
    // fix argument
    ThreadData *threadData = (ThreadData *)ptr;

    for (uint i = threadData->startIdx;
         i < (threadData->num + threadData->startIdx); i++) {
        // threadData->particles->xPos[i] = randDouble(0, threadData->simInfo->width);
        // threadData->particles->yPos[i] = randDouble(0, threadData->simInfo->height);
        // velocities are already set to 0

        threadData->particles->valid[i] = 1;

        uint y = i / threadData->simInfo->width;
        uint x = i % threadData->simInfo->width;
        threadData->particles->xPos[i] = x + .5;
        threadData->particles->yPos[i] = y + .5;
    }

    return NULL;
}

int ipowCpu(int base, int exp) {
    int result = 1;
    for (;;) {
        if (exp & 1) result *= base;
        exp >>= 1;
        if (!exp) break;
        base *= base;
    }

    return result;
}

void linterp2Cpu(ThreadData *threadData, uint i) {
    // extract coords
    double x = threadData->particles->xPos[i];
    double y = threadData->particles->yPos[i];

    // get the row and col of the particle
    double row = y / threadData->field->scl;
    double col = x / threadData->field->scl;
    double nextRow = row + 1;
    if (nextRow >= threadData->simInfo->rows) {
        nextRow -= threadData->simInfo->rows;
    }
    double nextCol = col + 1;
    if (nextCol >= threadData->simInfo->cols) {
        nextCol -= threadData->simInfo->cols;
    }

    // indices
    uint idx1 = (threadData->simInfo->cols) * ((uint)row) + ((uint)col);
    uint idx2 = (threadData->simInfo->cols) * ((uint)nextRow) + ((uint)col);
    uint idx3 = (threadData->simInfo->cols) * ((uint)row) + ((uint)nextCol);
    uint idx4 = (threadData->simInfo->cols) * ((uint)nextRow) + ((uint)nextCol);
    // get angles
    double angle1 = threadData->field->field[idx1];
    double angle2 = threadData->field->field[idx2];
    double angle3 = threadData->field->field[idx3];
    double angle4 = threadData->field->field[idx4];

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
    threadData->f_field[0] = (v6x - v5x) * (col - (uint)col) + v5x;
    threadData->f_field[1] = (v6y - v5y) * (col - (uint)col) + v5y;
}

void getAccelerationCpu(ThreadData *threadData, uint i) {
    // field force
    linterp2Cpu(threadData, i);

    // apply field strength and convert from grids to pixels
    threadData->f_field[0] *= threadData->field->fieldStrength * threadData->field->scl;
    threadData->f_field[1] *= threadData->field->fieldStrength * threadData->field->scl;

    // drag force
    double mag = sqrt(pow(threadData->particles->xVel[i], 2) +
                      pow(threadData->particles->yVel[i], 2));
    double f_drag[2] = {threadData->particles->xVel[i] * mag * -1 * threadData->particles->C_d, threadData->particles->yVel[i] * mag * -1 * threadData->particles->C_d};

    threadData->a_tot[0] = (threadData->f_field[0] + f_drag[0]) / threadData->particles->m;
    threadData->a_tot[1] = (threadData->f_field[1] + f_drag[1]) / threadData->particles->m;
}

void *verletCpu(void *ptr) {
    // fix argument
    ThreadData *threadData = (ThreadData *)ptr;

    for (uint i = threadData->startIdx; i < (threadData->startIdx + threadData->num); i++) {
        if (threadData->particles->valid[i] == 1) {
            // get acceleration due to drag and field
            getAccelerationCpu(threadData, i);

            // calculate v_(t/2)
            threadData->particles->xVel[i] += threadData->a_tot[0] * .5;
            threadData->particles->yVel[i] += threadData->a_tot[1] * .5;

            // calculate x_t
            threadData->particles->xPos[i] += threadData->particles->xVel[i];
            threadData->particles->yPos[i] += threadData->particles->yVel[i];

            // check if particle is out of bounds
            if (threadData->particles->xPos[i] < 0 ||
                threadData->particles->xPos[i] >= threadData->simInfo->width ||
                threadData->particles->yPos[i] < 0 ||
                threadData->particles->yPos[i] >= threadData->simInfo->height) {
                // invalid (or maybe replace - we'll experiment!)
                threadData->particles->valid[i] = 0;
            } else {
                // update the accumulator
                uint x = threadData->particles->xPos[i];
                uint y = threadData->particles->yPos[i];

                uint idx = x + y * threadData->simInfo->width;

                // pthread_mutex_lock(threadData->locks + idx);
                threadData->imageData->count[idx] += 1;
                threadData->imageData->countSorted[idx] += 1;
                // pthread_mutex_unlock(threadData->locks + idx);
            }

            // check valid again
            if (threadData->particles->valid[i] == 1) {
                // recalculate acceleration
                getAccelerationCpu(threadData, i);
            }

            // calculate v_t
            threadData->particles->xVel[i] += threadData->a_tot[0] * .5;
            threadData->particles->yVel[i] += threadData->a_tot[1] * .5;
        }
    }

    return NULL;
}

void *updateVelocities(void *ptr) {
    // fix argument
    ThreadData *threadData = (ThreadData *)ptr;

    for (uint i = threadData->startIdx; i < (threadData->startIdx + threadData->num); i++) {
        uint y = i / threadData->simInfo->cols;
        uint x = i - y * threadData->simInfo->cols;

        double xNoise = x * threadData->field->scl * threadData->field->noiseScl + threadData->field->scl / 2.0;
        double yNoise = y * threadData->field->scl * threadData->field->noiseScl + threadData->field->scl / 2.0;

        threadData->field->field[i] = open_simplex_noise3(threadData->simInfo->ctx, xNoise, yNoise, threadData->simInfo->t) * 2 * M_PI;
        // printf("x: %d, y: %d, noise: %.3lf\n", x, y, simData->field[i]);
    }

    return NULL;
}

void *updateCurl(void *ptr) {
    // fix argument
    ThreadData *threadData = (ThreadData *)ptr;

    double noiseScl = threadData->field->noiseScl;
    double scl = threadData->field->scl;
    double epsilon = threadData->field->epsilon;
    struct osn_context *ctx = threadData->simInfo->ctx;
    double t = threadData->simInfo->t;

    for (uint i = threadData->startIdx; i < (threadData->startIdx + threadData->num); i++) {
        uint y = i / threadData->simInfo->cols;
        uint x = i - y * threadData->simInfo->cols;

        double xNoise = x * scl * noiseScl + scl / 2.0;
        double yNoise = y * scl * noiseScl + scl / 2.0;

        double xFinal =
            (open_simplex_noise3(ctx, xNoise, yNoise + epsilon, t) -
             open_simplex_noise3(ctx, xNoise, yNoise - epsilon, t)) /
            (2 * epsilon);

        double yFinal =
            (open_simplex_noise3(ctx, xNoise - epsilon, yNoise, t) -
             open_simplex_noise3(ctx, xNoise + epsilon, yNoise, t)) /
            (2 * epsilon);

        threadData->field->field[i] = atan2(yFinal, xFinal);
    }

    // TODO: REMOVE DIVERGENCE

    return NULL;
}

void drawParticles(ImageData *imageData, SimInfo *simInfo, Particles *particles) {
    // image is black
    for (uint i = 0; i < simInfo->width * simInfo->height; i++) {
        imageData->img[i] = 0;
    }

    uint val = ipowCpu(2, 16) - 1;

    if (imageData->writeName == NULL) {
        // file name
        imageData->writeName = (char *)malloc(strlen(imageData->imgName) + 12 + 1);

        // copy the file name to the write name
        strcpy(imageData->writeName, imageData->imgName);
    }

    // write the frame num
    sprintf(imageData->writeName + strlen(imageData->imgName), "_%05d.tiff", simInfo->frameNum);

    // white at each particle
    for (uint i = 0; i < (particles->numParticles); i++) {
        uint idx = ((uint)particles->xPos[i]) + ((uint)particles->yPos[i]) * simInfo->width;
        imageData->img[idx] = val;
    }

    writeTiff16BW(imageData->img, imageData->writeName, simInfo->width, simInfo->height);
}

void drawField(Field *field, SimInfo *simInfo) {
    uint16_t *write = (uint16_t *)malloc(simInfo->rows * simInfo->cols * sizeof(uint16_t));
    uint val = ipowCpu(2, 16) - 1;
    double max = field->field[0];
    double min = field->field[0];
    for (uint i = 1; i < simInfo->rows * simInfo->cols; i++) {
        if (field->field[i] > max) {
            max = field->field[i];
        }
        if (field->field[i] < min) {
            min = field->field[i];
        }
    }
    double range = max - min;
    for (uint i = 0; i < simInfo->rows * simInfo->cols; i++) {
        write[i] = (uint16_t)(val * ((field->field[i] - min) / range));
    }

    writeTiff16BW(write, (char *)"out/noise.tiff", simInfo->cols, simInfo->rows);
    free(write);
}

double linterp(double x0, double y0, double x1, double y1, double x) {
    double m = (y1 - y0) / (x1 - x0);
    return x * m + y0;
}

double normLinterp(double y0, double y1, double x) {
    return x * (y1 - y0) + y0;
}

void writeAccumulator(SimInfo *simInfo, ImageData *imageData) {
    uint val = ipowCpu(2, 16) - 1;

    // number of pixels
    uint numPixels = simInfo->width * simInfo->height;

    // write array
    uint16_t *write = (uint16_t *)malloc(sizeof(uint16_t) * numPixels * 3);

    printf("numPixels: %u\n", numPixels);

    // sort it
    // TODO: FIX THIS SEGFAULTING
    mergeSort(imageData->countSorted, 0, numPixels - 1);

    // multiplicative factor of max position in sorted array
    double attenuation;
    switch (imageData->colorMap) {
        case 0:
            attenuation = .97;
            break;
        case 1:
            attenuation = 1.0;
            break;
        case 2:
            attenuation = 1.0;
            break;
        case 3:
            attenuation = 1.0;
            break;
        case 4:
            attenuation = 1.0;
            break;
        default:
            exit(EXIT_FAILURE);
    }

    double min = imageData->countSorted[0];
    double max = imageData->countSorted[(uint)(.97 * (numPixels - 1))];

    double range = max - min;

    for (int i = 0; i < numPixels; i++) {
        uint tiffIdx = i * 3;

        double normVal = (imageData->count[i] - min) / range;
        normVal = normVal > 1.0 ? 1.0 : normVal;
        normVal = sqrt(normVal);

        switch (imageData->colorMap) {
            case 0:  // black and white
                write[tiffIdx] = (uint16_t)(normVal * val);
                write[tiffIdx + 1] = (uint16_t)(normVal * val);
                write[tiffIdx + 2] = (uint16_t)(normVal * val);
                break;
            case 1:  // purple
                write[tiffIdx] = (uint16_t)(normVal * (180.0 / 255.0) * val);
                write[tiffIdx + 1] = (uint16_t)((20.0 / 255.0) * val);
                write[tiffIdx + 2] = (uint16_t)(normVal * (225.0 / 255.0) * val);
                break;
            case 2:  // blue
                write[tiffIdx] = (uint16_t)(normVal * (20.0 / 255.0) * val);
                write[tiffIdx + 1] = (uint16_t)(normVal * (180.0 / 255.0) * val);
                write[tiffIdx + 2] = (uint16_t)(normVal * (225.0 / 255.0) * val);
                break;
            case 3:  // wewow
                write[tiffIdx] = (uint16_t)(normVal * (200.0 / 255.0) * val);
                write[tiffIdx + 1] = (uint16_t)(normVal * (180.0 / 255.0) * val);
                write[tiffIdx + 2] = (uint16_t)(normVal * (90.0 / 255.0) * val);
                break;
            case 4: {
                double c2_r = (125.0 / 255.0);
                double c2_g = (186.0 / 255.0);
                double c2_b = (182.0 / 255.0);

                double c1_r = (46.0 / 255.0);
                double c1_g = (59.0 / 255.0);
                double c1_b = (65.0 / 255.0);

                write[tiffIdx] = (uint16_t)(normLinterp(c1_r, c2_r, normVal) * val);
                write[tiffIdx + 1] = (uint16_t)(normLinterp(c1_g, c2_g, normVal) * val);
                write[tiffIdx + 2] = (uint16_t)(normLinterp(c1_b, c2_b, normVal) * val);
                break;
            }
            default:
                printf("ERROR: colormap undefined\n");
                exit(EXIT_FAILURE);
        }
    }

    writeTiff16RGB(write, (char *)"out/accumulator.tiff", simInfo->width, simInfo->height);

    free(write);
}