#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <tiffio.h>
#include <unistd.h>

#include "CUDATest.cuh"
#include "CUDAUtils.cuh"
#include "Utils.hpp"
#include "open-simplex-noise.h"

// threading things
ulong NUM_CPUS;
pthread_t *threads;
ThreadData *threadArgs;
pthread_mutex_t *locks;

// structs and device structs
SimInfo simInfo;
// no array fields for CUDA

Field field;
// CUDA fields
double *d_field_field;

Particles particles;
// CUDA fields
volatile double *d_particles_xPos;
volatile double *d_particles_yPos;
volatile double *d_particles_xVel;
volatile double *d_particles_yVel;
volatile uint64_t *d_particles_colors;
volatile uint *d_particles_valid;

ImageData imageData;
// CUDA fields
volatile double *d_imageData_count;
volatile double *d_imageData_countSorted;

// CUDA struct
CudaData cudaData;
volatile SimInfo *d_simInfo;
volatile Field *d_field;
volatile Particles *d_particles;
volatile ImageData *d_imageData;
// on device
CudaData *d_cudaData;

// CUDA THINGS
uint NUM_BLOCKS;
// 1 warp
uint THREADS_PER_BLOCK = 32;

__host__ void tommyinnit() {
    // THREADING
    // define number of cores
    NUM_CPUS = sysconf(_SC_NPROCESSORS_ONLN);

    // INITIALIZE ALL CONSTANT PARAMETERS IN THE STRUCTS

    // SimInfo
    simInfo.frameNum = 0;         // current frame of the simulation
    simInfo.t = 0;                // current time
    simInfo.height = 1080;        // final height dimension
    simInfo.width = 1920;         // final width dimension
    simInfo.frameHold = 0;        // wait this many frames before drawing particles
    simInfo.frameInterval = 600;  // wait this many frames between draws

    // Field
    field.epsilon = .003;      // used in curl differentiation - increase to remove linear artifacts
    field.scl = 1;             // scale of the field grid
    field.timestep = .0005;    // increase of z noise coordinate per interval
    field.noiseScl = .0025;    // scale of noise used to generate field
    field.fieldStrength = .5;  // strength of the field's acceleration
    field.updateInterval = 0;  // number of frames between field updates

    // Particles
    particles.C_d = .7;  // drag coefficient for particles (v^2)
    particles.m = 1.0;   // mass of each particle
    // particles.numParticles = 10000;  // default - may or may not be calculated later

    // ImageData
    imageData.imgName = (char *)"out/particles\0";
    imageData.writeName = NULL;
    imageData.colorMap = 2;
}

__host__ void setup() {
    // engage random number generator
    srand(time(NULL));

    // THREADS

    // thread array (we only have to calculate it once!)
    threads = (pthread_t *)malloc(NUM_CPUS * sizeof(pthread_t *));
    // thread arguments
    threadArgs = (ThreadData *)malloc(NUM_CPUS * sizeof(ThreadData));

    // INITIALIZE NON-CONSTANT STRUCT FIELDS

    // SIMINFO
    // simplex
    open_simplex_noise(rand(), &simInfo.ctx);
    // field dimensions
    simInfo.rows = (uint)(simInfo.height / field.scl);
    simInfo.cols = (uint)(simInfo.width / field.scl);

    // FIELD
    // allocate field
    field.field = (double *)malloc(simInfo.rows * simInfo.cols * sizeof(double));

    // PARTICLES
    // number (1 per pixel)
    particles.numParticles = simInfo.height * simInfo.width;
    // physics
    particles.xPos = (double *)malloc(particles.numParticles * sizeof(double));
    particles.yPos = (double *)malloc(particles.numParticles * sizeof(double));
    particles.xVel = (double *)calloc(particles.numParticles, sizeof(double));
    particles.yVel = (double *)calloc(particles.numParticles, sizeof(double));
    // colors
    particles.colors = (uint64_t *)calloc(particles.numParticles, sizeof(uint64_t));
    // all start valid (we gotta fix)
    particles.valid = (uint *)malloc(particles.numParticles * sizeof(uint));
    NUM_BLOCKS = (particles.numParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // IMAGEDATA
    // actual image
    imageData.img = (uint16_t *)calloc(simInfo.height * simInfo.width, sizeof(uint16_t));
    // counts
    imageData.count = (double *)calloc(simInfo.width * simInfo.height, sizeof(double));
    imageData.countSorted = (double *)calloc(simInfo.width * simInfo.height, sizeof(double));
    imageData.maxCount = 0.0;

    // THREADDATA
    // locks - not needed, we atomic increment in the cuda thread thing
    locks = (pthread_mutex_t *)malloc(simInfo.height * simInfo.width * sizeof(pthread_mutex_t));
    for (uint i = 0; i < simInfo.height * simInfo.width; i++) {
        pthread_mutex_init(locks + i, NULL);
    }

    for (uint i = 0; i < NUM_CPUS; i++) {
        // array used in verlet - nope, each core passes two double* things
        threadArgs[i].f_field = (double *)malloc(2 * sizeof(double));
        threadArgs[i].a_tot = (double *)malloc(2 * sizeof(double));

        // set all of the struct pointers
        threadArgs[i].simInfo = &simInfo;
        threadArgs[i].field = &field;
        threadArgs[i].particles = &particles;
        threadArgs[i].imageData = &imageData;

        // locks - not used in CUDA
        threadArgs[i].locks = locks;
        // thread index
        threadArgs[i].threadId = i;
    }

    // set the pointers for CUDA
    cudaData.simInfo = &simInfo;
    cudaData.field = &field;
    cudaData.particles = &particles;
    cudaData.imageData = &imageData;

    // num per threads
    uint numPerThread = particles.numParticles / NUM_CPUS;

    printf("Initializing %u particles\n", particles.numParticles);

    // initialize the particles
    for (uint i = 0; i < NUM_CPUS; i++) {
        // start is always the same
        threadArgs[i].startIdx = i * numPerThread;

        // we always have at least numPerThread
        threadArgs[i].num = numPerThread;

        // for the last one, add the remainder (< NUM_CPUS)
        if (i == (NUM_CPUS - 1)) {
            threadArgs[i].num += particles.numParticles % NUM_CPUS;
        }

        // start the thread
        pthread_create(threads + i, NULL, initializeParticles, (void *)(threadArgs + i));
    }

    // S Y N C
    for (uint i = 0; i < NUM_CPUS; i++) {
        pthread_join(threads[i], NULL);
    }

    // draw the particles
    drawParticles(&imageData, &simInfo, &particles);
    simInfo.frameNum++;
}

__host__ void doUpdate() {
    printf("Update called...\n");

    // num per threads
    uint numPerThread = (simInfo.rows * simInfo.cols) / NUM_CPUS;

    // create the noise field
    for (uint i = 0; i < NUM_CPUS; i++) {
        // start is always the same
        threadArgs[i].startIdx = i * numPerThread;

        // we always have at least numPerThread
        threadArgs[i].num = numPerThread;

        // for the last one, add the remainder (< NUM_CPUS)
        if (i == (NUM_CPUS - 1)) {
            threadArgs[i].num += particles.numParticles % NUM_CPUS;
        }

        // start the thread
        pthread_create(threads + i, NULL, updateCurl, (void *)(threadArgs + i));
    }

    // S Y N C
    for (uint i = 0; i < NUM_CPUS; i++) {
        pthread_join(threads[i], NULL);
    }

    // copy updated field to device
    cudaMemcpy((void *)d_field_field, (void *)field.field, simInfo.rows * simInfo.cols * sizeof(double), cudaMemcpyHostToDevice);
}

__host__ uint draw() {
    if ((field.updateInterval != 0) &&
        (simInfo.frameNum % field.updateInterval == 0)) {
        doUpdate();
    } else if (field.updateInterval == 0 && simInfo.frameNum == 1) {
        doUpdate();
    }

    if (simInfo.frameNum == 1) {
        printf("Writing noise field...\n");
        drawField(&field, &simInfo);
    }
    // printf("verlet for frame %d\n", simInfo.frameNum);
    verlet<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_cudaData);
    // S Y N C
    cudaDeviceSynchronize();

    if (simInfo.frameNum >= simInfo.frameHold) {
        if (simInfo.frameInterval == 0 || simInfo.frameNum % simInfo.frameInterval == 0) {
            printf("Drawing for frame %d\n", simInfo.frameNum);
            // printf("numParticles: %d\n", particles.numParticles);
            // printf("d_particles_xPos: %p\n", d_particles_xPos);
            // printf("particles.xPos: %p\n", particles.xPos);
            cudaMemcpy((void *)particles.xPos, (void *)d_particles_xPos, particles.numParticles * sizeof(double), cudaMemcpyDeviceToHost);
            // printf("d_particles_yPos: %p\n", d_particles_yPos);
            // printf("particles.yPos: %p\n", particles.yPos);
            cudaMemcpy((void *)particles.yPos, (void *)d_particles_yPos, particles.numParticles * sizeof(double), cudaMemcpyDeviceToHost);
            // printf("cudaMemcpy complete\n");
            drawParticles(&imageData, &simInfo, &particles);
            // printf("draw complete\n");
        }
    }

    simInfo.t += field.timestep;
    simInfo.frameNum++;

    // 99999 frames max because arrays
    if (simInfo.frameNum == 3600) {
        printf("Writing accumulator for frame %d\n", simInfo.frameNum);
        cudaMemcpy((void *)imageData.count, (void *)d_imageData_count, simInfo.width * simInfo.height * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)imageData.countSorted, (void *)d_imageData_countSorted, simInfo.width * simInfo.height * sizeof(double), cudaMemcpyDeviceToHost);
        // printf("cudaMemcpy complete\n");
        writeAccumulator(&simInfo, &imageData);
        printf("Program complete!\n");
        return 0;
    }
    return 1;
}

__host__ void initCuda() {
    uint printStuff = 0;

    // SIMINFO
    // malloc struct
    if (printStuff) {
        printf("malloc SimInfo\n");
    }
    cudaMalloc(&d_simInfo, sizeof(SimInfo));
    // copy struct
    if (printStuff) {
        printf("copy SimInfo\n");
    }
    cudaMemcpy((void *)d_simInfo, &simInfo, sizeof(SimInfo), cudaMemcpyHostToDevice);

    // FIELD
    // malloc struct
    if (printStuff) {
        printf("malloc Field\n");
    }
    cudaMalloc(&d_field, sizeof(Field));
    // copy struct
    if (printStuff) {
        printf("copy Field\n");
    }
    cudaMemcpy((void *)d_field, &field, sizeof(Field), cudaMemcpyHostToDevice);
    // copy fields
    if (printStuff) {
        printf("copy field.field\n");
    }
    cudaStructarrcpy((void **)&d_field_field, (void *)field.field, simInfo.rows * simInfo.cols * sizeof(double), (void **)&(d_field->field));

    // PARTICLES
    // malloc struct
    if (printStuff) {
        printf("malloc Particles\n");
    }
    cudaMalloc(&d_particles, sizeof(Particles));
    // copy struct
    if (printStuff) {
        printf("copy particles\n");
    }
    cudaMemcpy((void *)d_particles, &particles, sizeof(Particles), cudaMemcpyHostToDevice);
    // copy fields
    // particles.xPos
    if (printStuff) {
        printf("copy particles.xPos\n");
    }
    cudaStructarrcpy((void **)&d_particles_xPos, (void *)particles.xPos, particles.numParticles * sizeof(double), (void **)&(d_particles->xPos));
    // particles.yPos
    if (printStuff) {
        printf("copy particles.yPos\n");
    }
    cudaStructarrcpy((void **)&d_particles_yPos, (void *)particles.yPos, particles.numParticles * sizeof(double), (void **)&(d_particles->yPos));
    // particles.xVel
    if (printStuff) {
        printf("copy particles.xVel\n");
    }
    cudaStructarrcpy((void **)&d_particles_xVel, (void *)particles.xVel, particles.numParticles * sizeof(double), (void **)&(d_particles->xVel));
    // particles.yVel
    if (printStuff) {
        printf("copy particles.yVel\n");
    }
    cudaStructarrcpy((void **)&d_particles_yVel, (void *)particles.yVel, particles.numParticles * sizeof(double), (void **)&(d_particles->yVel));
    // particles.colors
    if (printStuff) {
        printf("copy particles.colors\n");
    }
    cudaStructarrcpy((void **)&d_particles_colors, (void *)particles.colors, particles.numParticles * sizeof(uint64_t), (void **)&(d_particles->colors));
    // particles.valid
    if (printStuff) {
        printf("copy particles.valid\n");
    }
    cudaStructarrcpy((void **)&d_particles_valid, (void *)particles.valid, particles.numParticles * sizeof(uint), (void **)&(d_particles->valid));

    // IMAGEDATA
    // malloc struct
    if (printStuff) {
        printf("malloc imageData\n");
    }
    cudaMalloc(&d_imageData, sizeof(ImageData));
    // copy struct
    if (printStuff) {
        printf("copy imageData\n");
    }
    cudaMemcpy((void *)d_imageData, &imageData, sizeof(ImageData), cudaMemcpyHostToDevice);
    // copy fields
    // imageData.count
    if (printStuff) {
        printf("copy imageData.count\n");
    }
    cudaStructarrcpy((void **)&d_imageData_count, (void *)imageData.count, simInfo.width * simInfo.height * sizeof(double), (void **)&(d_imageData->count));
    // imageData.countSorted
    if (printStuff) {
        printf("copy imageData.countSorted\n");
    }
    cudaStructarrcpy((void **)&d_imageData_countSorted, (void *)imageData.countSorted, simInfo.width * simInfo.height * sizeof(double), (void **)&(d_imageData->countSorted));

    // CUDADATA
    // malloc struct
    if (printStuff) {
        printf("malloc cudaData\n");
    }
    cudaMalloc(&d_cudaData, sizeof(CudaData));
    // copy struct
    if (printStuff) {
        printf("copy cudaData\n");
    }
    cudaMemcpy((void *)d_cudaData, &cudaData, sizeof(CudaData), cudaMemcpyHostToDevice);
    // copy pointers - TODO: maybe method? add it in the cudaStructarrcpy
    // cudaData.simInfo
    if (printStuff) {
        printf("copy simInfo*\n");
    }
    cudaMemcpy(&(d_cudaData->simInfo), &d_simInfo, sizeof(SimInfo *), cudaMemcpyHostToDevice);
    // cudaData.field
    if (printStuff) {
        printf("copy field*\n");
    }
    cudaMemcpy(&(d_cudaData->field), &d_field, sizeof(Field *), cudaMemcpyHostToDevice);
    // cudaData.particles
    if (printStuff) {
        printf("copy particles*\n");
    }
    cudaMemcpy(&(d_cudaData->particles), &d_particles, sizeof(Particles *), cudaMemcpyHostToDevice);
    // cudaData.imageData
    if (printStuff) {
        printf("copy imageData*\n");
    }
    cudaMemcpy(&(d_cudaData->imageData), &d_imageData, sizeof(ImageData *), cudaMemcpyHostToDevice);
}

__host__ int main() {
    // TODO:
    // - implement helmholtz
    // - draw accumulator every frame
    // - arguments to control field type and color pallette
    // - maybe a cleanup/reinitalize method to launch from here?

    // remove old files
    if (system((char *)"rm -rf out/*") != 0) {
        printf("failed to remove old image files\n");
        exit(-1);
    }

    // change stack size
    setStackSize(16);

    // TIMING
    struct timeval start, end;
    long mtime, secs, usecs;

    // initialize values
    tommyinnit();
    // setup
    setup();

    printf("Initializing CUDA\n");
    // copy to CUDA
    initCuda();

    printf("Engaging simulation with %d blocks...\n", NUM_BLOCKS);
    // START STAMP
    gettimeofday(&start, NULL);
    uint cont = 1;
    while (cont) {
        cont = draw();
    }

    // END STAMP
    gettimeofday(&end, NULL);
    secs = end.tv_sec - start.tv_sec;
    usecs = end.tv_usec - start.tv_usec;
    mtime = ((secs)*1000 + usecs / 1000.0) + 0.5;

    printf("Elapsed time: %ld millisecs\n", mtime);
    return EXIT_SUCCESS;
}