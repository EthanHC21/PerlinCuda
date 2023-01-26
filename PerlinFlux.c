#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <tiffio.h>
#include <unistd.h>

#include "Utils.h"
#include "open-simplex-noise.h"

// threading things
ulong NUM_CPUS;
pthread_t *threads;
ThreadData *threadArgs;
pthread_mutex_t *locks;

// structs
SimInfo simInfo;
Field field;
Particles particles;
ImageData imageData;
ThreadData *threadData;

void tommyinnit() {
    // THREADING

    // define number of cores
    NUM_CPUS = sysconf(_SC_NPROCESSORS_ONLN);

    // INITIALIZE ALL CONSTANT PARAMETERS IN THE STRUCTS

    // SimInfo
    simInfo.frameNum = 0;         // current frame of the simulation
    simInfo.t = 0;                // current time
    simInfo.height = 800;         // final height dimension
    simInfo.width = 800;          // final width dimension
    simInfo.frameHold = 0;        // wait this many frames before drawing particles
    simInfo.frameInterval = 600;  // wait this many frames between draws

    // Field
    field.epsilon = .003;      // used in curl differentiation - increase to remove
                               // linear artifacts
    field.scl = .5;            // scale of the field grid
    field.timestep = .0005;    // increase of z noise coordinate per interval
    field.noiseScl = .0025;    // scale of noise used to generate field
    field.fieldStrength = .5;  // strength of the field's acceleration
    field.updateInterval = 0;  // number of frames between field updates

    // Particles
    particles.C_d = .7;              // drag coefficient for particles (v^2)
    particles.m = 1.0;               // mass of each particle
    particles.numParticles = 10000;  // can be calculated later

    // ImageData
    imageData.imgName = "out/particles\0";
    imageData.writeName = NULL;
}

void setup() {
    // engage random number generator
    srand(time(NULL));

    // THREADS

    // thread array (we only have to calculate it once!)
    threads = malloc(NUM_CPUS * sizeof(pthread_t *));
    // thread arguments
    threadArgs = malloc(NUM_CPUS * sizeof(ThreadData));

    // INITIALIZE NON-CONSTANT STRUCT FIELDS

    // SIMINFO
    // simplex
    open_simplex_noise(rand(), &simInfo.ctx);
    // field dimensions
    simInfo.rows = (uint)(simInfo.height / field.scl);
    simInfo.cols = (uint)(simInfo.width / field.scl);

    // FIELD
    // allocate field
    field.field = malloc(simInfo.rows * simInfo.cols * sizeof(double));

    // PARTICLES
    // number (1 per pixel)
    particles.numParticles = simInfo.height * simInfo.width;
    // physics
    particles.xPos = malloc(particles.numParticles * sizeof(double));
    particles.yPos = malloc(particles.numParticles * sizeof(double));
    particles.xVel = calloc(particles.numParticles, sizeof(double));
    particles.yVel = calloc(particles.numParticles, sizeof(double));
    // colors
    particles.colors = calloc(particles.numParticles, sizeof(uint64_t));
    // all start valid (we gotta fix)
    particles.valid = malloc(particles.numParticles * sizeof(uint));

    // IMAGEDATA
    // actual image
    imageData.img = calloc(simInfo.height * simInfo.width, sizeof(uint16_t));
    // counts
    imageData.count = calloc(simInfo.width * simInfo.height, sizeof(double));
    imageData.countSorted =
        calloc(simInfo.width * simInfo.height, sizeof(double));
    imageData.maxCount = 0.0;

    // THREADDATA
    // locks
    locks = malloc(simInfo.height * simInfo.width * sizeof(pthread_mutex_t));
    for (uint i = 0; i < simInfo.height * simInfo.width; i++) {
        pthread_mutex_init(locks + i, NULL);
    }

    for (uint i = 0; i < NUM_CPUS; i++) {
        // array used in verlet
        threadArgs[i].f_field = malloc(2 * sizeof(double));
        threadArgs[i].a_tot = malloc(2 * sizeof(double));
        // copy all of the structs
        threadArgs[i].simInfo = &simInfo;
        threadArgs[i].field = &field;
        threadArgs[i].particles = &particles;
        threadArgs[i].imageData = &imageData;
        // locks
        threadArgs[i].locks = locks;
        // thread index
        threadArgs[i].threadId = i;
    }

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
        pthread_create(threads + i, NULL, initializeParticles,
                       (void *)(threadArgs + i));
    }

    // S Y N C
    for (uint i = 0; i < NUM_CPUS; i++) {
        pthread_join(threads[i], NULL);
    }

    // TODO: TEST THIS - IT MAY NOT WORK
    drawParticles(&imageData, &simInfo, &particles);
    simInfo.frameNum++;
}

void doUpdate() {
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
        pthread_create(threads + i, NULL, updateVelocities,
                       (void *)(threadArgs + i));
    }

    // S Y N C
    for (uint i = 0; i < NUM_CPUS; i++) {
        pthread_join(threads[i], NULL);
    }
}

uint draw() {
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

    // num per threads
    uint numPerThread = particles.numParticles / NUM_CPUS;

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
        pthread_create(threads + i, NULL, verlet, (void *)(threadArgs + i));
    }

    // S Y N C
    for (uint i = 0; i < NUM_CPUS; i++) {
        pthread_join(threads[i], NULL);
    }

    if (simInfo.frameNum >= simInfo.frameHold) {
        if (simInfo.frameInterval == 0 ||
            simInfo.frameNum % simInfo.frameInterval == 0) {
            printf("Drawing for frame %d\n", simInfo.frameNum);
            drawParticles(&imageData, &simInfo, &particles);
        }
    }

    simInfo.t += field.timestep;
    simInfo.frameNum++;

    // 99999 frames max because arrays
    if (simInfo.frameNum == 3600) {
        writeAccumulator(&simInfo, &imageData);
        printf("Program complete!\n");
        return 0;
    }
    return 1;
}

int main() {
    // remove old files
    if (system((char *)"rm -rf out/*") != 0) {
        printf("failed to remove old image files\n");
        exit(-1);
    }

    // TIMING
    struct timeval start, end;
    long mtime, secs, usecs;

    tommyinnit();
    setup();

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