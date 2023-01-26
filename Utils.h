#ifndef UTILS_HEADER
#define UTILS_HEADER

#include <pthread.h>
#include <tiffio.h>

#include "open-simplex-noise.h"

typedef unsigned int uint;
typedef unsigned long ulong;

typedef struct {
    struct osn_context *ctx;  // simplex noise thing
    volatile double t;        // time variable
    volatile uint frameNum;   // current frame of the simulation
    uint rows;                // number of rows (field)
    uint cols;                // number of cols (field)
    uint height;              // figure it out
    uint width;               // figure it out
    uint frameHold;           // wait this many frames before drawing
    uint frameInterval;       // wait this many frames between draws
} SimInfo;

typedef struct {
    volatile double *field;  // actual field
    double epsilon;          // used in curl increase this to remove linear artifacts
    double scl;              // scale of the field grid
    double timestep;         // to increase in Z
    double noiseScl;         // scale of noise
    double fieldStrength;    // strength of the field
    uint updateInterval;     // number of frames between updates
} Field;

typedef struct {
    volatile double *xPos;      // particle x position
    volatile double *yPos;      // particle y position
    volatile double *xVel;      // particle x vel
    volatile double *yVel;      // particle y vel
    volatile uint64_t *colors;  // long color of each particle (not used)
    volatile uint *valid;       // store whether particles are valid
    double C_d;                 // drag coefficient
    double m;                   // mass of each particle
    uint numParticles;          // figure it out
} Particles;

typedef struct {
    volatile uint16_t *img;        // actual image to write
    char *imgName;                 // name of the image such that [name]
    char *writeName;               // name of the image actually written
    volatile double *count;        // number of particles in each cell
    volatile double *countSorted;  // sorted count for artistic purposes
    volatile double maxCount;      // maximum count
} ImageData;

typedef struct {
    volatile double *f_field;       // 2D array to return force vals without malloc
    volatile double *a_tot;         // 2D array to return acel vals without malloc
    volatile SimInfo *simInfo;      // global SimInfo struct
    volatile Field *field;          // global Field struct
    volatile Particles *particles;  // global Particles struct
    volatile ImageData *imageData;  // global ImageData struct
    pthread_mutex_t *locks;         // locks for the count array
    uint threadId;                  // thread id
    volatile uint startIdx;         // start idx for thread calculations
    volatile uint num;              // number of thread calculations
} ThreadData;

void writeTiff16BW(volatile uint16_t *img, char *name, uint32_t width,
                   uint32_t height);

void writeTiff16RGB(volatile uint16_t *img, char *name, uint32_t width,
                    uint32_t height);

void *initializeParticles(void *ptr);

double randDouble(double min, double max);

int ipow(int base, int exp);

void *verlet(void *ptr);

void *updateVelocities(void *ptr);

void *updateCurl(void *ptr);

void drawParticles(ImageData *imageData, SimInfo *simInfo,
                   Particles *particles);

void drawField(Field *field, SimInfo *simInfo);

void writeAccumulator(SimInfo *simInfo, ImageData *imageData);

#endif