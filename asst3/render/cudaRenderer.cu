#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
// #include <thrust/device_vector.h>
// #include <thrust/sort.h>
// #include <cub/device/device_radix_sort.cuh>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"


#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"
#include <list>


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, p, imgPtr);
            imgPtr++;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
        //  printf("%s Starting ...\n",argv[0]);
    // int deviceCount = 0;
    // cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    // if(error_id!=cudaSuccess)
    // {
    //     printf("cudaGetDeviceCount returned %d\n ->%s\n",
    //           (int)error_id,cudaGetErrorString(error_id));
    //     printf("Result = FAIL\n");
    //     exit(EXIT_FAILURE);
    // }
    // if(deviceCount==0)
    // {
    //     printf("There are no available device(s) that support CUDA\n");
    // }
    // else
    // {
    //     printf("Detected %d CUDA Capable device(s)\n",deviceCount);
    // }
    // int dev=0,driverVersion=0,runtimeVersion=0;
    // cudaSetDevice(dev);
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp,dev);
    // printf("Device %d:\"%s\"\n",dev,deviceProp.name);
    // cudaDriverGetVersion(&driverVersion);
    // cudaRuntimeGetVersion(&runtimeVersion);
    // printf("  CUDA Driver Version / Runtime Version         %d.%d  /  %d.%d\n",
    //     driverVersion/1000,(driverVersion%100)/10,
    //     runtimeVersion/1000,(runtimeVersion%100)/10);
    // printf("  CUDA Capability Major/Minor version number:   %d.%d\n",
    //     deviceProp.major,deviceProp.minor);
    // printf("  Total amount of global memory:                %.2f MBytes (%llu bytes)\n",
    //         (float)deviceProp.totalGlobalMem/pow(1024.0,3));
    // printf("  GPU Clock rate:                               %.0f MHz (%0.2f GHz)\n",
    //         deviceProp.clockRate*1e-3f,deviceProp.clockRate*1e-6f);
    // printf("  Memory Bus width:                             %d-bits\n",
    //         deviceProp.memoryBusWidth);
    // if (deviceProp.l2CacheSize)
    // {
    //     printf("  L2 Cache Size:                            	%d bytes\n",
    //             deviceProp.l2CacheSize);
    // }
    // printf("  Max Texture Dimension Size (x,y,z)            1D=(%d),2D=(%d,%d),3D=(%d,%d,%d)\n",
    //         deviceProp.maxTexture1D,deviceProp.maxTexture2D[0],deviceProp.maxTexture2D[1]
    //         ,deviceProp.maxTexture3D[0],deviceProp.maxTexture3D[1],deviceProp.maxTexture3D[2]);
    // printf("  Max Layered Texture Size (dim) x layers       1D=(%d) x %d,2D=(%d,%d) x %d\n",
    //         deviceProp.maxTexture1DLayered[0],deviceProp.maxTexture1DLayered[1],
    //         deviceProp.maxTexture2DLayered[0],deviceProp.maxTexture2DLayered[1],
    //         deviceProp.maxTexture2DLayered[2]);
    // printf("  Total amount of constant memory               %lu bytes\n",
    //         deviceProp.totalConstMem);
    // printf("  Total amount of shared memory per block:      %lu bytes\n",
    //         deviceProp.sharedMemPerBlock);
    // printf("  Total number of registers available per block:%d\n",
    //         deviceProp.regsPerBlock);
    // printf("  Wrap size:                                    %d\n",deviceProp.warpSize);
    // printf("  Maximun number of thread per multiprocesser:  %d\n",
    //         deviceProp.maxThreadsPerMultiProcessor);
    // printf("  Maximun number of thread per block:           %d\n",
    //         deviceProp.maxThreadsPerBlock);
    // printf("  Maximun size of each dimension of a block:    %d x %d x %d\n",
    //         deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    // printf("  Maximun size of each dimension of a grid:     %d x %d x %d\n",
    //         deviceProp.maxGridSize[0],
	//     deviceProp.maxGridSize[1],
	//     deviceProp.maxGridSize[2]);
    // printf("  Maximu memory pitch                           %lu bytes\n",deviceProp.memPitch);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

// __global__ void
// kernelCompareCircles(int curr, bool* is_intersect) {
//     if (*is_intersect)
//         return;

//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index >= curr)
//         return;

//     // read position and radius
//     float3 cp = ((float3*)cuConstRendererParams.position)[curr];
//     float  crad = cuConstRendererParams.radius[curr];
//     float3 ip = ((float3*)cuConstRendererParams.position)[index];
//     float  irad = cuConstRendererParams.radius[index];

//     float dist = sqrt((cp.x - ip.x) * (cp.x - ip.x) + (cp.y - ip.y) * (cp.y - ip.y)) - 0.1;

//     if (dist < crad + irad) {
//         *is_intersect = true;
//     }
// }

// inline bool&
// getIntersects(bool* intersects, int i, int j) {
//     return intersects[i * (i - 1) / 2 + j];
// }

// inline std::pair<int, int>
// getIntersectsCoordinates(int index) {
//     int i = 0.5 * (std::sqrt(8 * index + 1) + 1);
//     int j = index - i * (i - 1) / 2;
//     return std::make_pair(i, j);
// }

// __global__ void
// kernelGetIntersects(bool* intersects, int numCircles) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index >= numCircles * (numCircles - 1) / 2)
//         return;

//     auto [i, j] = getIntersectsCoordinates(index);

//     float3 ip = ((float3*)cuConstRendererParams.position)[i];
//     float  irad = cuConstRendererParams.radius[i];
//     float3 jp = ((float3*)cuConstRendererParams.position)[j];
//     float  jrad = cuConstRendererParams.radius[j];

//     float dist = sqrt((ip.x - jp.x) * (ip.x - jp.x) + (ip.y - jp.y) * (ip.y - jp.y)) - 0.1;

//     if (dist < irad + jrad) {
//         intersects[index] = true;
//     }
// }

// __global__ void
// kernelGetDependency(int* dependencies, bool* intersects, int numCircles) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
//     if (i > numCircles - 1)
//         return;

//     for(int j = i - 1; j >= 0; j--) {
//         if (getIntersects(intersects, i, j)) {
//             dependencies[i - 1] = j;
//             return;
//         }
//     }
// }

// inline int startOffset(int i, int numCircles) {
//     return (numCircles - 1 + numCircles - i) * i / 2;
// }

// __global__ void
// kernelInitialDependencyOffset(int* offset, int numCircles) {
//     int j = blockIdx.x * blockDim.x + threadIdx.x;
//     if (j > numCircles - 1)
//         return;
//     if (j == numCircles -1) {
//         offset[numCircles - 1] = 0;
//         return;
//     }
//     offset[j] = startOffset(j, numCircles);
// }

// __global__ void
// kernelGroupDependents(int* offset, int* dependencies, int* dependents, int* no_dependency, int numCircles) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
//     if (i > numCircles - 1)
//         return;
//     int dep = dependencies[i - 1];
//     if (dep == -1) {
//         int write_idx = atomicAdd(&offset[numCircles - 1], 1);
//         no_dependency[write_idx] = i;
//         return;
//     }
        
//     int write_idx = atomicAdd(&offset[dep], 1);
//     dependencies[write_idx] = i;
// }

// __global__ void
// kernelGroupCircles(int** groupIDs, int* groupIDSize, int* dependents, int* no_dependency, int numCircles) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= numCircles)
//         return;
//     if (no_dependency[i] != -1) {
//         int write_idx = atomicAdd(groupIDSize, 1);
//         groupIDs[write_idx] = &no_dependency[i];
//         return;
//     }
// }


__global__ void
kernelInitializePixelCircleList(unsigned int** pixel_circle_list,
                                unsigned int* pixel_circle_list_capacity,
                                unsigned int numCircles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.imageWidth * cuConstRendererParams.imageHeight)
        return;
    // pixel_circle_list[index] = new unsigned int[1];
    pixel_circle_list[index] = (unsigned int*) malloc(1024 * sizeof(unsigned int));
    if (pixel_circle_list[index] == nullptr) {
        // printf("malloc failed\n");
    }
    pixel_circle_list_capacity[index] = 1024;
    // pixel_circle_list[index] = (int*)malloc(numCircles * sizeof(int));
}

__global__ void
kernelDeletePixelCircleList(unsigned int** pixel_circle_list, int numCircles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.imageWidth * cuConstRendererParams.imageHeight)
        return;
    delete[] pixel_circle_list[index];
    // free(pixel_circle_list[index]);
}

__global__ void
kernelAssignPixelsHelper(unsigned int** pixel_circle_list,
                         unsigned int* pixel_circle_list_size,
                         unsigned int* pixel_circle_list_capacity,
                         unsigned int* pixel_circle_lock_list,
                         int circle_id,
                         int screenMinX,
                         int screenMaxX,
                         int screenMinY,
                         int screenMaxY) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int width = screenMaxX - screenMinX;
    int height = screenMaxY - screenMinY;
    if (index >= width * height)
        return;
    
    int pixelX = index % width + screenMinX;
    int pixelY = index / width + screenMinY;
    int i = pixelY * cuConstRendererParams.imageWidth + pixelX;
    // unsigned int write_idx = pixel_circle_list_size[i], assumed;
    // do {
    //     assumed = write_idx;
    //     write_idx = atomicCAS(&pixel_circle_list_size[i], assumed, assumed + 1);
    // } while (assumed != write_idx);
    // unsigned int lock = 0;
    // __syncthreads();
    bool blocked = true;
    while (blocked) {
        if (0 == atomicCAS(&pixel_circle_lock_list[i], 0, 1)) {
            unsigned int write_idx = atomicInc(&pixel_circle_list_size[i], UINT_MAX);
            pixel_circle_list[i][write_idx] = circle_id;
            atomicExch(&pixel_circle_lock_list[i], 0U);
            blocked = false;
        }
    }
    // while (atomicExch(&pixel_circle_lock_list[i], 1U) == 1U);
    // unsigned int write_idx = atomicInc_system(&pixel_circle_list_size[i], UINT_MAX);
    // while (atomicAdd(&pixel_circle_list_capacity[i], 0)< write_idx) {
    //     printf("=================hohoho============================\n");
    // }
    // if (pixel_circle_list_capacity[i] == write_idx) {
    //     printf("=================Wow============================\n");
    //     unsigned int* new_buf = (unsigned int*) malloc((pixel_circle_list_capacity[i] + 256) * sizeof(unsigned int));
    //     // unsigned int* new_buf = new unsigned int*) malloc((pixel_circle_list_capacity[i] + 256) * sizeof(int));
    //     if (new_buf == nullptr) {
    //         // printf("failed at assignment\n");
    //     }
    //     memcpy((void*) new_buf, (void *) pixel_circle_list[i], pixel_circle_list_capacity[i] * sizeof(unsigned int));

    //     free(pixel_circle_list[i]);
    //     pixel_circle_list[i] = new_buf;
    //     atomicAdd_system(&pixel_circle_list_capacity[i], 256);
    // }
    // pixel_circle_list[i][write_idx] = circle_id;
    // atomicExch(&pixel_circle_lock_list[i], 0U);
}

// #define CHUNK_SIZE 1024

__device__ inline unsigned int&
getCircle(unsigned int* pixel_circle_list, unsigned int* buf, int i, int j) {
    int index = buf[pixel_circle_list[i]];
    while (j >= 128) {
        j -= 128;
        index = buf[index + 128];
    }
    return buf[index + j];
}

__device__ inline unsigned int&
getCircle2(unsigned int* pixel_circle_list, unsigned int* buf, int i, int j) {
    int index = buf[pixel_circle_list[i]];
    while (j > 128) {
        j -= 128;
        index = buf[index + 128];
    }
    return buf[index + j];
}

__global__ void
kernelAssignPixels(unsigned int** pixel_circle_list,
                   unsigned int* pixel_circle_list_size,
                   unsigned int* pixel_circle_list_capacity,
                   unsigned int* pixel_circle_lock_list,
                   int numCircles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numCircles)
        return;
    
    // read position and radius
    float3 p = ((float3*) cuConstRendererParams.position)[index];
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    // fprintf(stderr, "%d", screenMaxX - screenMinX);

    // float invWidth = 1.f / imageWidth;
    // float invHeight = 1.f / imageHeight;

    // for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
    //     int i = pixelY * imageWidth + screenMinX;
    //     for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
    //         while (atomicExch_system(&pixel_circle_lock_list[i], 1U) == 0U);
    //         i++;
    //     }
    // }

    // for all pixels in the bonding box
    kernelAssignPixelsHelper<<<((screenMaxX - screenMinX) * (screenMaxY - screenMinY) + blockDim.x - 1) / blockDim.x, blockDim>>>(pixel_circle_list, pixel_circle_list_size, pixel_circle_list_capacity, pixel_circle_lock_list, index, screenMinX, screenMaxX, screenMinY, screenMaxY);

    // for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
    //     int i = pixelY * imageWidth + screenMinX;
    //     for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
    //         atomicExch_system(&pixel_circle_lock_list[i], 0U);
    //         i++;
    //     }
    // }

    // for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
    //     int i = pixelY * imageWidth + screenMinX;
    //     for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
    //         // int write_idx = atomicAdd(&pixel_circle_list_size[i], 1);
    //         int write_idx = atomicInc(&pixel_circle_list_size[i], UINT_MAX);
    //         // if (pixel_circle_list_capacity[index] < write_idx) {
    //         //     return;
    //         // }
    //         // if (10 <= write_idx) {
    //         //     i++;
    //         //     continue;
    //         // }
    //         while (pixel_circle_list_capacity[i] < write_idx);
    //         if (pixel_circle_list_capacity[i] == write_idx) {
    //             unsigned int* new_buf = (unsigned int*) malloc((pixel_circle_list_capacity[i] + 256) * sizeof(int));
    //             // unsigned int* new_buf = new unsigned int*) malloc((pixel_circle_list_capacity[i] + 256) * sizeof(int));
    //             if (new_buf == nullptr) {
    //                 printf("failed at assignment\n");
    //             }
    //             memcpy(new_buf, pixel_circle_list[i], pixel_circle_list_capacity[i] * sizeof(int));
    //             free(pixel_circle_list[i]);
    //             pixel_circle_list[i] = new_buf;
    //             atomicAdd(&pixel_circle_list_capacity[i], 256);
    //         }
    //         // if (!pixel_circle_list[i]) {
    //         //     pixel_circle_list[i] = (int*)malloc(numCircles * sizeof(int));
    //         // }
    //         // __syncthreads();
    //         // unsigned int* ptr = &pixel_circle_list[i][write_idx];
    //         // *ptr = index;
    //         atomicAdd(&pixel_circle_list[i][write_idx], index);
    //         i++;
    //     }
    // }
}

__global__ void
kernelAssignPixels2(unsigned int** pixel_circle_list,
                    unsigned int* pixel_circle_list_size,
                    unsigned int* pixel_circle_list_capacity,
                    unsigned int* pixel_circle_lock_list,
                    int numCircles,
                    int index) {
    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numCircles)
        return;
    
    // read position and radius
    float3 p = ((float3*) cuConstRendererParams.position)[index];
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    // fprintf(stderr, "%d", screenMaxX - screenMinX);

    // float invWidth = 1.f / imageWidth;
    // float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    kernelAssignPixelsHelper<<<((screenMaxX - screenMinX) * (screenMaxY - screenMinY) + blockDim.x - 1) / blockDim.x, blockDim>>>(pixel_circle_list, pixel_circle_list_size, pixel_circle_list_capacity, pixel_circle_lock_list, index, screenMinX, screenMaxX, screenMinY, screenMaxY);

    // for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
    //     int i = pixelY * imageWidth + screenMinX;
    //     for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
    //         // int write_idx = atomicAdd(&pixel_circle_list_size[i], 1);
    //         int write_idx = atomicInc(&pixel_circle_list_size[i], UINT_MAX);
    //         // if (pixel_circle_list_capacity[index] < write_idx) {
    //         //     return;
    //         // }
    //         // if (10 <= write_idx) {
    //         //     i++;
    //         //     continue;
    //         // }
    //         while (pixel_circle_list_capacity[i] < write_idx);
    //         if (pixel_circle_list_capacity[i] == write_idx) {
    //             unsigned int* new_buf = (unsigned int*) malloc((pixel_circle_list_capacity[i] + 256) * sizeof(int));
    //             // unsigned int* new_buf = new unsigned int*) malloc((pixel_circle_list_capacity[i] + 256) * sizeof(int));
    //             if (new_buf == nullptr) {
    //                 printf("failed at assignment\n");
    //             }
    //             memcpy(new_buf, pixel_circle_list[i], pixel_circle_list_capacity[i] * sizeof(int));
    //             free(pixel_circle_list[i]);
    //             pixel_circle_list[i] = new_buf;
    //             atomicAdd(&pixel_circle_list_capacity[i], 256);
    //         }
    //         // if (!pixel_circle_list[i]) {
    //         //     pixel_circle_list[i] = (int*)malloc(numCircles * sizeof(int));
    //         // }
    //         // __syncthreads();
    //         // unsigned int* ptr = &pixel_circle_list[i][write_idx];
    //         // *ptr = index;
    //         atomicAdd(&pixel_circle_list[i][write_idx], index);
    //         i++;
    //     }
    // }
}

__global__ void
kernelAssignPixels3(unsigned int** pixel_circle_list,
                    unsigned int* pixel_circle_list_size,
                    unsigned int* pixel_circle_list_capacity,
                    unsigned int* pixel_circle_lock_list,
                    int numCircles,
                    int index) {
    index += blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numCircles)
        return;
    
    // read position and radius
    float3 p = ((float3*) cuConstRendererParams.position)[index];
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    // fprintf(stderr, "%d", screenMaxX - screenMinX);

    // float invWidth = 1.f / imageWidth;
    // float invHeight = 1.f / imageHeight;

    // for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
    //     int i = pixelY * imageWidth + screenMinX;
    //     for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
    //         while (atomicExch_system(&pixel_circle_lock_list[i], 1U) == 0U);
    //         i++;
    //     }
    // }

    // for all pixels in the bonding box
    kernelAssignPixelsHelper<<<((screenMaxX - screenMinX) * (screenMaxY - screenMinY) + blockDim.x - 1) / blockDim.x, blockDim>>>(pixel_circle_list, pixel_circle_list_size, pixel_circle_list_capacity, pixel_circle_lock_list, index, screenMinX, screenMaxX, screenMinY, screenMaxY);

    // for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
    //     int i = pixelY * imageWidth + screenMinX;
    //     for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
    //         atomicExch_system(&pixel_circle_lock_list[i], 0U);
    //         i++;
    //     }
    // }

    // for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
    //     int i = pixelY * imageWidth + screenMinX;
    //     for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
    //         // int write_idx = atomicAdd(&pixel_circle_list_size[i], 1);
    //         int write_idx = atomicInc(&pixel_circle_list_size[i], UINT_MAX);
    //         // if (pixel_circle_list_capacity[index] < write_idx) {
    //         //     return;
    //         // }
    //         // if (10 <= write_idx) {
    //         //     i++;
    //         //     continue;
    //         // }
    //         while (pixel_circle_list_capacity[i] < write_idx);
    //         if (pixel_circle_list_capacity[i] == write_idx) {
    //             unsigned int* new_buf = (unsigned int*) malloc((pixel_circle_list_capacity[i] + 256) * sizeof(int));
    //             // unsigned int* new_buf = new unsigned int*) malloc((pixel_circle_list_capacity[i] + 256) * sizeof(int));
    //             if (new_buf == nullptr) {
    //                 printf("failed at assignment\n");
    //             }
    //             memcpy(new_buf, pixel_circle_list[i], pixel_circle_list_capacity[i] * sizeof(int));
    //             free(pixel_circle_list[i]);
    //             pixel_circle_list[i] = new_buf;
    //             atomicAdd(&pixel_circle_list_capacity[i], 256);
    //         }
    //         // if (!pixel_circle_list[i]) {
    //         //     pixel_circle_list[i] = (int*)malloc(numCircles * sizeof(int));
    //         // }
    //         // __syncthreads();
    //         // unsigned int* ptr = &pixel_circle_list[i][write_idx];
    //         // *ptr = index;
    //         atomicAdd(&pixel_circle_list[i][write_idx], index);
    //         i++;
    //     }
    // }
}

/* This function is same in both iterative and recursive*/
__device__ int
partition(unsigned int* arr, int l, int h) {
    unsigned int x = arr[h]; 
    int i = (l - 1); 

    for (int j = l; j <= h - 1; j++) { 
        if (arr[j] <= x) { 
            i++;
            unsigned int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        } 
    }
    unsigned int tmp = arr[i + 1];
    arr[i + 1] = arr[h];
    arr[h] = tmp;
    return (i + 1);
} 

/* A[] --> Array to be sorted, 
l --> Starting index, 
h --> Ending index */
__device__ void
quickSortIterative(unsigned int* arr, int l, int h) {
    // Create an auxiliary stack 
    unsigned int* stack = new unsigned int[h - l + 1];

    // initialize top of stack 
    int top = -1; 

    // push initial values of l and h to stack 
    stack[++top] = l; 
    stack[++top] = h; 

    // Keep popping from stack while is not empty 
    while (top >= 0) { 
        // Pop h and l 
        h = stack[top--]; 
        l = stack[top--]; 

        // Set pivot element at its correct position 
        // in sorted array 
        int p = partition(arr, l, h); 

        // If there are elements on left side of pivot, 
        // then push left side to stack 
        if (p - 1 > l) { 
            stack[++top] = l; 
            stack[++top] = p - 1; 
        } 

        // If there are elements on right side of pivot, 
        // then push right side to stack 
        if (p + 1 < h) { 
            stack[++top] = p + 1; 
            stack[++top] = h; 
        } 
    }
    delete[] stack;
}

__global__ void
kernelSortCircles(unsigned int** pixel_circle_list,
                  unsigned int* pixel_circle_list_size,
                  int numCircles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.imageWidth * cuConstRendererParams.imageHeight)
        return;
    int size = pixel_circle_list_size[index];
    if (size == 0)
        return;
    unsigned int* circle_list = pixel_circle_list[index];
    // std::sort(circle_list, circle_list + size);
    // quickSortIterative(circle_list, 0, size - 1);
    quickSortIterative(circle_list, 0, size - 1);
    // std::sort(circle_list, circle_list + size);
    // thrust::sort(circle_list, circle_list + size);
    // extern __shared__ int shared_data[];

    // // Copy data to shared memory
    // for (int i = threadIdx.x; i < size; i += blockDim.x) {
    //     shared_data[i] = circle_list[i];
    // }
    // __syncthreads();

    // // Use CUB's BlockRadixSort to sort the circle list
    // typedef cub::BlockRadixSort<int, 1024> BlockRadixSort;  // Adjust block size as needed
    // __shared__ typename BlockRadixSort::TempStorage temp_storage;

    // // Perform the sorting (in-place)
    // BlockRadixSort(temp_storage).Sort(shared_data, size);
    // __syncthreads();

    // // Copy sorted data back to the circle list
    // for (int i = threadIdx.x; i < size; i += blockDim.x) {
    //     circle_list[i] = shared_data[i];
    // }
}

__global__ void
kernelShadePixels(unsigned int** pixel_circle_list,
                  unsigned int* pixel_circle_list_size,
                  int imageWidth,
                  int imageHeight) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= imageWidth * imageHeight)
        return;
    unsigned int size = pixel_circle_list_size[index];
    if (size == 0)
        return;
    unsigned int* circle_list = pixel_circle_list[index];

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    int pixelX = index % imageWidth;
    int pixelY = index / imageWidth;

    float4 localPixel = {1.f, 1.f, 1.f, 1.f};
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));
    for (int i = 0; i < size; i++) {
            // read position and radius
        float3 p = ((float3*) cuConstRendererParams.position)[circle_list[i]];
        // float  rad = cuConstRendererParams.radius[i];
        // shadePixel(circle_list[i], pixelCenterNorm, p, &localPixel);
        shadePixel(circle_list[i], pixelCenterNorm, p, &(((float4*) cuConstRendererParams.imageData)[index]));
        // break;
    }
    // ((float4*) cuConstRendererParams.imageData)[index] = localPixel;
}

__global__ void
testMalloc() {
    int size = 0;
    while(1){
        int* a = (int*)malloc(512*1024*sizeof(int));
        if (a == nullptr) {
            break;
        }
        size += 512*1024*4;
    }
    // printf("malloc size = %d MB\n", size/(1024*1024));
}

__global__ void
kernelTestChecker(unsigned int** pixel_circle_list,
                  unsigned int* pixel_circle_list_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.imageWidth * cuConstRendererParams.imageHeight)
        return;
    int size = pixel_circle_list_size[index];
    if (size == 0)
        return;
    unsigned int* circle_list = pixel_circle_list[index];
    float prev = ((float3*) cuConstRendererParams.position)[circle_list[0]].z;
    for (int i = 1; i < size; i ++) {
        if (prev <= ((float3*) cuConstRendererParams.position)[circle_list[i]].z) {
            // printf("Sort compromized\n");
            break;
        }
        prev = ((float3*) cuConstRendererParams.position)[circle_list[i]].z;
    }
}

__global__ void
kernelProcessPixelCircleList(unsigned int** pixel_circle_list,
                             unsigned int*  pixel_circle_list_size,
                             unsigned int*  buf,
                             unsigned int*  buf_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.imageWidth * cuConstRendererParams.imageHeight)
        return;
    
    int write_idx = atomicAdd(buf_size, pixel_circle_list_size[index]);
    memcpy(buf + write_idx, pixel_circle_list[index], pixel_circle_list_size[index] * sizeof(unsigned int));
    free(pixel_circle_list[index]);
    pixel_circle_list[index] = buf + write_idx;
}

void 
exportPixelCircleListToTxt(unsigned int** d_pixel_circle_list,
                           unsigned int*  d_pixel_circle_list_size,
                           unsigned int*  d_buf,
                           int   width,
                           int   height,
                           int   numCircles,
                           const std::string& outFilename)
{
    std::vector<unsigned int> h_buf(numCircles * 512 * sizeof(unsigned int), 0);
    cudaCheckError(cudaMemcpy(h_buf.data(),
                              d_buf,
                              numCircles * 512 * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

    int totalPixels = width * height;

    // 1) 从 device 拷贝 pixel_circle_list (指针数组) 到 host
    std::vector<unsigned int*> h_pixel_circle_list(totalPixels, nullptr);
    cudaCheckError(cudaMemcpy(h_pixel_circle_list.data(),
                              d_pixel_circle_list,
                              totalPixels * sizeof(unsigned int*),
                              cudaMemcpyDeviceToHost));

    // 2) 从 device 拷贝 pixel_circle_list_size (每个像素的 circle 数量) 到 host
    std::vector<unsigned int> h_pixel_circle_list_size(totalPixels);
    cudaCheckError(cudaMemcpy(h_pixel_circle_list_size.data(),
                              d_pixel_circle_list_size,
                              totalPixels * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

    // 3) 打开输出文本文件
    std::ofstream outFile(outFilename);
    if (!outFile.is_open())
    {
        std::cerr << "Error: could not open output file " << outFilename << std::endl;
        return;
    }

    // 4) 对每个像素 i，读取它的 circle 列表并写入文件
    for (int i = 0; i < totalPixels; i++)
    {
        int size_i = h_pixel_circle_list_size[i];
        // 可以选择先写一下“像素索引”和“该像素中 circle 的数量”
        outFile << "Pixel " << i << " has " << size_i << " circles: ";
        // if (size_i > 64) {
        //     std::cout << "Wow-===========================" << std::endl;
        // }

        if (size_i > 0)
        {
            // 这时 h_pixel_circle_list[i] 是一个指向 GPU 上的 int 数组
            // 我们需要再分配一块 host 缓存，把这块数据拷回来。
            auto begin = h_buf.begin() + (h_pixel_circle_list[i] - d_buf);
            std::vector<unsigned int> circleData(begin, begin + size_i);

            // cudaCheckError(cudaMemcpy(circleData.data(),
            //                           h_pixel_circle_list[i],
            //                           size_i * sizeof(int),
            //                           cudaMemcpyDeviceToHost));

            // 将 circleData 中的内容写到文件
            for (int c = 0; c < size_i; c++)
            {
                outFile << circleData[c] << " ";
            }
        }

        outFile << "\n";  // 换行
    }

    outFile.close();

    std::cout << "Export done. Wrote file: " << outFilename << std::endl;
}

void
CudaRenderer::render() {
    cudaCheckError(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 6UL * 1024UL * 1024UL * 1024UL));

    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    // dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);
    dim3 gridDim((image->width * image->height + blockDim.x - 1) / blockDim.x);
    // fprintf(stderr, "grid: %d\n", gridDim.x);
    // fprintf(stderr, "blockDim: %d\n", blockDim.x);
    // return;

    //TODO: perform screen per pixel shading in a separate kernel
    //TODO: Compute radius parallelly: compare kernel
    //TODO: Build a data structure where every pixel has a list of circles that intersect it// 
    //: first group boundbox = per circle, then assign task kernel
    //TODO: Get Job Done // render kernel: per pixel 

    // testMalloc<<<1, 1>>>();
    // cudaDeviceSynchronize();
    // return;

    // fprintf(stderr, "float1 size: %d\n", sizeof(float1));
    // fprintf(stderr, "float2 size: %d\n", sizeof(float2));
    // fprintf(stderr, "float3 size: %d\n", sizeof(float3));
    // fprintf(stderr, "float4 size: %d\n", sizeof(float4));

    // fprintf(stderr, "float1 size: %d\n", ((float1*) nullptr) + 1);
    // fprintf(stderr, "float2 size: %d\n", ((float2*) nullptr) + 1);
    // fprintf(stderr, "float3 size: %d\n", ((float3*) nullptr) + 1);
    // fprintf(stderr, "float4 size: %d\n", ((float4*) nullptr) + 1);

    unsigned int** pixel_circle_list = nullptr;
    cudaCheckError(cudaMalloc(&pixel_circle_list, image->width * image->height * sizeof(unsigned int*)));
    cudaCheckError(cudaMemset(pixel_circle_list, 0, image->width * image->height * sizeof(unsigned int*)));

    unsigned int* pixel_circle_lock_list = nullptr;
    cudaCheckError(cudaMalloc(&pixel_circle_lock_list, image->width * image->height * sizeof(unsigned int)));
    cudaCheckError(cudaMemset(pixel_circle_lock_list, 0, image->width * image->height * sizeof(unsigned int)));

    unsigned int* pixel_circle_list_size = nullptr;
    cudaCheckError(cudaMalloc(&pixel_circle_list_size, image->width * image->height * sizeof(unsigned int)));
    cudaCheckError(cudaMemset(pixel_circle_list_size, 0, image->width * image->height * sizeof(unsigned int)));

    unsigned int* pixel_circle_list_capacity = nullptr;
    cudaCheckError(cudaMalloc(&pixel_circle_list_capacity, image->width * image->height * sizeof(unsigned int)));
    cudaCheckError(cudaMemset(pixel_circle_list_capacity, 0, image->width * image->height * sizeof(unsigned int)));

    kernelInitializePixelCircleList<<<gridDim, blockDim>>>(pixel_circle_list,
                                                           pixel_circle_list_capacity,
                                                           numCircles);    
    cudaCheckError(cudaDeviceSynchronize());

    kernelAssignPixels<<<(numCircles + blockDim.x - 1) / blockDim.x, blockDim>>>(pixel_circle_list,
                                                                                 pixel_circle_list_size,
                                                                                 pixel_circle_list_capacity,
                                                                                 pixel_circle_lock_list,
                                                                                 numCircles);
    cudaCheckError(cudaDeviceSynchronize());

    // bool flag = true;
    // for (int i = 420; i != 420 || flag; i = (i + numCircles - 17) % numCircles) {
    //     kernelAssignPixels2<<<1, 1>>>(pixel_circle_list,
    //                                   pixel_circle_list_size,
    //                                   pixel_circle_list_capacity,
    //                                   pixel_circle_lock_list,
    //                                   numCircles,
    //                                   i);
    //     cudaCheckError(cudaDeviceSynchronize());
    //     flag = false;
    // }

    // for (int i = 0; i < numCircles; i += 8 * blockDim.x) {
    //     kernelAssignPixels3<<<8, blockDim>>>(pixel_circle_list,
    //                                   pixel_circle_list_size,
    //                                   pixel_circle_list_capacity,
    //                                   pixel_circle_lock_list,
    //                                   numCircles,
    //                                   i);
    //     cudaCheckError(cudaDeviceSynchronize());
    // }

    // fprintf(stderr, "fuck\n");

    // kernelSortCircles<<<gridDim, blockDim>>>(pixel_circle_list, pixel_circle_list_size, buf, numCircles);
    kernelSortCircles<<<gridDim, blockDim>>>(pixel_circle_list, pixel_circle_list_size, numCircles);
    cudaCheckError(cudaDeviceSynchronize());

    // kernelTestChecker<<<gridDim, blockDim>>>(pixel_circle_list, pixel_circle_list_size);
    // cudaCheckError(cudaDeviceSynchronize());
    // return;

    // float4* localPixels = nullptr;
    // cudaMalloc(&localPixels, image->width * image->height * sizeof(float4));
    // cudaMemset(localPixels, 0, image->width * image->height * sizeof(float4));

    kernelShadePixels<<<gridDim, blockDim>>>(pixel_circle_list, pixel_circle_list_size, image->width, image->height);
    cudaCheckError(cudaDeviceSynchronize());

    // unsigned int* buf = nullptr;
    // cudaCheckError(cudaMalloc(&buf, numCircles * 512 * sizeof(unsigned int)));
    // unsigned int* buf_size = nullptr;
    // cudaCheckError(cudaMalloc(&buf_size, sizeof(unsigned int)));
    // kernelProcessPixelCircleList<<<gridDim, blockDim>>>(pixel_circle_list, pixel_circle_list_size, buf, buf_size);
    // cudaCheckError(cudaDeviceSynchronize());

    // exportPixelCircleListToTxt(pixel_circle_list, pixel_circle_list_size, buf, image->width, image->height, numCircles, "lala.txt");

    // int dims = image->width * image->height;
    // unsigned int** pixel_circle_list_host = new unsigned int*[dims];
    // cudaCheckError(cudaMemcpy(pixel_circle_list_host, pixel_circle_list, dims * sizeof(unsigned int*), cudaMemcpyDeviceToHost));
    // unsigned int* pixel_circle_list_size_host = new unsigned int[dims];
    // cudaCheckError(cudaMemcpy(pixel_circle_list_size_host, pixel_circle_list_size, dims * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // std::fstream outf("pixel_circle_list.txt", std::fstream::out);
    // for (int i = 0; i < dims; i++) {
    //     unsigned int* circle_list_host = new unsigned int[pixel_circle_list_size_host[i]];
    //     cudaCheckError(cudaMemcpy(circle_list_host, pixel_circle_list_host[i], pixel_circle_list_size_host[i] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    //     for (int j = 0; j < pixel_circle_list_size_host[i]; j++) {
    //         outf << circle_list_host[j] << ',';
    //     }
    //     outf << '\n';
    //     delete[] circle_list_host;
    // }
    // outf.close();
    // delete[] pixel_circle_list_host;
    // delete[] pixel_circle_list_size_host;

    // kernelDeletePixelCircleList<<<gridDim, blockDim>>>(pixel_circle_list, numCircles);
    // cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaFree(pixel_circle_list));
    cudaCheckError(cudaFree(pixel_circle_list_size));

    // cudaMemcpy(cudaDeviceImageData, localPixels, sizeof(float) * 4 * image->width * image->height, cudaMemcpyDeviceToDevice);

    // bool* intersects = nullptr;
    // cudaMalloc(&intersects, ((numCircles * (numCircles - 1)) / 2) * sizeof(bool));
    // cudaMemset(intersects, 0, ((numCircles * (numCircles - 1)) / 2) * sizeof(bool));
    // kernelGetIntersects<<<(numCircles * (numCircles - 1) / 2 + blockDim.x - 1) / blockDim.x, blockDim>>>(intersects, numCircles);
    // cudaDeviceSynchronize();

    // int* dependencies = nullptr;
    // cudaMalloc(&dependencies, (numCircles - 1) * sizeof(int));
    // cudaMemset(dependencies, -1, (numCircles - 1) * sizeof(int));
    // kernelGetDependency<<<(numCircles + blockDim.x - 2) / blockDim.x, blockDim>>>(dependencies, intersects, numCircles);
    // cudaDeviceSynchronize();

    // int* dependents = nullptr;
    // cudaMalloc(&dependents, ((numCircles * (numCircles - 1)) / 2) * sizeof(int));
    // cudaMemset(dependents, -1, ((numCircles * (numCircles - 1)) / 2) * sizeof(int));

    // int* offset = nullptr;
    // cudaMalloc(&offset, numCircles * sizeof(int));
    // kernelInitialDependencyOffset<<<(numCircles + blockDim.x - 1) / blockDim.x, blockDim>>>(offset, numCircles);
    // cudaDeviceSynchronize();

    // int* no_dependency = nullptr;
    // cudaMalloc(&no_dependency, numCircles * sizeof(int));
    // cudaMemset(no_dependency, -1, numCircles * sizeof(int));
    // kernelGroupDependents<<<(numCircles + blockDim.x - 2) / blockDim.x, blockDim>>>(offset, dependencies, dependents, numCircles);
    // cudaDeviceSynchronize();

    // int** groupIDs = nullptr;
    // cudaMalloc(&groupIDs, numCircles * sizeof(int*));
    // cudaMemset(groupIDs, 0, numCircles * sizeof(int*));

    // int* groupIDSize = nullptr;
    // cudaMalloc(&groupIDSize, sizeof(int));
    // cudaMemset(groupIDSize, 0, sizeof(int));

    // //compare
    // std::list<int> circle_list;
    // for (int i = 0; i < numCircles; i++) {
    //     circle_list.push_back(i);
    // }
    // std::vector<std::vector<int>> circle_groups;
    // while (!circle_list.empty()) {
    //     circle_groups.emplace_back();
    //     for (auto it = circle_list.begin(); it != circle_list.end();) {
    //         bool* is_intersect = nullptr;
    //         cudaMallocManaged(&is_intersect, sizeof(bool));
    //         *is_intersect = false;
    //         kernelCompareCircles<<<(*it + blockDim.x - 1) / blockDim.x, blockDim>>>(*it, is_intersect);
    //         cudaDeviceSynchronize();
    //         if (*is_intersect) {
    //             circle_groups.back().push_back(*it);
    //             it++;
    //         } else {
    //             it = circle_list.erase(it);
    //         }
    //     }
    // }
    // std::vector<std::vector<int>> pixel_circle_list(image->width * image->height);
    // for(every circle) {
    //     cur = cur_circle;
    //     parallel_for(every privious circlu) {
    //         //compare
    //     }
    //     if (cur_circle intersects with any privious circle) {
    //         assign task
    //     }
    // }



    // kernelRenderCircles<<<gridDim, blockDim>>>();
    // cudaDeviceSynchronize();
}
