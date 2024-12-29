#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

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
kernelInitializePixelCircleList(unsigned int* pixel_circle_list,
                                unsigned int* pixel_circle_list_capacity,
                                unsigned int* buf,
                                unsigned int* buf_size,
                                unsigned int numCircles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.imageWidth * cuConstRendererParams.imageHeight)
        return;
    // pixel_circle_list[index] = (unsigned int*) malloc(256 * sizeof(unsigned int));
    // pixel_circle_list_capacity[index] = 256;
    // pixel_circle_list[index] = (int*)malloc(numCircles * sizeof(int));
    pixel_circle_list[index] = atomicAdd(buf_size, 129);
    pixel_circle_list_capacity[index] = 128;
}

__global__ void
kernelDeletePixelCircleList(unsigned int** pixel_circle_list, int numCircles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.imageWidth * cuConstRendererParams.imageHeight)
        return;
    // delete[] pixel_circle_list[index];
    free(pixel_circle_list[index]);
}

// __global__ void
// kernelAssignPixelsHelper(int** pixel_circle_list,
//                          int* pixel_circle_list_size,
//                          int circle_id,
//                          int screenMinX,
//                          int screenMaxX,
//                          int screenMinY,
//                          int screenMaxY) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     int width = screenMaxX - screenMinX;
//     int height = screenMaxY - screenMinY;
//     if (index >= width * height)
//         return;
    
//     int pixelX = index % width + screenMinX;
//     int pixelY = index / width + screenMinY;
//     int i = pixelX * cuConstRendererParams.imageWidth + pixelY;
//     int write_idx = atomicAdd(&pixel_circle_list_size[i], 1);
//     pixel_circle_list[i][write_idx] = circle_id;
// }

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
kernelAssignPixels(unsigned int* pixel_circle_list,
                   unsigned int* pixel_circle_list_size,
                   unsigned int* pixel_circle_list_capacity,
                   unsigned int* buf,
                   unsigned int* buf_size,
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

    // for all pixels in the bonding box
    // kernelAssignPixelsHelper<<<((screenMaxX - screenMinX) * (screenMaxY - screenMinY) + blockDim.x - 1) / blockDim.x, blockDim>>>(pixel_circle_list, pixel_circle_list_size, index, screenMinX, screenMaxX, screenMinY, screenMaxY);

    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        int i = pixelY * imageWidth + screenMinX;
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            // int write_idx = atomicAdd(&pixel_circle_list_size[i], 1);
            int write_idx = atomicInc(&pixel_circle_list_size[i], UINT_MAX);
            // if (pixel_circle_list_capacity[index] < write_idx) {
            //     return;
            // }
            // if (10 <= write_idx) {
            //     i++;
            //     continue;
            // }
            while (pixel_circle_list_capacity[i] < write_idx);
            if (pixel_circle_list_capacity[i] == write_idx) {
                // unsigned int* new_buf = (unsigned int*) malloc((pixel_circle_list_capacity[i] + 256) * sizeof(int));
                // memcpy(new_buf, pixel_circle_list[i], pixel_circle_list_capacity[i] * sizeof(int));
                // free(pixel_circle_list[i]);
                // pixel_circle_list[i] = new_buf;
                // atomicAdd(&pixel_circle_list_capacity[i], 256);
                getCircle2(pixel_circle_list, buf, i, write_idx) = atomicAdd(buf_size, 128);
                atomicAdd(&pixel_circle_list_capacity[i], 128);
            }
            // if (!pixel_circle_list[i]) {
            //     pixel_circle_list[i] = (int*)malloc(numCircles * sizeof(int));
            // }
            // __syncthreads();
            // unsigned int* ptr = &pixel_circle_list[i][write_idx];
            // *ptr = index;
            // atomicAdd(&pixel_circle_list[i][write_idx], index);
            getCircle(pixel_circle_list, buf, i, write_idx) = index;
            i++;
        }
    }
}

/* This function is same in both iterative and recursive*/
__device__ int
partition(unsigned int* pixel_circle_list, unsigned int* buf, int index, int l, int h) {
    unsigned int x = getCircle(pixel_circle_list, buf, index, h); 
    int i = (l - 1); 

    for (int j = l; j <= h - 1; j++) { 
        if (getCircle(pixel_circle_list, buf, index, j) <= x) { 
            i++;
            unsigned int tmp = getCircle(pixel_circle_list, buf, index, i);
            getCircle(pixel_circle_list, buf, index, i) = getCircle(pixel_circle_list, buf, index, j);
            getCircle(pixel_circle_list, buf, index, j) = tmp;
        } 
    }
    unsigned int tmp = getCircle(pixel_circle_list, buf, index, i + 1);
    getCircle(pixel_circle_list, buf, index, i + 1) = getCircle(pixel_circle_list, buf, index, h);
    getCircle(pixel_circle_list, buf, index, h) = tmp;
    return (i + 1);
} 

/* A[] --> Array to be sorted, 
l --> Starting index, 
h --> Ending index */
__device__ void
quickSortIterative(unsigned int* pixel_circle_list, unsigned int* buf, int index, int l, int h) {
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
        int p = partition(pixel_circle_list, buf, index, l, h); 

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
kernelSortCircles(unsigned int* pixel_circle_list,
                  unsigned int* pixel_circle_list_size,
                  unsigned int* buf,
                  int numCircles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.imageWidth * cuConstRendererParams.imageHeight)
        return;
    int size = pixel_circle_list_size[index];
    if (size == 0)
        return;
    // unsigned int* circle_list = pixel_circle_list[index];
    // std::sort(circle_list, circle_list + size);
    // quickSortIterative(circle_list, 0, size - 1);
    quickSortIterative(pixel_circle_list, buf, index, 0, size - 1);
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
kernelShadePixels(unsigned int* pixel_circle_list,
                  unsigned int* pixel_circle_list_size,
                  unsigned int* buf,
                  int imageWidth,
                  int imageHeight) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= imageWidth * imageHeight)
        return;
    unsigned int size = pixel_circle_list_size[index];
    if (size == 0)
        return;
    // unsigned int* circle_list = pixel_circle_list[index];

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    int pixelX = index % imageWidth;
    int pixelY = index / imageWidth;

    float4 localPixel = {1.f, 1.f, 1.f, 1.f};
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));
    for (int i = 0; i < size; i++) {
            // read position and radius
        float3 p = ((float3*) cuConstRendererParams.position)[i];
        // float  rad = cuConstRendererParams.radius[i];
        // shadePixel(circle_list[i], pixelCenterNorm, p, &localPixel);
        shadePixel(getCircle(pixel_circle_list, buf, index, i), pixelCenterNorm, p, &localPixel);
    }
    ((float4*) cuConstRendererParams.imageData)[index] = localPixel;
}

void
CudaRenderer::render() {

    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    // dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);
    dim3 gridDim((image->width * image->height + blockDim.x - 1) / blockDim.x);

    //TODO: perform screen per pixel shading in a separate kernel
    //TODO: Compute radius parallelly: compare kernel
    //TODO: Build a data structure where every pixel has a list of circles that intersect it// 
    //: first group boundbox = per circle, then assign task kernel
    //TODO: Get Job Done // render kernel: per pixel 

    unsigned int* pixel_circle_list = nullptr;
    cudaCheckError(cudaMalloc(&pixel_circle_list, image->width * image->height * sizeof(unsigned int)));
    cudaCheckError(cudaMemset(pixel_circle_list, 0, image->width * image->height * sizeof(unsigned int)));

    unsigned int* buf = nullptr;
    cudaCheckError(cudaMalloc(&buf, image->width * image->height * 1024 * sizeof(unsigned int)));
    cudaCheckError(cudaMemset(buf, 0, image->width * image->height * 1024 * sizeof(unsigned int)));

    unsigned int* buf_size = nullptr;
    cudaCheckError(cudaMalloc(&buf_size, sizeof(unsigned int)));
    cudaCheckError(cudaMemset(buf_size, 0, sizeof(unsigned int)));

    unsigned int* pixel_circle_list_size = nullptr;
    cudaCheckError(cudaMalloc(&pixel_circle_list_size, image->width * image->height * sizeof(unsigned int)));
    cudaCheckError(cudaMemset(pixel_circle_list_size, 0, image->width * image->height * sizeof(unsigned int)));

    unsigned int* pixel_circle_list_capacity = nullptr;
    cudaCheckError(cudaMalloc(&pixel_circle_list_capacity, image->width * image->height * sizeof(unsigned int)));
    cudaCheckError(cudaMemset(pixel_circle_list_capacity, 0, image->width * image->height * sizeof(unsigned int)));

    kernelInitializePixelCircleList<<<gridDim, blockDim>>>(pixel_circle_list,
                                                           pixel_circle_list_capacity,
                                                           buf,
                                                           buf_size,
                                                           numCircles);    
    cudaCheckError(cudaDeviceSynchronize());

    kernelAssignPixels<<<(numCircles + blockDim.x - 1) / blockDim.x, blockDim>>>(pixel_circle_list,
                                                                                 pixel_circle_list_size,
                                                                                 pixel_circle_list_capacity,
                                                                                 buf,
                                                                                 buf_size,
                                                                                 numCircles);
    cudaCheckError(cudaDeviceSynchronize());

    fprintf(stderr, "fuck\n");

    // kernelSortCircles<<<gridDim, blockDim>>>(pixel_circle_list, pixel_circle_list_size, buf, numCircles);
    // cudaCheckError(cudaDeviceSynchronize());

    // float4* localPixels = nullptr;
    // cudaMalloc(&localPixels, image->width * image->height * sizeof(float4));
    // cudaMemset(localPixels, 0, image->width * image->height * sizeof(float4));

    kernelShadePixels<<<gridDim, blockDim>>>(pixel_circle_list, pixel_circle_list_size, buf, image->width, image->height);
    cudaCheckError(cudaDeviceSynchronize());

    // kernelDeletePixelCircleList<<<gridDim, blockDim>>>(pixel_circle_list, numCircles);
    // cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaFree(pixel_circle_list));
    cudaCheckError(cudaFree(buf));
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
