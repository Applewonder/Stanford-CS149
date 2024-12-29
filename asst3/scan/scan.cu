#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS 32768

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void
upsweep_kernel(int instance, int N, int next_pow2, int two_d, int* tmp, int* result) {
    int index = instance * MAX_BLOCKS + blockIdx.x * blockDim.x + threadIdx.x;
    int i = index * two_d * 2;
    // if (i < N && result[i+two_d-1] > 0) {
    //     int tmp = result[i+two_d*2-1] + result[i+two_d-1];
    //     if (tmp > 0) {
    //         result[i+two_d*2-1] = tmp;
    //     }
    // }
    if (i < next_pow2) {
        int index1 = i+two_d-1;
        int index2 = i+two_d*2-1;
        int* ptr1 = index1 < N ? result + index1 : tmp + index1 - N;
        int* ptr2 = index2 < N ? result + index2 : tmp + index2 - N;
        *ptr2 += *ptr1;
    }
}

__global__ void
downsweep_kernel(int instance, int N, int next_pow2, int two_d, int* tmp, int* result) {
    int index = instance * MAX_BLOCKS + blockIdx.x * blockDim.x + threadIdx.x;
    int i = index * two_d * 2;
    if (i < next_pow2) {
        int index1 = i+two_d-1;
        int index2 = i+two_d*2-1;
        int* ptr1 = index1 < N ? result + index1 : tmp + index1 - N;
        int* ptr2 = index2 < N ? result + index2 : tmp + index2 - N;
        int t = *ptr1;
        *ptr1 = *ptr2;
        *ptr2 += t;
    }
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.

    int* tmp = nullptr;
    int next_pow2 = nextPow2(N);
    int tmp_size = next_pow2 - N;
    int sign = cudaMalloc(&tmp, tmp_size * sizeof(int));
    if (sign == cudaErrorMemoryAllocation) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }

    // int N = end - start;
    // cudaMemcpy(tmp, input, N*sizeof(int), cudaMemcpyDeviceToDevice);
    // cudaMemset(tmp+N, 0, (next_pow2-N)*sizeof(int));
    // return;
    // memmove(result, input, N*sizeof(int));
    cudaMemcpy(result, input, N*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemset(tmp, 0, tmp_size * sizeof(int));

    // upsweep phase
    for (int two_d = 1; two_d <= next_pow2/2; two_d*=2) {
        int num_blocks = (((next_pow2 + 2 * two_d - 1)/(2 * two_d)) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        // int two_dplus1 = 2*two_d;
        // parallel_for (int i = 0; i < N; i += two_dplus1) {
        //     output[i+two_dplus1-1] += output[i+two_d-1];
        // }
        for (int i = 0; i < num_blocks; i += MAX_BLOCKS) {
            int remaining = num_blocks - i;
            upsweep_kernel<<<remaining < MAX_BLOCKS ? remaining : MAX_BLOCKS, THREADS_PER_BLOCK>>>(i, N, next_pow2, two_d, tmp, result);
            cudaDeviceSynchronize();
        }
    }
    // return;

    // result[N-1] = 0;
    // int zero = 0;
    // cudaMemcpy(result+N-1, &zero, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemset(tmp+next_pow2-1, 0, sizeof(int));
    if (N < next_pow2) {
        cudaMemset(tmp + next_pow2 - N - 1, 0, sizeof(int));
    } else {
        cudaMemset(result + next_pow2 - 1, 0, sizeof(int));
    }

    // downsweep phase
    for (int two_d = next_pow2/2; two_d >= 1; two_d /= 2) {
        int num_blocks = (((next_pow2 + 2 * two_d - 1)/(2 * two_d)) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        // int two_dplus1 = 2*two_d;
        // parallel_for (int i = 0; i < N; i += two_dplus1) {
        //     int t = output[i+two_d-1];
        //     output[i+two_d-1] = output[i+two_dplus1-1];
        //     output[i+two_dplus1-1] += t;
        // }
        // downsweep_kernel<<<(((next_pow2+2*two_d-1)/(2*two_d)) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(next_pow2, two_d, tmp);
        for (int i = 0; i < num_blocks; i += MAX_BLOCKS) {
            int remaining = num_blocks - i;
            downsweep_kernel<<<remaining < MAX_BLOCKS ? remaining : MAX_BLOCKS, THREADS_PER_BLOCK>>>(i, N, next_pow2, two_d, tmp, result);
            cudaDeviceSynchronize();
        }
    }

    // cudaMemcpy(result, tmp, N*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaFree(tmp);
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}

__global__ void
find_repeats_kernel(int N, int* input, int* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        result[index] = (input[index] == input[index+1]) ? 1 : 0;
    }
}

__global__ void
find_repeats_index_kernel(int N, int* repeat_i, int* index, int* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (repeat_i[i]) {
            result[index[i]] = i;
        }
    }
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    // parallel_for (int i = 0; i < length - 1; i++) {
    //     tmp_output[i] = (device_input[i] == device_input[i+1]) ? 1 : 0;
    // }
    int* tmp_output = nullptr;
    int* index = nullptr;
    cudaMalloc(&tmp_output, length * sizeof(int));
    cudaMalloc(&index, length * sizeof(int));

    find_repeats_kernel<<<(length + THREADS_PER_BLOCK - 2) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(length - 1, device_input, tmp_output);
    cudaDeviceSynchronize();

    exclusive_scan(tmp_output, length, index);
    cudaDeviceSynchronize();

    find_repeats_index_kernel<<<(length + THREADS_PER_BLOCK - 2) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(length - 1, tmp_output, index, device_output);
    cudaDeviceSynchronize();
    
    int total_repeat_num = 0;
    cudaMemcpy(&total_repeat_num, index + length - 1, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(tmp_output);
    cudaFree(index);

    return total_repeat_num; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
