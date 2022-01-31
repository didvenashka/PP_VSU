#ifndef __CUDACC__  
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <Windows.h>
using namespace std;

#define blockCount 1u
#define threadsCount 1024u
#define arraySize 128u
#define START 1
#define END 10
#define H float(END - START) / arraySize

#define F(x, y) (sin(x)*sin(x)+cos(y)*cos(y))
/* n = 128 максимум т.к.n x n 16384 = 16 блоков на 1024 нити в каждом блоке(параметры нашего устройства)
also it`ll cause constant memory overflow. as it only 64kb of constant memory avaible that`senough only for 16384 cells in float matrix*/

float a[arraySize * arraySize];
__constant__ float dev_a[arraySize * arraySize];

cudaError_t cudaRuner(float* in, unsigned int size);


float funcWithoutCuda(float* a, unsigned int size)
{
    //float result = 0; // <- ломает счет на наборах ~> (size:100, a[:][:] = 98)
    double result = 0;
    double a_elem = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            a_elem = a[i * size + j];
            a_elem = a_elem * H * H;
            result += a_elem;
        }
    }
    return result;
}

void printLine()
{
    printf("\n=========================\n\n");
}

int sum(float* a, int size)
{
    float res = 0;
    for (int i = 0; i < size * size; i++)
        res += a[i];
    return res;
}

__global__ void squareKernel_C_S(float* dev_out)
{
    int allthreadsCount = blockCount * threadsCount;
    int blockSize = blockDim.x;
    __shared__ float shared_a[threadsCount];
    int i = threadIdx.x;
    int j = blockIdx.x;
    int id = i + blockSize * j;
    int k = 0;
    float dev_a_elem;
    shared_a[i] = 0;
    while (k* allthreadsCount + id < arraySize * arraySize)
    {
        dev_a_elem = dev_a[id + k * allthreadsCount];
        shared_a[i] += dev_a_elem * H * H;
        k++;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (i < s)
            shared_a[i] += shared_a[i + s];
        __syncthreads();
    }
    if (i == 0)
        dev_out[j] += shared_a[0];
}

__global__ void squareKernel_C(float* dev_out)
{
    int allthreadsCount = blockCount * threadsCount;
    int blockSize = blockDim.x;
    float _a;
    int i = threadIdx.x;
    int j = blockIdx.x;
    int id = i + blockSize * j;
    int k = 0;
    float dev_a_elem;
    _a = 0;
    while (k * allthreadsCount + id < arraySize * arraySize)
    {
        dev_a_elem = dev_a[id + k * allthreadsCount];
        _a += dev_a_elem * H * H;
        k++;
    }
    atomicAdd(&dev_out[j], _a);
}

__global__ void squareKernel_G_S(float* in, float* dev_out)
{
    int allthreadsCount = blockCount * threadsCount;
    int blockSize = blockDim.x;
    __shared__ float shared_a[threadsCount];
    int i = threadIdx.x;
    int j = blockIdx.x;
    int id = i + blockSize * j;
    int k = 0;
    float in_elem;
    shared_a[i] = 0;
    while (k* allthreadsCount + id < arraySize * arraySize)
    {
        in_elem = in[id + k * allthreadsCount];
        shared_a[i] += in_elem * H * H;
        k++;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (i < s)
            shared_a[i] += shared_a[i + s];
        __syncthreads();
    }
    if (i == 0)
    {
        dev_out[j] += shared_a[0];
    }
}

__global__ void squareKernel_G(float* in, float* dev_out)
{
    int allthreadsCount = blockCount * threadsCount;
    int blockSize = blockDim.x;
    float _a;
    int i = threadIdx.x;
    int j = blockIdx.x;
    int id = i + blockSize * j;
    int k = 0;
    float dev_a_elem;
    _a = 0;
    while (k * allthreadsCount + id < arraySize * arraySize)
    {
        dev_a_elem = in[id + k * allthreadsCount];
        _a += dev_a_elem * H * H;
        k++;
    }
    atomicAdd(&dev_out[j], _a);
}

int main()
{
    //printf("%f\n", H);
    system("pause");
    float b[arraySize * arraySize] = {};
    //int c;

    for (int i = 0; i < arraySize; i++)
    {
        for (int j = 0; j < arraySize; j++)
        {
            a[i * arraySize + j] = F(START + i * H, START + j * H);
            b[i * arraySize + j] = F(START + i * H, START + j * H);
        }
    }



    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("%d - threads in block\n", prop.maxThreadsPerBlock);
    printf("%d mb - Avaible global memory\n", prop.totalGlobalMem / 1024 / 1024);
    printf("%d kb - Avaible constant memory\n", prop.totalConstMem / 1024);
    printf("%d - array size\n", sizeof(a));

    printLine();

    double start = omp_get_wtime();
    double result_on_cpu;
    result_on_cpu = funcWithoutCuda(b, arraySize); 
    double end = omp_get_wtime();
    printf("%.0f - result on cpu for %.3f ms\n", result_on_cpu, (end - start)*1000);
    printLine();

    // Add vectors in parallel.
    cudaError_t cudaStatus = cudaRuner(a, arraySize * arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t cudaRuner(float* in, unsigned int size)
{
    float* dev_out = 0;
    //int *dev_b = 0;
    //int *dev_c = 0;
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float gpuTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //-----------------------------------
    //------constant shared part---------
    //-----------------------------------


    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    float preOut[blockCount] = { 0 };


    cudaStatus = cudaMalloc((void**)&dev_out, blockCount * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaStatus = cudaMemcpyToSymbol(dev_a, a, sizeof(a), 0, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed!");
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    printf("\tTime spent constant + shared %.3f ms for move data from Host to Devise\n", gpuTime);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    squareKernel_C_S << <blockCount, threadsCount >> > (dev_out);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\tTime spent constant + shared % .3f ms for get result:  ", gpuTime);
    float* host_out = new float[blockCount];


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaStatus = cudaMemcpy(host_out, dev_out, blockCount * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float res = 0;
    for (int i = 0; i < blockCount; i++)
        res += host_out[i];
    printf("%.0f\n", res);

    printf("\tTime spent constant + shared % .3f ms move result from Devise to Host\n", gpuTime);


    //-----------------------------------
    //--------constant part--------------
    //-----------------------------------

    printf("--------------------------------\n");

    float* conts_dev_out = 0;
    //int *dev_b = 0;
    //int *dev_c = 0;

    float conts_preOut[blockCount] = { 0 };

    cudaStatus = cudaMalloc((void**)&conts_dev_out, blockCount * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaStatus = cudaMemcpyToSymbol(dev_a, a, sizeof(a), 0, cudaMemcpyHostToDevice);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("\tTime spent constant %.3f ms for move data from Host to Device\n", gpuTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    squareKernel_C << <blockCount, threadsCount >> > (conts_dev_out);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("\tTime spent constant %.3f ms for get result: ", gpuTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float* conts_host_out = new float[blockCount];



    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaStatus = cudaMemcpy(conts_host_out, conts_dev_out, blockCount * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float conts_res = 0;
    for (int i = 0; i < blockCount; i++)
        conts_res += conts_host_out[i];
    printf("%.0f\n", conts_res);

    printf("\tTime spent constant %.3f ms for move data from Device to Host\n", gpuTime);


    //-----------------------------------
    //------global shared part-----------
    //-----------------------------------

    printf("--------------------------------\n");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float* input_global_shared = NULL;
    cudaStatus = cudaMalloc(&input_global_shared, size * sizeof(in));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    float* output_global_shared;
    cudaStatus = cudaMalloc(&output_global_shared, blockCount * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaStatus = cudaMemcpy(input_global_shared, in, size * sizeof(in), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("\tTime spent global + shared %.3f ms for move data from Host to Device\n", gpuTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    squareKernel_G_S << < blockCount, threadsCount >> > (input_global_shared, output_global_shared);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("\tTime spent global + shared %.3f ms for get result: ", gpuTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float* out_global_shared = new float[blockCount];

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMemcpy(out_global_shared, output_global_shared, blockCount * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMssssemcpy failed!");
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float glob_shared_res = 0;
    for (int i = 0; i < blockCount; i++)
        glob_shared_res += out_global_shared[i];
    printf("%.0f\n", glob_shared_res);
    printf("\tTime spent global + shared %.3f ms for move data from Device to Host\n", gpuTime);


    //-----------------------------------
    //------global shared part-----------
    //-----------------------------------

    printf("--------------------------------\n");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float* input_global = NULL;
    cudaStatus = cudaMalloc(&input_global, size * sizeof(in));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    float* output_global;
    cudaStatus = cudaMalloc(&output_global, blockCount * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(input_global, in, size * sizeof(in), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("\tTime spent global %.3f ms for move data from Host to Device\n", gpuTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    squareKernel_G_S << < blockCount, threadsCount >> > (input_global, output_global);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("\tTime spent global %.3f ms for get result: ", gpuTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float* out_global = new float[blockCount];

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMemcpy(out_global, output_global, blockCount * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMssssemcpy failed!");
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float global_res = 0;
    for (int i = 0; i < blockCount; i++)
        global_res += out_global[i];
    printf("%.0f\n", global_res);
    printf("\tTime spent global %.3f ms for move data from Device to Host\n", gpuTime);



Error:
    //cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_out);
    cudaFree(conts_dev_out);
    //cudaFree(dev_b);

    return cudaStatus;
}
