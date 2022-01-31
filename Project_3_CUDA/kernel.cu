
#include "cuda_runtime.h"
#include <iostream>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>
#include <omp.h>

#define SIZE 40
#define xSIZE (SIZE * SIZE)
#define blockCount (SIZE/1024+1)
#define blockCount_ (xSIZE/1024+1)
#define min(a,b) (a > b ? b : a)
using namespace std;

cudaError_t addWithCuda(int* c, int* c_, unsigned int size);

void copy(int** from, int** to)
{
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            to[i][j] = from[i][j];
}
void copy(int** from, int* to)
{
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            to[i * SIZE + j] = from[i][j];
}

void print(int** matr)
{
    for (int i = 0; i < SIZE; i++)
    {
        printf("\n");
        for (int j = 0; j < SIZE; j++)
            printf("%d \t", matr[i][j]);
    }
}
void print(int* matr)
{
    for (int i = 0; i < SIZE; i++)
    {
        printf("\n");
        for (int j = 0; j < SIZE; j++)
            printf("%d \t", matr[i*SIZE + j]);
    }
}

void print_graph(int** matr)
{
    for (int i = 0; i < SIZE; i++)
        for (int j = i; j < SIZE; j++)
            if (matr[i][j])
                printf("\n %d - %d : %d\n", i + 1, j + 1, matr[i][j]);
}
void print_graph(int* matr)
{
    for (int i = 0; i < SIZE; i++)
        for (int j = i; j < SIZE; j++)
            if (matr[i * SIZE + j])
                printf("\n %d - %d : %d\n", i + 1, j + 1, matr[i * SIZE + j]);
}


float floyd(int** matr)
{
    double start_time, end_time, duration;
    start_time = omp_get_wtime();

    for (int v = 0; v < SIZE; v++)
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                if (matr[i][v] && matr[j][v])
                    matr[i][j] = min(matr[i][j], (matr[i][v] + matr[j][v]));

    end_time = omp_get_wtime();
    duration = end_time - start_time;
    return duration*1000;
}

void init(int** matr, string patern = "zeros", bool directed = false) // avaible paterns {zeros/ones/random/increasing} 
{
    if (patern == "zeros")
    {
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                matr[i][j] = 0;
    }
    else if (patern == "ones")
    {
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                matr[i][j] = 1;
    }
    else if (patern == "random")
    {
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                matr[i][j] = rand() % 20 + 5;
    }
    else if (patern == "increasing")
    {
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                matr[i][j] = i / 3 + j / 2 + 1;
    }
    else printf("\n::: Wrond parametr <patern> in init func\n");
    if (directed)
    {
        for (int i = 0; i < SIZE; i++)
            matr[i][i] = NULL;
    }
    else
    {
        for (int i = 0; i < SIZE; i++)
        {
            matr[i][i] = NULL;
            for (int j = i; j < SIZE; j++)
                matr[j][i] = matr[i][j];
        }
    }
}

__global__ void addKernel(int* matr) 
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    int id = j * blockDim.x + i;
    int v = id % SIZE;
    int smth, left, right, min_;
    if (id < SIZE)
    {
        for (int k = 0; k < SIZE; k++)
            for (int l = 0; l < SIZE; l++) 
            {
                //printf("%d==========\n", matr[k * SIZE + v]);
                if (matr[k * SIZE + v] && matr[l * SIZE + v])
                {
                    left = matr[k * SIZE + l];
                    right = (matr[k * SIZE + v] + matr[l * SIZE + v]);
                    min_ = min(left, right);
                    matr[k * SIZE + l] = int(min_);
                    //printf("%d <> %d   = %d  :  %d    - %d\n", left, right, matr[k * SIZE + l], min_, v);
                }
            }
    }
}

__global__ void addKernel_(int* matr) 
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    int id = j * blockDim.x + i;
    int m_i = id / SIZE;
    int m_j = id % SIZE;
    int smth, left, right, min_;
    if (id < xSIZE) 
    {
        for (int v = 0; v < SIZE; v++)
            if (matr[m_i * SIZE + v] && matr[m_i * SIZE + v])
            {
                left = matr[m_i * SIZE + m_j];
                right = (matr[m_i * SIZE + v] + matr[m_j * SIZE + v]);
                min_ = min(left, right);
                matr[m_i * SIZE + m_j] = min_;
                //printf("%d <> %d   = %d  :  %d    - %d\n", left, right, matr[m_i * SIZE + m_j], min_, v);
            }
    }
}

int main()
{
    system("pause");


    int** a;
    int** b;
    int* c;
    int* c_;
    a = new int* [SIZE];
    b = new int* [SIZE];
    c = new int[xSIZE];
    c_ = new int[xSIZE];
    for (int k = 0; k < SIZE; k++)
    {
        a[k] = new int[SIZE];
        b[k] = new int[SIZE];
    }
    init(a, "random", false);
    copy(a, b);
    copy(a, c);
    copy(a, c_);



    double cpuTime = floyd(b);
    printf("\t\t work time on CPU is %.3f ms\n", cpuTime);

    
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, c_,  xSIZE);
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
    
    if (SIZE < 6)
    {
        printf("start matrix : \n");
        print(a);
        printf("\n---------------------------\n");

        printf("\n\nafter non-parallel : \n");
        print(b);
        printf("\n---------------------------\n");
        print_graph(b);
        printf("\n---------------------------\n");

        cout << "\n\nafter parallel : \n";
        print(c_);
        cout << "\n---------------------------\n";
        print_graph(c_);
        cout << "\n---------------------------\n\n";

    }


    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, int* c_, unsigned int size)
{
    int *dev_c = 0;
    float gpuTime;
    cudaEvent_t start, stop;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc(&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_c, c, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<blockCount, 1024>>>(dev_c);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\t\t work time on GPU default is %.3f ms\n", gpuTime);

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

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }



    /*-----------------------------------------------------------
    -------------------------------------------------------------
    -------------------------------------------------------------
    -------------------------------------------------------------
    -----------------------------------------------------------*/


    int* dev_c_ = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc(&dev_c_, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_c_, c_, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch a kernel on the GPU with one thread for each element.
    addKernel_ <<<blockCount_, 1024 >>> (dev_c_);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);    
    cudaEventElapsedTime(&gpuTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\t\t work time on GPU modified is %.3f ms\n", gpuTime);

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

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c_, dev_c_, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


Error:
    cudaFree(dev_c);
    
    return cudaStatus;
}
