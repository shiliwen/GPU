#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include <iomanip>
#include <windows.h>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
using namespace std;
const int N = 3072;
const int BLOCK_SIZE = 32;
float A[N][N];

void reset()
{
    A[0][0] = 0;
    for (int i = 0; i < N; i++)
    {
        A[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
        {
            A[i][j] = rand();
        }
    }

    for (int k = 0; k < N; k++)
    {

        for (int i = k + 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A[i][j] += A[k][j];

            }
        }

    }
}


__global__ void division_kernel(float* data, int k, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;//计算线程索引
    int element = data[k * N + k];
    int temp = data[k * N + tid];
    data[k * N + tid] = (float)temp / element;

    return;
}
__global__ void eliminate_kernel(float* data, int k, int N) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tx == 0)  data[k * N + k] = 1.0;//对角线元素设为 1 

    int row = k + 1 + blockIdx.x;//每个块负责一行
    while (row < N) {
        int tid = threadIdx.x;
        while (k + 1 + tid < N) {
            int col = k + 1 + tid;
            float temp_1 = data[(row * N) + col];
            float temp_2 = data[(row * N) + k];
            float temp_3 = data[k * N + col];
            data[(row * N) + col] = temp_1 - temp_2 * temp_3;
            tid = tid + blockDim.x;
        }
        __syncthreads();//块内同步
        if (threadIdx.x == 0) {
            data[row * N + k] = 0;
        }
        row += gridDim.x;
    }
    return;
}
int main()
{
    float* gpudata;
    reset();

    dim3 dimBlock(BLOCK_SIZE, 1, 1);//dimBlock的三个参数分别表示线程块在 x、y、z 方向上的大小
    dim3 dimGrid(N / BLOCK_SIZE, 1, 1);//dimGrid的三个参数分别表示线程网格在 x、y、z 方向上的大小。

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaError_t ret;
    ret = cudaMalloc(&gpudata, N * N * sizeof(float));
    ret = cudaMemcpy(gpudata, A, N * N, cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);

    long long head, tail, freq; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    for (int k = 0; k < N; k++) {

        division_kernel << <dimGrid, dimBlock >> > (gpudata, k, N);//负责除法任务的核函数

        cudaDeviceSynchronize();//CPU 与 GPU 之间的同步函数
        ret = cudaGetLastError();
        if (ret != cudaSuccess) {
            printf("division_kernel failed, %s\n", cudaGetErrorString(ret));
        }

        eliminate_kernel << <dimGrid, dimBlock >> > (gpudata, k, N);//负责消去任务的核函数

        cudaDeviceSynchronize();
        ret = cudaGetLastError();
        if (ret != cudaSuccess) {
            printf("eliminate_kernel failed, %s\n", cudaGetErrorString(ret));
        }
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "GPU_Time:" << ((tail - head) * 1000.0) / freq << "ms" << endl;

    cudaEventRecord(stop, 0);
    ret = cudaMemcpy(A, gpudata, N * N, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}