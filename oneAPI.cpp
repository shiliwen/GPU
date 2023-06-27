#include <CL/sycl.hpp>
#include <iostream>
#include <random>

using namespace std;
using namespace sycl;

constexpr int N = 2048;
float A[N][N];

void reset()
{
    A[0][0] = 0.0;
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

void gauss_gpu(float* m, int n, queue& q)
{
    device my_device = q.get_device();
    for (int k = 0; k < n; k++)
    {
        q.submit([&](handler& h) {
            h.parallel_for(range(n - k), [=](auto idx) {
                int j = k + idx; m[k * n + j] = m[k * n + j] / m[k * n + k];
                });
            });
        q.submit([&](handler& h) {
            h.parallel_for(range(n - (k + 1), n - (k + 1)), [=](auto idx) {
                int i = k + 1 + idx.get_id(0);
                int j = k + 1 + idx.get_id(1);
                m[i * n + j] = m[i * n + j] - m[i * n + k] * m[k * n + j];
                });
            });
        q.submit([&](handler& h) {
            h.parallel_for(range(n - (k + 1)), [=](auto idx) {
                int i = k + 1 + idx; m[i * n + k] = 0;
                });
            });
    }
    q.wait();
}

int main() {


    default_selector selector;
    queue q{ property::queue::in_order() };
    reset();


    float* A_buffer = (float*)malloc_shared(N * N * sizeof(float), q);
    auto start = chrono::high_resolution_clock::now();

    gauss_gpu(A_buffer, N, q);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "time: " << duration.count() * 1000 << "ms" << std::endl;

    return 0;
}