#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cuda.h>

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#define RESOLUTION 1000
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)
#define HALF_HEIGHT ceil((float) HEIGHT/2)

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations

using namespace std;

__global__ void cuda_mandelbrot(int* image)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < WIDTH && y < HALF_HEIGHT)
    {        
        const int pos = y * WIDTH + x;
        image[pos] = 0;

        const double cr = x * STEP + MIN_X, 
                     ci = y * STEP + MIN_Y;
        
        // z = z^2 + c
        double zr = 0.0, zi = 0.0, zr2 = 0.0, zi2 = 0.0;
        for (int i = 1; i <= ITERATIONS; i++)
        {
            zi = 2 * zr * zi + ci;
            zr = zr2 - zi2 + cr;

            zr2 = zr * zr;
            zi2 = zi * zi;

            // If it is convergent
            if (zr2 + zi2 >= 4)
            {
                image[pos] = i;
                break;
            }
        }
    }
}

int main(int argc, char **argv)
{
    int *const image = new int[HEIGHT * WIDTH];

    const auto start = chrono::steady_clock::now();

    float stripeFact = 1;
    dim3 threads(128, 8);
    dim3 blocks((WIDTH - 1) * stripeFact/threads.x + 1, (HALF_HEIGHT - 1) * stripeFact/threads.y + 1);

    int *cuda_image;

    cudaMalloc(&cuda_image, HALF_HEIGHT * WIDTH * sizeof(int));
    cuda_mandelbrot<<<blocks, threads>>>(cuda_image);
    cudaMemcpy(image, cuda_image, HALF_HEIGHT * WIDTH * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(cuda_image);

    for (int y = 0; y < HALF_HEIGHT; y++)
    {
        const int y_comp = y * WIDTH,
                  y2 = HEIGHT - y - 1;
        for (int x = 0; x < WIDTH; x++)
        {
            image[y2] = image[y_comp + x];
        }
    }

    const auto end = chrono::steady_clock::now();
    cout << "Time elapsed: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " milliseconds." << endl;

    // Write the result to a file
    ofstream matrix_out;

    if (argc < 2)
    {
        cout << "Please specify the output file as a parameter." << endl;
        return -1;
    }

    matrix_out.open(argv[1], ios::trunc);
    if (!matrix_out.is_open())
    {
        cout << "Unable to open file." << endl;
        return -2;
    }

    for (int row = 0; row < HEIGHT; row++)
    {
        for (int col = 0; col < WIDTH; col++)
        {
            matrix_out << image[row * WIDTH + col];

            if (col < WIDTH - 1)
                matrix_out << ',';
        }
        if (row < HEIGHT - 1)
            matrix_out << endl;
    }
    matrix_out.close();

    delete[] image; // It's here for coding style, but useless
    return 0;
}