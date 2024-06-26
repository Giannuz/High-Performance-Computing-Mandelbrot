#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

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

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations

using namespace std;

// we no longer use the complex library

int main(int argc, char **argv)
{
    int *const image = new int[HEIGHT * WIDTH];

    const auto start = chrono::steady_clock::now();

    const int HALF_HEIGHT = ceil((float) HEIGHT/2);

    for (int y = 0; y < HALF_HEIGHT; y++)
    {
        const double ci = y * STEP + MIN_Y; // imaginary part of c in the form of double
        const int y2 = HEIGHT - y - 1;
        
        for (int x = 0; x < WIDTH; x++)
        {
            const int pos1 = y * WIDTH + x,
                      pos2 = y2 * WIDTH + x; // we exploit the simmetry of complex numbers: calculate only half figure and then mirror the results
            image[pos1] = image[pos2] = 0;

            const double cr = x * STEP + MIN_X;

            // z = z^2 + c
            double zr = 0.0, zi = 0.0, zr2 = 0.0, zi2 = 0.0;
            for (int i = 1; i <= ITERATIONS; i++)
            {
                zi = 2 * zr * zi + ci; // imaginary part of z in the form of double
                zr = zr2 - zi2 + cr;

                zr2 = zr * zr;
                zi2 = zi * zi;

                // If it is convergent
                if (zr2 + zi2 >= 4) // we no longer use the expensive abs(z) >= 2
                {
                    image[pos1] = image[pos2] = i;
                    break;
                }
            }
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