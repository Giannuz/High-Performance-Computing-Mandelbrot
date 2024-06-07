Vectorized:
- mandelbrot: original [not 10k-10k]
- mandelbrot_general: corrected DEGREE variable usage [not considered]
- mandelbrot1: moved one variable outside for loop (step_width) [1k-1k and 1k-10k only]
- mandelbrot2: splitted for loop in two for loops (x and y) [1k-1k and 1k-10k only]
- mandelbrot3: moved one variable outside for loop on x (row) [1k-1k and 1k-10k only]
- mandelbrot4: divided the calculation and mirrored the result on the real axis [1k-1k and 1k-10k only]
- mandelbrot5: moved one variable outside for loop on x (y2) [1k-1k and 1k-10k only]
- mandelbrot6: replaced complex library usage with four double variables (and squared the if condition for simplicity) [considered as reference, 1k-1k and 1k-10k only]
- mandelbrot7: splitted calculations in the iteration for loop [1k-1k and 1k-10k only]
- mandelbrot8a: moved the mirrored calculation of the results into a separate loop. [1k-1k and 1k-10k only]
- mandelbrot8b: implemented array allignment [not working for our RESOLUTION size, not considered]

Parallelized (Base):
- mandelbrot6: same as vectorized version, but with OpenMP pragma [1k-1k and 1k-10k only]
- mandelbrot7a: same as vectorized version, but with OpenMP pragma [1k-1k and 1k-10k only]
- mandelbrot7b: same as 7a, but with also simd option [1k-1k and 1k-10k only]
- mandelbrot8a: same as vectorized version, but with OpenMP pragma [1k-1k and 1k-10k only]
Parallelized (Scheduling):
- mandelbrot6: same as parallelized (base) version, but with also scheduling option (dynamic, 1)

Cuda:
- mandelbrot6a: same as vectorized version, but with one dimension on the CUDA function call (as before mandelbrot2) [1k-1k and 1k-10k only]
- mandelbrot6b: same as vectorized version, but with CUDA function call [1k-1k and 1k-10k only]
- mandelbrot6c: same as 6b, but with the mirrored calculation of the results on the CPU (like in mandelbrot8a)
- mandelbrot6d: same as 6c, but with also stripe factor [1k-1k and 1k-10k only]