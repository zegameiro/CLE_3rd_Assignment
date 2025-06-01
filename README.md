# Third Assignment for the Large-Scale Computation 2024/2025

This assignment consists in the implementation of a solution for the Canny Edge Detection algorithm using CUDA. This algorithm detects edges in  grayscale images using a multi-stage process.

---

## Table of Contents

- [Third Assignment for the Large-Scale Computation 2024/2025](#third-assignment-for-the-large-scale-computation-20242025)
    - [Table of Contents](#table-of-contents)
    - [File Structure](#file-structure)
    - [Implementation Details](#implementation-details)
        - [Gaussian Kernel Generation (`gaussianKernel`)](#gaussian-kernel-gneration-gaussiankernel)
        - [Convolution (`convolutionKernel`)](#convolution-convolutionkernel)
        - [Normalization (`normalizeKernel`)](#normalization-normalizekernel)
        - [Min/Max Reduction using Shared Memory (`minMaxKernel` and `cudaMinMax`)](#minmax-reduction-using-shared-memory-minmaxkernel-and-cudaminmax)
        - [Min/Max Reduction using Global Memory (`minMaxKernel_no_shared` and `cudaMinMax_noShared`)](#minmax-reduction-using-global-memory-minmaxkernel_no_shared-and-cudaminmax_noshared)
        - [Gradient Calculation (`convolutionKernel` with Sobel and `mergeGradientsKernel`)](#gradient-calculation-convolutionkernel-with-soberl-and-mergegradientskernel)
        - [Non-Maximum Suppression (`nonMaximumSuppressionKernel`)](#non-maximum-suppression-nonmaximumsuppressionkernel)
        - [First Edges (`firstEdgesKernel`)](#first-edges-firstedgeskernel)
        - [Hysteresis (`hysteresisKernel`)](#hysteresis-hysteresiskernel)
    - [Workflow](#workflow)
    - [Testing - Enhanced Performance Analysis](#testing---enhanced-performance-analysis)
        - [Performance Summary](#performance-summary)
        - [Key Conclusions from Performance Analysis](#key-conclusions-from-performance-analysis)
        - [Optimal Parameter Values](#optimal-parameter-values)
        - [Impact of Shared Memory Optimization](#impact-of-shared-memory-optimization)
    - [Compilation and Execution](#compilation-and-execution)
        - [Command-Line Arguments](#command-line-arguments)
        - [Example Usage](#example-usage)
        - [Run the tests](#run-the-tests)
    - [Authors](#authors)

---

## File Structure

| File | Purpose |
| :---: | :---: |
| `canny.cu` | Contains the main program logic, command-line parsing, timing measurements, and the reference CPU implementation |
| `image.c` | Contains some functions to load and save `.ppm` images  |
| `canny-device.cu` | Implements all CUDA kernels and the main GPU function that manages the GPU-based edge detection |

---

## Implementation Details

### Gaussian Kernel Gneration (`gaussianKernel`)

Computes the coefficients of a 2D Gaussian filter on the GPU, used for image smoothing.

- Each thread computes one coefficient based on its **(i, j)** position. The kernel is normalized after the generation.
- For large kernels, generating on the GPU avoids unnecesssary host-device transfers and leverages parallelism.

### Convolution (`convolutionKernel`)

Applies a convolution (Gaussian or Sobel) to the image.
- Each thread computes the convolution sum for one output pixel, iterating over the kernel window.
- Threads skip borders to avoid out-of-bounds memory access.

### Normalization (`normalizeKernel`)

Scales pixel values to the **[0, 255]** range after filtering or gradient calculation.

- Each thread normalizes one pixel, using **min/max values** found by `cudaMinMax`.
- Only valid pixels, meaning that borders are excluded, are normalized.

### Min/Max Reduction using Shared Memory (`minMaxKernel` and `cudaMinMax`)

Finds the minimum and maximum pixel values in an image for normalization.

- Through `extern __shared__ pixel_t sdata[]` each block loads a chunk of the image into shared memory arrays for min and max values.
- Threads within a block cooperate to find the local minimum and maximum values.
. The first thread in each block uses `atomicMin` and `atomicMax` to update global arrays, ensuring correctness when multiple blocks write results.
- In this kernel, using **shared memory** is much faster than **global memory**, because it avoids performing all reductions in global memory, meaning that it allows efficient reduction across blocks.
- **Atomics** are necessary to avoid race conditions when multiple blocks write their min/max to global memory. This ensures that the final min and max values are correct, even with many blocks running in parallel.

### Min/Max Reduction using Global Memory (`minMaxKernel_no_shared` and `cudaMinMax_noShared`)

Finds the minimum and maximum pixel values in an image for normalization, but using only global memory.

- Each thread reads a single pixel from the image and directly attempts to update the global minimum and maximum values in the device memory using `atomicMin` and `atomicMax`.

- All threads operate independently and perform atomic operations on the same global variables.

- **No shared memory** is used, so there is no intra-block cooperation or reduction; all reduction is performed globally with atomics.

### Gradient Calculation (`convolutionKernel` with Soberl and `mergeGradientsKernel`)

Computes horizontal and vertical gradients **(Gx, Gy)** and then the gradien magnitude.

- The Sobel convolution is performed usint the same `convolutionKernel`.
- `mergeGradientsKernel` computes the magnitude using `hypotf` for numerical stability, avoiding overflow issues with large gradients.

### Non-Maximum Suppression (`nonMaximumSuppressionKernel`)

It makes the edges in an image thinner and cleaner by keeping only the strongest parts.

- Each thread computes the gradient direction using `atan2f` and compares the current pixel to its neighbors along the gradient direction.
- If the current pixel is a local maximum in the gradient direction, it is kept; otherwise, it is set to zero.

### First Edges (`firstEdgesKernel`)

Marks strong edge pixels above the high threshold.

- Each thread checks if its pixel is above `tmax` and marks it as strong edge if the condition is met.

### Hysteresis (`hysteresisKernel`)

Connects pixels with weak edges to strong ones, so that important edges are not lost.

- Each thread checks if its pixel is a weak edge and if any of its 8 neighbors is a strong edge.
- If this happens, it promotes itself to a strong edge and sets a flag (`changed`) to indicate that another iteration is needed.

---

## Workflow

The CUDA-based Canny Edge Detection follows these steps:

- **1. Image transfer to GPU**
    - The input grayscale image is copied from host (CPU) memory to the device (GPU) memory.

- **2. Gaussian Smoothing**
    - A 2D Gaussian kernel is generated and normalized on the GPU.
    - The image is convolved with the Gaussian kernel using the `convolutionKernel` to reduce noise.
    - The result is normalized using `normalizeKernel`.

- **3. Gradient Calculation**
    - The smoothed image is convolved with Sobel kernels (horizontal and vertical) to compute the gradients Gx and Gy.
    - The gradient magnitude is computed using `mergeGradientsKernel`.
    - The gradient magnitude is also normalized using `normalizeKernel`.

- **4. Non-Maximum Suppression**
    - The `nonMaximumSuppressionKernel` examines each pixel and its neighbors along the gradient direction.
    - Only pixels that are considered local maxima or strong edges are retained, while others are set to zero.

- **5. Double Thresholding (First Edges)**
    - The `firstEdgesKernel` marks pixels with gradient magnitude above the high threshold as strong edges.

- **6. Edge Tracking by Hysteresis**
    - The `hysteresisKernel` iteratively checks for weak edge pixels (above the low threshold) that are connected to strong edges.
    - If a weak edge is connected to a strong one, then it is promoted to a strong edge.
    - This process continues until no more changes occur, ensuring that only meaningful connected edges remain.

- **7. Result Transfer to Host**
    - The final edge map is copied from device memory back to host memory to be saved.

---

## Testing - Enhanced Performance Analysis

To properly evaluate the CUDA Canny Edge Detection implementation, comprehensive performance measurements were conducted across varying images and algorithmic parameters. The testing methodology involved comparing the CUDA implementation's results with a CPU reference (`canny.cu`), focusing on correctness and performance metrics. Key measurements included processing time for both GPU and CPU implementations, as well as the difference in pixel output between the CUDA result and the CPU reference. Additionally, the algorithm's parameters, such as sigma and hysteresis thresholds, were varied to assess their impact on performance under diverse conditions.

The detailed results are visually presented in the following plots:

![cuda-canny/results](./cuda-canny/results/plots/parameter_analysis.png)
![cuda-canny/results](./cuda-canny/results/plots/parameter_heatmaps.png)
![cuda-canny/results](./cuda-canny/results/plots/timing_by_image.png)

### Performance Summary

**Overall Statistics:**
- **Total Tests:** 2592
- **Mean Speedup:** 21.91x
- **Max Speedup:** 27.60x
- **Min Speedup:** 14.31x
- **Std Speedup:** 1.69x
- **Mean Host Time:** 31.67 ms
- **Mean Device Time:** 1.44 ms

The system demonstrates a substantial performance improvement, achieving an average speedup of 21.91x when utilizing the device compared to the host. The significantly lower mean device time (1.44 ms) compared to the mean host time (31.67 ms) underscores the efficiency of the device in reducing execution time.

**Best Configuration:**
- **Image:** `jetplane.pgm`
- **Sigma:** 1.0
- **Tmin:** 50
- **Tmax:** 130
- **Speedup:** 27.60x

**Worst Configuration:**
- **Image:** `livingroom.pgm`
- **Sigma:** 2.5
- **Tmin:** 60
- **Tmax:** 100
- **Speedup:** 14.31x

### Key Conclusions from Performance Analysis:

* **Significant Speedup Achieved:** The implementation consistently provides substantial speedup, with a mean speedup of 21.91x and a peak of 27.60x. This highlights the effectiveness of CUDA for Canny Edge Detection.
* **Minimal Device Execution Time:** The device consistently completes operations in approximately 1.44 ms, demonstrating its high efficiency across various test conditions.
* **Parameter Impact on Speedup:**
    * Individual variations in `Sigma`, `Tmin`, and `Tmax` within the tested ranges do not drastically alter the absolute host or device execution times, which remain consistently high for the host and low for the device.
    * While raw times show minor changes, the normalized execution times indicate subtle influences of these parameters on the relative performance or workload distribution between host and device.
    * The heatmaps for "Speedup by Parameter Combinations" clearly illustrate that specific combinations of `Sigma`, `Tmin`, and `Tmax` yield higher speedup values, indicating an interplay between these parameters for optimal performance.
* **Image Characteristics are Crucial for Speedup:**
    * The type of input image significantly influences the achieved speedup.
    * The `jetplane.pgm` image, characterized by distinct features and potentially simpler background elements, yielded the highest speedup (27.60x). This suggests that images with strong, well-defined edges or patterns are highly favorable for the device's processing capabilities.
    * Conversely, the `livingroom.pgm` image resulted in the lowest speedup (14.31x). This could be attributed to its more complex and varied textures, numerous small details, or less distinct prominent edges, which may be less efficiently handled by the device's architecture.
    * The "Average Speedup by Image" chart confirms this variability, with other images like `mandril`, `lake`, `pirate`, `house`, `peppers`, and `walkbridge` showing intermediate speedup values.
* **Importance of Image Selection and Parameter Tuning:** The disparity between the best and worst configurations emphasizes the critical role of both the input image's characteristics and the tuning of `Sigma`, `Tmin`, and `Tmax` in achieving optimal performance with the CUDA Canny Edge Detection implementation.

### Optimal Parameter Values

Based on the comprehensive testing, the following parameter values were identified as contributing to the highest observed speedup for the CUDA Canny Edge Detection implementation:

* **Best Performing Single Configuration:**
    * **Sigma: 1.0**
    * **Tmin: 50**
    * **Tmax: 130**
    * These values, in combination with the `jetplane.pgm` image, resulted in the maximum speedup of 27.60x.

* **Average Best Performing Ranges:**
    While a single "best" configuration yielded the absolute maximum speedup, analysis of the parameter heatmaps and individual speedup plots reveals broader ranges of parameters that consistently deliver high performance:
    * **Sigma:** Values between **1.0 and 2.0** consistently show strong speedup.
    * **Tmin:** Values ranging from **50 to 80** generally contribute to high speedup.
    * **Tmax:** Values between **120 and 140** tend to result in excellent speedup when combined with appropriate `Tmin` and `Sigma` values.

The heatmaps visually confirm that these specific ranges, when used together, create a synergistic effect that optimizes performance. For instance, the red (highest speedup) regions in the heatmaps frequently align with `Sigma` values around 1.0-2.0, `Tmin` values around 50-80, and `Tmax` values around 120-140. This indicates that these settings enable the CUDA kernels to process image data most efficiently, maximizing the benefits of GPU acceleration across a range of conditions.

### Impact of Shared Memory Optimization

We decided to evaluate the impact of shared memory utilization in the detection of the minimum and maximum pixel values in an image. For this we executed our solution with the default parameters, but for all the available images, alternating between the shared memory and global memory implementations of the `cudaMinMax` function. The results are as follows:

| Image          | Device Time (ms) | Host Time (ms) | Different Pixels | Shared Memory |
| :------------: | :-------------: | :------------: | :--------------: | :-------------: |
| `house.pgm` | 2.3304 | 36.4083 | 0/262144 (0.00%) | ✔️ |
| `house.pgm`  | 2.5176| 37.9924 | 0/262144 (0.00%) | ❌ |
| `jetplane.pgm` | 2.2858 | 42.1376| 0/262144 (0.00%) | ✔️ |
| `jetplane.pgm`  | 2.4538 | 43.8364 | 0/262144 (0.00%) | ❌ |
| `lake.pgm` | 2.4104 | 44.8205 | 0/262144 (0.00%) | ✔️ |
| `lake.pgm`  | 2.4319 | 41.6696 | 0/262144 (0.00%) | ❌ |
| `livingroom.pgm` | 2.2589 | 38.4666 | 0/262144 (0.00%) | ✔️ |
| `livingroom.pgm`  | 2.4047 | 41.7956 | 0/262144 (0.00%) | ❌ |
| `mandril.pgm` | 2.1313 | 44.3914 | 0/262144 (0.00%) | ✔️ |
| `mandril.pgm`  | 2.5094 | 44.3136 | 0/262144 (0.00%) | ❌ |
| `peppers_gray.pgm` | 2.9987 | 43.7002 | 0/262144 (0.00%) | ✔️ |
| `peppers_gray.pgm` | 2.3057 | 41.6072 | 0/262144 (0.00%) | ❌ |
| `pirate.pgm` | 2.2102 | 42.8472 | 0/262144 (0.00%) | ✔️ |
| `pirate.pgm` | 2.3920 | 44.3238 | 0/262144 (0.00%) | ❌ |
| `walkbridge.pgm` | 2.4345 | 48.5294 | 0/262144 (0.00%) | ✔️ |
| `walkbridge.pgm` | 2.4094 | 49.2554 | 0/262144 (0.00%) | ❌ |

The results demonstrate that using the share memory approach for the `cudaMinMax` function provides a modest but measurable improvement in device execution time across most of the tested images, when compared to the global memory approach. This highlights that shared memory utilization, while it does not changes the overall runtime in this context, it can still provide performance benefits for maximizing CUDA kernel efficiency.

## Compilation and Execution

To compile the program and then execute it, use the following steps:

1. Navigate to the directory containing the `Makefile` file:

```bash
cd cuda-canny
```

2. Compile the program with the following command:

```bash
make
```

3. Run the program with the following command:

```bash
./canny
```
4. Two files will be generated: `out.pgm` (the CUDA result) and `reference.pgm` (the CPU reference result). And the following output will be printed to the terminal:

```bash 
Available devices
->0: NVIDIA GeForce RTX 3050 Laptop GPU (compute 8.6)

gaussian_filter: kernel size 7, sigma=1
Host processing time: 31.373312 (ms)
Device processing time: 2.427424 (ms)

Number of different pixels: 0/262144 (0.00%)
```


### Command-Line Arguments

It is possible to execute the program with some arguments

| Argument | Description |
| :---: | :---: |
| `-d <device>` | Select CUDA device (default: 0) |
| `-i <inputFile>` | Input image filename (default [lake.pgm](./cuda-canny/images/lake.pgm)) |
| `-o <outputFile>` | Output image filename for the CUDA result (default: `out.pgm`) |
| `-r <referenceFile>` | Output image filename for the CPU result (default: `referecen.pgm`) |
| `-n <tmin>` | Minimum threshold for hysteresis (default: 45) |
| `-x <tmax>` | Maximum threshold for hysteresis (default: 50) |
| `-s <sigma>` | Sigma value for Gaussian smoothing (default: 1.0) |
| `-u` | Use the shared memory approach (default: False) |
| `-h` | Show help message and exit |

#### Example Usage

```bash
./canny -i images/house.pgm -o house_out.pgm -r house_reference.pgm -n 35 -x 45 -s 1.0
```

> This will process the [house.pgm](./cuda-canny/images/house.pgm) with sigma = 1.0, tmin = 35, and tmax = 45, saving the CUDA result to `house_out.pgm` and the CPU reference result to `house_reference.pgm`.

---

### Run the tests
To run the tests, you can use the provided make target, which will execute the test script `performance_tests.sh`. To do this, run the following command:

```bash
make test
```
This will take some time, as it will run the Canny Edge Detection algorithm on multiple images with different parameters and save the results in the `results` directory.

To clean up the generated files after running the tests, you can use the following command:

```bash
make clean
```
which will remove the `out.pgm`, `reference.pgm`, and the `results` directory.

## Authors

- [Daniel Madureira](https://github.com/Dan1m4D)
- [José Gameiro](https://github.com/zegameiro)