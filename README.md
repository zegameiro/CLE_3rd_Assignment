# Third Assignment for the Large-Scale Computation 2024/2025

This assignment consists in the implementation of a solution for the Canny Edge Detection algorithm using CUDA. This algorithm detects edges in  grayscale images using a multi-stage process.

---

## Table of Contents

1. [File Structure](#file-structure)
2. [Implementation Details](#implementation-details)
    - [Gaussian Kernel Generation (`gaussianKernel`)](#gaussian-kernel-gneration-gaussiankernel)
    - [Convolution (`convolutionKernel`)](#convolution-convolutionkernel)
    - [Normalization (`normalizeKernel`)](#normalization-normalizekernel)
    - [Min/Max Reduction (`minMaxKernel` and `cudaMinMax`)](#minmax-reduction-minmaxkernel-and-cudaminmax)
    - [Gradient Calculation (`convolutionKernel` with Sobel and `mergeGradientsKernel`)](#gradient-calculation-convolutionkernel-with-soberl-and-mergegradientskernel)
    - [Non-Maximum Suppression (`nonMaximumSuppressionKernel`)](#non-maximum-suppression-nonmaximumsuppressionkernel)
    - [First Edges (`firstEdgesKernel`)](#first-edges-firstedgeskernel)
    - [Hysteresis (`hysteresisKernel`)](#hysteresis-hysteresiskernel)
3. [Workflow](#workflow)
4. [Compilation and Execution](#compilation-and-execution)
    - [Command-Line Arguments](#command-line-arguments)
    - [Example Usage](#example-usage)

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

### Min/Max Reduction (`minMaxKernel` and `cudaMinMax`)

Finds the minimum and maximum pixel values in an image for normalization.

- Through `extern __shared__ pixel_t sdata[]` each block loads a chunk of the image into shared memory arrays for min and max values.
- Threads within a block cooperate to find the local minimum and maximum values.
. The first thread in each block uses `atomicMin` and `atomicMax` to update global arrays, ensuring correctness when multiple blocks write results.
- In this kernel, using **shared memory** is much faster than **global memory**, because it avoids performing all reductions in global memory, meaning that it allows efficient reduction across blocks.
- **Atomics** are necessary to avoid race conditions when multiple blocks write their min/max to global memory. This ensures that the final min and max values are correct, even with many blocks running in parallel.

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
| `-h` | Show help message and exit |

#### Example Usage

```bash
./canny -i images/house.pgm -o house_out.pgm -r house_reference.pgm -n 35 -x 45 -s 1.0
```

> This will process the [house.pgm](./cuda-canny/images/house.pgm) with sigma = 1.0, tmin = 35, and tmax = 45, saving the CUDA result to `house_out.pgm` and the CPU reference result to `house_reference.pgm`.