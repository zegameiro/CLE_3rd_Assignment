# Third Assignment for the Large-Scale Computation 2024/2025

This assignment consists in the implementation of a solution for the Canny Edge Detection algorithm using CUDA. This algorithm detects edges in  grayscale images using a multi-stage process.

## File Structure

| File | Purpose |
| :---: | :---: |
| `canny.cu` | Contains the main program logic, command-line parsing, timing measurements, and the reference CPU implementation |
| `image.c` | Contains some functions to load and save `.ppm` images  |
| `canny-device.cu` | Implements all CUDA kernels and the main GPU function that manages the GPU-based edge detection |

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




