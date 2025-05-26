// CLE 24'25

// Forward declarations of kernels
__global__ void convolutionKernel(const int* in, int* out, const float* kernel, int nx, int ny, int kn);
__global__ void nonMaxSuppressionKernel(const int* Gx, const int* Gy, const int* G, int* out, int nx, int ny);
__global__ void firstEdgesKernel(const int* nms, int* edges, int nx, int ny, int tmax);
__global__ void hysteresisKernel(const int* nms, int* edges, int nx, int ny, int tmin, bool* changed);
__global__ void gaussianKernel(int nx, int ny, float sigma, float* kernel);
__global__ void minMaxKernel(const int* in, int nx, int ny, int* minMax);
__global__ void normalizeKernel(int* inout, int nx, int ny, int kn, int min, int max);
__global__ void gradientMagnitudeKernel(const int* Gx, const int* Gy, int* G, int nx, int ny);

// canny edge detector code to run on the GPU
void cannyDevice(const int *h_idata, const int w, const int h,
                 const int tmin, const int tmax,
                 const float sigma,
                 int * h_odata)
{
    // Calculate grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    
    // Allocate device memory
    int *d_idata, *d_odata, *d_afterGx, *d_afterGy, *d_G, *d_nms;
    float *d_gaussianKernel, *d_Gx, *d_Gy;
    bool *d_changed;
    int *d_minMax;
    
    // Memory allocation size
    int imgSize = w * h * sizeof(int);
    
    // Calculate Gaussian kernel size
    const int ksize = 2 * (int)(2 * sigma) + 3;
    const int kernelSize = ksize * ksize * sizeof(float);
    
    // Allocate memory for device arrays
    cudaMalloc((void**)&d_idata, imgSize);
    cudaMalloc((void**)&d_odata, imgSize);
    cudaMalloc((void**)&d_afterGx, imgSize);
    cudaMalloc((void**)&d_afterGy, imgSize);
    cudaMalloc((void**)&d_G, imgSize);
    cudaMalloc((void**)&d_nms, imgSize);
    cudaMalloc((void**)&d_gaussianKernel, kernelSize);
    cudaMalloc((void**)&d_Gx, 9 * sizeof(float)); // 3x3 kernel
    cudaMalloc((void**)&d_Gy, 9 * sizeof(float)); // 3x3 kernel
    cudaMalloc((void**)&d_changed, sizeof(bool));
    cudaMalloc((void**)&d_minMax, 2 * sizeof(int)); // [min, max]
    
    // Copy input data to device
    cudaMemcpy(d_idata, h_idata, imgSize, cudaMemcpyHostToDevice);
    
    // Initialize Gx and Gy kernels
    float h_Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float h_Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    cudaMemcpy(d_Gx, h_Gx, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Gy, h_Gy, 9 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize device memory
    cudaMemset(d_odata, 0, imgSize);
    
    // Step 1: Apply Gaussian filter
    // Create Gaussian kernel
    gaussianKernel<<<1, ksize*ksize>>>(ksize, ksize, sigma, d_gaussianKernel);
    
    // Apply Gaussian convolution
    convolutionKernel<<<gridSize, blockSize>>>(d_idata, d_odata, d_gaussianKernel, w, h, ksize);
    
    // Find min and max for normalization
    cudaMemset(d_minMax, 0, 2 * sizeof(int));
    minMaxKernel<<<gridSize, blockSize>>>(d_odata, w, h, d_minMax);
    
    // Normalize the filtered image
    int h_minMax[2];
    cudaMemcpy(h_minMax, d_minMax, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    normalizeKernel<<<gridSize, blockSize>>>(d_odata, w, h, ksize, h_minMax[0], h_minMax[1]);
    
    // Step 2: Compute gradients
    // Apply Gx and Gy convolution
    convolutionKernel<<<gridSize, blockSize>>>(d_odata, d_afterGx, d_Gx, w, h, 3);
    convolutionKernel<<<gridSize, blockSize>>>(d_odata, d_afterGy, d_Gy, w, h, 3);
    
    // Compute gradient magnitude
    gradientMagnitudeKernel<<<gridSize, blockSize>>>(d_afterGx, d_afterGy, d_G, w, h);
    
    // Step 3: Non-maximum suppression
    nonMaxSuppressionKernel<<<gridSize, blockSize>>>(d_afterGx, d_afterGy, d_G, d_nms, w, h);
    
    // Step 4: Trace edges with hysteresis
    // Reset output
    cudaMemset(d_odata, 0, imgSize);
    
    // Find strong edges
    firstEdgesKernel<<<gridSize, blockSize>>>(d_nms, d_odata, w, h, tmax);
    
    // Hysteresis thresholding
    bool h_changed;
    do {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
        
        hysteresisKernel<<<gridSize, blockSize>>>(d_nms, d_odata, w, h, tmin, d_changed);
        
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (h_changed);
    
    // Copy result back to host
    cudaMemcpy(h_odata, d_odata, imgSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaFree(d_afterGx);
    cudaFree(d_afterGy);
    cudaFree(d_G);
    cudaFree(d_nms);
    cudaFree(d_gaussianKernel);
    cudaFree(d_Gx);
    cudaFree(d_Gy);
    cudaFree(d_changed);
    cudaFree(d_minMax);
}

/**
 * @brief Applies convolution between the input image and a given kernel
 * @param in: Input image
 * @param out: Output image
 * @param kernel: Convolution kernel
 * @param nx: Image width
 * @param ny: Image height
 * @param kn: Kernel size
 * @return void
 */
__global__ void convolutionKernel(
    const int* in, int* out,
    const float* kernel,
    int nx, int ny, int kn
)
{
    // Calculate the pixel coordinates for the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int khalf = kn / 2;
    float pixel = 0.0f;

    // Ensures the thread operates only on valid pixels
    if (x >= khalf && x < nx - khalf && y >= khalf && y < ny - khalf)
    {
        for (int j = -khalf; j <= khalf; ++j)
        {
            for (int i = -khalf; i <= khalf; ++i)
            {
                int idx = (y - j) * nx + (x - i);
                int kid = (j + khalf) * kn + (i + khalf);
                pixel += in[idx] * kernel[kid];
            }
        }
        out[y * nx + x] = (int)pixel;
    }
}

/** 
 * @brief Compute gradient direction and suppress pixels not in the direction of maximum intensity
 * @param Gx: Gradient in x direction
 * @param Gy: Gradient in y direction
 * @param G: Gradient magnitude
 * @param out: Output image
 * @param nx: Image width
 * @param ny: Image height
 * @return void
*/
__global__ void nonMaxSuppressionKernel(
    const int* Gx, const int* Gy,
    const int* G, int* out,
    int nx, int ny
) 
{
    // Calculate the pixel coordinates for the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1)
    {
        int c = y * nx + x;
        const int nn = c - nx;
        const int ss = c + nx;
        const int ww = c + 1;
        const int ee = c - 1;
        const int nw = nn + 1;
        const int ne = nn - 1;
        const int sw = ss + 1;
        const int se = ss - 1;

        const float dir = (float)(fmodf(atan2f((float)Gy[c], (float)Gx[c]) + M_PI, M_PI) / M_PI) * 8;

        if (((dir <= 1 || dir > 7) && G[c] > G[ee] && G[c] > G[ww]) ||      // 0 deg
            ((dir > 1 && dir <= 3) && G[c] > G[nw] && G[c] > G[se]) ||      // 45 deg
            ((dir > 3 && dir <= 5) && G[c] > G[nn] && G[c] > G[ss]) ||      // 90 deg
            ((dir > 5 && dir <= 7) && G[c] > G[ne] && G[c] > G[sw]))        // 135 deg
            out[c] = G[c];
        else
            out[c] = 0;
    }
}

/**
 * @brief Traces edges where gradient magnitude is greater than tmax
 * @param nms: Non-max suppressed image
 * @param edges: Output image
 * @param nx: Image width
 * @param ny: Image height
 * @param tmax: Maximum threshold
 * @return void
*/
__global__ void firstEdgesKernel(
    const int* nms, int* edges,
    int nx, int ny, int tmax
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1)
    {
        int idx = y * nx + x;
        edges[idx] = (nms[idx] >= tmax) ? 255 : 0;
    }
}

/**
 * @brief Perform hysteresis thresholding to trace weak edges connected to strong edges
 * @param nms: Non-max suppressed image
 * @param edges: Output image with edges
 * @param nx: Image width
 * @param ny: Image height
 * @param tmin: Minimum threshold
 * @param changed: Flag to indicate if any pixel was changed
 * @return void
 */
__global__ void hysteresisKernel(
    const int* nms, int* edges, int nx,
    int ny, int tmin, bool* changed
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1)
    {
        int idx = y * nx + x;
        if (nms[idx] >= tmin && edges[idx] == 0)
        {
            for (int j = -1; j <= 1; ++j)
            {
                for (int i = -1; i <= 1; ++i)
                {
                    if (i == 0 && j == 0) continue;
                    int neighborIdx = (y + j) * nx + (x + i);
                    if (edges[neighborIdx] == 255)
                    {
                        edges[idx] = 255;
                        *changed = true;
                        break;
                    }
                }
                if (edges[idx] == 255) break;
            }
        }
    }
}

/**
 * @brief Generate Gaussian kernel
 * @param nx: Kernel width
 * @param ny: Kernel height
 * @param sigma: Gaussian sigma parameter
 * @param kernel: Output kernel
 * @return void
 */
__global__ void gaussianKernel(int nx, int ny, float sigma, float* kernel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nx * ny)
    {
        int x = idx % nx;
        int y = idx / nx;
        
        float mean = floorf(nx / 2.0f);
        kernel[idx] = expf(-0.5f * (powf((x - mean) / sigma, 2.0f) + 
                                    powf((y - mean) / sigma, 2.0f))) / 
                     (2.0f * M_PI * sigma * sigma);
    }
}

/**
 * @brief Find minimum and maximum values in an image
 * @param in: Input image
 * @param nx: Image width
 * @param ny: Image height
 * @param minMax: Output array [min, max]
 * @return void
 */
__global__ void minMaxKernel(const int* in, int nx, int ny, int* minMax)
{
    __shared__ int smin[256];
    __shared__ int smax[256];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Initialize with extreme values
    smin[tid] = INT_MAX;
    smax[tid] = INT_MIN;
    
    if (x < nx && y < ny)
    {
        int idx = y * nx + x;
        smin[tid] = in[idx];
        smax[tid] = in[idx];
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            smin[tid] = min(smin[tid], smin[tid + s]);
            smax[tid] = max(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result to global memory
    if (tid == 0)
    {
        atomicMin(&minMax[0], smin[0]);
        atomicMax(&minMax[1], smax[0]);
    }
}

/**
 * @brief Normalize image values to 0-255 range
 * @param inout: Input/output image
 * @param nx: Image width
 * @param ny: Image height
 * @param kn: Kernel size (for border handling)
 * @param min: Minimum value
 * @param max: Maximum value
 * @return void
 */
__global__ void normalizeKernel(int* inout, int nx, int ny, int kn, int min, int max)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int khalf = kn / 2;
    
    if (x >= khalf && x < nx - khalf && y >= khalf && y < ny - khalf)
    {
        int idx = y * nx + x;
        inout[idx] = (int)(255.0f * ((float)inout[idx] - (float)min) / ((float)max - (float)min));
    }
}

/**
 * @brief Compute gradient magnitude from x and y gradients
 * @param Gx: Gradient in x direction
 * @param Gy: Gradient in y direction
 * @param G: Output gradient magnitude
 * @param nx: Image width
 * @param ny: Image height
 * @return void
 */
__global__ void gradientMagnitudeKernel(const int* Gx, const int* Gy, int* G, int nx, int ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1)
    {
        int idx = y * nx + x;
        G[idx] = (int)hypotf((float)Gx[idx], (float)Gy[idx]);
    }
}
