
// CLE 24'25

typedef int pixel_t;

__global__ void convolutionKernel(
    const pixel_t *in, pixel_t *out, const float *kernel,
    const int nx, const int ny, const int kn
)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    const int khalf = kn / 2;

    if (m >= khalf && m < nx - khalf && n >= khalf && n < ny - khalf)
    {
        float pixel = 0.0f;

        for (int j = -khalf; j <= khalf; j++)
        {
            for (int i = -khalf; i <=khalf; i++)
            {
                int idx = (n - j) * nx + (m - i);
                int kid = (j + khalf) * kn + (i + khalf);
                pixel += in[idx] * kernel[kid];
            }
        }

        out[n * nx + m] = (pixel_t)pixel;
    }
}

__global__ void nonMaximumSuppressionKernel(
    const pixel_t *after_Gx, pixel_t *after_Gy,
    const pixel_t *G, pixel_t *nms,
    const int nx, const int ny
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1)
    {
        int c = i + nx * j;
        const int nn = c - nx;
        const int ss = c + nx;
        const int ww = c + 1;
        const int ee = c - 1;
        const int nw = nn + 1;
        const int ne = nn - 1;
        const int sw = ss + 1;
        const int se = ss - 1;

        const float dir = (float)(fmodf(atan2f((float)after_Gy[c], (float)after_Gx[c]) + M_PI, M_PI) / M_PI) * 8;

        if (((dir <= 1 || dir > 7) && G[c] > G[ee] && G[c] > G[ww]) || // 0 deg
            ((dir > 1 && dir <= 3) && G[c] > G[nw] && G[c] > G[se]) || // 45 deg
            ((dir > 3 && dir <= 5) && G[c] > G[nn] && G[c] > G[ss]) || // 90 deg
            ((dir > 5 && dir <= 7) && G[c] > G[ne] && G[c] > G[sw]))   // 135 deg
        {
            nms[c] = G[c];
        }
        else
        {
            nms[c] = 0;
        }
    }
}

__global__ void firstEdgesKernel(
    const pixel_t *nms, pixel_t *reference,
    const int nx, const int ny, const int tmax
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1)
    {
        int c = i + nx * j;
        if (nms[c] >= tmax)
        {
            reference[c] = 255;
        } 
        else
        {
            reference[c] = 0;
        }
    }
}

__global__ void hysteresisKernel(
    const pixel_t* nms, pixel_t* reference,
    const int nx, const int ny,
    const int tmin, bool *changed
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1)
    {
        int t = i + j * nx;

        int nbs[8];
        nbs[0] = t - nx; // nn
        nbs[1] = t + nx; // ss
        nbs[2] = t + 1; // ww
        nbs[3] = t - 1; // ee
        nbs[4] = nbs[0] + 1; // nw
        nbs[5] = nbs[0] - 1; // ne
        nbs[6] = nbs[1] + 1; // sw
        nbs[7] = nbs[1] - 1; // se

        if (nms[t] >= tmin && reference[t] == 0)
        {
            for (int k = 0; k < 8; k++)
            {
                if (reference[nbs[k]] >= tmin)
                {
                    reference[t] = 255;
                    *changed = true;
                }
            }
        }
    }
}

__global__ void gaussianKernel(
    float *kernel, const int n, const float sigma
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n)
    {
        const float mean = (float)floor(n / 2.0);
        int c = i + n * j;
        kernel[c] = expf(-0.5f * (powf((i - mean) / sigma, 2.0f) + powf((j - mean) / sigma, 2.0f))) /
                   (2 * M_PI * sigma * sigma);
    }
}

__global__ void minMaxKernel(
    const pixel_t *data, pixel_t *min_vals, pixel_t *max_vals,
    const int nx, const int ny
)
{
    extern __shared__ pixel_t sdata[];
    pixel_t *s_min = sdata;
    pixel_t *s_max = &sdata[blockDim.x];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory
    if (i < nx * ny)
    {
        s_min[tid] = data[i];
        s_max[tid] = data[i];
    }
    else
    {
        s_min[tid] = INT_MAX;
        s_max[tid] = INT_MIN;
    }

    __syncthreads();

    // Reduce to find min and max
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && i + s < nx * ny)
        {
            if (s_min[tid + s] < s_min[tid]) s_min[tid] = s_min[tid + s];
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
        }
        __syncthreads();
    }

    // Write results to global memory
    if (tid == 0)
    {
        atomicMin(&min_vals[blockIdx.x], s_min[0]);
        atomicMax(&max_vals[blockIdx.x], s_max[0]);
    }
}

__global__ void normalizeKernel(
    pixel_t *data, const int nx, const int ny,
    const int kn, const pixel_t min_val, const pixel_t max_val
)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    const int khalf = kn / 2;

    if (m >= khalf && m < nx - khalf && n >= khalf && n < ny - khalf)
    {
        int idx = m + nx * n;
        pixel_t pixel = 255 * ((int)data[idx] - (float)min_val) / ((float)max_val - (float)min_val);
        data[idx] = pixel;
    }
}

__global__ void mergeGradientsKernel(
    pixel_t *G, const pixel_t *Gx, const pixel_t *Gy,
    const int nx, const int ny
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nx * ny)
    {
        int i = idx % nx;
        int j = idx / nx;

        if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1)
        {
            G[idx] = (pixel_t)(hypotf((float)Gx[idx], (float)Gy[idx]));
        }
    }
}

void cudaMinMax(
    const pixel_t *d_data, const int nx, const int ny,
    pixel_t *min_val, pixel_t *max_val
)
{
    int num_elements = nx * ny;
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    pixel_t *d_min_vals, *d_max_vals;
    pixel_t *h_min_vals, *h_max_vals;

    cudaSafeCall(cudaMalloc(&d_min_vals, num_blocks * sizeof(pixel_t)));
    cudaSafeCall(cudaMalloc(&d_max_vals, num_blocks * sizeof(pixel_t)));

    h_min_vals = (pixel_t*)malloc(num_blocks * sizeof(pixel_t));
    h_max_vals = (pixel_t*)malloc(num_blocks * sizeof(pixel_t));

    minMaxKernel<<<num_blocks, block_size, 2 * block_size * sizeof(pixel_t)>>>(
        d_data, d_min_vals, d_max_vals, nx, ny
    );

    cudaSafeCall(cudaMemcpy(h_min_vals, d_min_vals, num_blocks * sizeof(pixel_t), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(h_max_vals, d_max_vals, num_blocks * sizeof(pixel_t), cudaMemcpyDeviceToHost));

    *min_val = h_min_vals[0];
    *max_val = h_max_vals[0];

    for (int i = 1; i < num_blocks; i++)
    {
        if (h_min_vals[i] < *min_val) *min_val = h_min_vals[i];
        if (h_max_vals[i] > *max_val) *max_val = h_max_vals[i];
    }

    cudaFree(d_min_vals);
    cudaFree(d_max_vals);
    free(h_min_vals);
    free(h_max_vals);
}

// canny edge detector code to run on the GPU
void cannyDevice( const int *h_idata, const int w, const int h,
                 const int tmin, const int tmax,
                 const float sigma,
                 int * h_odata)
{
    const int nx = w;
    const int ny = h;
    const int size = nx * ny * sizeof(pixel_t);

    // Device memory allocation
    pixel_t *d_input, *d_temp, *d_after_Gx, *d_after_Gy, *d_G, *d_nms, *d_reference;
    float *d_kernel, *d_Gx, *d_Gy;

    cudaSafeCall(cudaMalloc(&d_input, size));
    cudaSafeCall(cudaMalloc(&d_temp, size));
    cudaSafeCall(cudaMalloc(&d_after_Gx, size));
    cudaSafeCall(cudaMalloc(&d_after_Gy, size));
    cudaSafeCall(cudaMalloc(&d_G, size));
    cudaSafeCall(cudaMalloc(&d_nms, size));
    cudaSafeCall(cudaMalloc(&d_reference, size));

    // Copy input data to device
    cudaSafeCall(cudaMemcpy(d_input, h_idata, size, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemset(d_reference, 0, size));

    // Define thread block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // Apply Gaussian filter
    const int n = 2 * (int)(2 * sigma) + 3;
    cudaSafeCall(cudaMalloc(&d_kernel, n * n * sizeof(float)));

    dim3 block_kernel(16, 16);
    dim3 grid_kernel((n + block_kernel.x - 1) / block_kernel.x, (n + block_kernel.y - 1) / block_kernel.y);
    gaussianKernel<<<grid_kernel, block_kernel>>>(d_kernel, n, sigma);
    cudaCheckMsg("Gaussian kernel generation failed");

    convolutionKernel<<<grid, block>>>(d_input, d_temp, d_kernel, nx, ny, n);
    cudaCheckMsg("Gaussian convolution failed");

    // Normalize gaussian result
    pixel_t min_val, max_val;
    cudaMinMax(d_temp, nx, ny, &min_val, &max_val);
    normalizeKernel<<<grid, block>>>(d_temp, nx, ny, n, min_val, max_val);
    cudaCheckMsg("Gaussian normalization failed");

    // Gradient calculation

    // Sobel operators
    float h_Gx[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    float h_Gy[9] = {
        1, 2, 1,
        0, 0, 0,
        -1, -2, -1
    };

    cudaSafeCall(cudaMalloc(&d_Gx, 9 * sizeof(float)));
    cudaSafeCall(cudaMalloc(&d_Gy, 9 * sizeof(float)));
    cudaSafeCall(cudaMemcpy(d_Gx, h_Gx, 9 * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_Gy, h_Gy, 9 * sizeof(float), cudaMemcpyHostToDevice));

    // Gradient along x
    convolutionKernel<<<grid, block>>>(d_temp, d_after_Gx, d_Gx, nx, ny, 3);
    cudaCheckMsg("Gradient X convolution failed");

    // Gradient along y
    convolutionKernel<<<grid, block>>>(d_temp, d_after_Gy, d_Gy, nx, ny, 3);
    cudaCheckMsg("Gradient Y convolution failed");

    // Merge gradients
    int num_threads = 256;
    int num_blocks = (nx * ny + num_threads - 1) / num_threads;

    mergeGradientsKernel<<<num_blocks, num_threads>>>(d_G, d_after_Gx, d_after_Gy, nx, ny);
    cudaCheckMsg("Gradient merge failed");

    // Non-maximum suppression
    nonMaximumSuppressionKernel<<<grid, block>>>(d_after_Gx, d_after_Gy, d_G, d_nms, nx, ny);
    cudaCheckMsg("Non-maximum suppression failed");

    // First edges
    firstEdgesKernel<<<grid, block>>>(d_nms, d_reference, nx, ny, tmax);
    cudaCheckMsg("First edges failed");

    // Hysteresis edges
    bool *d_changed;
    bool h_changed;
    cudaSafeCall(cudaMalloc(&d_changed, sizeof(bool)));

    do 
    {
        h_changed = false;
        cudaSafeCall(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));

        hysteresisKernel<<<grid, block>>>(d_nms, d_reference, nx, ny, tmin, d_changed);
        cudaCheckMsg("Hysteresis failed");

        cudaSafeCall(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
    }
    while (h_changed);

    // Copy result to host
    cudaSafeCall(cudaMemcpy(h_odata, d_reference, size, cudaMemcpyDeviceToHost));

    // Clean up
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_after_Gx);
    cudaFree(d_after_Gy);
    cudaFree(d_G);
    cudaFree(d_nms);
    cudaFree(d_reference);
    cudaFree(d_kernel);
    cudaFree(d_Gx);
    cudaFree(d_Gy);
    cudaFree(d_changed);
}