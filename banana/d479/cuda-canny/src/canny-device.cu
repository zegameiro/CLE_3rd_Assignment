
// CLE 24'25

// canny edge detector code to run on the GPU
void cannyDevice( const int *h_idata, const int w, const int h,
                 const int tmin, const int tmax,
                 const float sigma,
                 int * h_odata)
{
    //TODO: insert your code here
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
__global__ void convulutionKernel(
    const int* in, int* out,
    const float* kernel,
    int nx, int ny, int kn
)
{
    // Calculate the pixel coordinates for the current thread
    int x = blockIdx.x * blockDim.y + threadIdx.y;
    int y = blockIdx.y * blockDim.x + threadIdx.x;

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
        out[y * nx + x] = pixel;
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

// convolution of in image to out image using kernel of kn width
void convolution(const pixel_t *in, pixel_t *out, const float *kernel,
    const int nx, const int ny, const int kn)
{
assert(kn % 2 == 1);
assert(nx > kn && ny > kn);
const int khalf = kn / 2;

for (int m = khalf; m < nx - khalf; m++)
{
for (int n = khalf; n < ny - khalf; n++)
{
float pixel = 0.0;
size_t c = 0;
for (int j = -khalf; j <= khalf; j++)
{
   for (int i = -khalf; i <= khalf; i++)
   {
       pixel += in[(n - j) * nx + m - i] * kernel[c];
       c++;
   }
}

out[n * nx + m] = (pixel_t)pixel;
}
}
}
    if (x > 0 && x < nx - 1 && y > 0 && y > 0 && y < ny - 1)
    {
        int c = y * nx + x;
        float angle = atan2f((float)Gy[c], (float)Gx[c]) * 100.0f / M_PI;
        if (angle < 0) angle += 180;

        int a = 0, b = 0;
        if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180))
        {
            a = G[c - 1];
            b = G[c + 1];        
        }
        else if (angle >= 22.5 && angle < 67.5)
        {
            a = G[c - nx + 1];
            b = G[c + nx - 1];
        }
        else if (angle >= 67.5 && angle < 112.5)
        {
            a = G[c - nx];
            b = G[c + nx];
        }
        else if (angle >= 112.5 && angle < 157.5)
        {
            a = G[c - nx - 1];
            b = G[c + nx + 1];
        }

        out[c] = (G[c] >= a && G[c] >= b) ? G[c] : 0; 
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

    if (x > 0 && x < nx - 1 && y > 0 &&  y < ny - 1)
    {
        int idx = y * nx + x;
        edges[idx] = (nms[idx] >= tmax) ? 255 : 0;
    }
}

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
                    }
                }
            }
        }
    }
}
