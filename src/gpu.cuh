#ifndef GPU_H
#define GPU_H

#include "utils.cuh"
#include "kernel.cuh"

// CUDA AREA
__constant__ float DEVICE_BLUR_KERNEL[BLUR_KERNEL_SIZE * BLUR_KERNEL_SIZE];
__constant__ float DEVICE_SOBELX_KERNEL[SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE];
__constant__ float DEVICE_SOBELY_KERNEL[SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE];

__global__ void convertRgb2Gray_kernel(uchar3 *inPixels, int width, int height, int *outPixels);
__global__ void computeMinEnergyRow_kernel(int *energy, int width, int height, int row, int *minIds);
__global__ void removeSeam(uchar3 *input, int width, int height, int *path, uchar3 *output);

class GPU
{
public:
    GPU() = default;
    ~GPU() = default;
    virtual void applySeamCarving(uchar3 *inPixels, int width, int height, int nSeams, uchar3 *&outPixels, int blocksize, Debuger *logger);
    virtual void removeSingleSeam(uchar3 *inPixels, int width, int height, int seam_order, uchar3 *outPixels, int blocksize, Debuger *logger){};
};

void GPU::applySeamCarving(uchar3 *inPixels, int width, int height, int nSeams, uchar3 *&outPixels, int blocksize, Debuger *logger)
{
    float *gx, *gy;
    createSobelFilters(gx, gy);
    float *gaussFilter;
    createGaussianFilter(gaussFilter);

    // copy data to CMEM
    CHECK(cudaMemcpyToSymbol(DEVICE_BLUR_KERNEL, gaussFilter, BLUR_KERNEL_SIZE * BLUR_KERNEL_SIZE * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(DEVICE_SOBELX_KERNEL, gx, SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(DEVICE_SOBELY_KERNEL, gy, SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE * sizeof(float)));

    int step_width = 0;
    uchar3 *src = inPixels;

    for (int seam_order = 0; seam_order < nSeams; seam_order++)
    {
        step_width = width - seam_order;
        if (seam_order == 0)
            outPixels = (uchar3 *)malloc((step_width - 1) * height * sizeof(uchar3));
        else
            outPixels = (uchar3 *)realloc(outPixels, (step_width - 1) * height * sizeof(uchar3));

        this->removeSingleSeam(src, step_width, height, seam_order + 1, outPixels, blocksize, logger);

        src = outPixels;
    }
}

// Convert image to grayscale kernel
__global__ void convertRgb2Gray_kernel(uchar3 *inPixels, int width, int height, int *outPixels)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < width && iy < height)
    {
        int idx = iy * width + ix;
        int r = inPixels[idx].x;
        int g = inPixels[idx].y;
        int b = inPixels[idx].z;
        outPixels[idx] = int(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// compute min energy for a row
__global__ void computeMinEnergyRow_kernel(int *energy, int width, int height, int row, int *minIds)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int y = row;
    int x = ix;
    int minEnergy = INT_MAX;
    int minIdx = -1;
    int neighbors[3] = {-1, 0, 1};
    if (ix < width)
    {
        for (int i = 0; i < 3; i++)
        {
            int x_ = min(max(0, ix + neighbors[i]), width - 1);
            int y_ = y + 1;

            int cost = energy[width * y_ + x_] + energy[width * y + x];
            if (cost < minEnergy)
            {
                minEnergy = cost;
                minIdx = x_;
            }
        }

        minIds[row * width + ix] = minIdx;
        energy[row * width + ix] = minEnergy;
    }
}

/*
Remove min seam
*/
__global__ void removeSeam(uchar3 *input, int width, int height, int *path, uchar3 *output)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    if (iy < height && ix < width)
    {
        int index_seam = path[iy];

        if (ix < index_seam)
        {
            output[iy * (width - 1) + ix].x = input[iy * width + ix].x;
            output[iy * (width - 1) + ix].y = input[iy * width + ix].y;
            output[iy * (width - 1) + ix].z = input[iy * width + ix].z;
        }
        else if (ix > index_seam && ix < width)
        {
            output[iy * (width - 1) + ix - 1].x = input[iy * width + ix].x;
            output[iy * (width - 1) + ix - 1].y = input[iy * width + ix].y;
            output[iy * (width - 1) + ix - 1].z = input[iy * width + ix].z;
        }
    }
}

#endif /* GPU_H */
