#ifndef GPU1_H
#define GPU1_H

#include "gpu.cuh"
#include "cpu.cuh"
__global__ void blurImg_kernel_v1(int *inPixels, int width, int height, int *outPixels);
__global__ void calcEnergyMap_kernel_v1(int *inPixels, int width, int height, int *outPixels);

class GPU1 : public GPU
{
public:
    GPU1() = default;
    ~GPU1();
    void removeSingleSeam(uchar3 *inPixels, int width, int height, int seam_order, uchar3 *outPixels, int blocksize, Debuger *logger);
};

void GPU1::removeSingleSeam(uchar3 *inPixels, int width, int height, int seam_order, uchar3 *outPixels, int blocksize, Debuger *logger)
{
    dim3 blockSize(blocksize, blocksize);
    // 0. Preparation
    /* Declare variables */
    uchar3 *src = inPixels;

    // Variables
    uchar3 *d_src;
    int *d_src_gray, *d_src_blur, *d_energies;

    int *min_path = (int *)malloc(height * sizeof(int));
    int *min_row0 = (int *)malloc(width * sizeof(int));
    int *min_array = (int *)malloc(width * height * sizeof(int));

    size_t pixelsSize_3channels = width * height * sizeof(uchar3);
    size_t pixelsSize_1channels = width * height * sizeof(int);

    CHECK(cudaMalloc(&d_src, pixelsSize_3channels));
    CHECK(cudaMemcpy(d_src, src, pixelsSize_3channels, cudaMemcpyHostToDevice));

    // 1. Convert img to gray
    CHECK(cudaMalloc(&d_src_gray, pixelsSize_1channels));
    dim3 gridSize_gray((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

    convertRgb2Gray_kernel<<<gridSize_gray, blockSize>>>(d_src, width, height, d_src_gray);

    // 2. Blur gray img
    CHECK(cudaMalloc(&d_src_blur, pixelsSize_1channels));
    dim3 gridSize_blur((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

    blurImg_kernel_v1<<<gridSize_blur, blockSize>>>(d_src_gray, width, height, d_src_blur);

    // 3. Calc Energies
    CHECK(cudaMalloc(&d_energies, pixelsSize_1channels));

    calcEnergyMap_kernel_v1<<<gridSize_blur, blockSize>>>(d_src_blur, width, height, d_energies);

    int *energy = (int *)malloc(pixelsSize_1channels);
    CHECK(cudaMemcpy(energy, d_energies, pixelsSize_1channels, cudaMemcpyDeviceToHost));
    if (seam_order == 1)
    {
        logger->drawEnergyImage(energy, width, height);
    }

    // 4. Find min cost each row iteratively (dp - iterative)
    CPU *host = new CPU();
    int min_seam_idx = -1;
    host->findSeam(energy, width, height, min_seam_idx, min_array);

    if (min_seam_idx == -1)
    {
        printf("cannot find min seam at %d\n", seam_order);
    }

    // 5. Calculate min col idx on each row
    logger->drawSeamImage(inPixels, width, height, seam_order, min_seam_idx, min_array);
    min_path[0] = min_seam_idx;

    for (int i = 1; i < height; ++i)
    {
        min_path[i] = min_array[(i - 1) * width + min_path[i - 1]];
    }

    // 6. Remove seam
    uchar3 *d_output;
    int *d_min_path;

    CHECK(cudaMalloc(&d_output, (width - 1) * height * sizeof(uchar3)));
    CHECK(cudaMalloc(&d_min_path, height * sizeof(int)));
    CHECK(cudaMemcpy(d_min_path, min_path, height * sizeof(int), cudaMemcpyHostToDevice));

    removeSeam<<<gridSize_blur, blockSize>>>(d_src, width, height, d_min_path, d_output);
    CHECK(cudaMemcpy(outPixels, d_output, (width - 1) * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

    // Clean memory
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_energies));
    CHECK(cudaFree(d_src_blur));
    CHECK(cudaFree(d_src_gray));

    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_min_path));

    free(min_array);
    free(min_path);
    free(min_row0);
}

__global__ void blurImg_kernel_v1(int *inPixels, int width, int height, int *outPixels) //DONE
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // apply convolution on each pixel
    if (ix < width && iy < height)
    {
        int idx_1d = iy * width + ix;
        int ele = 0;

        for (int dy = -BLUR_KERNEL_SIZE / 2; dy <= BLUR_KERNEL_SIZE / 2; dy++)
        {
            for (int dx = -BLUR_KERNEL_SIZE / 2; dx <= BLUR_KERNEL_SIZE / 2; dx++)
            {

                int conv_x = max(min(ix + dx, width - 1), 0);
                int conv_y = max(min(iy + dy, height - 1), 0);
                int filter_x = dx + BLUR_KERNEL_SIZE / 2;
                int filter_y = dy + BLUR_KERNEL_SIZE / 2;
                float ele_conv = DEVICE_BLUR_KERNEL[filter_y * BLUR_KERNEL_SIZE + filter_x];

                ele += int(inPixels[conv_y * width + conv_x] * ele_conv);
            }
        }

        outPixels[idx_1d] = ele;
    }
}

__global__ void calcEnergyMap_kernel_v1(int *inPixels, int width, int height, int *outPixels) //DONE
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // apply convolution on each pixel
    if (ix < width && iy < height)
    {
        int idx_1d = iy * width + ix;
        int ele_x = 0, ele_y = 0;

        for (int dy = -SOBEL_KERNEL_SIZE / 2; dy <= SOBEL_KERNEL_SIZE / 2; dy++)
        {
            for (int dx = -SOBEL_KERNEL_SIZE / 2; dx <= SOBEL_KERNEL_SIZE / 2; dx++)
            {

                int conv_x = max(min(ix + dx, width - 1), 0);
                int conv_y = max(min(iy + dy, height - 1), 0);
                int filter_x = dx + SOBEL_KERNEL_SIZE / 2;
                int filter_y = dy + SOBEL_KERNEL_SIZE / 2;

                int ele_gx = DEVICE_SOBELX_KERNEL[filter_y * SOBEL_KERNEL_SIZE + filter_x];
                int ele_gy = DEVICE_SOBELY_KERNEL[filter_y * SOBEL_KERNEL_SIZE + filter_x];

                ele_x += int(inPixels[conv_y * width + conv_x] * ele_gx);
                ele_y += int(inPixels[conv_y * width + conv_x] * ele_gy);
            }
        }

        outPixels[idx_1d] = (int)sqrt(float(ele_x * ele_x + ele_y * ele_y));
    }
}

#endif /* GPU1_H */
