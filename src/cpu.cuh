#ifndef CPU_H
#define CPU_H

#include <stdio.h>
#include <stdint.h>
#include "kernel.cuh"
#include "utils.cuh"

class CPU
{
private:
    void convertRgb2Gray(uchar3 *inPixels, int width, int height, int *out);
    void calConvolution(int *grayPixels, int width, int height, float *filter, int filterWidth, int *outPixels);
    void calConvolutionInt(int *grayPixels, int width, int height, int *filter, int filterWidth, int *outPixels);
    void calEnergies(int *gx, int *gy, int width, int height, int *energies);
    void copyRow(uchar3 *inPixels, int width, int height, int delimIdx, int rowIdx, uchar3 *outPixels);
    void removeSeam(uchar3 *inPixels, int width, int height, int seamIdx, int *path, uchar3 *outPixels);

    int getMinCost(int *energy, int width, int height, int x, int y);

public:
    CPU() = default;
    ~CPU();
    void findSeam(int *energy, int width, int height, int &seamIdx, int *path);
    void applySeamCarving(uchar3 *inPixels, int width, int height, int nSeams, uchar3 *&outPixels, Debuger *logger);
};

void CPU::applySeamCarving(uchar3 *inPixels, int width, int height, int nSeams, uchar3 *&outPixels, Debuger *logger)
{
    uchar3 *src = inPixels;
    uchar3 *out = (uchar3 *)malloc((width - 1) * height * sizeof(uchar3));
    // int outHeight = height;
    int srcWidth = width, srcHeight = height;
    float *gx, *gy;
    createSobelFilters(gx, gy);

    float *gaussFilter;
    createGaussianFilter(gaussFilter);

    for (int i = 1; i <= nSeams; i++)
    {
        int outWidth = width - i;
        if (i > 1)
        {
            out = (uchar3 *)realloc(out, outWidth * height * sizeof(uchar3));
        }

        // 1. Convert img to grayscale
        int *grayscaleImg = (int *)malloc(srcWidth * srcHeight * sizeof(int));
        CPU::convertRgb2Gray(src, srcWidth, srcHeight, grayscaleImg);

        // 2. Calculate energy value for each pixels: blur --> dx, dy --> energy = |dx| + |dy|
        int *blurImg = (int *)malloc(srcWidth * srcHeight * sizeof(int));
        CPU::calConvolution(grayscaleImg, srcWidth, srcHeight, gaussFilter, BLUR_KERNEL_SIZE, blurImg);

        int *dx = (int *)malloc(srcWidth * srcHeight * sizeof(int));
        int *dy = (int *)malloc(srcWidth * srcHeight * sizeof(int));

        CPU::calConvolution(blurImg, srcWidth, srcHeight, gx, SOBEL_KERNEL_SIZE, dx);
        CPU::calConvolution(blurImg, srcWidth, srcHeight, gy, SOBEL_KERNEL_SIZE, dy);

        int *energy = (int *)malloc(srcWidth * srcHeight * sizeof(int));
        calEnergies(dx, dy, srcWidth, srcHeight, energy);
        if (i == 1)
        {
            logger->drawEnergyImage(energy, srcWidth, srcHeight);
        }

        // 3. Find seam given energy values above

        int *path = (int *)malloc(srcWidth * srcHeight * sizeof(int));
        int seamIdx = -1;
        CPU::findSeam(energy, srcWidth, srcHeight, seamIdx, path);
        logger->drawSeamImage(src, srcWidth, srcHeight, i, seamIdx, path);

        // 4. Remove seam
        CPU::removeSeam(src, srcWidth, srcHeight, seamIdx, path, out);

        // 5. Reassign variables for next iteration
        src = out;
        srcWidth--;

        free(grayscaleImg);
        free(blurImg);
        free(energy);
        free(path);
        free(dx);
        free(dy);
    }

    outPixels = out;

    // Free allocated memory
    free(gx);
    free(gy);
    free(gaussFilter);
}

void CPU::convertRgb2Gray(uchar3 *inPixels, int width, int height, int *outPixels)
{
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            int r = inPixels[i].x;
            int g = inPixels[i].y;
            int b = inPixels[i].z;
            outPixels[i] = int(0.299f * r + 0.587f * g + 0.114f * b);
            ;
        }
    }
}

void CPU::calConvolution(int *inPixels, int width, int height, float *filter, int filterWidth, int *outPixels)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx_1d = y * width + x;
            int ele = 0;

            for (int dy = -filterWidth / 2; dy <= filterWidth / 2; dy++)
            {
                for (int dx = -filterWidth / 2; dx <= filterWidth / 2; dx++)
                {
                    int conv_x = max(min(x + dx, width - 1), 0);
                    int conv_y = max(min(y + dy, height - 1), 0);

                    int filter_x = dx + filterWidth / 2;
                    int filter_y = dy + filterWidth / 2;
                    float ele_conv = filter[filter_y * filterWidth + filter_x];

                    ele += int(inPixels[conv_y * width + conv_x] * ele_conv);
                }
            }

            outPixels[idx_1d] = (int)ele;
        }
    }
}

void CPU::calEnergies(int *gx, int *gy, int width, int height, int *energies)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = width * y + x;
            energies[idx] = sqrt(gx[idx] * gx[idx] + gy[idx] * gy[idx]);

            // energies[idx] = abs(gx[idx]) + abs(gy[idx]);
        }
    }
}

int CPU::getMinCost(int *energy, int width, int height, int x, int y)
{
    int minEnergy = INT_MAX;
    int minIdx = -1;
    int neighbor[3] = {-1, 0, 1};
    for (int i = 0; i < 3; i++)
    {
        int x_ = min(max(0, x + neighbor[i]), width - 1);
        int y_ = y + 1;

        int cost = energy[width * y_ + x_] + energy[width * y + x];
        if (cost < minEnergy)
        {
            minEnergy = cost;
            minIdx = x_;
        }
    }

    energy[width * y + x] = minEnergy;
    return minIdx;
}

void CPU::findSeam(int *energy, int width, int height, int &seamIdx, int *path)
{
    // 1. dp
    for (int y = height - 2; y >= 0; y--)
    {
        for (int x = 0; x < width; x++)
        {
            int minIdx = getMinCost(energy, width, height, x, y);
            path[width * y + x] = minIdx;
        }
    }

    // 2. Choose min seam
    int minSeamIdx = -1;
    int minSeamCost = INT_MAX;
    for (int i = 0; i < width; i++)
    {
        if (energy[i] < minSeamCost)
        {
            minSeamCost = energy[i];
            minSeamIdx = i;
        }
    }

    seamIdx = minSeamIdx;
    // printf("CPU min val: %d\n", minSeamCost);
}

void CPU::copyRow(uchar3 *inPixels, int width, int height, int delimIdx, int rowIdx, uchar3 *outPixels)
{
    int idx = -1;
    int outIdx = rowIdx * (width - 1);
    for (int i = 0; i < width; i++)
    {
        if (i == delimIdx)
            continue;
        idx = width * rowIdx + i;
        outPixels[outIdx++] = inPixels[idx];
    }
}

void CPU::removeSeam(uchar3 *inPixels, int width, int height, int seamIdx, int *path, uchar3 *outPixels)
{
    int delimIdx = seamIdx;
    copyRow(inPixels, width, height, delimIdx, 0, outPixels);

    for (int i = 1; i < height; i++)
    {
        delimIdx = path[(i - 1) * width + delimIdx];
        copyRow(inPixels, width, height, delimIdx, i, outPixels);
    }
}

#endif /* CPU_H */
