#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <stdint.h>

#define SOBEL_KERNEL_SIZE 3
#define BLUR_KERNEL_SIZE 3

void createGaussianFilter(float *&filter);
void createSobelFilter(int *&gx, int *&gy);

void createGaussianFilter(float *&filter)
{
    /*
        Create a 3x3 Gaussian filter
        Expected output with KERNEL_SIZE = 3
        [1/16, 2/16, 1/16]
        [2/16, 4/16, 2/16]
        [1/16, 2/16, 1/16]
    */

    int filterWidth = BLUR_KERNEL_SIZE;
    filter = (float *)malloc(filterWidth * filterWidth * sizeof(float));
    for (int filterR = 0; filterR < filterWidth; filterR++)
    {
        for (int filterC = 0; filterC < filterWidth; filterC++)
        {
            filter[filterR * filterWidth + filterC] = 1.0f / (filterWidth * filterWidth);
        }
    }
}

void createSobelFilters(float *&gx, float *&gy)
{
    /*
    gy =    [-1, 0, 1] 
            [-2, 0, 2] 
            [-1, 0, 1] 
            
    gx =    [-1,-2,-1] 
            [ 0, 0, 0] 
            [ 1, 2, 1] 
    
    G = |gx| + |gy|

    */
    gx = (float *)malloc(SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE * sizeof(float));
    gy = (float *)malloc(SOBEL_KERNEL_SIZE * SOBEL_KERNEL_SIZE * sizeof(float));

    gx[3 * 0 + 0] = -1, gx[3 * 0 + 1] = 0, gx[3 * 0 + 2] = 1;
    gx[3 * 1 + 0] = -2, gx[3 * 1 + 1] = 0, gx[3 * 1 + 2] = 2;
    gx[3 * 2 + 0] = -1, gx[3 * 2 + 1] = 0, gx[3 * 2 + 2] = 1;

    gy[3 * 0 + 0] = -1, gy[3 * 0 + 1] = -2, gy[3 * 0 + 2] = -1;
    gy[3 * 1 + 0] = 0, gy[3 * 1 + 1] = 0, gy[3 * 1 + 2] = 0;
    gy[3 * 2 + 0] = 1, gy[3 * 2 + 1] = 2, gy[3 * 2 + 2] = 1;
}

#endif /* KERNEL_H */
