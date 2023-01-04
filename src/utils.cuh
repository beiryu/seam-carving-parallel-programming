#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <sys/stat.h>

char *concatStr(const char *s1, const char *s2);
void drawImage(int *dImg, int width, int height, char *savePath);
void writePnm(uchar3 *pixels, int width, int height, char *fileName);

class Debuger
{
private:
    bool debug;
    int maxSeams;
    char *energy_filename, *debug_folder, *seams_folder;

public:
    Debuger()
    {
        this->debug = false;
    }
    Debuger(char *debug_folder, char *energy_filename, char *seams_folder, int maxSeams, bool debug)
    {
        this->debug_folder = debug_folder;
        this->energy_filename = energy_filename;
        this->seams_folder = concatStr(debug_folder, seams_folder);
        this->debug = debug;
        this->maxSeams = maxSeams;

        if (debug)
        {
            mkdir(this->debug_folder, 0777);
            mkdir(this->seams_folder, 0777);
        }
    }
    ~Debuger();
    void printDeviceInfo();
    void drawEnergyImage(int *dImg, int width, int height);
    void drawSeamImage(uchar3 *dImg, int width, int height, int iter, int seamIdx, int *path);
};

char *intToStr(int data)
{
    std::string strData = std::to_string(data);

    char *temp = new char[strData.length() + 1];
    strcpy(temp, strData.c_str());
    return temp;
}

char *concatStr(const char *s1, const char *s2)
{
    char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void Debuger::drawEnergyImage(int *dImg, int width, int height)
{
    if (!debug)
        return;
    printf("Drawing energy image\n");
    char *savePath = concatStr(this->debug_folder, this->energy_filename);
    printf("Saving energy image to %s\n", savePath);
    drawImage(dImg, width, height, savePath);
}

void Debuger::drawSeamImage(uchar3 *dImg, int width, int height, int iter, int seamIdx, int *path)
{
    // printf("%d", seamIdx);
    if (!debug || iter > this->maxSeams)
        return;
    printf("Drawing %d / %d seams image\n", iter, this->maxSeams);
    char *savePath = concatStr(this->seams_folder, concatStr(intToStr(iter), ".pnm"));
    uchar3 *img2draw = (uchar3 *)malloc(width * height * sizeof(uchar3));
    memcpy(img2draw, dImg, width * height * sizeof(uchar3));

    img2draw[0 * width + seamIdx] = make_uchar3(220, 0, 0);
    for (int i = 0; i < height - 1; i++)
    {
        seamIdx = path[i * width + seamIdx];
        img2draw[(i + 1) * width + seamIdx] = make_uchar3(220, 0, 0); // red pixel
    }

    writePnm(img2draw, width, height, savePath);
}

void drawImage(int *dImg, int width, int height, char *savePath)
{
    // clipping image to 0-255 before saving
    uchar3 *img2draw = (uchar3 *)malloc(width * height * sizeof(uchar3));
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;
            img2draw[idx] = make_uchar3(min(int(dImg[idx]), 255), min(int(dImg[idx]), 255), min(int(dImg[idx]), 255));
        }
    }
    writePnm(img2draw, width, height, savePath);
}

void readPnm(char *fileName, int &width, int &height, uchar3 *&pixels)
{
    FILE *f = fopen(fileName, "r");
    if (f == NULL)
    {
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    char type[3];
    fscanf(f, "%s", type);

    if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);

    int max_val;
    fscanf(f, "%i", &max_val);
    if (max_val > 255) // In this exercise, we assume 1 byte per value
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    for (int i = 0; i < width * height; i++)
        fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

    fclose(f);
}

void writePnm(uchar3 *pixels, int width, int height, char *fileName)
{
    FILE *f = fopen(fileName, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fprintf(f, "P3\n%i\n%i\n255\n", width, height);

    for (int i = 0; i < width * height; i++)
        fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);

    fclose(f);
}

// GPU utils
#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void Debuger::printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("****************************\n\n");
}

#endif /* UTILS_H */
