#include "cpu.cuh"
#include "gpu1.cuh"
#include "gpu2.cuh"
#include "gpu3.cuh"
#include "utils.cuh"

void run_cpu(char *image_path, Debuger *debuger, int nSeams)
{
    GpuTimer timer;
    char result_savePath[] = "cpu.pnm";

    int width, height;
    uchar3 *inPixels = NULL;
    uchar3 *outPixels = NULL;

    CPU *host = new CPU();

    readPnm(image_path, width, height, inPixels);

    for (int i = 0; i < 1; i++)
    {

        timer.Start();
        host->applySeamCarving(inPixels, width, height, nSeams, outPixels, debuger);
        timer.Stop();

        printf("Version CPU, %d seams: %f ms\n", nSeams, timer.Elapsed());
        writePnm(outPixels, width - nSeams, height, result_savePath);
    }
}

void run_gpu1(char *image_path, Debuger *debuger, int nSeams)
{
    int blocksize = 32;

    char gpu_savepath[] = "gpu_1.pnm";
    GpuTimer timer;
    GPU1 *device = new GPU1();

    // int list_nseams[5] = {300, 50, 100, 150, 300};
    int width, height;
    uchar3 *inPixels = NULL;
    uchar3 *outPixels = NULL;

    readPnm(image_path, width, height, inPixels);
    for (int i = 0; i < 1; i++)
    {

        timer.Start();
        device->applySeamCarving(inPixels, width, height, nSeams, outPixels, blocksize, debuger);
        timer.Stop();

        float kernelTime = timer.Elapsed();
        printf("Version GPU 1, %d seams: %f ms\n", nSeams, kernelTime);

        writePnm(outPixels, width - nSeams, height, gpu_savepath);
    }
}
void run_gpu2(char *image_path, Debuger *debuger, int nSeams)
{
    int blocksize = 32;

    char gpu_savepath[] = "gpu_2.pnm";
    GpuTimer timer;
    GPU2 *device = new GPU2();

    // int list_nseams[5] = {300, 50, 100, 150, 300};
    int width, height;
    uchar3 *inPixels = NULL;
    uchar3 *outPixels = NULL;

    readPnm(image_path, width, height, inPixels);
    for (int i = 0; i < 1; i++)
    {

        timer.Start();
        device->applySeamCarving(inPixels, width, height, nSeams, outPixels, blocksize, debuger);
        timer.Stop();

        float kernelTime = timer.Elapsed();
        printf("Version GPU 2, %d seams: %f ms\n", nSeams, kernelTime);

        writePnm(outPixels, width - nSeams, height, gpu_savepath);
    }
}

void run_gpu3(char *image_path, Debuger *debuger, int nSeams)
{
    int blocksize = 32;

    char gpu_savepath[] = "gpu_3.pnm";
    GpuTimer timer;
    GPU3 *device = new GPU3();

    int width, height;
    uchar3 *inPixels = NULL;
    uchar3 *outPixels = NULL;

    readPnm(image_path, width, height, inPixels);
    for (int i = 0; i < 1; i++)
    {

        timer.Start();
        device->applySeamCarving(inPixels, width, height, nSeams, outPixels, blocksize, debuger);
        timer.Stop();

        float kernelTime = timer.Elapsed();
        printf("Version GPU 3, %d seams: %f ms\n", nSeams, kernelTime);

        writePnm(outPixels, width - nSeams, height, gpu_savepath);
    }
}
int main(int argc, char **argv)
{
    char debug_folder[] = "debug/";
    char seams_folder[] = "seams/";
    char energy_filename[] = "energy.pnm";
    Debuger *debuger = new Debuger(debug_folder, energy_filename, seams_folder, 10, false);

    run_cpu(argv[1], debuger, atoi(argv[2]));
    run_gpu1(argv[1], debuger, atoi(argv[2]));
    run_gpu2(argv[1], debuger, atoi(argv[2]));
    run_gpu3(argv[1], debuger, atoi(argv[2]));
    return 0;
}