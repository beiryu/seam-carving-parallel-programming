#include "gpu3.cuh"
#include "utils.cuh"

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
    char debug_folder[] = "debug_device3/";
    char seams_folder[] = "seams/";
    char energy_filename[] = "energy.pnm";
    Debuger *debuger = new Debuger(debug_folder, energy_filename, seams_folder, 10, false);

    run_gpu3(argv[1], debuger, atoi(argv[2]));
    return 0;
}