#include "cpu.cuh"
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

int main(int argc, char **argv)
{
    char debug_folder[] = "debug_cpu/";
    char seams_folder[] = "seams/";
    char energy_filename[] = "energy.pnm";
    Debuger *debuger = new Debuger(debug_folder, energy_filename, seams_folder, 10, false);

    run_cpu(argv[1], debuger, atoi(argv[2]));
    return 0;
}