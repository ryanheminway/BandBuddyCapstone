#include "shared_mem.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <poll.h>
#include <alsa/asoundlib.h>
#include <signal.h>
#include <atomic>
#include <condition_variable>
#include <mutex>

int main(int argc, char** argv)
{
    if (argc != 2) 
    {
        fprintf(stderr, "forgot to pass in the size in bytes!\n");
        return 1;
    }

    char* size_str = argv[1];
    int size = atoi(size_str);

    // Open the shared memory block 
    uint8_t* shared_mem_blk = (uint8_t*)get_midi_mem_blk(size);
    if (!shared_mem_blk)
    {
        fprintf(stderr, "Failed to open shared memory block!\n");
        return 1;
    }

    // Memcpy to a file 
    FILE* file = fopen("zzz.wav", "wb");
    if (!file) 
    {
        fprintf(stderr, "Could not open wav file!\n"); return 1;
    }

    if (fwrite(shared_mem_blk, sizeof(uint8_t), size, file) != size)
    {
        fprintf(stderr, "fwrite failed!\n"); return 1;
    }

    detach_mem_blk(shared_mem_blk);
    fclose(file);
    return 0;
}
