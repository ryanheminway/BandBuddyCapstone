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


// The key of the shared memory block; should probably be an env variable?
#define SHARED_MEMORY_BLOCK_KEY ((char*)"/home/patch/BandBuddyCapstone/Firmware/code/common/shared_mem_key")


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
    uint8_t* shared_mem_blk = (uint8_t*)attach_mem_blk(SHARED_MEMORY_BLOCK_KEY, size);
    if (!shared_mem_blk)
    {
        fprintf(stderr, "Failed to open shared memory block!\n");
        return 1;
    }

    // Memcpy to a file 
    FILE* file = fopen("/home/patch/BandBuddyCapstone/Firmware/code/stage1/zzz.wav", "w");
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