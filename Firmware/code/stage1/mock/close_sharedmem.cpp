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

#include "shared_mem.h"

// The key of the shared memory block; should probably be an env variable?
#define SHARED_MEMORY_BLOCK_KEY ((char*)"/home/patch/BandBuddyCapstone/Firmware/code/common/shared_mem_key")

int main(int argc, char** argv)
{
    bool success = destroy_mem_blk(SHARED_MEMORY_BLOCK_KEY);
    if (!success)
    {
        fprintf(stdout, "Failed to destroy!\n");
    }

    return success ? 0 : 1;
}