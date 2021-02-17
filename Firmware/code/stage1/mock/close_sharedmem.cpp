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

int main(int argc, char** argv)
{
    bool success = destroy_wav_mem_blk();
    if (!success)
    {
        fprintf(stdout, "Failed to destroy!\n");
    }

    return success ? 0 : 1;
}