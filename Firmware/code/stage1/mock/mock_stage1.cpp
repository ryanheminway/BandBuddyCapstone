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
#include <sys/stat.h>
#include <sys/types.h>

#include "shared_mem.h"
#include "band_buddy_server.h"
#include "band_buddy_msg.h"

#define DUMMY_FILE ((char*)"/home/bandbuddy/BandBuddyCapstone/Data/footsteps.wav")

int main()
{
    // Get mock music file size 
    int size;
    struct stat st;
    if (stat(DUMMY_FILE, &st) != 0)
    {
        fprintf(stderr, "%s\n", "stat failed!"); return 1;
    }

    size = (int)st.st_size;

    // Open the mock music file 
    FILE* wav = fopen(DUMMY_FILE, "r");
    if (!wav)
    {
        fprintf(stderr, "%s\n", "fopen failed!"); return 1;
    }
    
    // Acquire the shared memory
    uint8_t* shared_mem = (uint8_t*)get_wav_mem_blk(size);
    if (!shared_mem)
    {
        fprintf(stderr, "%s\n", "attach_mem_blk failed!"); return 1;
    }

    // Copy the wav into the shared memory 
    size_t wrote;
    if ((wrote = fread(shared_mem, sizeof(uint8_t), size, wav)) != size)
    {
        fprintf(stderr, "fread: expected %ld, wrote %u!\n", size, wrote);
    }

    int stage = STAGE1;
    int bb_fd; 
    if (connect_and_register(stage, bb_fd) != SUCCESS)
    {
	   fprintf(stderr, "%s\n", "agh"); return 1;
    }

    if (stage1_data_ready(bb_fd, size) != SUCCESS)
    {
	    fprintf(stderr, "%s\n", "ghafdh"); return 1;
    }
    

    // Cleanup 
    detach_mem_blk(shared_mem);
    fclose(wav);
    close(bb_fd);

    fprintf(stdout, "Size of test wav: %ld bytes\n", size);

    return 0;
}
