#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <alsa/asoundlib.h>
#include <thread>
#include <chrono>

#include "shared_mem.h"


// The name of the env var containing the key of the shared memory block
#define SHARED_MEMORY_ENV_VAR "BANDBUDDY_SHARED_MEMORY_KEY"


#define SYNC_BUFFER_SIZE (1024 * 1024 * 256)
// The buffer into which to synchronize wav data
static uint8_t sync_buffer[SYNC_BUFFER_SIZE];


// Sample rate: use 48k for now
#define SAMPLE_RATE 48000

// 24-bit input/output channels, but only use 16 bits for now
#define BYTES_PER_SAMPLE 2

// Stereo audio
#define NUM_CHANNELS 2

// Frame size is defined by bytes/sample/channel
#define BYTES_PER_FRAME (BYTES_PER_SAMPLE * NUM_CHANNELS)

// The number of frames to record per transfer
#define FRAMES_PER_PERIOD 128

// The number of bytes in a period
#define BYTES_PER_PERIOD (BYTES_PER_FRAME * FRAMES_PER_PERIOD)


// The ALSA playback handle
static snd_pcm_t* playback_handle;

// The ALSA capture device name to use
static const char* alsa_capture_device_name = "plughw:CARD=pisound";


// Print an error and its snd string message to stderr.
void print_error(int err, const char* message, ...)
{
    va_list args;
    va_start(args, message);
 
    vfprintf(stderr, message, args);
    fprintf(stderr, " (error: %s)\n", snd_strerror(err));

    va_end(args);
}


// Awaits a message from the backbone containing the size of the wav file in shared memory. 
// Returns the size, in bytes
static int await_message_from_backbone()
{
    // Nothing happens here yet!
    fprintf(stdout, "%s\n", "WARNING: backbone await is commented out! Proceeding directly now.");
#warning *** WAV FILE SIZE NOT REPORTED FROM BACKBONE - BACKBONE NOT READY YET! PASS SIZE IN BYTES TO ARGV *** 
    return 1;
}

static int spoof_wav_size(int argc, char** argv)
{
    if (argc != 2) 
    {
        fprintf(stderr, "Forgot to pass in the size of the wav in bytes!\n");
        return 0;
    }

    char* size_str = argv[1];
    int size = atoi(size_str);
    return size;
}

static int synchronize_wavs(uint8_t* shared_mem, int shared_mem_size)
{
    // For now just copy the shared mem into the sync buffer
    if (shared_mem_size > SYNC_BUFFER_SIZE)
    {
        fprintf(stderr, "%s\n", "The file in shared memory is larger than the sync buffer! oops :)");
        return 1;
    }

    memcpy(sync_buffer, shared_mem, shared_mem_size);
    return 0;
}


static int init_capture_handle()
{
    int err; 

    // Open the pisound audio device
    if ((err = snd_pcm_open(&playback_handle, alsa_capture_device_name, SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK)) < 0)
    {
        print_error(err, "Cannot open audio device \"%s\"!", alsa_capture_device_name);
        return err;
    }

    // Allocate hardware parameters for this device
    snd_pcm_hw_params_t* hw_params;
    if ((err = snd_pcm_hw_params_malloc(&hw_params)) < 0)
    {
        print_error(err, "Cannot allocate hardware parameters!"); return err;
    }

    // Initialize the hardware params 
    if ((err = snd_pcm_hw_params_any(playback_handle, hw_params)) < 0)
    {
        print_error(err, "Cannot initialize hardware parameters!"); return err;
    }

    // Receive data in interleaved format (vs each channel in completion at a time) to directly write data as WAV
    if ((err = snd_pcm_hw_params_set_access(playback_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0)
    {
        print_error(err, "Cannot set access type to interleaved!"); return err;
    }

    // Receive data as unsigned 16-bit frames
    if ((err = snd_pcm_hw_params_set_format(playback_handle, hw_params, SND_PCM_FORMAT_S16)) < 0)
    {
        print_error(err, "Cannot set frame format to unsigned 16-bit!"); return err;
    }

    // Target 48KHz; if that isn't possible, something has gone wrong
    unsigned int rate = SAMPLE_RATE;
    if ((err = snd_pcm_hw_params_set_rate_near(playback_handle, hw_params, &rate, 0)) < 0)
    {
        print_error(err, "Could not set sample rate: pcm call failed!\n"); return err;
    }
    if (rate != SAMPLE_RATE)
    {
        fprintf(stderr, "Could not set sample rate: target %d, returned %d!\n", SAMPLE_RATE, rate); return 1;
    }

    // Capture stereo audio 
    if ((err = snd_pcm_hw_params_set_channels(playback_handle, hw_params, 2)) < 0)
    {
        print_error(err, "Could not request stereo audio!\n"); return err;
    }

    // Set the period size
    snd_pcm_uframes_t num_frames = FRAMES_PER_PERIOD;
    if ((err = snd_pcm_hw_params_set_period_size_near(playback_handle, hw_params, &num_frames, 0)) < 0)
    {
        print_error(err, "Could not set the period size!\n"); return err;
    }
    if (num_frames != FRAMES_PER_PERIOD) {
        fprintf(stderr, "Could not set frames/period: target %d, returned %lu!\n", FRAMES_PER_PERIOD, num_frames); 
        return 1;
    }

    // Deliver the hardware params to the handle
    if ((err = snd_pcm_hw_params(playback_handle, hw_params)) < 0)
    {
        print_error(err, "Could not deliver hardware parameters to the capture device!"); return err;
    }

    // Free the hw params
    snd_pcm_hw_params_free(hw_params);

    // Allocate software parameters 
    snd_pcm_sw_params_t* sw_params;
    if ((err = snd_pcm_sw_params_malloc(&sw_params)) < 0)
    {
        print_error(err, "Could not allocate software parameters for the capture device!"); return err;
    }

    // Initialize the software parameters
    if ((err = snd_pcm_sw_params_current(playback_handle, sw_params)) < 0)
    {
        print_error(err, "Could not initialize the software parameters for the capture device!"); return err;
    }

    // Set the software parameters
    if ((err = snd_pcm_sw_params(playback_handle, sw_params)) < 0)
    {
        print_error(err, "Could not deliver the software parameters to the capture device!"); return err;
    }

    // Free the sw params
    snd_pcm_sw_params_free(sw_params);

    // Prepare and start the device
    snd_pcm_prepare(playback_handle);
    //snd_pcm_start(playback_handle);

    return 0;
}

static int play_loop(int loop_size_bytes)
{
    int sample_index = 0, err = 0;

    while (sample_index < loop_size_bytes)
    {
        if ((err = snd_pcm_wait(playback_handle, 1000)) < 0)
        {
            print_error(err, "Poll failed!\n");
        }

        int frames_to_deliver; 
        if ((frames_to_deliver = snd_pcm_avail_update(playback_handle)) < 0) 
        {
            if (frames_to_deliver == -EPIPE)
            {
                print_error(frames_to_deliver, "An xrun occurred!"); return frames_to_deliver;
            }
            else 
            {
                print_error(frames_to_deliver, "An unknown error occurred!\n"); return frames_to_deliver;
            }
        }

        // Cap the frames to write 
        frames_to_deliver = (frames_to_deliver > FRAMES_PER_PERIOD) ? FRAMES_PER_PERIOD : frames_to_deliver;

        int frames_written; 
        if ((frames_written = snd_pcm_writei(playback_handle, sync_buffer + sample_index, FRAMES_PER_PERIOD)) != frames_to_deliver)
        {
            if (frames_written == -EPIPE) 
            {
                fprintf(stdout, "%s\n", "underrun!");
                snd_pcm_prepare(playback_handle);
            }
            else 
            {
                fprintf(stderr, "writei (wrote %d): expected to write %d frames, actually wrote %d!\n", 
                   sample_index, FRAMES_PER_PERIOD, frames_written);
                return 1;
            }
        }
        sample_index += BYTES_PER_FRAME * frames_written;
    }    

    return 0;                 
}

int close_playback_handle()
{
    // Flush the playback handle - this is a BUSY wait! Do better! 
    while (snd_pcm_drain(playback_handle) == -EAGAIN);
    return snd_pcm_close(playback_handle);
}

static int loop_audio_until_cancelled(int loop_size)
{
    int err; 

    // Initialize the playback handle
    if ((err = init_capture_handle()))
    {
        return 1;
    }

    // Loop until what? 
    int loops = 0;
#warning *** PLAYBACK LOOP REPEATS ONLY 3 TIMES - NEED TO INTEGRATE W/ BUTTON! ***
    while (loops++ < 3)
    {
        if ((err = play_loop(loop_size)))
        {
            break;
        }
    }

    return err | close_playback_handle();
}

int delete_shared_memory(char* mem_block_addr, void* mem)
{
    bool err = detach_mem_blk(mem);
    #warning *** TEMPORARILY NOT DESTROYING MEMBLK AFTER PLAYTHROUGH! ***
    // err &= destroy_mem_blk(mem_block_addr);
    return err;
}

int main(int argc, char** argv)
{
    // This will change as it is integrated with the network backbone
    while (1)
    {
        // Await instruction from the network backbone
        int wav_size;
        if (!(wav_size = await_message_from_backbone()))
        {
            fprintf(stderr, "%s\n", "Backbone message receive failed!");
            return 1;
        }

        // Spoof the size of the 
        if (!(wav_size = spoof_wav_size(argc, argv)))
        {
            return 1;
        }

        // Retrieve the shared memory 
        char* mem_addr = getenv(SHARED_MEMORY_ENV_VAR);
        uint8_t* mem = (uint8_t*)attach_mem_blk(mem_addr, wav_size);
        if (!mem)
        {
            fprintf(stderr, "%s\n", "Unable to retrieve shared memory pointer!");
            return 1;
        }

        // Synchronize the buffered wav and the return data
        if (synchronize_wavs(mem, wav_size))
        {
            fprintf(stderr, "%s\n", "Failed to synchronize the wav files!");
            return 1;
        }

        // Loop the audio until told not to (!TODO ???)
        int err;
        if ((err = loop_audio_until_cancelled(wav_size)))
        {
            fprintf(stderr, "loop audio returned %d!\n", err);
        }

        // Close and delete the shared memory - we're done with the old data
        delete_shared_memory(mem_addr, mem);

        return 0;
    }
}
