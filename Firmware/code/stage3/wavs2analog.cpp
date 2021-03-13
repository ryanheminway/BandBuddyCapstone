#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <alsa/asoundlib.h>
#include <atomic>
#include <condition_variable>
#include <signal.h>
#include <pthread.h>
#include <iostream>
#include <thread>

#include "shared_mem.h"
#include "band_buddy_server.h"
#include "band_buddy_msg.h"

#define SYNC_BUFFER_SIZE (1024 * 1024 * 256)
// The buffer into which to synchronize wav data
static uint8_t output_buffer_combined[SYNC_BUFFER_SIZE];
// The buffer into which to write stereo drum data
static uint8_t output_buffer_drum[SYNC_BUFFER_SIZE];
// A buffer filled with zeros
static uint8_t output_buffer_silence[SYNC_BUFFER_SIZE] = {0};
// A pointer to the Stage 1 shared memory buffer
static uint8_t* output_buffer_recorded;

// Sample rate: use 48k for now
#define SAMPLE_RATE 48000

// 24-bit input/output channels, but only use 16 bits for now
#define BYTES_PER_SAMPLE 2

// Stereo audio
#define NUM_CHANNELS 2

// Frame size is defined by bytes/sample/channel
#define BYTES_PER_FRAME (BYTES_PER_SAMPLE * NUM_CHANNELS)

// The number of frames to record per transfer
#define FRAMES_PER_PERIOD 1024

// The number of bytes in a period
#define BYTES_PER_PERIOD (BYTES_PER_FRAME * FRAMES_PER_PERIOD)

// The size of the internal ring buffer in the capture device: fit at least two periods
#define PCM_RING_BUFFER_SIZE (BYTES_PER_FRAME * FRAMES_PER_PERIOD * 2)

// // The buffer into which to put recorded audio from the user]
static uint8_t recording_buffer[SYNC_BUFFER_SIZE];

// // The head of the recording buffer 'queue'
// static int recording_buffer_head;
// // The tail of the recording buffer 'queue'
// static int recording_buffer_tail;

// The ALSA playback handle
static snd_pcm_t *playback_handle;

// The ALSA capture handle
static snd_pcm_t* capture_handle;

// The ALSA capture device name to use
static const char *alsa_capture_device_name = "plughw:CARD=pisound";

// Cancel atomic: set high when the button is pressed
std::atomic_bool is_button_pressed, is_audio_playing;

// The mutex upon which to lock the condition variable
std::mutex is_button_pressed_mutex;

// The condition variable upon which to alert a button press
static std::condition_variable is_button_pressed_cv;

// Set high when listening thread gets a stage2_data_ready message
static std::atomic_bool midi_data_ready;

// High when the audio recorded during Stage 1 should be played.
static std::atomic_bool output_recorded_audio;
// High when the backing track received from Stage 2 should be played.
static std::atomic_bool output_generated_audio;

void await_network_backbone()
{
    // Acquire the mutex and await the condition variable
    std::unique_lock<std::mutex> lock(is_button_pressed_mutex);

    // Lambda prevents spurious wakeups
    is_button_pressed_cv.wait(lock, [&]() { return midi_data_ready.load(std::memory_order::memory_order_seq_cst); });
}

// The socket descriptor for the network backbone
static int networkbb_fd;

//size of midi data from network_bb
static uint32_t midi_size = 0;

void start_recording()
{
    //set to high to notify that stage2_data is in shared mem block
    midi_data_ready.store(true, std::memory_order::memory_order_seq_cst);
    is_button_pressed_cv.notify_one();
}

void stop_recording()
{
    //set to high to breakout of looping
    is_button_pressed.store(true, std::memory_order::memory_order_seq_cst);
    midi_data_ready.store(false, std::memory_order::memory_order_seq_cst);
}

void *wait_button_pressed(void *thread_args)
{
#warning "Clean up wait_button_pressed function\n"
    int cmd;
    int destination;
    int stage_id;
    int payload_size;
    char buffer[1024];

    int this_stage_id = STAGE3;
    int this_destination = BIG_BROTHER;

    while (1)
    {

        retrieve_header(buffer, networkbb_fd);
        parse_header(buffer, destination, cmd, stage_id, payload_size);
        switch (cmd)
        {
        case STAGE2_DATA_READY:
            recieve_stage2_fbb(networkbb_fd, payload_size, midi_size);
            start_recording();
            break;
        case STOP:
            stop_recording();
            send_ack(networkbb_fd, this_destination, this_stage_id);
            break;
        default:
            std::cout << " Sorrry kid wrong command\n";
            break;
        }
    }
}
// Button press handler
/*void button_pressed(int sig)
{
    // Ignore button presses that occur before audio is playing - these are for Stage 1
    if (is_audio_playing.load(std::memory_order::memory_order_relaxed)) {
        is_button_pressed.store(true, std::memory_order::memory_order_seq_cst);
    }
}*/

// Print an error and its snd string message to stderr.
void print_error(int err, const char *message, ...)
{
    va_list args;
    va_start(args, message);

    vfprintf(stderr, message, args);
    fprintf(stderr, " (error: %s)\n", snd_strerror(err));

    va_end(args);
}

// Form the network backbone connection, return success, store the fd in networkbb_fd
int connect_networkbb()
{
    int id = STAGE3;
    int err = connect_and_register(id, networkbb_fd);
    return err;
}

// Awaits a message from the backbone containing the size of the wav file in shared memory.
// Returns the size, in bytes, or 0 if read failed
static int await_message_from_backbone()
{
    // Retrieve the info from the backbone
    uint32_t midi_size;
    if (recieve_header_and_stage2_fbb(networkbb_fd, midi_size) == FAILED)
    {
        fprintf(stderr, "%s\n", "Awaiting backbone ping failed!");
        return 1;
    }

    return midi_size;
}

static int synchronize_wavs(uint8_t *midi, int midi_size, uint8_t *wav, int wav_size)
{
    // Ensure that the two sizes are the same - if not, we have a lineup issue
    if (midi_size != wav_size)
    {
        fprintf(stderr, "Audio source size mismatch! midi is %d bytes; wav is %d bytes!\n", midi_size, wav_size);
        return 1;
    }

    // Determine the normalization constant for each buffer
    int16_t norm_midi_max = INT16_MIN, norm_midi_min = INT16_MAX;
    int16_t norm_wav_max = INT16_MIN, norm_wav_min = INT16_MAX;

    for (int i = 44; i < midi_size; i += 2)
    {
        int midi_index = i / 2;
        int16_t midi_word_int = midi[midi_index] | (midi[midi_index + 1] << 8);
        int16_t wav_word_int = wav[i] | (wav[i + 1] << 8);
        if (midi_word_int > norm_midi_max)
        {
            norm_midi_max = midi_word_int;
        }
        else if (midi_word_int < norm_midi_min)
        {
            norm_midi_min = midi_word_int;
        }
        if (wav_word_int > norm_wav_max)
        {
            norm_wav_max = wav_word_int;
        }
        else if (wav_word_int < norm_wav_min)
        {
            norm_wav_min = wav_word_int;
        }
    }

    fprintf(stdout, "midi: [%d, %d]\twav: [%d, %d]\n", norm_midi_min, norm_midi_max, norm_wav_min, norm_wav_max);

    double norm_max_avg = (norm_midi_max + norm_wav_max / 2);

    // Skip the headers - loop over the rest of the content
    for (int i = 44, j = 42; i < midi_size; i += 2)
    {
        // Iterate over the drum data half as fast to simulate mono->stereo conversion
        if (i % 4 == 0)
        {
            j += 2;
        }

        int16_t midi_word_int = midi[j] | (midi[j + 1] << 8);
        double midi_word = (double)midi_word_int;
        int16_t wav_word_int = wav[i] | (wav[i + 1] << 8);
        double wav_word = (double)wav_word_int;

        double norm_midi_word = (midi_word / (double)norm_midi_max) * norm_max_avg;
        double norm_wav_word = (wav_word / (double)norm_wav_max) * norm_max_avg;

        double avg_word = (norm_midi_word + norm_wav_word) / 2;
        int16_t avg = (int16_t)avg_word;

        output_buffer_combined[i - 44] = avg & 0xFF;
        output_buffer_combined[i + 1 - 44] = (avg >> 8) & 0xFF;
        
        // Also write the drum data to the drum output 
        output_buffer_drum[i - 44] = ((int)midi_word) & 0xFF;
        output_buffer_drum[i + 1 - 44] = (((int)midi_word) >> 8) & 0xFF;
    }

    // Update the recorded audio buffer pointer to point to the shared memory
    output_buffer_recorded = midi;

    return 0;
}

static int init_playback_handle()
{
    int err;

    // Open the pisound audio device
    if ((err = snd_pcm_open(&playback_handle, alsa_capture_device_name, SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK)) < 0)
    {
        print_error(err, "Cannot open audio device \"%s\"!", alsa_capture_device_name);
        return err;
    }

    // Allocate hardware parameters for this device
    snd_pcm_hw_params_t *hw_params;
    if ((err = snd_pcm_hw_params_malloc(&hw_params)) < 0)
    {
        print_error(err, "Cannot allocate hardware parameters!");
        return err;
    }

    // Initialize the hardware params
    if ((err = snd_pcm_hw_params_any(playback_handle, hw_params)) < 0)
    {
        print_error(err, "Cannot initialize hardware parameters!");
        return err;
    }

    // Receive data in interleaved format (vs each channel in completion at a time) to directly write data as WAV
    if ((err = snd_pcm_hw_params_set_access(playback_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0)
    {
        print_error(err, "Cannot set access type to interleaved!");
        return err;
    }

    // Receive data as unsigned 16-bit frames
    if ((err = snd_pcm_hw_params_set_format(playback_handle, hw_params, SND_PCM_FORMAT_S16)) < 0)
    {
        print_error(err, "Cannot set frame format to unsigned 16-bit!");
        return err;
    }

    // Target 48KHz; if that isn't possible, something has gone wrong
    unsigned int rate = SAMPLE_RATE;
    if ((err = snd_pcm_hw_params_set_rate_near(playback_handle, hw_params, &rate, 0)) < 0)
    {
        print_error(err, "Could not set sample rate: pcm call failed!\n");
        return err;
    }
    if (rate != SAMPLE_RATE)
    {
        fprintf(stderr, "Could not set sample rate: target %d, returned %d!\n", SAMPLE_RATE, rate);
        return 1;
    }

    // Capture stereo audio
    if ((err = snd_pcm_hw_params_set_channels(playback_handle, hw_params, 2)) < 0)
    {
        print_error(err, "Could not request stereo audio!\n");
        return err;
    }

    // Set the period size
    snd_pcm_uframes_t num_frames = FRAMES_PER_PERIOD;
    if ((err = snd_pcm_hw_params_set_period_size_near(playback_handle, hw_params, &num_frames, 0)) < 0)
    {
        print_error(err, "Could not set the period size!\n");
        return err;
    }
    if (num_frames != FRAMES_PER_PERIOD)
    {
        fprintf(stderr, "Could not set frames/period: target %d, returned %lu!\n", FRAMES_PER_PERIOD, num_frames);
        return 1;
    }

    // Set the pcm ring buffer size
    if ((err = snd_pcm_hw_params_set_buffer_size(playback_handle, hw_params, BYTES_PER_PERIOD * 2)) < 0)
    {
        print_error(err, "Cannot set pcm ring buffer size!");
        return err;
    }


    // Deliver the hardware params to the handle
    if ((err = snd_pcm_hw_params(playback_handle, hw_params)) < 0)
    {
        print_error(err, "Could not deliver hardware parameters to the capture device!");
        return err;
    }

    // Free the hw params
    snd_pcm_hw_params_free(hw_params);

    // Allocate software parameters
    snd_pcm_sw_params_t *sw_params;
    if ((err = snd_pcm_sw_params_malloc(&sw_params)) < 0)
    {
        print_error(err, "Could not allocate software parameters for the capture device!");
        return err;
    }

    // Initialize the software parameters
    if ((err = snd_pcm_sw_params_current(playback_handle, sw_params)) < 0)
    {
        print_error(err, "Could not initialize the software parameters for the capture device!");
        return err;
    }

    // Set the software parameters
    if ((err = snd_pcm_sw_params(playback_handle, sw_params)) < 0)
    {
        print_error(err, "Could not deliver the software parameters to the capture device!");
        return err;
    }

    // Free the sw params
    snd_pcm_sw_params_free(sw_params);

    // Prepare and start the device
    snd_pcm_prepare(playback_handle);
    //snd_pcm_start(playback_handle);

    return 0;
}

// Initalize the capture handle for audio capture. Returns 0 on success, errno on failure.
int init_capture_handle()
{
    int err;

    // Open the pisound audio device to capture
    if ((err = snd_pcm_open(&capture_handle, alsa_capture_device_name, SND_PCM_STREAM_CAPTURE, SND_PCM_NONBLOCK)) < 0)
    {
        print_error(err, "Cannot open audio device \"%s\"!", alsa_capture_device_name);
        return err;
    }

    // Allocate hardware parameters for this device
    snd_pcm_hw_params_t *hw_params;
    if ((err = snd_pcm_hw_params_malloc(&hw_params)) < 0)
    {
        print_error(err, "Cannot allocate hardware parameters!");
        return err;
    }

    // Initialize the hardware params
    if ((err = snd_pcm_hw_params_any(capture_handle, hw_params)) < 0)
    {
        print_error(err, "Cannot initialize hardware parameters!");
        return err;
    }

    // Receive data in interleaved format (vs each channel in completion at a time) to directly write data as WAV
    if ((err = snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0)
    {
        print_error(err, "Cannot set access type to interleaved!");
        return err;
    }

    // Receive data as unsigned 24-bit frames
    if ((err = snd_pcm_hw_params_set_format(capture_handle, hw_params, SND_PCM_FORMAT_S16)) < 0)
    {
        print_error(err, "Cannot set frame format to unsigned 24-bit!");
        return err;
    }

    // Set the pcm ring buffer size
    if ((err = snd_pcm_hw_params_set_buffer_size(capture_handle, hw_params, PCM_RING_BUFFER_SIZE)) < 0)
    {
        print_error(err, "Cannot set pcm ring buffer size!");
        return err;
    }

    // Target 48KHz; if that isn't possible, something has gone wrong
    unsigned int rate = SAMPLE_RATE;
    if ((err = snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &rate, 0)) < 0)
    {
        print_error(err, "Could not set sample rate: pcm call failed!");
        return err;
    }
    if (rate != SAMPLE_RATE)
    {
        fprintf(stderr, "Could not set sample rate: target %d, returned %d!\n", SAMPLE_RATE, rate);
        return 1;
    }

    // Capture stereo audio
    if ((err = snd_pcm_hw_params_set_channels(capture_handle, hw_params, 2)) < 0)
    {
        print_error(err, "Could not request stereo audio!");
        return err;
    }

    // Deliver the hardware params to the handle
    if ((err = snd_pcm_hw_params(capture_handle, hw_params)) < 0)
    {
        print_error(err, "Could not deliver hardware parameters to the capture device!");
        return err;
    }

    // Free the hw params
    snd_pcm_hw_params_free(hw_params);

    // Allocate software parameters
    snd_pcm_sw_params_t *sw_params;
    if ((err = snd_pcm_sw_params_malloc(&sw_params)) < 0)
    {
        print_error(err, "Could not allocate software parameters for the capture device!");
        return err;
    }

    // Initialize the software parameters
    if ((err = snd_pcm_sw_params_current(capture_handle, sw_params)) < 0)
    {
        print_error(err, "Could not initialize the software parameters for the capture device!");
        return err;
    }

    // Set the minimum available frames for a wakeup to the ring buffer size
    if ((err = snd_pcm_sw_params_set_avail_min(capture_handle, sw_params, FRAMES_PER_PERIOD)) < 0)
    {
        print_error(err, "Could not set the frame wakeup limit!");
        return err;
    }

    // Set the software parameters
    if ((err = snd_pcm_sw_params(capture_handle, sw_params)) < 0)
    {
        print_error(err, "Could not deliver the software parameters to the capture device!");
        return err;
    }

    // Free the sw params
    snd_pcm_sw_params_free(sw_params);

    return 0;
}


static int play_loop(int loop_size_bytes)
{
    int sample_index = 0, err = 0;
    uint8_t* buffer_to_play;

    while (sample_index < loop_size_bytes)
    {
        // If the button has been pressed, time to stop
        if (is_button_pressed.load(std::memory_order::memory_order_relaxed))
        {
            return 1;
        }

        // Using the flags, determine which playback buffer to use
        bool should_output_recorded = output_recorded_audio.load(std::memory_order::memory_order_relaxed);
        bool should_output_generated = output_generated_audio.load(std::memory_order::memory_order_relaxed);

        if (should_output_recorded) 
        {
            if (should_output_generated)
            {
                // Use both
                buffer_to_play = output_buffer_combined;
            }
            else 
            {
                // Use just the recorded
                buffer_to_play = output_buffer_recorded;
            }
        }
        else 
        {
            if (should_output_generated)
            {
                // Use just the drums
                buffer_to_play = output_buffer_drum;
            }
            else 
            {
                // Use silence
                buffer_to_play = output_buffer_silence;
            }
        }

        if ((err = snd_pcm_wait(playback_handle, 1000)) < 0)
        {
            print_error(err, "Poll failed!\n");
            return err;
        }

        int frames_to_deliver;
        if ((frames_to_deliver = snd_pcm_avail_update(playback_handle)) < 0)
        {
            if (frames_to_deliver == -EPIPE)
            {
                print_error(frames_to_deliver, "An xrun occurred!");
                return frames_to_deliver;
            }
            else
            {
                print_error(frames_to_deliver, "An unknown error occurred!\n");
                return frames_to_deliver;
            }
        }

        // Cap the frames to write
        frames_to_deliver = (frames_to_deliver > FRAMES_PER_PERIOD) ? FRAMES_PER_PERIOD : frames_to_deliver;

        int frames_written;
        if ((frames_written = snd_pcm_writei(playback_handle, buffer_to_play + sample_index, frames_to_deliver)) != frames_to_deliver)
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
                return -frames_written;
            }
        }
        sample_index += BYTES_PER_FRAME * frames_written;
    }

    return 0;
}

static void async_record_until_button_press(int loop_size_bytes)
{
    int err = 0;
    int num_bytes_read = 0;
    bool overtook = false;

    if ((err = snd_pcm_start(capture_handle)) < 0)
    {
        print_error(err, "Could not start capture handle!");
    }

    fprintf(stdout, "%s\n", "Starting async record (not async rn  tho)");
    while (!is_button_pressed.load(std::memory_order::memory_order_relaxed))
    {
        // if (!overtook && num_bytes_written >= num_bytes_read)
        // {
        //     fprintf(stderr, "Write went too fast: wrote %d bytes.\n", num_bytes_written);
        //     overtook = true;
        // }
        // else if (overtook && num_bytes_written < num_bytes_read)
        // {
        //     fprintf(stderr, "%s\n", "write back behind read");
        //     overtook = false;
        // }

        // if (!overtook) 
        // {
        //     fprintf(stdout, "bytes written: %d\n", num_bytes_written);
        // }

        // if ((err = snd_pcm_wait(capture_handle, 1000)) < 0)
        // {
        //     print_error(err, "Poll failed!\n");
        //     return;
        // }
        
        // snd_pcm_sframes_t frames_to_deliver;
        // if ((frames_to_deliver = snd_pcm_avail_update(capture_handle)) < FRAMES_PER_PERIOD)
        // {
        //     if (frames_to_deliver == -EPIPE)
        //     {
        //         print_error(frames_to_deliver, "An xrun occurred!");
        //         snd_pcm_prepare(playback_handle);
        //         continue;
        //     }
        //     else
        //     {
        //         print_error(frames_to_deliver, "An unknown error occurred!\n");
        //         return;
        //     }
        // }

        // // Read one period, even if more is available
        // if ((err = snd_pcm_readi(capture_handle, recording_buffer + num_bytes_read, frames_to_deliver)) != frames_to_deliver)
        // {
        //     print_error(err, "Frame read failed: %d!", err);
        //     return;
        // }        

        // num_bytes_read += frames_to_deliver * BYTES_PER_FRAME;
        // fprintf(stdout, "num bytes read: %d\n", num_bytes_read);

// Await a new set of data
        if ((err = snd_pcm_wait(capture_handle, 1000)) < 0)
        {
            print_error(err, "Capture wait failed!");
            return;
        }

        snd_pcm_sframes_t frames_available = snd_pcm_avail_update(capture_handle);
        if (frames_available < FRAMES_PER_PERIOD)
        {
            fprintf(stderr, "Too few frames received: expected %d, received %lu!\n", FRAMES_PER_PERIOD, frames_available);
            return;
        }

        // If there were more frames available than expected, report it - if this happens many times in a row, we might get an overrun
        if (frames_available != FRAMES_PER_PERIOD)
        {
            fprintf(stderr, "Expected %d frames, but %lu are ready. Monitor for overflow?", FRAMES_PER_PERIOD, frames_available);
        }

        // Read one period, even if more is available
        if ((err = snd_pcm_readi(capture_handle, recording_buffer + num_bytes_read, FRAMES_PER_PERIOD)) != FRAMES_PER_PERIOD)
        {
            print_error(err, "Frame read failed!");
            return;
        }

        //fprintf(stdout, "Num_bytes_Read = %d\n", num_bytes_read);

        num_bytes_read += BYTES_PER_PERIOD; 

        if (num_bytes_read == BYTES_PER_PERIOD * 32)
        {
            std::thread t([&]() { 
                    fprintf(stdout, "Starting to play: recorded %d bytes\n", num_bytes_read); 
                    while (play_loop(loop_size_bytes) == 0); 
                });
            t.detach();
        }
    }

    if ((err = snd_pcm_close(playback_handle)) < 0)
    {
        print_error(err, "Could not close the playback device!");
        return;
    } else 
    {
        fprintf(stdout, "%s\n", "playback handle closed");
    }   
}

static int close_playback_handle()
{
    // Flush the playback handle - this is a BUSY wait! Do better!
    //while (snd_pcm_drain(playback_handle) == -EAGAIN);
    return snd_pcm_close(playback_handle);
}

static int loop_audio_until_cancelled(int loop_size)
{
    int err;

    // Initialize the playback handle
    if ((err = init_playback_handle()))
    {
        return 1;
    }

    if ((err = init_capture_handle()))
    {
        return 1;
    } 

    // Spawn the producer thread
    //std::thread producer_thread(async_record_until_button_press);
    //producer_thread.detach();

    // Mark that audio is playing
    //is_audio_playing.store(true, std::memory_order::memory_order_seq_cst);

    // Loop until the button is pressed
    async_record_until_button_press(loop_size);
    //while ((err = play_loop(loop_size)) == 0);

    // Clear the successful exit value from the error code - error handling needs to be done much better throughout Stages 1 and 3
    if (err == 1)
        err = 0;

    // Mark that audio is no longer playing
    //is_audio_playing.store(false, std::memory_order::memory_order_seq_cst);

    return err | close_playback_handle();
}

int delete_shared_memory(void *mem)
{
    bool err = detach_mem_blk(mem);
#warning *** TEMPORARILY NOT DESTROYING MEMBLK AFTER PLAYTHROUGH! ***
    // err &= destroy_mem_blk(mem_block_addr);
    return err;
}

int main(int argc, char **argv)
{
    // Register button press signal handler
    //signal(SIGINT, button_pressed);
    pthread_t thread;
    int err;

    // !TEMP 
    output_generated_audio.store(true, std::memory_order_seq_cst);
    output_recorded_audio.store(true, std::memory_order_seq_cst);

    // Connect to the network backbone
    int failed = FAILED;
    int what = connect_networkbb();
    if (what == failed)
    {
        fprintf(stderr, "%s\n", "Could not connect to the network backbone!");
        return 1;
    }

    err = pthread_create(&thread, NULL, wait_button_pressed, NULL);

    if (err)
    {
        std::cout << "Error:unable to create thread," << err << std::endl;
        exit(-1);
    }

    // This will change as it is integrated with the network backbone
    while (1)
    {
        // Reset the button press status
        is_button_pressed.store(false, std::memory_order::memory_order_seq_cst);

        // Await instruction from the network backbone
        //int midi_size;
        /*if (!(midi_size = await_message_from_backbone()))
        {
            fprintf(stderr, "%s\n", "Backbone message receive failed!");
            return 1;
        }*/

        await_network_backbone();

        // Retrieve the shared memory pointers
        uint8_t *midi = (uint8_t *)get_midi_mem_blk(midi_size);
        if (!midi)
        {
            fprintf(stderr, "%s\n", "Unable to retrieve midi shared memory pointer!");
            return 1;
        }

#warning *** WAV MEMBLK SIZE ASSUMED TO BE == TO MIDI MEMBLK!!! ***
        int wav_size = midi_size;
        uint8_t *wav = (uint8_t *)get_wav_mem_blk(0);
        if (!wav)
        {
            fprintf(stderr, "%s\n", "Unable to retrieve wav shared memory pointer!");
            return 1;
        }

        // Synchronize the buffered wav and the return data
        if (synchronize_wavs(midi, midi_size, wav, wav_size))
        {
            fprintf(stderr, "%s\n", "Failed to synchronize the wav files!");
            return 1;
        }

        // Loop the audio until told not to (!TODO ???)

        if ((err = loop_audio_until_cancelled(midi_size)))
        {
            fprintf(stderr, "loop audio returned %d!\n", err);
        }

        // Close and delete the shared memory - we're done with the old data
        detach_mem_blk(midi);
        detach_mem_blk(wav);
        //delete_shared_memory(mem);

        if (err)
        {
            break;
        }
    }

    return 0;
}


        // int bytes_to_deliver = frames_to_deliver * BYTES_PER_FRAME;
        // int recording_queue_start = recording_buffer_tail;
        // int recording_queue_end = (recording_queue_start + bytes_to_deliver) % RECORDING_BUFFER_SIZE;

        // for (int i = recording_queue_start; i != recording_queue_end; i = (i + 2) % RECORDING_BUFFER_SIZE)
        // {
        //     // Grab one sample from the sync buffer and one sample from the recorded buffer
        //     int16_t sample_to_play_int = buffer_to_play[sample_index] | (buffer_to_play[sample_index + 1] << 8);
        //     int16_t sample_recorded_int = recording_buffer[i] | (recording_buffer[i + 1] << 8);

        //     double sample_to_play = (double)sample_to_play_int;
        //     double sample_recorded = (double)sample_recorded_int;

        //     double avg = (sample_to_play + sample_recorded) / 2;
        //     int16_t avg_int = (int16_t)avg;

        //     // Put it back into the recording buffer
        //     recording_buffer[i] = (avg_int & 0xFF);
        //     recording_buffer[i + 1] = (avg_int >> 8) & 0xFF;
        // }

        // recording_buffer_tail = recording_queue_end;
