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

static std::thread producer_thread;

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

static void handle_webserverstage3_data(int &socket_fd, int &payload_size) {
    uint8_t drums = 0;
    uint8_t guitar = 0;

    recieve_webserverstage3_data(socket_fd, payload_size, drums, guitar);
    output_recorded_audio.store(guitar, std::memory_order::memory_order_seq_cst);
    output_generated_audio.store(drums, std::memory_order::memory_order_seq_cst);
}

static void send_webserverstage3_data(int &socket_fd, int &payload_size) {
    uint8_t drums = 0;
    uint8_t guitar = 0;
    int this_stage_id = STAGE3;
    int this_destination = WEBSERVER;

    guitar = output_recorded_audio.load(std::memory_order::memory_order_relaxed);
    drums = output_generated_audio.load(std::memory_order::memory_order_relaxed);
    send_webserverstage3_data(socket_fd, this_stage_id, this_destination, drums, guitar);
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
        switch(cmd){
            case STAGE2_DATA_READY:
                recieve_stage2_fbb(networkbb_fd, payload_size, midi_size);
                start_recording();
                break;
            case STOP:
                stop_recording();
                break;
            case WEBSERVER_DATA:
                handle_webserverstage3_data(networkbb_fd, payload_size);
                break;
            case WEBSERVER_REQUEST:
                send_webserverstage3_data(networkbb_fd, payload_size);
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
    if (midi_size * 2 != wav_size)
    {
        fprintf(stderr, "Audio source size mismatch! midi is %d bytes; wav is %d bytes!\n", midi_size, wav_size);
        return 1;
    }

    // Determine the normalization constant for each buffer
    int16_t norm_midi_max = INT16_MIN, norm_midi_min = INT16_MAX;
    int16_t norm_wav_max = INT16_MIN, norm_wav_min = INT16_MAX;

    for (int i = 44; i < wav_size; i += 2)
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
    for (int i = 44, j = 42; i < wav_size; i += 2)
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

static int init_playback_handle(uint32_t ring_buffer_size)
{
    int err;

    // Open the pisound audio device
    if ((err = snd_pcm_open(&playback_handle, alsa_capture_device_name, SND_PCM_STREAM_PLAYBACK,  SND_PCM_NONBLOCK)) < 0)
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
    snd_pcm_uframes_t num_frames = ((ring_buffer_size / BYTES_PER_FRAME) / 2);
    if ((err = snd_pcm_hw_params_set_period_size_near(playback_handle, hw_params, &num_frames, 0)) < 0)
    {
        print_error(err, "Could not set the period size!\n");
        return err;
    }
    if (num_frames != ((ring_buffer_size / BYTES_PER_FRAME) / 2))
    {
        fprintf(stderr, "Could not set frames/period: target %d, returned %lu!\n", FRAMES_PER_PERIOD, num_frames);
        return 1;
    }

    // Set the pcm ring buffer size
    // (NOTE Ryan Heminway) dividing by 8 instead of 2 
    if ((err = snd_pcm_hw_params_set_buffer_size(playback_handle, hw_params, ring_buffer_size)) < 0)
    {
        print_error(err, "Cannot set playback handle ring buffer size!");
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

    // Set the minimum available frames for a wakeup to the ring buffer size
    if ((err = snd_pcm_sw_params_set_avail_min(playback_handle, sw_params, ring_buffer_size / 2)) < 0)
    {
        print_error(err, "Could not set the frame wakeup limit!");
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
int init_capture_handle(uint32_t ring_buffer_size)
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
    if ((err = snd_pcm_hw_params_set_buffer_size(capture_handle, hw_params, ring_buffer_size)) < 0)
    {
        print_error(err, "Cannot set capture handle ring buffer size!");
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

     // Set the period size
    snd_pcm_uframes_t num_frames = ((ring_buffer_size / BYTES_PER_FRAME) / 2);
    if ((err = snd_pcm_hw_params_set_period_size_near(capture_handle, hw_params, &num_frames, 0)) < 0)
    {
        print_error(err, "Could not set the period size!\n");
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

// The path of the input wav on disk
static const char* wav_input_path = "/home/patch/input.wav";
static const char* wav_drums_path = "/home/patch/drums.wav";
static const char* wav_sync_path = "/home/patch/sync.wav";

// Big-Endian
void uint32_to_uint8_array_BE(uint8_t *array, uint32_t value)
{
    array[0] = (value >> 0) & 0xFF;
    array[1] = (value >> 8) & 0xFF;
    array[2] = (value >> 16) & 0xFF;
    array[3] = (value >> 24) & 0xFF;
}

void uint16_to_uint8_array_BE(uint8_t *array, uint16_t value)
{
    array[0] = (value >> 0) & 0xFF;
    array[1] = (value >> 8) & 0xFF;
}

void calculate_header_values(uint16_t *num_channels, uint32_t *sample_rate, uint32_t *byte_rate,
                             uint16_t *block_align, uint16_t *bits_per_sample)
{
    // Some of these are #defined for now
    *num_channels = NUM_CHANNELS;
    *sample_rate = SAMPLE_RATE;
    *bits_per_sample = BYTES_PER_SAMPLE * 8;

    // Block align = # channels * bytes/sample
    *block_align = NUM_CHANNELS * BYTES_PER_SAMPLE;

    // Byte rate = Sample rate * # channels * bytes/sample
    *byte_rate = SAMPLE_RATE * NUM_CHANNELS * BYTES_PER_SAMPLE;
}

int write_wav_header(FILE* file, int file_size)
{
    uint32_t chunk_size, sample_rate, byte_rate, subchunk_size;
    uint16_t num_channels, block_align, bits_per_sample;
    
    calculate_header_values(&num_channels, &sample_rate, &byte_rate, &block_align, &bits_per_sample);
    chunk_size = file_size + 36;
    subchunk_size = file_size;

    uint8_t data[4];
    uint8_t mem[44];

    // RIFF bytes
    memcpy(mem, "RIFF", 4);
    //memcpy(data, "RIFF", 4);
    //fwrite(data, sizeof(uint8_t), 4, file);

    // Chunk size, big-endian
    uint32_to_uint8_array_BE(data, chunk_size);
    memcpy(mem + 4, data, 4);
    //fwrite(data, sizeof(uint8_t), 4, file);

    // WAVE bytes
    memcpy(mem + 8, "WAVE", 4);
    //memcpy(data, "WAVE", 4);
    //fwrite(data, sizeof(uint8_t), 4, file);

    // fmt  bytes
    memcpy(mem + 12, "fmt ", 4);
    //memcpy(data, "fmt ", 4);
    //fwrite(data, sizeof(uint8_t), 4, file);

    // Subchunk1 size is always 16 for WAV
    data[0] = 16;
    data[1] = 0x00;
    data[2] = 0x00;
    data[3] = 0x00;
    memcpy(mem + 16, data, 4);
    //fwrite(data, sizeof(uint8_t), 4, file);

    // Audio format is always 1 for PCM
    data[0] = 0x01;
    data[1] = 0x00;
    data[2] = 0x00;
    data[3] = 0x00;
    memcpy(mem + 20, data, 4);
    //fwrite(data, sizeof(uint8_t), 2, file);

    // Channel count, big-endian
    uint16_to_uint8_array_BE(data, num_channels);
    memcpy(mem + 22, data, 2);
    //fwrite(data, sizeof(uint8_t), 2, file);

    // Sample rate, big-endian
    uint32_to_uint8_array_BE(data, sample_rate);
    memcpy(mem + 24, data, 4);
    //fwrite(data, sizeof(uint8_t), 4, file);

    // Byte rate, big-endian
    uint32_to_uint8_array_BE(data, byte_rate);
    memcpy(mem + 28, data, 4);
    //fwrite(data, sizeof(uint8_t), 4, file);

    // Block align, big-endian
    uint16_to_uint8_array_BE(data, block_align);
    memcpy(mem + 32, data, 4);
    //fwrite(data, sizeof(uint8_t), 2, file);

    // Bits per sample, big-endian
    uint16_to_uint8_array_BE(data, bits_per_sample);
    memcpy(mem + 34, data, 2);
    //fwrite(data, sizeof(uint8_t), 2, file);

    // data bytes
    memcpy(mem + 36, "data", 4);
    //memcpy(data, "data", 4);
    //fwrite(data, sizeof(uint8_t), 4, file);

    // Subchunk 2 size, big-endian
    uint32_to_uint8_array_BE(data, subchunk_size);
    memcpy(mem + 40, data, 4);
    //fwrite(data, sizeof(uint8_t), 4, file);

    // Write the header to file 
    return (fwrite(mem, sizeof(uint8_t), 44, file) != 44);
}

int write_wav_to_disk(FILE* file, uint8_t* buffer, int file_size)
{
    return write_wav_header(file, file_size) || 
    (fwrite(buffer, sizeof(uint8_t), file_size, file) != file_size);
}

int write_wavs_to_disk(int file_size)
{
    // Open the file, write the header, write the contents
    FILE *file = fopen(wav_input_path, "w");
    if (!file)
    {
        fprintf(stderr, "Failed to open file %s!\n", wav_input_path);
        return 1;
    }

    if (write_wav_to_disk(file, output_buffer_recorded, file_size))
    {
        fprintf(stderr, "Failed to write to file %s!\n", wav_input_path);
        return 1;
    }

    fclose(file);

    file = fopen(wav_drums_path, "w");
    if (!file)
    {
        fprintf(stderr, "Failed to open file %s!\n", wav_input_path);
        return 1;
    }

    if (write_wav_to_disk(file, output_buffer_drum, file_size))
    {
        fprintf(stderr, "Failed to write file %s!\n", wav_drums_path);
        return 1;
    }
    fclose(file);

    file = fopen(wav_sync_path, "w");
    if (!file)
    {
        fprintf(stderr, "Failed to open file %s!\n", wav_sync_path);
        return 1;
    }

    if (write_wav_to_disk(file, output_buffer_combined, file_size))
    {
        fprintf(stderr, "Failed to write file %s!\n", wav_sync_path);
        return 1;
    }
    fclose(file);

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
        //frames_to_deliver = (frames_to_deliver > FRAMES_PER_PERIOD) ? FRAMES_PER_PERIOD : frames_to_deliver;

        for (int i = sample_index; i < sample_index + (frames_to_deliver * BYTES_PER_FRAME); i += 2)
        {
            // Pull a sample from the record buffer and the buffer to play 
            int16_t record_sample = recording_buffer[i] | (recording_buffer[i + 1] << 8);
            int16_t sync_sample = buffer_to_play[i] | (buffer_to_play[i + 1] << 8);

            int32_t sum = record_sample + sync_sample; 
            int16_t avg = (int16_t)(sum / 2);
            recording_buffer[i] = avg & 0xFF;
            recording_buffer[i + 1] = (avg >> 8) & 0xFF;
        }

        int frames_written;
        if ((frames_written = snd_pcm_writei(playback_handle, recording_buffer + sample_index, frames_to_deliver)) != frames_to_deliver)
        {
            if (frames_written == -EPIPE)
            {
                fprintf(stdout, "%s\n", "underrun!");
                snd_pcm_prepare(playback_handle);
            }
            else if (frames_written == -EAGAIN)
            {
                fprintf(stdout, "%s\n", "writei: EAGAIN!");
            }
            else
            {
                fprintf(stderr, "writei (wrote %d): expected to write %d frames, actually wrote %d!\n",
                        sample_index, frames_to_deliver, frames_written);
                sample_index = BYTES_PER_FRAME * frames_written;
                //return -frames_written;
            }
        }
        else 
        {
            sample_index += BYTES_PER_FRAME * frames_written;
        }
    }

    return 0;
}

static void async_record_until_button_press(int loop_size_bytes)
{
    int err = 0;
    int num_bytes_read = 0;
    bool overtook = false;
    bool spawned_thread = false;

    if ((err = snd_pcm_start(capture_handle)) < 0)
    {
        print_error(err, "Could not start capture handle!");
    }

    fprintf(stdout, "%s\n", "Starting async record (not async rn  tho)");
    while (!is_button_pressed.load(std::memory_order::memory_order_relaxed))
    {
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

        if (!spawned_thread && num_bytes_read == BYTES_PER_PERIOD * 3)
        {
            producer_thread = std::thread([&]() { 
                    fprintf(stdout, "Starting to play: recorded %d bytes\n", num_bytes_read); 
                    while (play_loop(loop_size_bytes) == 0); 
                });
            spawned_thread = true;
        }

        if (num_bytes_read >= loop_size_bytes)
        {
            num_bytes_read = 0;
        }
    }  
}

static int close_ALSA_handles()
{
    // Flush the playback handle - this is a BUSY wait! Do better!
    //while (snd_pcm_drain(playback_handle) == -EAGAIN);
    return snd_pcm_close(playback_handle) | snd_pcm_close(capture_handle);
}

static int loop_audio_until_cancelled(int loop_size)
{
    int err;

    // Initialize the playback handle
    if ((err = init_playback_handle(BYTES_PER_PERIOD / 4)))
    {
        return 1;
    }

    if ((err = init_capture_handle(BYTES_PER_PERIOD)))
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

    return err;
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
        int wav_size = 921600;
        midi_size = wav_size / 2;
        
        //int wav_size = midi_size * 2;
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

        // Write just input, just drums, and syncronized audio to disk
        if ((err = write_wavs_to_disk(wav_size))) 
        {
            return 1;
        }

        // Loop the audio until told not to (!TODO ???)
        if ((err = loop_audio_until_cancelled(wav_size)))
        {
            fprintf(stderr, "loop audio returned %d!\n", err);
        }

        // Close the ALSA handles
        if ((err = close_ALSA_handles())) 
        {
            print_error(err, "Closing ALSA handles at the end of a loop failed!");
            return 1;
        }

        // Send the ACK to big brother
        producer_thread.join();
        int this_stage_id = STAGE3;
        int this_destination = BIG_BROTHER; 
        send_ack(networkbb_fd, this_destination, this_stage_id);

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