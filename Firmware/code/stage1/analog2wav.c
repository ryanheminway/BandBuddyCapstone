#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <poll.h>
#include <alsa/asoundlib.h>

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

// The number of bytes per period
#define BYTES_PER_PERIOD (FRAMES_PER_PERIOD * BYTES_PER_FRAME)

// The size of the internal ring buffer in the capture device: fit at least two periods
#define PCM_RING_BUFFER_SIZE (BYTES_PER_FRAME * FRAMES_PER_PERIOD * 2)

// For now, hold 64 periods in the mem buffer
#define PERIODS_IN_WAV_BUFFER 1024
#define WAV_BUFFER_SIZE BYTES_PER_PERIOD * PERIODS_IN_WAV_BUFFER
static uint8_t buffer[WAV_BUFFER_SIZE];


// The ALSA capture handle
static snd_pcm_t* capture_handle;

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

// Initalize the capture handle for audio capture. Returns 0 on success, errno on failure.
int init_capture_handle()
{
    int err; 

    // Open the pisound audio device
    if ((err = snd_pcm_open(&capture_handle, alsa_capture_device_name, SND_PCM_STREAM_CAPTURE, SND_PCM_NONBLOCK)) < 0)
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
    if ((err = snd_pcm_hw_params_any(capture_handle, hw_params)) < 0)
    {
        print_error(err, "Cannot initialize hardware parameters!"); return err;
    }

    // Receive data in interleaved format (vs each channel in completion at a time) to directly write data as WAV
    if ((err = snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0)
    {
        print_error(err, "Cannot set access type to interleaved!"); return err;
    }

    // Receive data as unsigned 24-bit frames
    if ((err = snd_pcm_hw_params_set_format(capture_handle, hw_params, SND_PCM_FORMAT_S16)) < 0)
    {
        print_error(err, "Cannot set frame format to unsigned 24-bit!"); return err;
    }

    // Set the pcm ring buffer size
    if ((err = snd_pcm_hw_params_set_buffer_size(capture_handle, hw_params, PCM_RING_BUFFER_SIZE)) < 0)
    {
        print_error(err, "Cannot set pcm ring buffer size!"); return err;
    }

    // Target 48KHz; if that isn't possible, something has gone wrong
    unsigned int rate = SAMPLE_RATE;
    if ((err = snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &rate, 0)) < 0)
    {
        print_error(err, "Could not set sample rate: pcm call failed!"); return err;
    }
    if (rate != SAMPLE_RATE)
    {
        fprintf(stderr, "Could not set sample rate: target %d, returned %d!\n", SAMPLE_RATE, rate); return 1;
    }

    // Capture stereo audio 
    if ((err = snd_pcm_hw_params_set_channels(capture_handle, hw_params, 2)) < 0)
    {
        print_error(err, "Could not request stereo audio!"); return err;
    }

    // Deliver the hardware params to the handle
    if ((err = snd_pcm_hw_params(capture_handle, hw_params)) < 0)
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
    if ((err = snd_pcm_sw_params_current(capture_handle, sw_params)) < 0)
    {
        print_error(err, "Could not initialize the software parameters for the capture device!"); return err;
    }

    // Set the minimum available frames for a wakeup to the ring buffer size 
    if ((err = snd_pcm_sw_params_set_avail_min(capture_handle, sw_params, FRAMES_PER_PERIOD)) < 0)
    {
        print_error(err, "Could not set the frame wakeup limit!"); return err;
    }

    // Set the software parameters
    if ((err = snd_pcm_sw_params(capture_handle, sw_params)) < 0)
    {
        print_error(err, "Could not deliver the software parameters to the capture device!"); return err;
    }

    // Free the sw params
    snd_pcm_sw_params_free(sw_params);

    return 0;
}

int prepare_capture_device()
{
    int err = 0; 
    if ((err = snd_pcm_prepare(capture_handle)) < 0)
    {
        print_error(err, "Could not prepare the capture device!");
    }

    return err;
}

int start_device_and_record()
{
    int err; 
    if ((err = snd_pcm_start(capture_handle)) < 0)
    {
        print_error(err, "Could not start the capture device!"); return err;
    }

    int buffer_index = 0;
    while (buffer_index + BYTES_PER_PERIOD < WAV_BUFFER_SIZE)
    { 
        if (buffer_index >= 6000) 
        {
            int i = 1; i++;
        }
        // Await a new set of data
        if ((err = snd_pcm_wait(capture_handle, 1000)) < 0)
        {
            print_error(err, "Capture wait failed!"); return err;
        }

        snd_pcm_sframes_t frames_available = snd_pcm_avail_update(capture_handle);
        if (frames_available != FRAMES_PER_PERIOD)
        {
            fprintf(stderr, "Incorrect available frames: expected %d, received %d!\n", FRAMES_PER_PERIOD, frames_available);
            return 1;
        }

        // Print here etc
        if ((err = snd_pcm_readi(capture_handle, buffer + buffer_index, frames_available)) != frames_available)
        {
            print_error(err, "Frame read failed!"); return err;
        }

        buffer_index += BYTES_PER_PERIOD;
    }

    return 0;
}

int record_audio()
{
    // Init the capture handle 
    if (init_capture_handle())
    {
        return 1;
    }

    // Prepare the capture device
    if (prepare_capture_device())
    {
        return 1;
    }

    // Start the device and record
    return start_device_and_record();
}

uint8_t TEMP_WAVE_HEADER[44] = {
    'R', 'I', 'F', 'F',             // 'RIFF'
    0x24, 0x00, 0x18, 0x00,         // ChunkSize
    0x57, 0x41, 0x56, 0x45,         // 'WAVE'
    0x66, 0x6d, 0x74, 0x20,         // 'fmt '
    0x10, 0x00, 0x00, 0x00,         // Subchunk1Size
    0x01, 0x00,                     // AudioFormat (1 = PCM)
    0x02, 0x00,                     // NumChannels
    0x80, 0xbb, 0x00, 0x00,         // SampleRate
    0x00, 0xee, 0x02, 0x00,         // ByteRate
    0x04, 0x00,                     // BlockAlign
    0x10, 0x00,                     // BitsPerSample
    0x64, 0x61, 0x74, 0x61,         // 'data'
    0x00, 0x00, 0x18, 0x00          // Subchunk2Size
}; 

int calculate_header_values(uint32_t* chunk_size, uint16_t* num_channels, uint32_t* sample_rate, uint32_t* byte_rate, 
    uint16_t* block_align, uint16_t* bits_per_sample, uint32_t* subchunk_size)
{
    // Some of these are #defined for now
    *num_channels = NUM_CHANNELS;
    *sample_rate = SAMPLE_RATE;
    *bits_per_sample = BYTES_PER_SAMPLE * 8;

    // Block align = # channels * bytes/sample
    *block_align = NUM_CHANNELS * BYTES_PER_SAMPLE;

    // Byte rate = Sample rate * # channels * bytes/sample
    *byte_rate = SAMPLE_RATE * NUM_CHANNELS * BYTES_PER_SAMPLE;

    // Subchunk size = # samples * # channels * bytes/sample = sizeof(buffer) I believe? 
    *subchunk_size = WAV_BUFFER_SIZE;

    // Chunk size = subchunk size * header size 
    *chunk_size = WAV_BUFFER_SIZE + 36;
}

// Big-Endian
void uint32_to_uint8_array_BE(uint8_t* array, uint32_t value)
{
    array[0] = (value >> 0) & 0xFF; array[1] = (value >> 8) & 0xFF; 
    array[2] = (value >> 16) & 0xFF; array[3] = (value >> 24) & 0xFF;
}

void uint16_to_uint8_array_BE(uint8_t* array, uint16_t value)
{
    array[0] = (value >> 0) & 0xFF; 
    array[1] = (value >> 8) & 0xFF;
}

int write_wav_header(FILE* file)
{
    uint8_t data[4]; 

    // Must calculate: chunk size, num channels, sample rate, byte rate, block align, bits/sample, subchunk2 size
    uint32_t chunk_size, sample_rate, byte_rate, subchunk_size;
    uint16_t num_channels, block_align, bits_per_sample;
    calculate_header_values(&chunk_size, &num_channels, &sample_rate, &byte_rate,
        &block_align, &bits_per_sample, &subchunk_size);

    // RIFF bytes
    memcpy(data, "RIFF", 4);
    fwrite(data, sizeof(uint8_t), 4, file);

    // Chunk size, big-endian
    uint32_to_uint8_array_BE(data, chunk_size);
    fwrite(data, sizeof(uint8_t), 4, file);

    // WAVE bytes
    memcpy(data, "WAVE", 4);
    fwrite(data, sizeof(uint8_t), 4, file);

    // fmt  bytes
    memcpy(data, "fmt ", 4);
    fwrite(data, sizeof(uint8_t), 4, file);

    // Subchunk1 size is always 16 for WAV
    data[0] = 16; data[1] = 0x00; data[2] = 0x00; data[3] = 0x00;
    fwrite(data, sizeof(uint8_t), 4, file);

    // Audio format is always 1 for PCM
    data[0] = 0x01; data[1] = 0x00; data[2] = 0x00; data[3] = 0x00;
    fwrite(data, sizeof(uint8_t), 2, file);

    // Channel count, big-endian
    uint16_to_uint8_array_BE(data, num_channels);
    fwrite(data, sizeof(uint8_t), 2, file);

    // Sample rate, big-endian
    uint32_to_uint8_array_BE(data, sample_rate);
    fwrite(data, sizeof(uint8_t), 4, file);

    // Byte rate, big-endian
    uint32_to_uint8_array_BE(data, byte_rate);
    fwrite(data, sizeof(uint8_t), 4, file);

    // Block align, big-endian
    uint16_to_uint8_array_BE(data, block_align);
    fwrite(data, sizeof(uint8_t), 2, file);

    // Bits per sample, big-endian
    uint16_to_uint8_array_BE(data, bits_per_sample);
    fwrite(data, sizeof(uint8_t), 2, file);

    // data bytes 
    memcpy(data, "data", 4);
    fwrite(data, sizeof(uint8_t), 4, file); 

    // Subchunk 2 size, big-endian
    uint32_to_uint8_array_BE(data, subchunk_size);
    fwrite(data, sizeof(uint8_t), 4, file);

    return 0;
}

int write_wav_data(FILE* file)
{
    int num_bytes_written;
    if ((num_bytes_written = fwrite(buffer, sizeof(uint8_t), WAV_BUFFER_SIZE, file)) != WAV_BUFFER_SIZE)
    {
        fprintf(stderr, "Wav data write failed: Expected %d, wrote %d!", WAV_BUFFER_SIZE, num_bytes_written);
        return 1;
    }

    return 0;
}

int write_to_wav(const char* path)
{
    // Open the file, write the header, write the contents
    FILE* file = fopen(path, "w");
    if (!file)
    {
        fprintf(stderr, "Failed to open file %s!", path);
        return 1;
    }

    if (write_wav_header(file))
    {
        fprintf(stderr, "Failed to write WAV header!");
        fclose(file);
        return 1;
    }

    if (write_wav_data(file))
    {
        fprintf(stderr, "Failed to write WAV data!");
        fclose(file);
        return 1;
    }

    return 0;
}

void clean_up()
{ 
    // Close the capture device
    snd_pcm_close(capture_handle);
}

int main(int argc, char* argv[])
{
    // This will change as the module evolves
    int err = record_audio();

    if (err)
    {
        clean_up();
        return 1;
    }

    err = write_to_wav("from_pi.wav");
 
    clean_up();

    return err;
}