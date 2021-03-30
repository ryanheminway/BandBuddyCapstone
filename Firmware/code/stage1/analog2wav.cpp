#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <poll.h>
#include <alsa/asoundlib.h>
#include <signal.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <pthread.h>
#include <iostream>
#include <thread>
#include <fstream>
#include <ios>

#include "shared_mem.h"
#include "band_buddy_msg.h"
#include "band_buddy_server.h"

#define DEFAULT_BPM 100

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

// For now, hold 1024 periods in the mem buffer
#define PERIODS_IN_WAV_BUFFER 1024
#define WAV_BUFFER_SIZE BYTES_PER_PERIOD * PERIODS_IN_WAV_BUFFER
static uint8_t buffer[WAV_BUFFER_SIZE * 32];

// The ALSA capture handle
static snd_pcm_t *capture_handle;
static snd_pcm_t *playback_handle;

// The ALSA capture device name to use
static const char *alsa_capture_device_name = "plughw:CARD=pisound";

// Cancel atomic: set high when the button is pressed
static std::atomic_bool is_button_pressed;

// Set high when the main thread is finished responding to a STOP command
static std::atomic_bool main_thread_stop_status;

//beats per minute. Will be updated by the webserver
static std::atomic_uint8_t bpm;

// The mutex upon which to lock the condition variable
static std::mutex is_button_pressed_mutex;

// The condition variable upon which to alert a button press
static std::condition_variable is_button_pressed_cv;

static constexpr int met_delay_at_start = 192000 / 200;

// Metronome buffer into which clicks are populated
static uint8_t* metronome_buffer;
static int metronome_buffer_size;

//click high array and size
static uint8_t *click_high = nullptr;
static int click_high_size;

//click low array and size
static uint8_t *click_low = nullptr;
static int click_low_size;

// The number of bytes read in total
static int num_bytes_read = 0;

// The socket descriptor for the network backbone
static int networkbb_fd;

// Button press handler
void start_recording()
{
    // Acquire the mutex and await the condition variable
    //std::unique_lock<std::mutex> lock(is_button_pressed_mutex);
    is_button_pressed.store(true, std::memory_order::memory_order_seq_cst);
    is_button_pressed_cv.notify_one();
}
void stop_recording()
{

    // Acquire the mutex and await the condition variable
    //std::unique_lock<std::mutex> lock(is_button_pressed_mutex);
    is_button_pressed.store(false, std::memory_order::memory_order_seq_cst);
}

void await_button_press()
{
    // Acquire the mutex and await the condition variable
    std::unique_lock<std::mutex> lock(is_button_pressed_mutex);

    // Lambda prevents spurious wakeups
    is_button_pressed_cv.wait(lock, [&]() { return is_button_pressed.load(std::memory_order::memory_order_seq_cst); });
}

void await_network_backbone_notify_complete()
{
    // Acquire the mutex and await the condition variable
    std::unique_lock<std::mutex> lock(is_button_pressed_mutex);

    // Lambda prevents spurious wakeups
    is_button_pressed_cv.wait(lock, [&]() { return main_thread_stop_status.load(std::memory_order::memory_order_seq_cst); });
}

static void handle_webserver_data(int &socket_fd, int &payload_size) {
    uint32_t genre = 0;
    uint32_t timbre = 0;
    uint32_t tempo = 0;
    double temperature = 0;
    uint32_t bars = 0;

    recieve_webserver_data(socket_fd, payload_size, genre, timbre, tempo, temperature, bars);
    fprintf(stdout, "temp  = %d\n", tempo);
    fprintf(stdout, "bars = %d\n", bars);
    bpm.store(tempo, std::memory_order::memory_order_seq_cst);
}


void *wait_button_pressed(void *thread_args)
{
#warning "Clean up wait_button_pressed function\n"
    int cmd;
    int destination;
    int stage_id;
    int payload_size;
    char buffer[1024];

    int this_stage_id = STAGE1;
    int this_destination = BIG_BROTHER;

    while (1)
    {

        retrieve_header(buffer, networkbb_fd);
        parse_header(buffer, destination, cmd, stage_id, payload_size);

        switch (cmd)
        {
        case START:
            start_recording();
            send_ack(networkbb_fd, this_destination, this_stage_id);
            break;
        case STOP:
            stop_recording();

            //wait for main thread to finish
            await_network_backbone_notify_complete();
            send_ack(networkbb_fd, this_destination, this_stage_id);
            break;
        case WEBSERVER_DATA:
            handle_webserver_data(networkbb_fd, payload_size);
            break;
        default:
            std::cout << " Sorrry kid wrong command\n";
            break;
        }
    }
}

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
    int id = STAGE1;
    return connect_and_register(id, networkbb_fd);
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

int prepare_capture_device()
{
    int err = 0;

    //prepare capture
    if ((err = snd_pcm_prepare(capture_handle)) < 0)
    {
        print_error(err, "Could not prepare the capture device!");
    }


    return err;
}
int close_capture_handle()
{
    int ret;
    ret  = snd_pcm_close(capture_handle);
    ret |= snd_pcm_close(playback_handle);

    return ret;
}

void async_playback_until_button_press();

int record_until_button_press(int num_bytes_to_record)
{
    int err;
    if ((err = snd_pcm_start(capture_handle)) < 0)
    {
        print_error(err, "Could not start the capture device!");
        return err;
    }

    bool is_consumer_thread_spawned = false;
    num_bytes_read = 0;
    while (num_bytes_read + BYTES_PER_PERIOD < num_bytes_to_record && is_button_pressed.load(std::memory_order::memory_order_relaxed))
    {
        // Await a new set of data
        if ((err = snd_pcm_wait(capture_handle, 1000)) < 0)
        {
            print_error(err, "Capture wait failed!");
            return err;
        }

        snd_pcm_sframes_t frames_available = snd_pcm_avail_update(capture_handle);
        if (frames_available < FRAMES_PER_PERIOD)
        {
            fprintf(stderr, "Too few frames received: expected %d, received %lu!\n", FRAMES_PER_PERIOD, frames_available);
            return 1;
        }

        // If there were more frames available than expected, report it - if this happens many times in a row, we might get an overrun
        if (frames_available != FRAMES_PER_PERIOD)
        {
            fprintf(stderr, "Expected %d frames, but %lu are ready. Monitor for overflow?", FRAMES_PER_PERIOD, frames_available);
        }

        // Read one period, even if more is available
        if ((err = snd_pcm_readi(capture_handle, buffer + num_bytes_read, FRAMES_PER_PERIOD)) != FRAMES_PER_PERIOD)
        {
            print_error(err, "Frame read failed!");
            close_capture_handle();
            return err;
        }

        //fprintf(stdout, "Num_bytes_Read = %d\n", num_bytes_read);

        num_bytes_read += BYTES_PER_PERIOD;
	
	// (NOTE Ryan Heminway) listening for 1 periods instead
        if (num_bytes_read == BYTES_PER_PERIOD * 3)
        {
            // Spawn the consumer thread for async playback 
            std::thread playback_thread(async_playback_until_button_press);
            playback_thread.detach();
        }
    }

    // If the button has not been pressed, we must notify big brother to cycle its state machine
    if (is_button_pressed.load(std::memory_order::memory_order_relaxed))
    {
        //fprintf(stderr, "%s", "Recording ran out of memory!\n");
        fprintf(stdout, "%s\n", "Recording duration reached - notify BB that button was pressed!");
        is_button_pressed.store(false, std::memory_order_seq_cst);
    }

#warning *** MEMORY OVERFLOW TEMPORARILY DOES NOT ERROR - FIX THIS POST DEBUG SESSION!!!

    // !TODO flush the capture buffer!!!

    if ((err = snd_pcm_close(capture_handle)) < 0)
    {
        print_error(err, "Could not pause the capture device!");
        return err;
    }

    return 0;
}

void async_playback_until_button_press()
{
    fprintf(stdout, "Playback spawned: read %d bytes\n", num_bytes_read);

    int err = 0;
    int num_bytes_written = 0;
    bool overtook = false;

    uint8_t* sync_buffer = new uint8_t[BYTES_PER_PERIOD];

    while (num_bytes_written + BYTES_PER_PERIOD < WAV_BUFFER_SIZE && is_button_pressed.load(std::memory_order::memory_order_relaxed))
    {
        if (!overtook && num_bytes_written >= num_bytes_read)
        {
            fprintf(stderr, "Write went too fast: wrote %d bytes.\n", num_bytes_written);
            overtook = true;
        }
        else if (overtook && num_bytes_written < num_bytes_read)
        {
            fprintf(stderr, "%s\n", "write back behind read");
            overtook = false;
        }

        if ((err = snd_pcm_wait(playback_handle, 1000)) < 0)
        {
            fprintf(stdout, "num bytes written = %d\n", num_bytes_written);
            print_error(err, "Poll failed! normal_playback\n");
            snd_pcm_close(playback_handle);
            return;
        }
        
        int frames_to_deliver;
        if ((frames_to_deliver = snd_pcm_avail_update(playback_handle)) < 0)
        {
            if (frames_to_deliver == -EPIPE)
            {
                print_error(frames_to_deliver, "An xrun occurred!");
                snd_pcm_prepare(playback_handle);
                continue;
            }
            else
            {
                print_error(frames_to_deliver, "An unknown error occurred!\n");
                return;
            }
        }

        // Cap the frames to write
        //frames_to_deliver = (frames_to_deliver > FRAMES_PER_PERIOD / 2) ? FRAMES_PER_PERIOD / 2 : frames_to_deliver;

        // Copy the data to write into the sync buffer and add the metronome data at the current write pointer mod one measure length
        memcpy(sync_buffer, buffer + num_bytes_written, frames_to_deliver * BYTES_PER_FRAME);
        for (int i = 0; i < frames_to_deliver * BYTES_PER_FRAME; i++)
        {
            sync_buffer[i] += metronome_buffer[(num_bytes_written + i) % metronome_buffer_size];
        }

        int frames_written;
        if ((frames_written = snd_pcm_writei(playback_handle, sync_buffer, frames_to_deliver)) != frames_to_deliver)
        {
            if (frames_written == -EPIPE)
            {
                fprintf(stdout, "%s\n", "underrun!");
                snd_pcm_prepare(playback_handle);
                continue;
            }
            else
            {
                fprintf(stderr, "writei (wrote %d): expected to write %d frames, actually wrote %d!\n",
                        num_bytes_written, FRAMES_PER_PERIOD, frames_written);
                return;
            }
        }

        num_bytes_written += frames_written * BYTES_PER_FRAME;
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


static inline int calculate_num_sample(int bpm){
    float bpm_float = (float)bpm;
    float denom = bpm_float / 60.;
    return (int)((SAMPLE_RATE * 4) / denom);
}

static inline int calculate_met_measure_buffer_size(int samples_per_measure)
{
    return (samples_per_measure *  NUM_CHANNELS * BYTES_PER_SAMPLE) + met_delay_at_start;
}

static void fill_metronome_buffer(uint8_t *metronome_buffer, int buffer_size)
{
    //memset(metronome_buffer, 0, buffer_size * sizeof(uint8_t));

    int l0 = met_delay_at_start, 
        l1 = met_delay_at_start + ((buffer_size - met_delay_at_start) / 4), 
        l2 = met_delay_at_start + ((buffer_size - met_delay_at_start) / 2), 
        l3 = met_delay_at_start + ((buffer_size - met_delay_at_start) * 3 / 4);

    fprintf(stdout, "Clicks @ %d, %d, %d, %d (%d delay, %d bpm)", 
            l0, l1, l2, l3, met_delay_at_start, bpm.load(std::memory_order_relaxed));

    memcpy(metronome_buffer + l0, click_high, click_high_size);
    memcpy(metronome_buffer + l1, click_low, click_low_size);
    memcpy(metronome_buffer + l2, click_low, click_low_size);
    memcpy(metronome_buffer + l3, click_low, click_low_size);
}

static void open_tick_files(){
    if (click_high != nullptr && click_low != nullptr) 
    {
        return;
    }

    auto high_click_name = "/home/patch/click_high.wav";
    auto low_click_name = "/home/patch/click_low.wav";

    std::ifstream high_click_file;
    high_click_file.open(high_click_name, std::ios::in | std::ios::binary);

    std::ifstream low_click_file;
    low_click_file.open(low_click_name, std::ios::in | std::ios::binary);

    //seek to the end so we can file size
    high_click_file.seekg(0, std::ios::end);
    low_click_file.seekg(0, std::ios::end);

    click_high_size = high_click_file.tellg();
    click_low_size = low_click_file.tellg();

    //allocate buffers and fill them
    click_low = new uint8_t[click_low_size];
    click_high = new uint8_t[click_high_size];

    // Skip the 44-byte header
    high_click_file.seekg(44, std::ios::beg);
    low_click_file.seekg(44, std::ios::beg);
    high_click_file.read((char*)click_high, click_high_size);
    low_click_file.read((char*)click_low, click_low_size); 

    high_click_file.close();
    low_click_file.close();
}

static void play_countin(uint8_t* metronome_buffer, int buffer_size)
{
    int err = 0;
    int num_bytes_written = 0;
    while (num_bytes_written + BYTES_PER_PERIOD < buffer_size)
    {
        if ((err = snd_pcm_wait(playback_handle, 1000)) < 0)
        {
            fprintf(stdout,  "err = %d\n", err);
            print_error(err, "Poll failed! plat_counting\n");
            snd_pcm_close(playback_handle);
            return;
        }
        
        int frames_to_deliver;
        if ((frames_to_deliver = snd_pcm_avail_update(playback_handle)) < 0)
        {
            if (frames_to_deliver == -EPIPE)
            {
                print_error(frames_to_deliver, "An xrun occurred!");
                snd_pcm_prepare(playback_handle);
                continue;
            }
            else
            {
                print_error(frames_to_deliver, "An unknown error occurred!\n");
                return;
            }
        }

        // Cap the frames to write
        //frames_to_deliver = (frames_to_deliver > FRAMES_PER_PERIOD ) ? FRAMES_PER_PERIOD : frames_to_deliver;

        int frames_written;
        if ((frames_written = snd_pcm_writei(playback_handle, 
                metronome_buffer + num_bytes_written, frames_to_deliver)) != frames_to_deliver)
        {
            if (frames_written == -EPIPE)
            {
                fprintf(stdout, "%s\n", "underrun!");
                snd_pcm_prepare(playback_handle);
                continue;
            }
            else
            {
                fprintf(stderr, "writei (wrote %d): expected to write %d frames, actually wrote %d!\n",
                        num_bytes_written, FRAMES_PER_PERIOD, frames_written);
                return;
            }
        }

        fprintf(stdout, "frames written =%d\n", frames_written);

        num_bytes_written += frames_written * BYTES_PER_FRAME;
    }

    if ((err = snd_pcm_close(playback_handle)) < 0)
    {
        print_error(err, "Could not close the playback device!");
        return;
    }
} 

static int metronome()
{
    //get number of samples
    int samples_per_measure = calculate_num_sample(bpm.load(std::memory_order_relaxed));

    //total number of bytes
    metronome_buffer_size =   calculate_met_measure_buffer_size(samples_per_measure);

    //allocate buffer 
    delete[] metronome_buffer;
    metronome_buffer = new uint8_t[metronome_buffer_size];

    // Fill the click buffers
    open_tick_files();

    //fill array with correct data
    fill_metronome_buffer(metronome_buffer, metronome_buffer_size);

    fprintf(stdout, "Buffer size: %d, high click size: %d, low click size:%d\n", 
            metronome_buffer_size, click_high_size, click_low_size);

    // Play the metronome buffer in its entirety 
    play_countin(metronome_buffer, metronome_buffer_size);

    // Return the number of samples for 'n' measures of recording
    #warning "*** NUM SAMPLES ASSUMED TO BE 2 FOR NOW ***"
    return metronome_buffer_size * 2;
}

int close_networkbb_fd()
{
    return close(networkbb_fd);
}

void calculate_header_values(uint32_t *chunk_size, uint16_t *num_channels, uint32_t *sample_rate, uint32_t *byte_rate,
                             uint16_t *block_align, uint16_t *bits_per_sample, uint32_t *subchunk_size)
{
    // Some of these are #defined for now
    *num_channels = NUM_CHANNELS;
    *sample_rate = SAMPLE_RATE;
    *bits_per_sample = BYTES_PER_SAMPLE * 8;

    // Block align = # channels * bytes/sample
    *block_align = NUM_CHANNELS * BYTES_PER_SAMPLE;

    // Byte rate = Sample rate * # channels * bytes/sample
    *byte_rate = SAMPLE_RATE * NUM_CHANNELS * BYTES_PER_SAMPLE;

    // Subchunk size = # samples * # channels * bytes/sample = num_bytes_read I believe?
    *subchunk_size = num_bytes_read;

    // Chunk size = subchunk size * header size
    *chunk_size = WAV_BUFFER_SIZE + 36;
}

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

int write_wav_header_old(FILE *file)
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
    data[0] = 16;
    data[1] = 0x00;
    data[2] = 0x00;
    data[3] = 0x00;
    fwrite(data, sizeof(uint8_t), 4, file);

    // Audio format is always 1 for PCM
    data[0] = 0x01;
    data[1] = 0x00;
    data[2] = 0x00;
    data[3] = 0x00;
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

int write_wav_data_old(FILE *file)
{
    int num_bytes_written;
    if ((num_bytes_written = fwrite(buffer, sizeof(uint8_t), num_bytes_read, file)) != num_bytes_read)
    {
        fprintf(stderr, "Wav data write failed: Expected %d, wrote %d!", WAV_BUFFER_SIZE, num_bytes_written);
        return 1;
    }

    return 0;
}

int write_to_wav(const char *path)
{
    // Open the file, write the header, write the contents
    FILE *file = fopen(path, "w");
    if (!file)
    {
        fprintf(stderr, "Failed to open file %s!\n", path);
        return 1;
    }

    if (write_wav_header_old(file))
    {
        fprintf(stderr, "Failed to write WAV header!\n");
        fclose(file);
        return 1;
    }

    if (write_wav_data_old(file))
    {
        fprintf(stderr, "Failed to write WAV data!\n");
        fclose(file);
        return 1;
    }

    return 0;
}

int write_wav_header(uint8_t *mem)
{
    uint8_t data[4];

    // Must calculate: chunk size, num channels, sample rate, byte rate, block align, bits/sample, subchunk2 size
    uint32_t chunk_size, sample_rate, byte_rate, subchunk_size;
    uint16_t num_channels, block_align, bits_per_sample;
    calculate_header_values(&chunk_size, &num_channels, &sample_rate, &byte_rate,
                            &block_align, &bits_per_sample, &subchunk_size);

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

    return 0;
}

int write_wav_data(uint8_t *mem)
{
    // Memcpy into the shared memory; skip the 44-byte header
    memcpy(mem + 44, buffer, num_bytes_read);
    return 0;
}

int write_to_shared_mem()
{
    // Open the shared memory block
    int size = num_bytes_read + 44;
    uint8_t *shared_mem_blk = (uint8_t *)get_wav_mem_blk(size);
    if (!shared_mem_blk)
    {
        fprintf(stderr, "Failed to open shared memory block!\n");
        return 1;
    }

    if (write_wav_header(shared_mem_blk))
    {
        fprintf(stderr, "Failed to write wav header into shared memory block!\n");
        detach_mem_blk(shared_mem_blk);
        return 1;
    }

    if (write_wav_data(shared_mem_blk))
    {
        fprintf(stderr, "Failed to write wav data into shared memory block!\n");
        detach_mem_blk(shared_mem_blk);
        return 1;
    }

    detach_mem_blk(shared_mem_blk);
    return 0;
}

int ping_network_backbone()
{
    int destination = BACKBONE_SERVER;
    int err = (stage1_data_ready(networkbb_fd, destination, num_bytes_read + 44) == SUCCESS) ? 0 : 1;
    if (err)
    {
        fprintf(stderr, "%s\n", "Failed to notify network backbone!");
    }

    return err;
}

int main(int, char *[])
{
    // Register button press signal handler
    //signal(SIGINT, button_pressed);
    pthread_t thread;
    int err = 0;

    // Set BPM to default 
    bpm.store(DEFAULT_BPM, std::memory_order::memory_order_seq_cst);

    // Initialize the server connection
    if (connect_networkbb() == FAILED)
    {
        fprintf(stderr, "%s\n", "Failed to connect to network backbone!");
        return 1;
    }
    err = pthread_create(&thread, NULL, wait_button_pressed, NULL);

    if (err)
    {
        std::cout << "Error:unable to create thread," << err << std::endl;
        exit(-1);
    }

    //button has not been pressed before
    is_button_pressed.store(false, std::memory_order::memory_order_seq_cst);
    main_thread_stop_status.store(false, std::memory_order::memory_order_seq_cst);

    while (1)
    {
        await_button_press();

        // Init the capture handle
        err = init_capture_handle(BYTES_PER_PERIOD);
        if (err)
        {
            return err;
        }

         // Init the capture handle
        err = init_playback_handle(BYTES_PER_PERIOD);
        if (err)
        {
            return err;
        }

        // Play the countin 
        int num_bytes_to_record = metronome();

        // Prepare the capture handle
        if ((err = prepare_capture_device()))
        {
            return err;
        }

        //Re-open the playback device and record
        if ((err = init_playback_handle(BYTES_PER_PERIOD / 4)))
        {
            return err;
        }

        if ((err = record_until_button_press(num_bytes_to_record)))
        {
            break;
        }

        // Write to shared memory
        if ((err = write_to_shared_mem()))
        {
            break;
        }

        // Inform the network backbone that data is ready
        if ((err = ping_network_backbone()))
        {
            break;
        }

        main_thread_stop_status.store(true, std::memory_order_seq_cst);
        is_button_pressed_cv.notify_one();
        // !TEMP
        fprintf(stdout, "Num bytes read: %d\n", num_bytes_read);

    }

    // Cleanup
    if (err)
    {
        close_capture_handle();
        close_networkbb_fd();
    }

    return err;
}
