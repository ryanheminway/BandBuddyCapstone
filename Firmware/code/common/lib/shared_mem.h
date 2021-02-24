#ifndef SHARED_MEM
#define SHARED_MEM

#include <stdbool.h>

#define BLK_SIZE 4096   //word align 
constexpr const char* WAV_DATA_KEY = "/home/bandbuddy/BandBuddyCapstone/Firmware/code/common/lib/shared_wav.txt";  
constexpr const char* MIDI_DATA_KEY = "/home/bandbuddy/BandBuddyCapstone/Firmware/code/common/lib/shared_midi.txt";

bool detach_mem_blk(void *blk_ptr);
void *get_wav_mem_blk(int size);
void *get_midi_mem_blk(int size);
bool destroy_wav_mem_blk();
bool destroy_midi_mem_blk();

#endif //SHARED_MEM
