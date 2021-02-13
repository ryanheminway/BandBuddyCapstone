#ifndef BAND_BUDDY_MSG
#define BAND_BUDDY_MSG 

#include <stdint.h>

#define SUCCESS (1)
#define FAILED  (-1)

//#define stages and cmds for user 
#define STAGE1  (0)
#define STAGE2  (1)
#define STAGE3  (2)

//Available commands
#define REGISTER            (0)
#define STAGE1_DATA_READY   (1)
#define STAGE2_DATA_READY   (2)
#define STAGE3_DATA_READY   (3)
#define STAGE1_DATA         (4)

#pragma pack(push, 1)
struct wave_header
{
  uint32_t ChunkID, ChunkSize, Format, Subchunk1ID, Subchunk1Size;
  uint16_t AudioFormat, NumChannels;
  uint32_t SampleRate, ByteRate;
  uint16_t BlockAlign, BitsPerSample;
  uint32_t Subchunk2ID, Subchunk2Size;
};
#pragma pack(pop)

#define WAVE_HEADER_SIZE sizeof(struct wave_header)


int get_header_size();
int connect_and_register(int &stage_id, int &socket_fd);
int stage1_data_ready(int &socket_fd, int &size);
int send_wav_file(int &socket_fd, struct wave_header &wav_hdr, int8_t *raw_data, int &size);
int send_wav_shared_mem(int &socket_fd, int &size);

#endif //BAND_BUDDY_MSG