#ifndef BAND_BUDDY_SERVER
#define BAND_BUDDY_SERVER

#include <stdint.h> 

int retrieve_header(char *buffer, int sockfd);
int parse_header(char *buffer, int &destination, int &cmd, int &stage_id, int &size);
int register_client(int *client_lst, int id, int sockfd);
int recieve_stage1_fbb(int &sock_fd, int &payload_sz, uint32_t &wave_data_sz);
int recieve_stage2_fbb(int &sock_fd, int &payload_sz, uint32_t &midi_data_sz);
int recieve_header_and_stage2_fbb(int sockfd, char *buffer, uint32_t &midi_data_sz);
int recieve_and_mem_shared_stage2_data(int &sock_fd, int &payload_size);
#endif
