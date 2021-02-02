#ifndef BAND_BUDDY_MSG
#define BAND_BUDDY_MSG 

#define SUCCESS (1)
#define FAILED  (-1)

int get_header_size();
int get_socket_discriptor();
int create_header();
int register_stage(const int &socket_fd, int stage);

#endif //BAND_BUDDY_MSG