#ifndef BAND_BUDDY_MSG
#define BAND_BUDDY_MSG 

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

int get_header_size();
int get_socket_discriptor();
int create_and_send_header(int &socket_fd, int &payload_size, int &destination, int &cmd, int &stage_id);
int register_stage(const int &socket_fd, int &stage);
int connect_and_register(int &stage_id);

#endif //BAND_BUDDY_MSG