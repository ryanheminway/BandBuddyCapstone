#ifndef BAND_BUDDY_SERVER
#define BAND_BUDDY_SERVER

int retrieve_header(char *buffer, int sockfd);
int parse_header(char *buffer, int &destination, int &cmd, int &stage_id);
int register_client(int *client_lst, int id, int sockfd);

#endif
