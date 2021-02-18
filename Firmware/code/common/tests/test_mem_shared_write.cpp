#include "shared_mem.h"
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "band_buddy_msg.h"
#include "band_buddy_server.h"
#include <unistd.h>

int main(void){
    const char *message = "This is a test write into mem shared block";
    int size = BLK_SIZE;
    char *mem_blk = (char *)get_wav_mem_blk(size);
    int ret = FAILED;
    int sock_fd;
    int stage_id = STAGE1;

    ret = connect_and_register(stage_id, sock_fd);

    if(ret == FAILED){
        std::cout << " Could not register stage\n";
        return 0;
    }

    if (mem_blk == NULL)
    {
        printf("Could not get memory block\n");
        close(sock_fd);
        return 1;
    }

    int message_sz = strlen(message) + 1;

    printf("Writing data to shared memory\n");
    memcpy(mem_blk, message, message_sz);

    stage1_data_ready(sock_fd, message_sz);

    detach_mem_blk(mem_blk);
    close(sock_fd);
    
    return 0;
}