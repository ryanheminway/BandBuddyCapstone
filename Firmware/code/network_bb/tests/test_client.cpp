#include "band_buddy_msg.h"
#include <iostream>
#include <stdio.h>


int main(void){
    int socket_fd, ret = FAILED;
    int stage1 = STAGE3;
    std::cout << "stage: " << stage1 << std::endl;
    ret = connect_and_register(stage1, socket_fd);

    if(ret == FAILED){
        printf("Could not connect to server\n");
    } else{
       printf("Successfull connection.  Socket_fd = %d\n", socket_fd); 
    }

    return 0;
}