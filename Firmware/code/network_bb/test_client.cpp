#include "band_buddy_msg.h"
#include <iostream>
#include <stdio.h>


int main(void){
    int socket_fd;
    int stage1 = STAGE3;
    std::cout << "stage: " << stage1 << std::endl;
    socket_fd = connect_and_register(stage1);

    if(socket_fd == FAILED){
        printf("Could not connect to server\n");
    } else{
       printf("Successfull connection.  Socket_fd = %d\n", socket_fd); 
    }

    return 0;
}