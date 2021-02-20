#include "band_buddy_msg.h"
#include "band_buddy_server.h"
#include "shared_mem.h"
#include <iostream>
#include <unistd.h>
#include <stdint.h>

int main(void) {
    int ret = FAILED;
    int sock_fd = 0;
    int stage_id = STAGE3;
    uint32_t midi_data_sz = 0;
    char *midi_data_ptr = NULL;

    //register stage with server 
    ret = connect_and_register(stage_id, sock_fd);

    if (ret == FAILED){
        std::cout << " Could not connect to the server\n";
        close(sock_fd);
        return 1;
    }

    //listen for incoming fbb message
    //recieve_header_and_stage2_fbb(int sockfd, char *buffer, uint32_t &midi_data_sz)
    ret = recieve_header_and_stage2_fbb(sock_fd, midi_data_sz);

    if(ret == FAILED) {
        std::cout << " Could not get recieve stage2 fbb\n";
        close(sock_fd);
        return 1;
    }


    std::cout << "Midi data size = " << midi_data_sz << std::endl;
 
    midi_data_ptr = (char *)get_midi_mem_blk(midi_data_sz);

    if (midi_data_sz == NULL){
        std::cout << "Could not get midi block\n";
        close(sock_fd);
        return 1;
    }

    std::cout << midi_data_ptr << std::endl;

    close(sock_fd);
    return 0;

}