#include "band_buddy_server.h"
#include "band_buddy_msg.h"
#include "header_generated.h"
#include "stage1_generated.h"
#include "shared_mem.h"
#include <iostream>
#include <netdb.h> 
#include <stdio.h> 
#include <unistd.h>
#include <stdlib.h> 
#include <string.h> 
#include <sys/socket.h> 
#include <arpa/inet.h>
#include <stdint.h>

#define IP_ADDR "127.0.0.1"

using namespace Server::Header; 
using namespace Server::Stage1;

int retrieve_header(char *buffer, int sockfd) {
    int ret = -1;
    int header_size = get_header_size();

#ifdef DEBUG
    std::cout << "Header size: " << header_size << std::endl;
#endif

    ret = read(sockfd, buffer, header_size);
    #ifdef DEBUG
    std::cout << "Msg: " << buffer << std::endl;
    #endif
    if(ret < 0) {
        std::cout << "Error in receiving header" << std::endl;
    }
    return ret;
}

int parse_header(char *buffer, int &destination, int &cmd, int &stage_id, int &size) {
    auto header = GetHeader(buffer);
    // bool check = VerifyHeaderBuffer();
    destination = static_cast<int>(header->destination());
    cmd = static_cast<int>(header->cmd());
    stage_id = static_cast<int>(header->stage_id());
    size = static_cast<int>(header->size());

    std::cout << "stage id" << stage_id << std::endl;
    std::cout << "dest: " << destination << std::endl;
    std::cout << "cmd: " << cmd << std::endl;
    std::cout << "size: " << size << std::endl;
    return 0;
}

int register_client(int *client_lst, int id, int sockfd) {
    if(client_lst[id]) {
        // For sanity check
        std::cout << "Client already exists" << std::endl;
        return -1;
    }
    client_lst[id] = sockfd;
    return 0;
}

static int retrieve_payload(int &sock_fd, int &payload_size, uint8_t **raw_data){
    int ret = FAILED;

    *(raw_data) = (uint8_t *)malloc(payload_size);

    if(*(raw_data) == NULL){
        return FAILED;
    }

    ret = read(sock_fd, *(raw_data), payload_size);

    if(ret < 0) {
        std::cout << "Error in recieving payload\n";
        return FAILED;
    }

    return SUCCESS;
}

int recieve_stage1_fbb(int &sock_fd, int &payload_sz, uint32_t &wave_data_sz){
    int ret = FAILED;
    uint8_t *buffer_ptr = NULL;

    ret = retrieve_payload(sock_fd, payload_sz, &buffer_ptr);

    if(ret == FAILED){
        return ret;
    }

    auto stage1_fb = GetStage1(buffer_ptr);
    wave_data_sz = stage1_fb->wave_data_sz();

    ret = SUCCESS;
    free(buffer_ptr);

    return ret;
}

int recieve_and_mem_shared_stage2_data(int &sock_fd, int &payload_size){
    int ret = FAILED;
    uint8_t *buffer_ptr = NULL;
    uint8_t *shared_mem_blk = NULL;

    ret = retrieve_payload(sock_fd, payload_size, &buffer_ptr);

    if(ret == FAILED){
        return ret;
    }

    //get shared_mem block and write it 
    shared_mem_blk = (uint8_t *)get_midi_mem_blk(payload_size);
    if ( shared_mem_blk == NULL)
    {
        printf("Could not get memory block\n");
        return FAILED;
    }

    //copy data
    memcpy(shared_mem_blk, buffer_ptr, payload_size);

    detach_mem_blk(shared_mem_blk);
    free(buffer_ptr);

    return ret;
} 
