#include "band_buddy_server.h"
#include "band_buddy_msg.h"
#include "header_generated.h"
#include "stage1_generated.h"
#include "stage2_generated.h"
#include "web_server_generated.h"
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
using namespace Server::Stage2;
using namespace Server::WebServer;

int retrieve_header(char *buffer, int sockfd) {
    int ret = FAILED;
    int header_size = 0; 

#ifdef DEBUG
    std::cout << "Header size: " << header_size << std::endl;
#endif
    ret = read(sockfd, &header_size, sizeof(header_size));

    if (header_size == 0){
        std::cout << " Error while retrieving header size\n";
        return FAILED;
    }
    ret = read(sockfd, buffer, header_size);
    #ifdef DEBUG
    //std::cout << "Msg: " << buffer << std::endl;
    #endif
    if(ret <= 0) {
        std::cout << "Error in receiving header" << std::endl;
        return FAILED;
    }

    return ret;
}

int parse_header(char *buffer, int &destination, int &cmd, int &stage_id, int &size) {
    auto header = GetHeader(buffer);
    // bool check = VerifyHeaderBuffer();
    destination = static_cast<int>(header->destination());
    cmd = static_cast<int>(header->cmd());
    stage_id = static_cast<int>(header->stage_id());
    size = static_cast<int>(header->payload_size());

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
    std::cout << "Wave_data_size = " << wave_data_sz << std::endl;

    ret = SUCCESS;
    free(buffer_ptr);

    return ret;
}

int recieve_stage2_fbb(int &sock_fd, int &payload_sz, uint32_t &midi_data_sz){
    int ret = FAILED;
    uint8_t *buffer_ptr = NULL;

    ret = retrieve_payload(sock_fd, payload_sz, &buffer_ptr);

    if(ret == FAILED){
        return ret;
    }

    auto stage2_fb = GetStage2(buffer_ptr);
    midi_data_sz = stage2_fb->midi_data_sz();
    std::cout << "Wave_data_size = " << midi_data_sz << std::endl;

    ret = SUCCESS;
    free(buffer_ptr);

    return ret;
}

int recieve_header_and_stage2_fbb(int &sockfd, uint32_t &midi_data_sz){
    int ret = FAILED;
    int destination, cmd, stage_id, payload_size;
    char buffer[1024];

    ret = retrieve_header(buffer, sockfd);

    if (ret == FAILED){
        std::cout << "Failed to retrieve header\n";
        return FAILED;
    }

    parse_header(buffer, destination, cmd, stage_id, payload_size);

    //sanity check 
    if( destination != STAGE3 && cmd != STAGE2_DATA_READY){
        std::cout << "Stage3 cannot process this message\n";
        return FAILED;
    }
    

    ret = recieve_stage2_fbb(sockfd, payload_size, midi_data_sz);

    return ret;
}

int recieve_and_send_webserver_fbb(int &sock_fd, int &payload_size, int &destination_socket)
{
    int ret = FAILED;
    uint8_t *buffer_ptr = NULL;

    ret = retrieve_payload(sock_fd, payload_size, &buffer_ptr);

    if (ret == FAILED)
    {
        return ret;
    }

    // don't need to do this. Debug only
    auto webserver_fb = GetWebServer(buffer_ptr);

    std::cout << "Genre = " << webserver_fb->genre() << std::endl;
    
    ret = send_webserver_data(destination_socket, buffer_ptr, payload_size);

    return ret != FAILED ? SUCCESS : FAILED;
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