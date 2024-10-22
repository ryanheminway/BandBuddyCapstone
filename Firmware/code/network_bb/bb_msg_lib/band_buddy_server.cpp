#include "band_buddy_server.h"
#include "band_buddy_msg.h"
#include "header_generated.h"
#include "stage1_generated.h"
#include "stage2_generated.h"
#include "web_server_generated.h"
#include "web_server_stage3_generated.h"
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
using namespace Server::WebServerStage3;

int retrieve_header(char *buffer, int sockfd) {
    int ret = FAILED;
    int header_size = 0; 

#ifdef DEBUG
    std::cout << "Header size: " << header_size << std::endl;
#endif
    ret = read(sockfd, &header_size, sizeof(header_size));

    if (header_size == 0){
        std::cout << " Error while retrieving header size BBBBBBBBBBB\n";
        return FAILED;
    }
    ret = read(sockfd, buffer, header_size);
    #ifdef DEBUG
    //std::cout << "Msg: " << buffer << std::endl;
    #endif
    if(ret <= 0) {
        std::cout << "Error in receiving header AAAAAAAAAAAA" << std::endl;
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

static int retrieve_payload(int &sock_fd, int &payload_size, uint8_t *raw_data){
    int bytes_recv = 0;

	
    while(bytes_recv < payload_size){

	     bytes_recv += recv(sock_fd, raw_data + bytes_recv, payload_size - bytes_recv, MSG_WAITALL);

	    std::cout << " ret = " << bytes_recv << std::endl;
    }

    if(bytes_recv < payload_size) {
        std::cout << "Error in recieving payload\n";
        return FAILED;
    }

    return SUCCESS;
}

int recieve_stage1_fbb(int &sock_fd, int &payload_sz, uint32_t &wave_data_sz){
    int ret = FAILED;
    uint8_t *buffer_ptr = NULL;

    buffer_ptr = (uint8_t *)malloc(payload_sz);

    ret = retrieve_payload(sock_fd, payload_sz, buffer_ptr);

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

    buffer_ptr = (uint8_t *)malloc(payload_sz);

    ret = retrieve_payload(sock_fd, payload_sz, buffer_ptr);

    if(ret == FAILED){
        return ret;
    }
    std::cout << "retrieve_payload bytes: " << ret << std::endl;

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

int recieve_header_and_stage1_fbb(int &sockfd, uint32_t &wave_data_sz){
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
    if( destination != BIG_BROTHER && cmd != STAGE1_DATA_READY){
        std::cout << "Big brother cannot process this message\n";
        return FAILED;
    }
    

    ret = recieve_stage1_fbb(sockfd, payload_size, wave_data_sz);

    return ret;
}

int recieve_and_mem_shared_stage2_data(int &sock_fd, int &payload_size){
    int ret = FAILED;
    uint8_t *shared_mem_blk = NULL;


    shared_mem_blk = (uint8_t *)get_midi_mem_blk(payload_size);

    //get shared_mem block and write it 
    if ( shared_mem_blk == NULL)
    {
        printf("Could not get memory block\n");
        return FAILED;
    }

    ret = retrieve_payload(sock_fd, payload_size, shared_mem_blk);

    if(ret == FAILED){
        return ret;
    }


    detach_mem_blk(shared_mem_blk);

    return ret;
} 

int recieve_ack(int &sock_fd, int &stage_id){

    int ret = FAILED;
    int destination; 
    int cmd; 
    int payload_size;
    int thi_stage_id; //do not really care where it came from

    char buffer[1024];
    ret = retrieve_header(buffer, sock_fd);
    parse_header(buffer, destination, cmd, thi_stage_id, payload_size);

    if(cmd != ACK || destination != stage_id || ret == FAILED){
        std::cout << " Could not processed ACK\n";
        return FAILED;
    }

    return SUCCESS;
}


int recieve_through_message(int &sock_fd, uint8_t *buff, int &payload_size){
   int ret = FAILED; 

   ret = retrieve_payload(sock_fd, payload_size, buff);

   return ret;
}

int recieve_webserver_data(int &sock_fd, int &payload_sz, uint32_t &genre, uint32_t &timbre, uint32_t &tempo, double &temperature, uint32_t &bars){
    int ret = FAILED;
    uint8_t *buffer_ptr = NULL;

    buffer_ptr = (uint8_t *)malloc(payload_sz);

    ret = retrieve_payload(sock_fd, payload_sz, buffer_ptr);

    if(ret == FAILED){
        return ret;
    }

    std::cout << "retrieve_payload bytes: " << ret << std::endl;

    auto webserver_fb = GetWebServer(buffer_ptr);
    genre = webserver_fb->genre();
    timbre = webserver_fb->timbre();
    tempo = webserver_fb->tempo();
    temperature = webserver_fb->temperature();
    bars = webserver_fb->bars();

#ifdef DEBUG
    std::cout << "genre: " << genre << std::endl;
    std::cout << "timbre: " << timbre << std::endl;
    std::cout << "tempo: " << tempo << std::endl;
    std::cout << "temperature: " << temperature << std::endl;
#endif

    ret = SUCCESS;
    free(buffer_ptr);

    return ret;
}

int recieve_webserverstage3_data(int &sock_fd, int &payload_sz, uint8_t &drums, uint8_t &guitar) {
    int ret = FAILED;
    uint8_t *buffer_ptr = NULL;

    buffer_ptr = (uint8_t *)malloc(payload_sz);

    ret = retrieve_payload(sock_fd, payload_sz, buffer_ptr);

    if(ret == FAILED){
        return ret;
    }

    std::cout << "retrieve_payload bytes: " << ret << std::endl;

    auto webserver_fb = GetWebServerStage3(buffer_ptr);
    drums = webserver_fb->drums();
    guitar = webserver_fb->guitar();

#ifdef DEBUG
    std::cout << "drums: " << drums << std::endl;
    std::cout << "guitar: " << guitar << std::endl;
#endif

    ret = SUCCESS;
    free(buffer_ptr);

    return ret;
}
