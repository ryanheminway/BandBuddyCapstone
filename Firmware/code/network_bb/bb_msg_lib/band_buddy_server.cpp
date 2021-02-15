#include "band_buddy_server.h"
#include "band_buddy_msg.h"
#include "header_generated.h"
#include <iostream>
#include <netdb.h> 
#include <stdio.h> 
#include <unistd.h>
#include <stdlib.h> 
#include <string.h> 
#include <sys/socket.h> 
#include <arpa/inet.h>
#define IP_ADDR "127.0.0.1"

using namespace Server::Header; 

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
