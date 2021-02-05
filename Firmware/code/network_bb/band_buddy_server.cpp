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
    // int header_size = -1;
    // ret = read(sockfd, &header_size, sizeof(header_size));
    // #ifdef DEBUG
    std::cout << "Header size: " << header_size << std::endl;
    // #endif
    // if(ret < 0) {
    //     std::cout << "Error in receiving header size" << std::endl;
    //     return ret;
    // }
    ret = read(sockfd, buffer, header_size);
    #ifdef DEBUG
    std::cout << "Msg: " << buffer << std::endl;
    #endif
    if(ret < 0) {
        std::cout << "Error in receiving header" << std::endl;
    }
    return ret;
}

int parse_header(char *buffer, int &destination, int &cmd, int &stage_id) {
    auto header = GetHeader(buffer);
    // bool check = VerifyHeaderBuffer();
    destination = static_cast<int>(header->destination());
    cmd = static_cast<int>(header->cmd());
    stage_id = static_cast<int>(header->stage_id());
    std::cout << stage_id << std::endl;
    std::cout << destination << std::endl;
    std::cout << cmd << std::endl;
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
