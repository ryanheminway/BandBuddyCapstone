#include "band_buddy_msg.h"
#include "header_generated.h"
#include <netdb.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <sys/socket.h> 
#define MAX 80 
#define PORT 8080 
#define SERVER_IP "127.0.0.1"
#define SA struct sockaddr 

using namespace Server::Header; 

int *get_socket_discriptor(){
    int *sockfd_ptr = (int *)malloc(sizeof(int));
    struct sockaddr_in servaddr; 
  
    // socket create and varification 
    *sockfd_ptr = socket(AF_INET, SOCK_STREAM, 0); 
    if (*sockfd_ptr == -1) { 
        printf("socket creation failed...\n"); 
        exit(0); 
    } 
    else
        printf("Socket successfully created..\n"); 
    bzero(&servaddr, sizeof(servaddr)); 
  
    // assign IP, PORT 
    servaddr.sin_family = AF_INET; 
    servaddr.sin_addr.s_addr = inet_addr("SERVER_IP"); 
    servaddr.sin_port = htons(PORT); 
  
    // connect the client socket to server socket 
    if (connect(*sockfd_ptr, (SA*)&servaddr, sizeof(servaddr)) != 0) { 
        printf("connection with the server failed...\n"); 
        exit(0); 
    } 
    else
        printf("connected to the server..\n"); 

    return sockfd_ptr;
}

int get_header_size(){
    int ret = FAILED;
    flatbuffers::FlatBufferBuilder builder;
    auto header = CreateHeader(builder);
    builder.Finish(header);

    ret = builder.GetSize();
    builder.Clear();

    return ret;
}

int register_stage(const int &socket_fd, int stage_id){
    int ret = FAILED;
    flatbuffers::FlatBufferBuilder builder; 
    Stages this_stage = static_cast<Stages>(stage_id);
    Cmds cmd = Cmds_Register;
    
    auto header = CreateHeader(builder, 0, Stages_Stage1, cmd, this_stage); 
    builder.Finish(header);

    auto header_ptr = builder.GetBufferPointer();
    int header_size = builder.GetSize();

    //send over network
    ret = write(socket_fd, header_ptr, header_size);

    return ret != FAILED ? SUCCESS : FAILED;

}