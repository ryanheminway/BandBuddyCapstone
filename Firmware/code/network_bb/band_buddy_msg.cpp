#include "band_buddy_msg.h"
#include "header_generated.h"
#include <netdb.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <sys/socket.h> 
#include <arpa/inet.h>
#define MAX 80 
#define PORT 8080 
#define SERVER_IP "127.0.0.1"
#define SA struct sockaddr 

using namespace Server::Header; 

int get_socket_discriptor(){
    int sockfd = 0; 
    struct sockaddr_in servaddr; 
  
    // socket create and varification 
    sockfd = socket(AF_INET, SOCK_STREAM, 0); 
    if (sockfd == -1) { 
        printf("socket creation failed...\n"); 
        exit(0); 
    } 
    else
        printf("Socket successfully created..\n"); 
    bzero(&servaddr, sizeof(servaddr)); 
  
    // assign IP, PORT 
    servaddr.sin_family = AF_INET; 
    servaddr.sin_addr.s_addr = inet_addr(SERVER_IP); 
    servaddr.sin_port = htons(PORT); 
  
    // connect the client socket to server socket 
    if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr)) != 0) { 
        printf("connection with the server failed...\n"); 
        exit(0); 
    } 
    else
        printf("connected to the server..\n"); 

    return sockfd;
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


int create_and_send_header(int &socket_fd, int &payload_size, int &destination, int &cmd, int &stage_id){
    int ret = FAILED;
    flatbuffers::FlatBufferBuilder builder; 

    //static cast all args
    Stages dest = static_cast<Stages>(destination);
    Cmds command = static_cast<Cmds>(cmd);
    Stages this_stage = static_cast<Stages>(stage_id);

    //create header
    auto header = CreateHeader(builder, payload_size, dest, command, this_stage);
    builder.Finish(header);

    auto header_ptr = builder.GetBufferPointer();
    int size = builder.GetSize();

    // ret = write(socket_fd, &size, sizeof(size));
    ret = write(socket_fd, header_ptr, size);

    return ret != FAILED ? SUCCESS : FAILED;

}

int register_stage(const int &socket_fd, int &stage_id){
    int ret = FAILED;
    flatbuffers::FlatBufferBuilder builder; 
    Stages this_stage = static_cast<Stages>(stage_id);
    Cmds cmd = Cmds_Register;
    
    auto header = CreateHeader(builder, 0, Stages_Stage1, cmd, this_stage); 
    builder.Finish(header);
    //TODO: use create_and_sendd header function
    auto header_ptr = builder.GetBufferPointer();
    int header_size = builder.GetSize();

    //send size along with payload over network
    // ret = write(socket_fd, &header_size, sizeof(int));
    ret = write(socket_fd, header_ptr, header_size);

    builder.Clear();

    return ret != FAILED ? SUCCESS : FAILED;
}

int connect_and_register(int &stage_id){
    int socket_fd = 0, ret = FAILED;

    //connect to server to get socket descriptor
    socket_fd = get_socket_discriptor();
    ret = register_stage(socket_fd, stage_id);

    return ret != FAILED ? socket_fd : FAILED;

}