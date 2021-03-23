#include "band_buddy_msg.h"
#include "band_buddy_server.h"
#include "header_generated.h"
#include "stage1_generated.h"
#include "stage2_generated.h"
#include "wave_file_generated.h"
#include "web_server_generated.h"
#include "web_server_stage3_generated.h"
#include "shared_mem.h"
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
using namespace Server::Stage1;
using namespace Server::Stage2;
using namespace Server::Wave;

using namespace Server::WebServer;
using namespace Server::WebServerStage3;

static int get_socket_discriptor(){
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
    int stage_id = 2;
    Stages this_stage = static_cast<Stages>(stage_id);
    Cmds cmd = Cmds_Register;
    
    auto header = CreateHeader(builder, 0, Stages_Stage3, cmd, this_stage); 
    // auto header = CreateHeader(builder);
    builder.Finish(header);

    ret = builder.GetSize();
    builder.Clear();

    return ret;
}


static int create_and_send_header(int &socket_fd, int &payload_size, int &destination, int &cmd, int &stage_id){
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

    printf("Header size in create_and_send_header %d\n", size);

    // ret = write(socket_fd, &size, sizeof(size));
    ret = write(socket_fd, &size, sizeof(size));
    ret = write(socket_fd, header_ptr, size);

    return ret != FAILED ? SUCCESS : FAILED;

}

static int register_stage(int &socket_fd, int &stage_id){
    int ret = FAILED;
    int cmd = REGISTER;
    int destination = STAGE1; // DUMMY value. it does not matter 
    int payload_size = 0; // no payload message

    ret = create_and_send_header(socket_fd, payload_size, destination, cmd, stage_id);
    return ret;
}

int connect_and_register(int &stage_id, int &socket_fd){
    int ret = FAILED;
    //connect to server to get socket descriptor
    socket_fd = get_socket_discriptor();
    ret = register_stage(socket_fd, stage_id);
    return ret; 

}

static int send_payload(int &socket_fd, unsigned char *data, int size){
    int ret = FAILED;    
    ret = write(socket_fd, data, size);

    return ret != FAILED ? ret : FAILED;

}

int stage1_data_ready(int &socket_fd, int &destination, int size){
    int ret = FAILED;
    int cmd = STAGE1_DATA_READY;
    int payload_size; 
    int stage_id = STAGE1;
    int data_ready = 1;
    flatbuffers::FlatBufferBuilder builder;  

    auto stage1_msg = CreateStage1(builder, data_ready, size);
    builder.Finish(stage1_msg);

    auto stage2_msg_ptr = builder.GetBufferPointer();
    payload_size = builder.GetSize();

    ret = create_and_send_header(socket_fd, payload_size, destination, cmd, stage_id);
    if(ret == FAILED){
        printf("Error while sending header message to server\n");
        return ret;
    }

    ret = send_payload(socket_fd, stage2_msg_ptr, payload_size);

    builder.Clear();

    return (ret != FAILED) ? SUCCESS : FAILED;
}


int stage2_data_ready(int &socket_fd, int &size){
    int ret = FAILED;
    int cmd = STAGE2_DATA_READY;
    int destination = STAGE3;
    int payload_size; 
    int stage_id = BACKBONE_SERVER;
    int data_ready = 1;
    flatbuffers::FlatBufferBuilder builder;  

    auto stage2_msg = CreateStage2(builder, data_ready, size);
    builder.Finish(stage2_msg);

    auto stage2_msg_ptr = builder.GetBufferPointer();
    payload_size = builder.GetSize();

    ret = create_and_send_header(socket_fd, payload_size, destination, cmd, stage_id);
    if(ret == FAILED){
        printf("Error while sending header message to server\n");
        return ret;
    }

    ret = send_payload(socket_fd, stage2_msg_ptr, payload_size);

    builder.Clear();

    return (ret != FAILED) ? SUCCESS : FAILED;
}

int send_wav_file(int &socket_fd, struct wave_header &wav_hdr, int8_t *raw_data, int &size){
    int ret = FAILED;
    int cmd = STAGE1_DATA;
    int destination = STAGE2;
    int payload_size;
    int stage_id = STAGE1;
    flatbuffers::FlatBufferBuilder builder;

    auto wave_header = CreateWaveHeader(builder, wav_hdr.ChunkID, wav_hdr.ChunkSize, wav_hdr.Format,
                                        wav_hdr.Subchunk1ID, wav_hdr.Subchunk1Size, wav_hdr.AudioFormat,
                                        wav_hdr.NumChannels, wav_hdr.SampleRate, wav_hdr.ByteRate, wav_hdr.BlockAlign,
                                        wav_hdr.BitsPerSample, wav_hdr.Subchunk2ID, wav_hdr.Subchunk2Size);

    auto data = builder.CreateVector(raw_data, size);

    auto wave_file_msg = CreateWaveFile(builder, wave_header, data);
    builder.Finish(wave_file_msg);

    auto wave_file_msg_ptr = builder.GetBufferPointer();
    payload_size = builder.GetSize();

    ret = create_and_send_header(socket_fd, payload_size, destination, cmd, stage_id);
    if(ret == FAILED){
        printf("Error while sending header\n");
        return ret;
    }

    ret = send_payload(socket_fd, wave_file_msg_ptr, payload_size);
    
    return ret;
}

int send_wav_shared_mem(int &socket_fd, uint32_t &size){
    int ret = FAILED;
    int cmd = STAGE1_DATA;
    int destination = STAGE2;
    int payload_size;
    int stage_id = STAGE1;
    unsigned char *shared_mem_blk = NULL;
    flatbuffers::FlatBufferBuilder builder;

    payload_size = size; 

    ret = create_and_send_header(socket_fd, payload_size, destination, cmd, stage_id);
    if(ret == FAILED){
        printf("Error while sending header\n");
        return ret;
    } 

    //get shared_mem buffer  
    shared_mem_blk = (unsigned char *)get_wav_mem_blk(payload_size);

    if(shared_mem_blk == NULL){
        printf("Could not get memory block\n");
        return FAILED;
    }

   ret = send_payload(socket_fd, shared_mem_blk, payload_size);

   printf("Byte sent through send_payload = %d\n", ret);

   //detach 
   detach_mem_blk(shared_mem_blk);

   return ret;
   
}

int send_ack(int &socket_fd, int &destination, int &stage_id){
    int ret = FAILED;
    int cmd = ACK;
    int payload_size = 0;

    ret = create_and_send_header(socket_fd, payload_size, destination, cmd, stage_id);
    if(ret == FAILED){
        printf("Error while sending header\n");
        return ret;
    } 

    return ret;
}


int send_through_message(int &socket_fd, int &destination, int &cmd, int &stage_id, int &payload_size, int &dst_sock){

    int ret = FAILED;
    
    if(payload_size == 0){
        ret = create_and_send_header(dst_sock, payload_size, destination, cmd, stage_id);
    } else {
        uint8_t *buff = (uint8_t *)malloc(payload_size);

        ret = recieve_through_message(socket_fd, buff, payload_size);
        if(ret == FAILED){
            return FAILED;
        }

        create_and_send_header(dst_sock, payload_size, destination, cmd, stage_id);
        ret = send_payload(dst_sock, buff, payload_size);

        return ret;
    }


    return ret;
}


int stage1_start(int &socket_fd, int& stage_id)
{
    int ret = FAILED;
    
    int payload_size = 0, cmd = START, destination = STAGE1;
    ret = create_and_send_header(socket_fd, payload_size, destination, cmd, stage_id);

    return ret;
}

int stage1_stop(int &socket_fd, int& stage_id)
{
    int ret = FAILED;
    
    int payload_size = 0, cmd = STOP, destination = STAGE1;
    ret = create_and_send_header(socket_fd, payload_size, destination, cmd, stage_id);

    return ret;
}

int stage3_stop(int &socket_fd, int& stage_id)
{
    int ret = FAILED;
    
    int payload_size = 0, cmd = STOP, destination = STAGE3;
    ret = create_and_send_header(socket_fd, payload_size, destination, cmd, stage_id);

    return ret;
}

int send_webserver_data(int &socket_fd, int &stage_id, int &destination, uint32_t &genre,
                        uint32_t &timbre, uint32_t &tempo, double &temperature) {
    int ret = FAILED;
    int cmd = WEBSERVER_DATA;
    int payload_size;
    flatbuffers::FlatBufferBuilder builder;

    auto webserver_msg = CreateWebServer(builder, genre, timbre, tempo, temperature);
    builder.Finish(webserver_msg);

    auto webserver_msg_ptr = builder.GetBufferPointer();
    payload_size = builder.GetSize();

    ret = create_and_send_header(socket_fd, payload_size, destination, cmd, stage_id);
    if(ret == FAILED){
        printf("Error while sending header message to server\n");
        return ret;
    }

    ret = send_payload(socket_fd, webserver_msg_ptr, payload_size);

    builder.Clear();

    return (ret != FAILED) ? SUCCESS : FAILED;
}

int send_webserverstage3_data(int &socket_fd, int &stage_id, int &destination, uint8_t &drums, uint8_t &guitar) {
    int ret = FAILED;
    int cmd = WEBSERVER_DATA;
    int payload_size;
    flatbuffers::FlatBufferBuilder builder;

    auto webserverstage3_msg = CreateWebServerStage3(builder, drums, guitar);
    builder.Finish(webserverstage3_msg);

    auto webserverstage3_msg_ptr = builder.GetBufferPointer();
    payload_size = builder.GetSize();

    ret = create_and_send_header(socket_fd, payload_size, destination, cmd, stage_id);
    if(ret == FAILED){
        printf("Error while sending header message to server\n");
        return ret;
    }

    ret = send_payload(socket_fd, webserverstage3_msg_ptr, payload_size);

    builder.Clear();

    return (ret != FAILED) ? SUCCESS : FAILED;
}
