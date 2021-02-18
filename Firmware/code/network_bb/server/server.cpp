/*

Thanks geeksforgeeks.com for providing the skeleton for our server.
The skeleton code can be found at https://www.geeksforgeeks.org/socket-programming-in-cc-handling-multiple-clients-on-server-without-multi-threading/
*/


#include "band_buddy_server.h"
#include "band_buddy_msg.h"
#include <iostream>
#include <stdio.h>  
#include <string.h>   //strlen  
#include <stdlib.h>  
#include <errno.h>  
#include <unistd.h>   //close  
#include <arpa/inet.h>    //close  
#include <sys/types.h>  
#include <sys/socket.h>  
#include <netinet/in.h>  
#include <sys/time.h> //FD_SET, FD_ISSET, FD_ZERO macros  
     
#define TRUE   1  
#define FALSE  0  
#define PORT   8080 
#define MAX_CLIENTS 3
#define MAX_BUFFER_SIZE 1024
     
int main(int argc , char *argv[])   
{   
    int opt = TRUE;   
    int master_socket , addrlen , new_socket , client_socket[MAX_CLIENTS],  
        activity, i , valread , sd;   
    int max_sd;
    int destination, cmd, stage_id, payload_size;
    struct sockaddr_in address;
         
    char buffer[MAX_BUFFER_SIZE];  //data buffer of 1K  
         
    //set of socket descriptors  
    fd_set readfds;   
         
    //a message  
    const char *message = "ECHO Daemon v1.0 \r\n";   
     
    //initialise all client_socket[] to 0 so not checked  
    for (i = 0; i < MAX_CLIENTS; i++)   
    {   
        client_socket[i] = 0;   
    }   
         
    //create a master socket  
    if( (master_socket = socket(AF_INET , SOCK_STREAM , 0)) == 0)   
    {   
        perror("socket failed");   
        exit(EXIT_FAILURE);   
    }   
     
    //set master socket to allow multiple connections ,  
    //this is just a good habit, it will work without this  
    if( setsockopt(master_socket, SOL_SOCKET, SO_REUSEADDR, (char *)&opt,  
          sizeof(opt)) < 0 )   
    {   
        perror("setsockopt");   
        exit(EXIT_FAILURE);   
    }   
     
    //type of socket created  
    address.sin_family = AF_INET;   
    address.sin_addr.s_addr = INADDR_ANY;   
    address.sin_port = htons( PORT );   
         
    //bind the socket to localhost port 8888  
    if (bind(master_socket, (struct sockaddr *)&address, sizeof(address))<0)   
    {   
        perror("bind failed");   
        exit(EXIT_FAILURE);   
    }   
    printf("Listener on port %d \n", PORT);   
         
    //try to specify maximum of 3 pending connections for the master socket  
    if (listen(master_socket, 3) < 0)   
    {   
        perror("listen");   
        exit(EXIT_FAILURE);   
    }   
         
    //accept the incoming connection  
    addrlen = sizeof(address);   
    puts("Waiting for connections ...");   
         
    while(TRUE)   
    {   
        //clear the socket set  
        FD_ZERO(&readfds);   
     
        //add master socket to set  
        FD_SET(master_socket, &readfds);   
        max_sd = master_socket;   
             
        //add child sockets to set  
        for ( i = 0 ; i < MAX_CLIENTS; i++)   
        {   
            //socket descriptor  
            sd = client_socket[i];   
                 
            //if valid socket descriptor then add to read list  
            if(sd > 0)   
                FD_SET( sd , &readfds);   
                 
            //highest file descriptor number, need it for the select function  
            if(sd > max_sd)   
                max_sd = sd;   
        }   
     
        //wait for an activity on one of the sockets , timeout is NULL ,  
        //so wait indefinitely  
        activity = select( max_sd + 1 , &readfds , NULL , NULL , NULL);   
       
        if ((activity < 0) && (errno!=EINTR))   
        {   
            printf("select error");   
        }   
             
        //If something happened on the master socket ,  
        //then its an incoming connection  
        if (FD_ISSET(master_socket, &readfds))   
        {   
            if ((new_socket = accept(master_socket,  
                    (struct sockaddr *)&address, (socklen_t*)&addrlen))<0)   
            {   
                perror("accept");   
                exit(EXIT_FAILURE);   
            }   
             
            //inform user of socket number - used in send and receive commands  
            printf("New connection received from socket: %d (%s:%d)\n", 
                    new_socket , inet_ntoa(address.sin_addr) , ntohs(address.sin_port));

            // Receive register message (Header flatbuffer)
            if(retrieve_header(buffer, new_socket) < 0) {
                std::cout << "Error in retrieving header" << std::endl;
                //exit(1);
            }

            // Extract information from flatbuffer
            parse_header(buffer, destination, cmd, stage_id, payload_size);
                 
            // Register client by saving its sockfd based on stage_id
           if(register_client(client_socket, stage_id, new_socket) < 0) {
               std::cout << "Error registering client socket" << std::endl;
               //exit(1);
           }
        }   
             
        //else its some IO operation on some other socket 
        for (i = 0; i < MAX_CLIENTS; i++)   
        {   
            sd = client_socket[i];   
                 
            if (FD_ISSET( sd , &readfds))   
            {   
                //Check if it was for closing , and also read the  
                //incoming message  
                /*
                    get header  --> return header or the info that you need
                    switch(cmd):
                    stage1_data_ready:
                    send_wav_shared_mem(socket_fd, size);
                    ...
                */

                if (retrieve_header(buffer, sd) == FAILED)
                {   
                    //Somebody disconnected , get his details and print  
                    getpeername(sd , (struct sockaddr*)&address , (socklen_t*)&addrlen);
                    printf("Host disconnected , ip %s , port %d \n" ,  
                          inet_ntoa(address.sin_addr) , ntohs(address.sin_port));   
                         
                    //Close the socket and mark as 0 in list for reuse  
                    close( sd );   
                    client_socket[i] = 0;   
                }   
                     
                // Parse header for header size
                // Run appropriate functions based on cmd in header
                else 
                {
                    parse_header(buffer, destination, cmd, stage_id, payload_size);

                    switch(cmd) {
                        case STAGE1_DATA_READY:
                            std::cout << "Processing stage 1 data ready" << std::endl;
                            uint32_t wav_data_sz;

                            if (recieve_stage1_fbb(sd, payload_size, wav_data_sz) != FAILED){
                                send_wav_shared_mem(client_socket[destination], wav_data_sz);
                            } else{
                                std::cout<< " Could not recieve payload\n";
                            }

                            break;
                        case STAGE2_DATA_READY:
                            std::cout << "Processing stage 2 data ready" << std::endl;
                            recieve_and_mem_shared_stage2_data(sd, payload_size);
                            break;
                        case STAGE3_DATA_READY:
                            // TODO: stage3_data_ready function
                            std::cout << "Processing stage 2 data ready" << std::endl;
                            break;
                    }
                }
            }   
        }   
    }   
         
    return 0;   
}
