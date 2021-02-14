//all includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "shared_mem.h"

//ERROR codes
#define IPC_ERROR   (-1)

static int get_blk_id(char *file_name, int size){
    key_t file_key;

    file_key = ftok(file_name, 0);
    if(file_key == IPC_ERROR){
        return IPC_ERROR;
    }

    //create or get shared memory block
    return shmget(file_key, size, 0644 | IPC_CREAT);
}

//attaches to shared memory block. 
//@param file_name ---> name of the file associated with block
//@param size ----> size of block of memory. keep it word align
//@return ----> pointer to memory block
void *attach_mem_blk(char *file_name, int size){
    int shared_blk_id = get_blk_id(file_name, size);
    void *ret = NULL;

    //erro checking 
    if(shared_blk_id == IPC_ERROR){
        return ret;
    }

    //we got valid id, so map it to out current memory space 
    ret = shmat(shared_blk_id, NULL, 0);
    if(ret == (void *)IPC_ERROR){
        return NULL;
    }
    return ret;
}

//after proccess is done with memory, it should detach using this function 
//@param blk ---> pointer to shared mem block
 bool detach_mem_blk(void *blk_ptr){
    return (shmdt(blk_ptr) != IPC_ERROR);
 }

//when there are no more processes using the memory block, it should be destroyed to free up memory using this function
//@param file_name ---> file name associiated with shared memory block
//@return success/fail(true/false)
bool destroy_mem_blk(char *file_name){
    int shared_blk_id = get_blk_id(file_name, 0);

    if(shared_blk_id == IPC_ERROR){
        return false;
    }
    
    return (shmctl(shared_blk_id, IPC_RMID, NULL) != IPC_ERROR);
}

