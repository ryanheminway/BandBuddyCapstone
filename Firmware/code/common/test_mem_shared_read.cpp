#include "shared_mem.h"
#include <stdio.h>
#include <string.h>

int main(void){
    char message[100];
    char *mem_blk = (char *)attach_mem_blk(FILE_NAME, BLK_SIZE);

    if (mem_blk == NULL)
    {
       printf("Could not get memory block\n");
    }

    memcpy(message, mem_blk, strlen(message)+1);

    printf("Message read from block: %s", message);

    //detach from block
    detach_mem_blk(mem_blk);
    
    return 0;
}