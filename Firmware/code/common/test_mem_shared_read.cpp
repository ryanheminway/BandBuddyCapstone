#include "shared_mem.h"
#include <stdio.h>
#include <string.h>

int main(void){
    char *mem_blk = (char *)attach_mem_blk(FILE_NAME, BLK_SIZE);

    if (mem_blk == NULL)
    {
       printf("Could not get memory block\n");
    }

    printf("Message read from block: %s\n", mem_blk);

    //detach from block
    detach_mem_blk(mem_blk);
    
    return 0;
}