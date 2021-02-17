#include "shared_mem.h"
#include <stdio.h>
#include <string.h>

int main(void){
    int size = BLK_SIZE;

    char *mem_blk = (char *)get_wav_mem_blk(size);

    if (mem_blk == NULL)
    {
       printf("Could not get memory block\n");
    }

    printf("Message read from block: %s\n", mem_blk);

    //detach from block
    detach_mem_blk(mem_blk);
    
    return 0;
}