#include "shared_mem.h"
#include <stdio.h>
#include <string.h>

int main(void){
    const char *message = "This is a test write into mem shared block";
    int size = BLK_SIZE;
    char *mem_blk = (char *)get_wav_mem_blk(size);

    if (mem_blk == NULL)
    {
        printf("Could not get memory block\n");
        return 1;
    }

    printf("Writing data to shared memory\n");
    memcpy(mem_blk, message, strlen(message)+1);

    detach_mem_blk(mem_blk);
    
    return 0;
}