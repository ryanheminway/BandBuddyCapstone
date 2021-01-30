#include "shared_mem.h"
#include <stdio.h>



int main(void){
    //destroy block
    if(destroy_mem_blk(FILE_NAME)){
        printf("Succesfully destroyed mem shared block\n");
    } else {
        printf("Could not destroy block. Did you create the block\n");
    }

    return 0;
}