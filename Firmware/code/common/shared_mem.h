#ifndef SHARED_MEM
#define SHARED_MEM

#include <stdbool.h>

#define BLK_SIZE 4096   //word align 
#define FILE_NAME   "shared_memory.txt"   

void *attach_mem_blk(char *file_name, int size);
bool detach_mem_blk(void *blk_ptr);
bool destroy_mem_blk(char *file_name);


#endif //SHARED_MEM