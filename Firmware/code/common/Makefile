CROSS_PREFIX := /pitools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-
CC           := $(CROSS_PREFIX)g++

OBJS 		 := shared_mem.o test_mem_shared_destroy.o test_mem_shared_write.o test_mem_shared_read.o 
CFLAGS		 := -std=c++11
TARGETS		 := test_shared_destroy test_shared_write test_shared_read

all: $(TARGETS)

test_shared_destroy: shared_mem.o test_mem_shared_destroy.o
	$(CC) $(CFLAGS)  -o  $@ $^ 
	
test_shared_write: shared_mem.o test_mem_shared_write.o
	$(CC) $(CFLAGS)  -o  $@ $^ 

test_shared_read: shared_mem.o test_mem_shared_read.o
	$(CC) $(CFLAGS)  -o  $@ $^ 

%.o: %.cpp
	$(CC) $(CFLAGS) -c  $(INCLUDE) $< 

clean:
	rm -f $(TARGETS) $(OBJS)