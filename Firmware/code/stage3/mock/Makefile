# CROSS_PREFIX := /pitools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-
CC           := $(CROSS_PREFIX)g++

OBJS 		 := dummy_wav_to_shared_mem.o ../../common/shared_mem.o
CFLAGS		 := -std=c++11 -Wall -g
TARGET		 := dummy_wav_to_shared_mem
LIB			 := -lasound
INCLUDE		 := -I/usr/include/ -I../../common
LIB_PATH	 := -L/usr/lib

all: $(TARGET)

$(TARGET): $(OBJS)
	mv shared_mem.o ../../common
	$(CC) $(CFLAGS)  -o  $@ $^ $(LIB_PATH) $(LIB)

%.o: %.cpp
	$(CC) $(CFLAGS) -c  $(INCLUDE) $< 

clean:
	rm -f $(TARGET) $(OBJS)