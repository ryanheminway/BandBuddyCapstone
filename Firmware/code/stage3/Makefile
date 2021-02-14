# CROSS_PREFIX := /pitools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-
CC 			:= $(CROSS_PREFIX)g++

OBJS		:= wavs2analog.o ../common/shared_mem.o
CFLAGS		:= -Wall -g -std=c++11
TARGET		:= wavs2analog
LIB			:= -lasound
INCLUDE 	:= -I/usr/include -I../common
LIB_PATH	:= -L/usr/lib

all: $(TARGET)

$(TARGET): $(OBJS)
	mv shared_mem.o ../common
	$(CC) $(CFLAGS)  -o  $@ $^ $(LIB_PATH) $(LIB)

%.o: %.cpp
	$(CC) $(CFLAGS) -c  $(INCLUDE) $< 

clean:
	rm -f $(TARGET) $(OBJS)