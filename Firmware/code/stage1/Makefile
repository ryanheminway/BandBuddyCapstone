CROSS_PREFIX := /pitools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-
CC           := $(CROSS_PREFIX)g++

OBJS 		 := analog2wav.o shared_mem.o
CFLAGS		 := -std=c++11
TARGET		 := analog2wave
LIB			 := -lasound
INCLUDE		 := -I/usr/include/ -I../common
LIB_PATH	 := -L/usr/lib

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS)  -o  $@ $^ $(LIB_PATH) $(LIB)

%.o: %.cpp
	$(CC) $(CFLAGS) -c  $(INCLUDE) $< 

clean:
	rm -f $(TARGET) $(OBJS)