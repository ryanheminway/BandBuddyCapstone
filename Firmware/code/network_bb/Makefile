#CROSS_PREFIX := /pitools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-
CC           := $(CROSS_PREFIX)g++

OBJS 		 := server.o 
CFLAGS		 := -std=c++11
TARGET		 :=	ban_buddy_server 
LIB			 := 
INCLUDE		 := 
LIB_PATH	 := 

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS)  -o  $@ $^ $(LIB_PATH) $(LIB)

%.o: %.cpp
	$(CC) $(CFLAGS) -c  $(INCLUDE) $< 

clean:
	rm -f $(TARGET) $(OBJS)