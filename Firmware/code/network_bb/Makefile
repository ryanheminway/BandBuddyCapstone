#CROSS_PREFIX := /pitools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-
CC           := $(CROSS_PREFIX)g++ 

OBJS 		 := server.o band_buddy_msg.o band_buddy_server.o test_client.o
CFLAGS		 := -std=c++14 -DDebug -ggdb3
TARGETS		 :=	ban_buddy_server test_client
LIB			 := 
INCLUDE		 := -Iflatbuffer_messages/ 
LIB_PATH	 := 

all: $(TARGETS)

ban_buddy_server: server.o band_buddy_server.o band_buddy_msg.o
	$(CC) $(CFLAGS)  -o  $@ $^ $(LIB_PATH) $(LIB)

test_client: test_client.o band_buddy_msg.o
	$(CC) $(CFLAGS)  -o  $@ $^ $(LIB_PATH) $(LIB)	

%.o: %.cpp
	$(CC) $(CFLAGS) -c  $(INCLUDE) $< 

clean:
	rm -f $(TARGETS) $(OBJS)
