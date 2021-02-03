# CROSS_PREFIX := /pitools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-
CC           := $(CROSS_PREFIX)gcc

OBJS 		 := analog2wav.o
CFLAGS		 := -Wall -Werror
TARGET		 := test


all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< 

clean:
	rm -f $(TARGET) $(OBJS)