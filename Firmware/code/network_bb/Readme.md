# API for network backbone 
wg: client 
- fun. for writing to backbone 
    - pass in network descriptor for as arg and send flatbuffer 
- register flatbuffer 
    - register function(fd, data need to create flatbuffer)
- wave data flatbuffer 
    - function that takes in all data required to represent .wavfile 
        - wave file header + raw data

rb: server
 - read the flatbuffer header and extract relevant info.
    -size ----> when the message ends

- excute the command specified by the flatbuffer 
    - e.g send wav data to stage2
     - get access to shared memory  

- status update
    - send back a message like an ack.

- if there is a way to get rid off the looping 
- Go over TODOs in code to check that everything is there