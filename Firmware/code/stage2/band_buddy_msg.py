import sys
sys.path.insert(0, '../network_bb/flatbuffer_messages')
import socket
import flatbuffers
import Server.Header.Cmds as cmds 
import Server.Header.Header as header 
import Server.Header.Stages as stages 

FAILED = -1
SUCCESS = 1
#socket descriptor that can be used by the client to send/recieve info
socket_fd = -1

def create_and_send_header(sck_fd, payload_size, destination, cmd, stage_id):
    ret = FAILED
    builder = flatbuffers.Builder(0)

    header.Start(builder)
    header.AddPayloadSize(payload_size)
    header.AddDestination(destination)
    header.AddCmd(cmd)
    header.AddStageId(stage_id)
    header_msg = header.End(builder)

    builder.Finish(header_msg)

    buf = builder.Output()
    buf_size = len(buf)

    sck_fd.send()



def connect_and_register(stage_id, host, port):
    socket_fd = socket.socket()
    socket_fd.connect((host, port))

def header_size():
   pass 

def get_header():
    pass
def get_payload():
    pass

if __name__ == "__main__":
    print("This module should not be ran as main")