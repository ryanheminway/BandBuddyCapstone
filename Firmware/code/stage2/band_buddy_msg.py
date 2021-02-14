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
socket_fd = FAILED

def get_socket_descriptor(host, port):
    if socket_fd == -1 :
        socket_fd = socket.socket()
        socket_fd.connect((host, port))
        return socket_fd
    else:
        return socket_fd

def create_header(payload_size, destination, cmd, stage_id):
    builder = flatbuffers.Builder(0)

    header.Start(builder)
    header.AddPayloadSize(builder, payload_size)
    header.AddDestination(builder, destination)
    header.AddCmd(builder, cmd)
    header.AddStageId(builder, stage_id)
    header_msg = header.End(builder)

    builder.Finish(header_msg)

    return builder.Output() 
        
def create_and_send_header(sck_fd, payload_size, destination, cmd, stage_id):
    ret = FAILED
    buf = create_header(payload_size, destination, cmd, stage_id)
    sck_fd.sendall(buf)

def connect_and_register(host, port):
    payload_size = 0
    destination = stages.Stages().Stage2
    cmd = cmds.Cmds().Register
    stage_id = stages.Stages().Stage2

    sfd = get_socket_descriptor(host, port)
    
    if sfd != FAILED :
        create_and_send_header(sfd, payload_size, destination, cmd, stage_id)
        return sfd
    else : 
        return FAILED

def get_header_size():
    payload_size = 0
    destination = stages.Stages().Stage1
    cmd = cmds.Cmds().Register
    stage_id = stages.Stages().Stage2 

    buf = create_header(payload_size, destination, cmd, stage_id)
    return len(buf)

def recv_msg(sock_fd, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def recv_header(sock_fd):
    header_raw = recv_msg(sock_fd, get_header_size())
    header_msg = header.Header.GetRootAs(header_raw, 0)
    return header_msg

def get_payload(sock_fd, size):
    raw_data = recv_msg(socket_fd, size)
    return raw_data

if __name__ == "__main__":
    header_len = get_header_size() 
    print(header_len)
