import sys
sys.path.insert(0, '../network_bb/flatbuffer_messages')
import socket
import flatbuffers
import Server.Header.Cmds as cmds 
import Server.Header.Header as header 
import Server.Header.Stages as stages 

FAILED = -1
SUCCESS = 1

def get_socket_descriptor(host, port):
    socket_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_fd.connect((host, port))
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
    header_size = int(len(buf))
    sck_fd.sendall(header_size.to_bytes(4, byteorder="little"))
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
        packet = sock_fd.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def recv_header(sock_fd):
    header_size_array = recv_msg(sock_fd, 4)
    header_size = int.from_bytes(header_size_array, "little")

    print("Header size after conversion %d" %header_size)
    header_raw = recv_msg(sock_fd, header_size)
    header_msg = header.Header.GetRootAs(header_raw, 0)
    #wav_data = get_payload(sock_fd, header_msg.PayloadSize())
    #print('Wave data %s' %wav_data)
    return header_msg

def get_payload(sock_fd, size):
    raw_data = recv_msg(sock_fd, size)
    return raw_data

def send_payload(sock_fd, buf):
    sock_fd.sendall(buf)

def send_midi_data(sock_fd, raw_data, size):
    payload_size = size
    destination = stages.Stages().Stage3
    this_cmd = cmds.Cmds().Stage2_data_ready
    stage_id = stages.Stages().Stage2

    create_and_send_header(sock_fd, payload_size, destination, this_cmd, stage_id)
    send_payload(sock_fd, raw_data)

def recv_wav_msg(sock_fd):
    header_fbb = recv_header(sock_fd)

    ##error checking 
    if header_fbb.Destination() != stages.Stages().Stage2 and header_fbb.Cmd() != cmds.Cmds().Stage1_data :
        return FAILED
    else :
        print("Payload size %d" %header_fbb.PayloadSize())
        buf = get_payload(sock_fd, header_fbb.PayloadSize())
        return buf


def test():
    header_len = get_header_size() 
    print(header_len)
    host = "127.0.0.1"
    port = 8080

    socket_fd = connect_and_register(host, port)

    while True:
        try:
            #Recieve wav data
            print("Waiting for wav data\n")
            wav_data = recv_wav_msg(socket_fd)
            
            if wav_data == None:
                print("Could not get payload")

            print(wav_data)
            ##send "midi data back to stage 3"
            print("Sendind midi data\n")
            send_midi_data(socket_fd, wav_data, len(wav_data))

        except KeyboardInterrupt:
            print("Shutting down stage2")
            socket_fd.close()
            sys.exit(0)

if __name__ == "__main__":
    test()
    