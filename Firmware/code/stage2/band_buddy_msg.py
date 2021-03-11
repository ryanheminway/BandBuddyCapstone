import sys
sys.path.insert(0, '/home/patch/BandBuddyCapstone/Firmware/code/network_bb/flatbuffer_messages')
import socket
import flatbuffers
import Server.Header.Cmds as cmds 
import Server.Header.Header as header 
import Server.Header.Stages as stages 
import Server.WebServer.WebServer as webserver
import Server.WebServer_Stage3.WebServerStage3 as websever_stage3
# My defines so users have easy access to flatbuffers types
STAGE1 = stages.Stages().Stage1
STAGE2 = stages.Stages().Stage2
STAGE3 = stages.Stages().Stage3
WEB_SERVER_STAGE = stages.Stages().WebServer

REGISTER = cmds.Cmds().Register
STAGE1_DATA = cmds.Cmds().Stage1_data
STAGE2_DATA_READY = cmds.Cmds().Stage2_data_ready
WEBSERVER_DATA = cmds.Cmds().Web_server_data

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


def create_webserver_fbb(genre, timbre, tempo, temperature):
    builder = flatbuffers.Builder(0)
    
    webserver.Start(builder)
    webserver.AddGenre(builder, genre)
    webserver.AddTimbre(builder, timbre)
    webserver.AddTempo(builder, tempo)
    webserver.AddTemperature(builder, temperature)
    webserver_msg = webserver.End(builder)

    builder.Finish(webserver_msg) 

    return builder.Output() 

def create_webserver_stage3_fbb(drums, guitar):
    builder = flatbuffers.Builder(0)

    websever_stage3.Start(builder)
    websever_stage3.AddDrums(builder, drums)
    websever_stage3.AddGuitar(builder, guitar)
    websever_stage3_msg = websever_stage3.End(builder)

    builder.Finish(websever_stage3_msg)

    return builder.Output()



def create_and_send_header(sck_fd, payload_size, destination, cmd, stage_id):
    ret = FAILED
    buf = create_header(payload_size, destination, cmd, stage_id)
    header_size = int(len(buf))
    ret = sck_fd.sendall(header_size.to_bytes(4, byteorder="little"))
    print("header size: ", header_size)
    print("sent from sendall: ", sck_fd.sendall(buf))
    #return ret == None ? SUCCESS : FAILED
    if ret == None:
        return SUCCESS
    else:
        return FAILED

def connect_and_register(host, port, stage_id):
    payload_size = 0
    destination = STAGE2 
    cmd = REGISTER 

    sfd = get_socket_descriptor(host, port)
    
    if sfd != FAILED:
        create_and_send_header(sfd, payload_size, destination, cmd, stage_id)
        return sfd
    else:
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


def send_payload(sock_fd, data):
    ret = FAILED
    ret = sock_fd.sendall(data)
    #return ret == None ? SUCCESS : FAILED
    if ret == None:
        return SUCCESS
    else:
        return FAILED


def send_midi_data(sock_fd, raw_data, destination):
    ret = FAILED
    # payload_size = size
    # destination = stages.Stages().Stage3
    this_cmd = cmds.Cmds().Stage2_data_ready
    stage_id = stages.Stages().Stage2
    payload_size = len(raw_data)

    ret = create_and_send_header(sock_fd, payload_size, destination, this_cmd, stage_id)
    ret = send_payload(sock_fd, raw_data)
    return ret

def send_webserver_data(sock_fd, genre, timbre, tempo, temperature, drums, guitar, destination, stage_id):
    ret = FAILED

    if destination == STAGE3:
        webserver_fbb = create_webserver_stage3_fbb(drums, guitar)
    elif destination == STAGE2:
        webserver_fbb = create_webserver_fbb(genre, timbre, tempo, temperature)
    else:
        print("Only stage2 and stage3 can process this message")
        return FAILED

    payload_size = len(webserver_fbb)
    this_cmd = WEBSERVER_DATA 

    ret = create_and_send_header(sock_fd, payload_size, destination, this_cmd, stage_id)
    ret = send_payload(sock_fd, webserver_fbb)
    return ret


def recv_wav_msg(sock_fd, header_fbb):
    # header_fbb = recv_header(sock_fd)

    # error checking
    if header_fbb.Destination() != stages.Stages().Stage2 and header_fbb.Cmd() != cmds.Cmds().Stage1_data:
        return FAILED
    else:
        print("Payload size %d" % header_fbb.PayloadSize())
        buf = get_payload(sock_fd, header_fbb.PayloadSize())
        return buf

def recv_webserver_fbb(sock_fd, header_fbb):
    #header_fbb = recv_header(sock_fd)

    ##error checking 
    if header_fbb.Destination() != stages.Stages().Stage2 and header_fbb.Cmd() != cmds.Cmds().Web_server_data :
        return FAILED
    else :
        print("Payload size %d" %header_fbb.PayloadSize())
        buf = get_payload(sock_fd, header_fbb.PayloadSize())
        webserver_fbb = webserver.WebServer.GetRootAs(buf, 0)
        return webserver_fbb


def send_msg(sock_fd, data1, destination, cmd):
    ret = FAILED
    if cmd == WEBSERVER_DATA:
        ret = send_webserver_data(sock_fd, data1, destination)
    elif cmd == STAGE2_DATA_READY:
        ret = send_midi_data(sock_fd, data1, destination)
    return ret

def recv_msg(sock_fd):
    header_fbb = recv_header(sock_fd)
    buf = ''

    if header_fbb.Cmd() == cmds.Cmds().Stage1_data:
        buf = recv_wav_msg(sock_fd, header_fbb)
    elif header_fbb.Cmd() == cmds.Cmds().Web_server_data:
        buf = recv_webserver_fbb(sock_fd, header_fbb)

    return header_fbb.Cmd(), buf


# Unnecessary with stage2 module
def test():
    header_len = get_header_size() 
    print(header_len)
    host = "127.0.0.1"
    port = 8080

    socket_fd = connect_and_register(host, port, STAGE2)

    while True:
        try:
            #Recieve wav data
            print("Waiting for wav data\n")
            wav_data = recv_wav_msg(socket_fd)
            
            if wav_data == None:
                print("Could not get payload")

            # print(wav_data) 
            ##send "midi data back to stage 3"
            print("Sending ***MOCK MIDI DATA*** data\n")
            with open("/home/patch/BandBuddyCapstone/Firmware/code/stage2/mock/hcb_drums.wav", mode="rb") as wav:
                raw_data = wav.read()
                print(f"MIDI data length: {len(raw_data)}")
                send_midi_data(socket_fd, raw_data, len(raw_data))

        except KeyboardInterrupt:
            print("Shutting down stage2")
            socket_fd.close()
            sys.exit(0)

def test_webserver():
   host = "127.0.0.1"
   port = 8080 

   socket_fd = connect_and_register(host, port, STAGE2)

   #send_webserver_data(socket_fd, 45, STAGE2)

   webserver_fbb = recv_webserver_fbb(socket_fd)

   print("Recieved genre = %d" %webserver_fbb.Genre())


def test_large_data():
    host = "129.10.159.188" 
    port = 8080 

    f = open("model_out_drums.wav", "rb")
    f_test = open("model_out_test.wav", "wb")
    socket_fd = connect_and_register(host, port, STAGE2)

    wav_data = f.read()

    print(wav_data[0x5d0:0x5e0])

    f_test.write(wav_data)

    send_midi_data(socket_fd, wav_data, len(wav_data))

    f.close()
    f_test.close()
if __name__ == "__main__":
    test_webserver()
    
