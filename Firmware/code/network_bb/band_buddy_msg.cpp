#include "band_buddy_msg.h"
#include "header_generated.h"

using namespace Server::Header; 
int get_header_size(){
    int ret = FAILED;
    flatbuffers::FlatBufferBuilder builder;
    auto header = CreateHeader(builder);
    builder.Finish(header);

    ret = builder.GetSize();
    builder.Clear();

    return ret;
}

int register_stage(int stage_id, const int &socket_fd){
    int ret = FAILED;
    flatbuffers::FlatBufferBuilder builder; 
    Stages this_stage = static_cast<Stages>(stage_id);
    Cmds cmd = Cmds_Register;
    
    auto header = CreateHeader(builder, 0, Stages_Stage1, cmd, this_stage); 
    builder.Finish(header);

    auto header_ptr = builder.GetBufferPointer();
    int header_size = builder.GetSize();

    //send over network
    ret = write(socket_fd, header_ptr, header_size);

    return ret != FAILED ? SUCCESS : FAILED;

}