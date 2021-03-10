import sys
import numpy as np
from scipy.io.wavfile import read, write 
sys.path.insert(0, '../../../ML/Magenta/NanoMagenta')
import band_buddy_msg as network
import nano_audio_utils as audio
import nano_configs as configs
from nano_trained_model import TrainedModel
import soundfile as sf
import io


"""
Handles all basic Stage2 operations. This includes:
- Connecting to network backbone
- Waiting for messages to come in
- Handling data messages (commands) from network backbone
    - Configuring params via webserver
    - Applying model to wav data 
    - Sending responses via network backbone
"""
class Stage2Handler():
    def __init__(self):
        self.tempo = 100  # (TODO) Control via webserver
        self.genre = 0  # (TODO) Control via webserver, also tie to models
        self.soundpack = 0  # (TODO) Control via Webserver, also tie to soundpacks
        self.temperature = 0.8  # (TODO) Need to experiment with these parameters. Also need to make them configurable via a network msg
        self.velocity_threshold = 0.08
        self.drum_track = None
        self.socket_fd = None
        self.soundpack_path = None # (TODO) Control via webserver

    # Handles wav data representing an input track for the drum generation model:
    # 1) Applies given model to data to create drum track
    # 2) Applies soundpack to midi drums to generate wav track
    # 3) Convert wav track to byte array before returning
    def handle_wav_data(self, data, model):
        # Trying to convert raw wav bytearray to nparray
        np_wav_data, sr = audio.read_wav_data(data=data)
        # print("Converted to np array: ", np_wav_data)
        # print("Len of converted: ", len(np_wav_data))
        # print("SR of converted: ", sr)

        print("Generating drum track...\n")
        # Apply trained model to input track. Only care about drum audio on return
        full_drum_audio = audio.audio_to_drum(np_wav_data, sr, velocity_threshold=self.velocity_threshold,
                                                          temperature=self.temperature, model=model)

        #print("Got generated drum track: ", full_drum_audio)
        # full_drum_audio = np.repeat(full_drum_audio, 2)
        # print("Duplicated into stereo data")
        print("Turning it into wav!")
        full_drum_audio = audio.midi_to_wav(full_drum_audio, sr)

        print("Writing drum track to disk for good measure")
        sf.write("model_out_drums.wav", full_drum_audio, sr, subtype='PCM_24')

        # with open("model_out_drums.wav", 'rb') as f:
        #    drum_bytes = f.read()

        drum_bytes = bytes()
        byte_io = io.BytesIO(drum_bytes)
        write(byte_io, sr, (full_drum_audio * 32767).astype(np.int16))
        output_drum_wav = byte_io.read()

        return output_drum_wav

    # Handles webserver data by either changing parameters as requested or sharing current parameter settings
    def handle_webserver_data(self, data):
        print("Got webserver message: ", data)


def main():
    host = "129.10.159.188"
    port = 8080
    handler = Stage2Handler()

    # Load model checkpoint
    GROOVAE_2BAR_TAP_FIXED_VELOCITY = "/home/bandbuddy/BandBuddyCapstone/ML/model_checkpoints/groovae_rock/groovae_rock.tar"
    config_2bar_tap = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity']
    # Build GrooVAE model from checkpoint variables and config model definition
    model = TrainedModel(config_2bar_tap, 1, checkpoint_dir_or_path=GROOVAE_2BAR_TAP_FIXED_VELOCITY)

    print("Stage2 connected")
    socket_fd = network.connect_and_register(host, port, network.STAGE2)

    while True:
        try:
            print("Waiting for commands from network backbone\n")
            command, buff = network.recv_msg(socket_fd)

            if buff == None:
                print("Could not get payload... program dying")
                exit(1)

            print("got command: ", command)
            print("we want command: ", network.STAGE1_DATA)

            if (command == network.STAGE1_DATA):
                print("STAGE1 DATA")
                drum_data = handler.handle_wav_data(buff, model)

                # print("SENDING DATA: ", drum_data)
                print("SENDING LEN: ", len(drum_data))

                # send wav data back to stage 3
                print("Sending drum wav data\n")
                network.send_msg(socket_fd, drum_data, network.BACKBONE_SERVER, network.STAGE2_DATA_READY)
            elif (command == network.WEBSERVER_DATA):
                print("WEBSERVER DATA")
                handler.handle_webserver_data(buff)

        except KeyboardInterrupt:
            print("Shutting down stage2")
            socket_fd.close()
            sys.exit(0)

if __name__ == "__main__":
    main()
