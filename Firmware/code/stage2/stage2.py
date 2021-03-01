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


def main():
    host = "129.10.159.188"
    port = 8080

    # Load model checkpoint
    GROOVAE_2BAR_TAP_FIXED_VELOCITY = "/home/bandbuddy/BandBuddyCapstone/ML/model_checkpoints/groovae_rock/groovae_rock.tar"
    config_2bar_tap = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity']
    # Build GrooVAE model from checkpoint variables and config model definition
    groovae_2bar_tap = TrainedModel(config_2bar_tap, 1, checkpoint_dir_or_path=GROOVAE_2BAR_TAP_FIXED_VELOCITY)

    # (TODO) Need to experiment with these parameters. Also need to make them configurable via a network msg
    temperature = 0.8
    velocity_threshold = 0.08
    
    # Connect to network backbone
    socket_fd = network.connect_and_register(host, port, network.STAGE2)

    while True:
        try:
            # Recieve wav data
            print("Waiting for wav data\n")
            wav_data = network.recv_wav_msg(socket_fd)

            if wav_data == None:
                print("Could not get payload")
            
            # Trying to convert raw wav bytearray to nparray
            np_wav_data, sr = audio.read_wav_data(data=wav_data)
            #print("Converted to np array: ", np_wav_data)
            #print("Len of converted: ", len(np_wav_data))
            #print("SR of converted: ", sr)

            print("Generating drum track...\n")
            # Apply trained model to input track. Only care about drum audio on return
            full_drum_audio, _, _, _, _ = audio.audio_to_drum(np_wav_data, sr, velocity_threshold=velocity_threshold,
                                                              temperature=temperature, model=groovae_2bar_tap)

            print("Got generated drum track: ", full_drum_audio)
            #full_drum_audio = np.repeat(full_drum_audio, 2)
            #print("Duplicated into stereo data")
            
            #print("Writing drum track to disk for good measure")
            #sf.write("model_out_drums.wav", full_drum_audio, sr, subtype='PCM_24')

            #with open("model_out_drums.wav", 'rb') as f:
            #    drum_bytes = f.read()

            drum_bytes = bytes()
            byte_io = io.BytesIO(drum_bytes)
            write(byte_io, sr, (full_drum_audio * 32767).astype(np.int16))
            output_drum_wav = byte_io.read()
            #print("SENDING DATA: ", output_drum_wav)
            print("SENDING LEN: ", len(output_drum_wav))

            # send wav data back to stage 3
            print("Sending drum wav data\n")
            network.send_midi_data(socket_fd, output_drum_wav, len(output_drum_wav))

        except KeyboardInterrupt:
            print("Shutting down stage2")
            socket_fd.close()
            sys.exit(0)

if __name__ == "__main__":
    main()
