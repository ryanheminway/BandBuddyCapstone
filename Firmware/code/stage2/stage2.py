import sys
sys.path.insert(0, '../../../ML/Magenta/NanoMagenta')
import band_buddy_msg as network
import nano_audio_utils as audio
import nano_configs as configs
from nano_trained_model import TrainedModel
import soundfile as sf
import io


def main():
    host = "127.0.0.1"
    port = 8080

    # Connect to network backbone
    socket_fd = network.connect_and_register(host, port)

    # Load model checkpoint
    GROOVAE_2BAR_TAP_FIXED_VELOCITY = "../../model_checkpoints/groovae_rock/groovae_rock.tar"
    config_2bar_tap = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity']
    # Build GrooVAE model from checkpoint variables and config model definition
    groovae_2bar_tap = TrainedModel(config_2bar_tap, 1, checkpoint_dir_or_path=GROOVAE_2BAR_TAP_FIXED_VELOCITY)

    # (TODO) Need to experiment with these parameters. Also need to make them configurable via a network msg
    temperature = 0.8
    velocity_threshold = 0.08

    while True:
        try:
            # Recieve wav data
            print("Waiting for wav data\n")
            wav_data = network.recv_wav_msg(socket_fd)

            if wav_data == None:
                print("Could not get payload")

            print("Got wav data: ", wav_data)

            # Trying to convert raw wav bytearray to nparray
            np_wav_data, sr = sf.read(io.BytesIO(wav_data))
            print("Converted to np array: ", np_wav_data)

            print("Generating drum track...\n")
            # Apply trained model to input track. Only care about drum audio on return
            #full_drum_audio, _, _, _, _ = audio.audio_to_drum(f, velocity_threshold=velocity_threshold,
            #                                                  temperature=temperature, model=groovae_2bar_tap)

            #print("Got generated drum track: ", full_drum_audio)


            # send "midi data back to stage 3"
            print("Sending midi data\n")
            network.send_midi_data(socket_fd, wav_data, len(wav_data))

        except KeyboardInterrupt:
            print("Shutting down stage2")
            socket_fd.close()
            sys.exit(0)


if __name__ == "__main__":
    main()