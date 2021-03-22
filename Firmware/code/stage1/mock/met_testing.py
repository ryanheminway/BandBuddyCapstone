import wave

def calculate_buffer_size_in_samples(bpm):
    # Formula from white board: sample rate / bps * 4 beats per measure 
    return int((48000 * 4) / (bpm / 60))

def calculate_buffer_size(samples_per_measure):
    # Multiply by # channels and sample size 
    return int(samples_per_measure * 2 * 2)

def go():
    # Assume 60BPM for now
    bpm = 120

    # 1) Calculate buffer size 
    print(f"BPM: {bpm}")
    buffer_size_samples = calculate_buffer_size_in_samples(bpm)
    print(f"Buffer size, in samples: {buffer_size_samples}")

    buffer_size = calculate_buffer_size(buffer_size_samples)
    print(f"Buffer size, in bytes: {buffer_size}")

    # 2) Allocate the buffer
    buffer = []
    for i in range(0, buffer_size + 44):
        buffer.append(0)

    # 3) Open the click files
    with open("/home/patch/click_high.wav", "rb") as click_high_f:
        with open("/home/patch/click_low.wav", "rb") as click_low_f:
            click_high = click_high_f.read()
            click_low = click_low_f.read();

            # 4) Put the high click at position 0 in the buffer
            for i in range(0, len(click_high)): 
                buffer[i] = click_high[i]
            

            # 5) Put 3 low clicks at positions n/4, n/2, 3n/4 in the buffer
            l1 = buffer_size // 4
            l2 = buffer_size // 2
            l3 = buffer_size * 3 // 4
            for i in range(0, len(click_low)):
                buffer[l1 + i] = click_low[i] 
                buffer[l2 + i] = click_low[i] 
                buffer[l3 + i] = click_low[i] 

            with wave.open("/home/patch/met_out.wav", "wb") as out:
                out.setnchannels(2)
                out.setsampwidth(2)
                out.setframerate(48000)
                bb = bytes(buffer)
                out.writeframes(bb)

            # 6) Write WAV header 
            # wav_header = []
            # wav_header.append(click_high[0:44])
            # wav_header[4:8] = buffer_size + 44 - 8
            # wav_header[40:44] = buffer_size
            # buffer.insert(0, wav_header)

            # 7) Write WAV to a file
            # with open("/home/patch/met_out.wav", "wb") as out:
            #     out.write(buffer)
            

if __name__ == "__main__": 
    go()