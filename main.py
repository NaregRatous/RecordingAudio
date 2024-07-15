import sounddevice as sd
import numpy as np
import wave

# List all available audio input devices
print("Available audio input devices:")
devices = sd.query_devices()
input_devices = [device for device in devices if device['max_input_channels'] > 0]

for idx, device in enumerate(input_devices):
    print(f"{idx}: {device['name']}")

# Prompt user to select an input device
device_index = int(input("Select the device index to use for recording: "))

# Check if the selected index is valid
if device_index < 0 or device_index >= len(input_devices):
    print("Invalid device index.")
else:
    selected_device = input_devices[device_index]
    print(f"Selected device: {selected_device['name']}")

    # Parameters
    duration = 10  # seconds
    sample_rate = 44100  # Hertz
    channels = selected_device['max_input_channels']  # Use the maximum number of input channels

    # Record audio
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype=np.int16, device=selected_device['index'])
    sd.wait()  # Wait until the recording is finished
    print("Recording finished.")

    # Save the audio data as a .wav file
    output_filename = 'my_record.wav'
    with wave.open(output_filename, 'w') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM format
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    print(f"Audio recorded and saved as {output_filename}.")
