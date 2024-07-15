import sounddevice as sd
import numpy as np
import wave
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

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
    duration = 1  # seconds
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

    # Convert audio data to a numpy array
    audio_data = np.array(audio_data, dtype=np.float32)

    # If stereo, take one channel
    if channels > 1:
        audio_data = audio_data[:, 0]

    # Perform FFT
    N = len(audio_data)
    yf = fft(audio_data)
    xf = fftfreq(N, 1 / sample_rate)

    # Get the positive frequencies
    pos_mask = np.where(xf >= 0)
    xf = xf[pos_mask]
    yf = np.abs(yf[pos_mask])

    # Plot the frequency and amplitude
    plt.figure(figsize=(12, 6))
    plt.plot(xf, yf)
    plt.title("Frequency vs Amplitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    # Print the main frequencies and amplitudes
    main_freqs = xf[np.argsort(yf)[-10:]]  # Get top 10 frequencies
    main_amps = yf[np.argsort(yf)[-10:]]  # Get top 10 amplitudes
    max_frequency = 0
    max_amplitute = 0
    for freq, amp in zip(main_freqs, main_amps):
        print(f"Frequency: {freq:.2f} Hz, Amplitude: {amp:.2f}")
        if freq>max_frequency:
            max_frequency = freq
        if amp > max_amplitute:
            max_amplitute = amp
    print(f"max freq={max_frequency}, max amp={max_amplitute}")
