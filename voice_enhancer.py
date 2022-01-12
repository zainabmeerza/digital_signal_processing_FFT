from scipy.io import wavfile
from scipy.fft import ifft
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Read the audio files
    audio_file = r'original.wav'
    sample_rate, audio_data = wavfile.read(audio_file)

    # DATA PROCESSING

    # Data is 2 dimensional. Averaging data to get in Mono.
    avg_data = (audio_data[:, 0] + audio_data[:, 1]) / 2
    # Normalizing Data
    normalized_audio_data = avg_data / (np.max(np.abs(avg_data)))
    N = len(normalized_audio_data)
    # Duration and time vector of the audio sample
    duration = N / sample_rate
    time = np.linspace(0, duration, N)

    # Spectrum in Frequency Domain and Frequency Vector
    fft_spectrum = np.fft.fft(normalized_audio_data)
    fft_spectrum_abs = np.abs(fft_spectrum)
    frequency = np.linspace(0, sample_rate, len(avg_data))
  

    # VOICE ENHANCER QUESTION 3
    enhanced_fft_spectrum = np.fft.fft(avg_data)
    enhanced_fft_spectrum_plot = np.fft.fft(normalized_audio_data)

    for lower, upper, enhancer in ((70, 250, 2.25), (6000, 10000, 10.0)):
        x_lower = abs((frequency - lower)).argmin()
        x_upper = abs((frequency - upper)).argmin()
        enhanced_fft_spectrum[x_lower:x_upper + 1] *= enhancer
        enhanced_fft_spectrum[N - x_upper:N - x_lower + 1] *= enhancer
        enhanced_fft_spectrum_plot[x_lower:x_upper + 1] *= enhancer
        enhanced_fft_spectrum_plot[N - x_upper:N - x_lower + 1] *= enhancer

    improved_audio = ifft(enhanced_fft_spectrum).real
    improved_audio = improved_audio.astype(np.int16)

    wavfile.write("improved.wav", sample_rate, improved_audio)

    # PLOTS

    # Normalized Amplitude vs Time Plot
    plt.figure(1)
    plt.plot(time, normalized_audio_data)
    plt.title("Normalised Audio Signal: Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig('normalized_time.eps', format='eps')

    # Normalized Amplitude vs Frequency Plot
    plt.figure(2)
    plt.semilogx(frequency, 20 * np.log10(fft_spectrum_abs / len(fft_spectrum)), label='original')
    plt.title("Normalised Audio Signal: Frequency Domain")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.savefig('normalized_freq.eps', format='eps')

    # Intensity vs Frequency Plot
    plt.figure(3)
    plt.plot(frequency[:len(frequency) // 2], abs(fft_spectrum[:len(frequency) // 2]))
    plt.title("Intensity of Normalised Audio Signal: Frequency Domain")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Intensity: |FFT()| [-]')
    plt.savefig('intensity_original.eps', format='eps')

    # Intensity vs Frequency Plot with Voice Enhancer Added
    plt.figure(4)
    plt.plot(frequency[:len(frequency) // 2], abs(enhanced_fft_spectrum_plot[:len(frequency) // 2]))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Intensity: |FFT()| [-]')
    plt.title('Intensity of Normalised Audio Signal + Voice Enhancer: Frequency Domain')
    plt.savefig('intensity_enhanced.eps', format='eps')

    # Normalized Amplitude vs Frequency Plot with Voice Enhancer Added
    plt.figure(5)
    plt.semilogx(frequency, 20 * np.log10((np.abs(enhanced_fft_spectrum_plot)) / len(enhanced_fft_spectrum_plot)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Audio Signal + Voice Enhancer: Frequency Domain')
    plt.savefig('enhanced_signal_frequency_domain.eps', format='eps')

    # Normalized Amplitude vs Time Plot with Voice Enhancer Added
    plt.figure(6)
    plt.plot(time, improved_audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Signal + Voice Enhancer: Time Domain')
    plt.savefig('enhanced_signal_time_domain.eps', format='eps')

    plt.show()
    pass


if __name__ == "__main__":
    main()

