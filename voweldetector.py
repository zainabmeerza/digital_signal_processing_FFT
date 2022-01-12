import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def vowel_detector(audio_file, freq_constant, label):
    # VOWEL SIGNAL PROCESSING
    sample_rate, audio_data = wavfile.read(audio_file)
    avg_data = (audio_data[:, 0] + audio_data[:, 1]) / 2
    normalized_audio_data = avg_data / (np.max(np.abs(avg_data)))
    duration = len(normalized_audio_data) / sample_rate
    time = np.linspace(0, duration, len(normalized_audio_data))
    # PLOTTING THE AUDIO FILE IN THE TIME DOMAIN
    plt.figure()
    plt.plot(time, normalized_audio_data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title('Normalized Audio Signal, Time Domain: Vowel {}'.format(label))
    plt.savefig('vowel_{}_time.eps'.format(label), format='eps')

    # CALCULATING THE FFT & FREQUENCY OF THE SIGNAL
    fft_spectrum = np.fft.fft(normalized_audio_data)
    fft_spectrum = fft_spectrum / normalized_audio_data.shape[0]
    frequency_axis = np.linspace(0, sample_rate, len(avg_data))
    # FFT CONVERTED TO dB
    fft_spectrum_in_dB = 20 * np.log10(abs(fft_spectrum))

    # PLOTTING THE VOWELS TOGETHER IN THE FREQUENCY DOMAIN
    plt.figure()
    plt.semilogx(frequency_axis, fft_spectrum_in_dB)
    plt.title('Frequency Spectrum of Vowel {}'.format(label))
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.savefig('vowel_{}_frequency.eps'.format(label), format='eps')
    plt.show()
    # PERFORMING LINEAR INTERPOLATION TO EXTRACT THE AMPLITUDE OF THE SIGNALS AT A GIVEN FREQUENCY
    # THE FREQUENCY WAS DETECTED EARLIER BY INTERPRETING THE FREQUENCY SPECTRUM
    amplitude = np.interp(freq_constant, frequency_axis, fft_spectrum_in_dB)
    # With np.interp, we find the corresponding amplitude to the frequency we detected by inspection
    print('amplitude:', amplitude)
    vowel_detected = ''
    if amplitude > -35: # At amplitudes greater than -35, we can safely say we found the corresponding vowel
        vowel_detected = label + ' detected'
    # FUNCTION RETURNS A STRING
    return vowel_detected


def main():
    audio_file_a = r'vowel1.wav'
    audio_file_e = r'vowel2.wav'
    frequency_constant_a = 1220
    frequency_constant_e = 460
    # We find the frequency value where we can differentiate between different vowels
    print('Analysing vowel...')
    vowel = vowel_detector(audio_file_a, frequency_constant_a, 'a')
    print(vowel)
    print('Analysing next vowel...')
    vowel = vowel_detector(audio_file_e, frequency_constant_e, 'e')
    print(vowel)

    pass


if __name__ == "__main__":
    main()
