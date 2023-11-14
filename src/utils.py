import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from moviepy.video.io.VideoFileClip import VideoFileClip


def extract_audio_from_video(video_path, audio_path):
    """
    Extract audio from a video file using moviepy.

    Args:
        video_path (str): Path to the video file.
        audio_path (str): Path to save the extracted audio file.
    """
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)


def load_audio_file(file_path):
    """
    Load an audio file from disk.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        tuple: A tuple containing the audio data and the sample rate.
    """
    audio_data, sample_rate = sf.read(file_path)
    return audio_data, sample_rate

def plot_spectrogram(audio_data, sample_rate):
    """
    Plot the spectrogram of an audio signal.

    Args:
        audio_data (ndarray): Audio signal.
        sample_rate (int): Sample rate of the audio signal.
    """
    frequencies, times, spectrogram = signal.spectrogram(audio_data, sample_rate)
    plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


