import os
from utils import extract_audio_from_video, denoise_audio_file, get_noise_files, transcribe_audio_file, compute_metrics, save_metrics

# Set extensions
video_ext = '.mp4'
audio_ext = '.wav'

# Set paths
video_directory = "data/videos"
audio_directory = "data/audios"
denoised_audio_directory = "data/denoised"
noise_audio_directory = "data/noise"
transcriptions_directory = "data/transcriptions"

# Create audio directories if they don't exist
if not os.path.exists(audio_directory):
    os.makedirs(audio_directory)
if not os.path.exists(denoised_audio_directory):
    os.makedirs(denoised_audio_directory)
if not os.path.exists(noise_audio_directory):
    os.makedirs(noise_audio_directory)
if not os.path.exists(transcriptions_directory):
    os.makedirs(transcriptions_directory)

'''# Get video paths and extract audio from each video
video_paths = [os.path.join(video_directory, filename) for filename in os.listdir(video_directory) if filename.endswith(video_ext)]
for video_path in video_paths:
    extract_audio_from_video(video_path, audio_directory)

# Get audio paths and denoise each audio
audio_paths = [os.path.join(audio_directory, filename) for filename in os.listdir(audio_directory) if filename.endswith(audio_ext)]
for audio_path in audio_paths:
    denoise_audio_file(audio_path, denoised_audio_directory)

# Get denoised audio paths and subtract them from original audio to get noise
denoised_audio_paths = [os.path.join(denoised_audio_directory, filename) for filename in os.listdir(denoised_audio_directory) if filename.endswith(audio_ext)]
for denoised_audio_path in denoised_audio_paths:
    audio_path = audio_directory + '/' + denoised_audio_path.split('/')[-1]
    noise_path = noise_audio_directory + '/' + denoised_audio_path.split('/')[-1]
    get_noise_files(audio_path, denoised_audio_path, noise_path)

# Transcribe each audio file and denoised audio file and save it to transcriptions_directory
for audio_path in audio_paths:
    transcriptions_path = transcriptions_directory + '/' + audio_path.split('/')[-1].split('.')[0] + '.txt'
    transcribe_audio_file(audio_path, transcriptions_path)

for denoised_audio_path in denoised_audio_paths:
    transcriptions_path = transcriptions_directory + '/' + denoised_audio_path.split('/')[-1].split('.')[0] + '_denoised.txt'
    transcribe_audio_file(denoised_audio_path, transcriptions_path)'''

# (Optional) Compute metrics and save them to a csv file
denoised_audio_paths = [os.path.join(denoised_audio_directory, filename) for filename in os.listdir(denoised_audio_directory) if filename.endswith(audio_ext)]
metrics_dict = {}
for denoised_audio_path in denoised_audio_paths:
    original_audio_path = audio_directory + '/' + denoised_audio_path.split('/')[-1]
    metrics_dict = compute_metrics(metrics_dict, denoised_audio_path, original_audio_path)

metrics_path = 'data/metrics.csv'
if os.path.exists(metrics_path):
    os.remove(metrics_path)
else:
    save_metrics(metrics_dict, metrics_path)

