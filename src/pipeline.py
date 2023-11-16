import os
import time
from utils import (
    clear_make_directory,
    extract_audio_from_video,
    downsample_and_mono_from_path,
    denoise_audio_file,
    get_noise_files,
    transcribe_audio_file,
    compute_audio_quality_metrics,
    compute_transcription_quality_metrics,
    save_metrics,
)


# Separation models
model = "FAIR"  # 'FAIR' or 'CleanUNet'
print("Using {} model".format(model))

# Set extensions
video_ext = ".mp4"
audio_ext = ".wav"
text_ext = ".txt"

# Set paths
video_directory = "data/videos"
audio_directory = "data/audios"
denoised_audio_directory = "data/denoised"
noise_audio_directory = "data/noise"
transcriptions_directory = "data/transcriptions"

# Create audio directories if they don't exist
clear_make_directory(audio_directory, audio_ext)
clear_make_directory(denoised_audio_directory, audio_ext)
clear_make_directory(noise_audio_directory, audio_ext)
clear_make_directory(transcriptions_directory, text_ext)

# STEP 1 - Get video paths and extract audio from each video
video_paths = [
    os.path.join(video_directory, filename)
    for filename in os.listdir(video_directory)
    if filename.endswith(video_ext)
]
for video_path in video_paths:
    extract_audio_from_video(video_path, audio_directory)

# STEP 2 - Convert all audio files to 16kHz mono
audio_paths = [
    os.path.join(audio_directory, filename)
    for filename in os.listdir(audio_directory)
    if filename.endswith(audio_ext)
]
for audio_path in audio_paths:
    downsample_and_mono_from_path(audio_path)

# STEP 3 - Get audio paths and denoise each audio
start_time = time.time()
for audio_path in audio_paths:
    denoise_audio_file(audio_path, denoised_audio_directory, model)
print("Denoising took {} seconds".format(time.time() - start_time))

# STEP 4 - Get denoised audio paths and subtract them from original audio to get noise
denoised_audio_paths = [
    os.path.join(denoised_audio_directory, filename)
    for filename in os.listdir(denoised_audio_directory)
    if filename.endswith(audio_ext)
]
for audio_path in audio_paths:
    denoised_audio_path = denoised_audio_directory + "/" + audio_path.split("/")[-1]
    noise_path = noise_audio_directory + "/" + denoised_audio_path.split("/")[-1]
    get_noise_files(audio_path, denoised_audio_path, noise_path)

# STEP 5 - Transcribe audio files and denoised audio files and save it to transcriptions_directory
for audio_path in audio_paths:
    transcriptions_path = (
        transcriptions_directory
        + "/"
        + audio_path.split("/")[-1].split(".")[0]
        + ".txt"
    )
    transcribe_audio_file(audio_path, transcriptions_path)

for denoised_audio_path in denoised_audio_paths:
    transcriptions_path = (
        transcriptions_directory
        + "/"
        + denoised_audio_path.split("/")[-1].split(".")[0]
        + "_denoised.txt"
    )
    transcribe_audio_file(denoised_audio_path, transcriptions_path)

# STEP 6 (Optional) - Compute audio quality metrics and save them to a csv file
denoised_audio_paths = [
    os.path.join(denoised_audio_directory, filename)
    for filename in os.listdir(denoised_audio_directory)
    if filename.endswith(audio_ext)
]
audio_metrics_dict: dict = {}
for denoised_audio_path in denoised_audio_paths:
    original_audio_path = audio_directory + "/" + denoised_audio_path.split("/")[-1]
    audio_metrics_dict = compute_audio_quality_metrics(
        audio_metrics_dict,
        denoised_audio_path,
        original_audio_path,
        abs_metrics=[],
        rel_metrics=["stoi", "sisdr"],
    )

metrics_path = "data/audio_metrics_" + model + ".csv"
if os.path.exists(metrics_path):
    os.remove(metrics_path)
    save_metrics(audio_metrics_dict, metrics_path)
else:
    save_metrics(audio_metrics_dict, metrics_path)

# STEP 7 (Optional) - Compute transcription quality metrics and save them to a csv file
transcription_metrics_dict: dict = {}
for denoised_audio_path in denoised_audio_paths:
    transcription_gt_path = (
        transcriptions_directory
        + "_gt/"
        + denoised_audio_path.split("/")[-1].split(".")[0]
        + "_gt.txt"
    )
    transcription_path = (
        transcriptions_directory
        + "/"
        + denoised_audio_path.split("/")[-1].split(".")[0]
        + ".txt"
    )
    transcription_denoised_path = transcription_path[:-4] + "_denoised.txt"

    transcription_metrics_dict = compute_transcription_quality_metrics(
        transcription_metrics_dict,
        transcription_gt_path,
        transcription_path,
    )
    transcription_metrics_dict = compute_transcription_quality_metrics(
        transcription_metrics_dict,
        transcription_gt_path,
        transcription_denoised_path,
    )

metrics_path = "data/transcription_metrics_" + model + ".csv"
if os.path.exists(metrics_path):
    os.remove(metrics_path)
    save_metrics(transcription_metrics_dict, metrics_path)
else:
    save_metrics(transcription_metrics_dict, metrics_path)
