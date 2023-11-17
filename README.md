# Dubme Assignment - Alejandro Delgado

This is the code repository for the [dubme assignment](Dubme_ML_engineer_-_problem_resolution_-_Alejandro_Delgado_Luezas.docx) with audio denoising routines from [video sources](data/videos/). The repository has three core scripts in the `src` folder:

- [`separation_program.py`](src/separation_program.py): This script would be the minimum one required for the assignment. The users input a video file in `.mp4` format and they get two `.wav` files, one containing the clean (denoised) signal and another containing the noise signal. This is the basic script for assignment completion. To use it from the repository's main directory:

```sh
python src/separation_program.py -vp [path_to_video]
```

- [`research_pipeline.py`](src/research_pipeline.py): This script contains all the research routines that have been implemented in order to select the best separation algorithm and extract insights from both the separation and the transcription routines based on objective metrics. This is an extended script for the assignment.

- [`utils.py`](src/utils.py): Contains utility functions for the two scripts described above.

All three scripts are commented on and structured to improve both readability and comprehensiveness.

# Research

## Methodology

The following has been included in this brief research study:

- 2 State-of-the-art models based on neural network algorithms: [`FAIR Denoiser`](https://github.com/facebookresearch/denoiser) and [`NVIDIA's CleanUNet`](https://github.com/NVIDIA/CleanUNet). I selected those based on their [performance](https://nv-adlr.github.io/projects/cleanunet/). It can be seen there that authors do not compare the model's performance to those from classical signal processing-based methods like Weiner filtering, hence I deduce that it's likely that deep learning-based methods significantly outperform classical signal processing-based ones.

- 1 State-of-the-art STT transcription model based on a neural network algorithm: [`OpenAI's Whisper`](https://github.com/openai/whisper). This open-source model is known for outputting almost perfect transcriptions even with heavy background noise.

- 2 Audio quality relative metrics via the [speechmetrics](https://github.com/aliutkus/speechmetrics) toolbox: `STOI` and `SI-SNR`. These are very well-known in the field of audio quality assessment and relatively simple to interpret.

- 2 Transcription quality relative metrics via the [jiwer](https://github.com/jitsi/jiwer) toolbox: `WER` and `CER`. Again, these are very well-known in the field of transcription quality assessment and relatively simple to interpret.

- 4 [Video sources](data/videos/) with different environmental noise and levels of speech intelligibility.

The [`research_pipeline.py`](src/research_pipeline.py) contains all associated research routines step by step.

## Results

Here, we present the results I gathered without analysing them (they are discussed in the [Discussion](#discussion) section below). All videos, original audios from videos, denoised files, noise files, trasncriptions, and ground truth transcriptions can be found in the [data](data/) directory.

### Noise Separation Timing

To time the two noise separation algorithms, I used a MacBook Pro 16" 2021 and performed separation 10 times, from which the average was taken.

Processing a total of 218 seconds in 4 videos, FAIR's Denoiser took 16.80 seconds while NVIDIA's CleanUNet took 16.75, which both equal to nearly 13 times faster than real-time.

### Noise Separation Audio Quality

The tables below show the values for the retrieved audio quality metrics for noise removal using FAIR's Denoiser and NVIDIA's CleanUNet. The SI-SDR is given in dB (the more, the merrier), and the STOI is normalised from 0 (worst) to 1 (best).

FAIR's Denoiser | inside_and_reverb | outside_and_wind | inside_and_low_volume | music_background
--- | --- | --- | --- |---
SI-SDR | -1.23 | 9.804 | 1.299 | 13.693 
STOI | 0.869 | 0.686 | 0.539 | 0.963

NVIDIA's CleanUNet | inside_and_reverb | outside_and_wind | inside_and_low_volume | music_background
--- | --- | --- | --- |---
SI-SDR | -1.24 | 9.674 | 0.619 | 13.044 
STOI | 0.874 | 0.662 | 0.498 | 0.96 

The content of these tables can be found in the [data](data/) directory in `.csv` format.

### Speech Transcription Quality

The tables below show the values for the retrieved transcription quality metrics using OpenAI's Whisper. Both for WER and CER, the lower the value the better. Ground truth transcriptions were hand-crafted and written in .txt files.

Original Audios | inside_and_reverb | outside_and_wind | inside_and_low_volume | music_background
--- | --- | --- | --- |---
WER | 0.0 | 0.009 | 0.408 | 0.027 
CER | 0.0 | 0.006 | 0.274 | 0.009

FAIR's Denoiser | inside_and_reverb | outside_and_wind | inside_and_low_volume | music_background
--- | --- | --- | --- |---
WER | 0.006 | 0.28 | 0.606 | 0.071 
CER | 0.001 | 0.149 | 0.416 | 0.018

NVIDIA's CleanUNet | inside_and_reverb | outside_and_wind | inside_and_low_volume | music_background
--- | --- | --- | --- |---
WER | 0.022 | 0.28 | 0.62 | 0.054 
CER | 0.008 | 0.136 | 0.373 | 0.018

The content of these tables can be found in the [data](data/) directory in `.csv` format.

## Discussion

It can be seen that, in terms of speed, both algorithms perform considerably faster than in real time and no significant difference between speed performances was found.

In terms of noise separation audio quality, FAIR's Denoiser generally performs slightly better than NVIDIA's CleanUNet for all videos and metrics. Then, when considering transcription quality, they are usually on pair, with no significant differences in performance.

It is also worth noting that the OpenAI's Whisper transcriber algorithm performs significantly better when applied to original audio than to denoised audio. This is due to the fact that the algorithm was trained with noisy videos and in an end-to-end fashion, so the denoising process already takes place there implicitly.

Hence, I decided to use FAIR's Denoiser in the main [`separation_program.py`](src/separation_program.py) script.

Finally, some limitations that this study has are:

- Low video sample size: Increasing the number of analysed videos taken with different environmental conditions would clarify the veracity and impact of the insights above.

- Low model sample size: Increasing the number of models would clarify the veracity and impact of the insights above. For instance, experiments with NVIDIA's CleanUNet were done with its `high` version of the model (fewer parameters), so it would be interesting to see how the `full` version of the model performs in relation to FAIR's Denoiser.

- Low number of metrics: Having more metrics would help to get clearer conclusions.

- Unverified [ground truth transcripts](data/transcriptions_gt/): Getting verified ground truth transcriptions and/or more people to verify one's ground truth transcriptions would make findings on speech transcription quality more robust.

Tackling these limitations among others would improve the quality of the research study and would pave the way to better STT algorithms. The challenge, then, would be in (i) identifying where end-to-end STT algorithms like OpenAI's Whisper fail (e.g. audios with too much noise), (ii) see whether noise removal makes a difference, and (iii) applying noise removal if it improves performance.


# Repository's Usage 

## Dependencies

I use `conda` to handle Python dependencies, the default `conda` environment creation is currently supported for 64-bit Linux operating system.  To create/update an environment:

```sh
conda env create -f environment/[op_system].yml
```

```sh
conda env update -f environment/[op_system].yml
```

All dependencies are either handled by `conda` or by `pip` via `conda`. The `pip` dependencies are all listed in the `pyproject.toml` file.

## Code Quality

### Code Style
I use `flake8` for linting and `black` for formatting.

```sh
make code-style
```

### Static Typing
I check static types using `mypy`.
```sh
make type-check
```
