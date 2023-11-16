# Dubme Assignment - Alejandro Delgado
This is the code repository for the [dubme assignment](Dubme_ML_engineer_-_problem_resolution_-_Alejandro_Delgado_Luezas.docx) with audio denoising routines from [video sources](data/videos/). The repository has three core scripts in the `src` folder:

- [`separation_program.py`](src/separation_program.py): This script would be the minimum one required for the assignment. The users input a video file in `.mp4` format and they get two `.wav` files, one containing the clean (denoised) signal and another containing the noise signal. This is the basic script for assignment completion. To use it from the repository's main directory:

```sh
python src/separation_program.py -vp [path_to_video]
```

- [`research_pipeline.py`](src/research_pipeline.py): This script contains all the research routines that have been implemented in order to select the best separation algorithm and extract insights from both the sepatation and the transcription routines based on objective metrics. This is an extended script for the assignment.

- [`utils.py`](src/utils.py): Contains utility functions for the two scripts described above.


# Research

The following has been implemented:

1. 2 State-of-the-art models based on neural network algorithms: [FAIR Denoiser](https://github.com/facebookresearch/denoiser) and [NVIDIA's CleanUNet](https://github.com/NVIDIA/CleanUNet). I selected those based on their [performance](https://nv-adlr.github.io/projects/cleanunet/). It can be seen there that authors do not compare the model's performance to those from signal processing-based methods like Weiner filtering, hence I deduce that it's likely that deep learning-based methods significantly outperform signal processing-based ones.

1. 1 State-of-the-art STT transcription model based on a neural network algorithm: [OpenAI's Whisper](https://github.com/openai/whisper). This open-source model is known for outputing almost perfect transcriptions even with heavy background noise.

1. 2 Audio quality relative metrics via the [speechmetrics](https://github.com/aliutkus/speechmetrics) toolbox: STOI and SI-SNR. These are very well-known in the field of audio quality assessment and relatively simple to interpret.

1. 2 Transcription quality relative metrics via the [jiwer](https://github.com/jitsi/jiwer) toolbox: WER and CER. Again, these are very well-known in the field of transcription quality assessment and relatively simple to interpret.

# Dependencies
## conda
I use `conda` to handle python dependencies, the default `conda` environment creation is currently supported for 64-bit Linux operating system.  To create/update an environment:

```sh
conda env create -f environment/[op_system].yml
```

```sh
conda env update -f environment/[op_system].yml
```

All dependencies are either handled by `conda` or by `pip` via `conda`. The `pip` dependencies are all listed in the `pyproject.toml` file.

# Code Quality
## Code Style
I use `flake8` for linting and `black` for formatting.

```sh
make code-style
```

## Static Typing
I check static types using `mypy`.
```sh
make type-check
```