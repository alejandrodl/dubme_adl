# Dubme Assignment - Alejandro Delgado
This is the code repository for the [dubme assignment](Dubme_ML_engineer_-_problem_resolution_-_Alejandro_Delgado_Luezas.docx) consisting on



# Dependencies
## conda
We use `conda` to handle python dependencies, the default `conda` environment creation is currently supported for 64-bit Linux operating system.  To create/update an environment:

```sh
conda env create -f environment/[op_system].yml
```

```sh
conda env update -f environment/[op_system].yml
```

All dependencies are either handled by `conda` or by `pip` via `conda`. The `pip` dependencies are all listed in the `pyproject.toml` file.

# Code Quality
## Code Style
We use `flake8` for linting and `black` for formatting.

```sh
make code-style
```

## Static Typing
We check static types using `mypy`.
```sh
make type-check
```