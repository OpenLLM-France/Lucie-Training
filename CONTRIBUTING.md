# Contributing

The project dependencies are managed with [poetry](https://python-poetry.org/). To prepare your 
development environment, create a Python (>=3.9) virtual environment with your preferred tool 
(e.g. [conda](https://docs.conda.io/en/latest/),
 [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [pyenv](https://github.com/pyenv/pyenv)).
Then install the project by invoking the poetry install command:

```shell script
$ cd <project-directory>
$ poetry install
```

Poetry ensures that every developer in the project will have the same Python environment. It also 
resolves dependency conflicts automatically.

If you want to add a library, use the add function of poetry. If the library is needed only for the
development phase, add the `--group dev` flag. More information can be found in the poetry documentation.
```shell script
$ cd <project-directory>
$ poetry add <library> [--group dev]
```

Once you have added a dependency, commit the files `pyproject.toml`, `poetry.lock` to the 
repository.

## Code formatting

Code consistent formatting is ensured by using [Ruff](https://github.com/astral-sh/ruff). Please configure your workflow to ensure that every file that you commit to the repository has been processed with Ruff. The project Ruff configuration is contained in the `pyproject.toml` file at the root of the repository.

## Code quality

Code quality is ensured by using [Ruff](https://github.com/astral-sh/ruff). Please configure your workflow to ensure that every file that you commit to the repository has been processed with Ruff.
The project Ruff configuration is contained in the `pyproject.toml` file at the root of the repository.

## Pre-commit hooks

For those who tend to forget to pass their code through Ruff, we provide a 
[pre-commit](https://pre-commit.com/) configuration (`.pre-commit-config.yaml`). In short, pre-commit will
pass the staged code that you try to commit in Ruff. It will yell at you if you didn't do it
before this point. This workflow ensures that every file committed to the repository is correctly formatted and respect project quality levels. Feel free to use it.