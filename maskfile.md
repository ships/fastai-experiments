# Commands and tasks

This file is designed to be parsed and used by [mask](https://github.com/jacobdeichert/mask)

## setup

> sets up local workstation and/or shell.

~~~zsh
conda activate handwriting_drawing
~~~

## lab

> Runs the jupyter-lab locally and opens browser window. For local development.

~~~zsh
jupyter-lab
~~~

## test

> Run unit tests.

~~~zsh
python -m unittest
~~~

## lint

> Check and autofix linting issues.

~~~zsh
black .
~~~
