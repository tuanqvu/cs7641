# Installation

Install Poetry and add it to `PATH`:

```
curl -sSL https://install.python-poetry.org | python -
echo 'export PATH="/home/$USER/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Test installation of poetry
poetry --version
```

Install project dependencies:
```
poetry install
```

# Prepare datasets

1) Download the [Adult dataset](https://archive.ics.uci.edu/dataset/2/adult) and the [Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
2) Extract both zip files into the `data/` folder
3) Navigate to `data/adult/adult.test` CSV file and remove the first line: `|1x3 Cross validator`
4) Install project dependencies as specified in [Installation](#installation)
5) Run `poetry run python prepare_dataset.py`

# Run the project
Run the command:
```
poetry run python main.py
```