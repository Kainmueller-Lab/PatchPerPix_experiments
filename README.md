# PatchPerPix for Instance Segmentation

## Example experiment scripts for PatchPerPix
To use this instance segmentation method, first get and install the core code for [PatchPerPix](https://github.com/Kainmueller-Lab/PatchPerPix).

## Disclaimer

The readme will be extended in the near future.
In the meanwhile, if you have questions regarding the code, on how to run it or run into problems, please open an issue!


## Organization
- run_ppp.py:
  - master script to start the experiments
  - command line arguments are used to select the experiment and the sub-task to be executed (training, inference etc)
  - specific parameters for the network and segmentation are supplied in a config file
- config: contains an example configuration file
- wormbodies: an example experiment
  - 01_data: script to preprocess the raw input and convert it to the right format
  - 02_setups: here the different experiment setups are placed, the python script should not be called manually, but will be called by the master script
    - mknet.py: create the tensorflow graph based on the config file
    - train.py: trains the network
    - predict.py: inference after training
    - decode.py: if ppp+dec is used, decode the predicted foreground codes to the full patches
    - config.toml: the configuration

## Installation

### Core
First install the core code: [PatchPerPix](https://github.com/Kainmueller-Lab/PatchPerPix).
Then continue here.

### Experiments
Clone this repository and install the dependencies.
The recommended way is to install them into your conda/python virtual environment.

```
conda activate <<your-env-name>>
git clone https://github.com/Kainmueller-Lab/PatchPerPix_experiments.git
cd PatchPerPix_experiments
pip install -r requirements.txt
```

### Evaluation

For validation and evaluation this package is required:
[Evaluation](https://github.com/Kainmueller-Lab/evaluate-instance-segmentation)


## Usage
The master script `run_ppp.py` can be used to control all aspects of the experiments.

Example call:
```
python run_ppp.py --setup setup08 --config wormbodies/02_setups/setup08/config.toml --do mknet train validate_checkpoints predict decode label evaluate --app wormbodies --root ~/data/patchPerPix/experiments
```

With `--do TASK` you can set the sub task that should be executed (or `all` for the whole pipeline), `--root PATH` sets the output directory, `--app APP` the experiment (e.g. wormbodies) and `--setup SETUP` the specific setup of that experiment (e.g. setup01).

Within the respective experiment folder (e.g. wormbodies) you can find some information on where to get the data and how to convert into the correct format.
