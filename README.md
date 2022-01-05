# PatchPerPix for Instance Segmentation

## Example experiment scripts for PatchPerPix
To use this instance segmentation method, first get and install the core code for [PatchPerPix](https://github.com/Kainmueller-Lab/PatchPerPix).

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


### Data preperation

The code expects the data to be in the zarr format (similar to hdf5, but uses the underlying file system to enable parallel read and write)
The dataset specific subfolders (e.g. [wormbodies](wormbodies)) contain further information on how to get and preprocess the data.


## Usage
The master script `run_ppp.py` can be used to control all aspects of the experiments.

Example call:
```
python run_ppp.py --setup setup08 --config wormbodies/02_setups/setup08/config.toml --do mknet train validate_checkpoints predict decode label evaluate --app wormbodies --root ~/data/patchPerPix/experiments
```

With `--do TASK` you can set the sub task that should be executed (or `all` for the whole pipeline), `--root PATH` sets the output directory, `--app APP` the experiment (e.g. wormbodies) and `--setup SETUP` the specific setup of that experiment (e.g. setup01).

The command above creates a time stamped experiment folder under the path specified by `--root`.
To continue training or for further validation adapt the command. Change the `--config` parameter to point to the config file in the experiment folder and remove the `--root` flag and replace it with the `-id` flag and point it to the experiment folder. The tasks specified after `--do` depend on what you want to do:
```
python run_ppp.py --setup setup08 --config experiments/wormbodies_setup08_201125_120524/config.toml --do  validate_checkpoints predict decode label evaluate --app wormbodies -id experiments/wormbodies_setup08_201125_120524
```
