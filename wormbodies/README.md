# Wormbodies Experiments

## Data
You can get the data here:
https://data.broadinstitute.org/bbbc/BBBC010/
(\_images.zip and \_foreground\_eachworm.zip)


## Preprocessing
The data is provided in tif-files, however our code expects zarr-files

```
python consolidate_data.py -i ~/data/datasets/data_wormbodies/ -o ~/data/datasets/data_wormbodies --raw-gfp-min 0 --raw-gfp-max 4095 --raw-bf-min 0 --raw-bf-max 3072 --out-format zarr --parallel 50
```
(adapt the input and output paths)


## Setups

### setup10
This is the setup used for the results of our ppp model in the paper.


### setup08
This is the setup used for the results of our ppp+dec model in the paper.
