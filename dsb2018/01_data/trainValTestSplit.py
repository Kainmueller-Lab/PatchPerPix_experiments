import argparse
import glob
import os
import random
import sys
import shutil
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-itr', dest="in_train_dir", required=True)
    parser.add_argument('-ite', dest="in_test_dir", required=True)
    parser.add_argument('-o', dest="out_dir", required=True)
    parser.add_argument('-f', dest="out_format", default="hdf")
    args = parser.parse_args()

    trainD = os.path.join(args.out_dir, "train")
    valD = os.path.join(args.out_dir, "val")
    testD = os.path.join(args.out_dir, "test")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(trainD, exist_ok=True)
    os.makedirs(testD, exist_ok=True)
    os.makedirs(valD, exist_ok=True)

    fmt = "." + args.out_format
    trainFls = sorted(list(
        map(os.path.basename,
            glob.glob(os.path.join(args.in_train_dir, "*" + fmt)))))

    rng = np.random.RandomState(42)
    ind = rng.permutation(len(trainFls))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    valFls = [trainFls[i] for i in ind_val]
    trainFls = [trainFls[i] for i in ind_train]

    # valFls = trainFls[:int(0.15*len(trainFls))]
    # trainFls = trainFls[int(0.15*len(trainFls)):]
    testFls = sorted(list(
        map(os.path.basename,
            glob.glob(os.path.join(args.in_test_dir, "*" + fmt)))))

    if args.out_format == "hdf":
        copy_func = shutil.copy2
    elif args.out_format == "zarr":
        copy_func = shutil.copytree

    for fl in trainFls:
        copy_func(os.path.join(args.in_train_dir, fl),
                  os.path.join(trainD, fl))
        shutil.copy2(os.path.join(args.in_train_dir, os.path.splitext(fl)[0]+".csv"),
                     trainD)
    for fl in valFls:
        copy_func(os.path.join(args.in_train_dir, fl),
                  os.path.join(valD, fl))
        shutil.copy2(os.path.join(args.in_train_dir, os.path.splitext(fl)[0]+".csv"),
                     valD)
    for fl in testFls:
        copy_func(os.path.join(args.in_test_dir, fl),
                  os.path.join(testD, fl))
        shutil.copy2(os.path.join(args.in_test_dir, os.path.splitext(fl)[0]+".csv"),
                     testD)

if __name__ == "__main__":
    main()
