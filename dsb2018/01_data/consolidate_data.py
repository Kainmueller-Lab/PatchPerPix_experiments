import argparse
import glob
import os

import h5py
from joblib import Parallel, delayed
import numpy as np
import scipy.ndimage
import scipy.stats
import skimage.io as io
import tifffile
import zarr

# python consolidate_data.py -i ~/data/datasets/dsb2018Full/original/train -o ~/data/datasets/dsb2018Full/consolidated/original/train --raw-min 0 --raw-max 255 --out-format zarr --parallel 30 -t fixed --overlap_limit 5

def load_array(filename):
    if filename.endswith(".tif") or \
       filename.endswith(".tiff") or \
       filename.endswith(".TIF") or \
       filename.endswith(".TIFF"):
        image = tifffile.imread(filename)
    elif filename.endswith(".png"):
        image = io.imread(filename, plugin="simpleitk")
    else:
        print("invalid file type")
        raise ValueError("invalid input file type", filename)

    print("{} shape {}".format(filename, image.shape))
    if len(image.shape) > 2 and image.shape[-1] > 1:
        image = rgb2gray(image,
                         np.iinfo(image.dtype).min, np.iinfo(image.dtype).max)
        print("{} shape {}".format(filename, image.shape))
    return image


def rgb2gray(rgb, mi, mx):
    return np.clip(np.dot(rgb[..., :3], [1.0/3.0, 1.0/3.0, 1.0/3.0]), mi, mx)


# https://github.com/CSBDeep/CSBDeep/blob/master/csbdeep/utils/utils.py
def normalize_percentile(x, pmin=3, pmax=99.8, axis=None, clip=False,
                         eps=1e-20, dtype=np.float32):
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    print("min/max", mi, ma, np.min(x), np.max(x))
    return normalize_min_max(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_min_max(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if mi is None:
        mi = np.min(x)
    if ma is None:
        ma = np.max(x)
    if dtype is not None:
        x   = x.astype(dtype, copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x, 0, 1)

    return x


def normalize(args, raw, sample):
    print("{} before norm {}: min {}, max {}, mean {}, std {}, median {}".format(
        sample, args.normalize, np.min(raw), np.max(raw), np.mean(raw),
        np.std(raw), np.median(raw)))

    if args.normalize == "minmax":
        raw = normalize_min_max(raw, args.raw_min, args.raw_max)
    elif args.normalize == "percentile":
        raw = normalize_percentile(raw, args.raw_min, args.raw_max)

    print("{} after norm {}:  min {}, max {}, mean {}, std {}, median {}".format(
        sample, args.normalize, np.min(raw), np.max(raw), np.mean(raw),
        np.std(raw), np.median(raw)))
    return raw


def preprocess(args, raw, sample):
    print("{} before preproc {}: skew {}".format(
        sample, args.preprocess, scipy.stats.skew(raw.ravel())))
    if args.preprocess is None or args.preprocess == "no":
        pass
    elif args.preprocess == "square":
        raw = np.square(raw)
    elif args.preprocess == "cuberoot":
        raw = np.cbrt(raw)
    print("{} after preproc {}:  skew {}".format(
        sample, args.preprocess, scipy.stats.skew(raw.ravel())))
    return raw


def pad(args, array, mode):
    if args.padding != 0:
        array = np.pad(array,
                       ((args.padding, args.padding),
                        (args.padding, args.padding),
                        (args.padding, args.padding)),
                       mode)
    return array


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in-dir', dest='in_dir', required=True,
                        help='location of input files')
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True,
                        help='where to place output files')
    parser.add_argument('--out-format', dest='out_format', default="hdf",
                        help='format of output files')
    parser.add_argument('-p', '--parallel', default=1, type=int)
    parser.add_argument('--raw-min', dest='raw_min', type=int)
    parser.add_argument('--raw-max', dest='raw_max', type=int)
    parser.add_argument('--scale-sdt', dest='scale_sdt', type=float, default=-9)
    parser.add_argument('--sigma', type=float, default=2)
    parser.add_argument('-t', '--type', required=True,
                        choices=['stardist', 'original', 'fixed'])
    parser.add_argument('--normalize', default='minmax',
                        choices=['minmax', 'percentile', 'meanstd'])
    parser.add_argument('--preprocess', default='no',
                        choices=['no', 'square', 'cuberoot'])
    parser.add_argument('--padding', default=0, type=int)
    parser.add_argument("--overlap_limit", type=int, default=0,
                        help=('limit to how many pixels of two instances '
                              'can overlap (last one wins, negative for '
                              'unlimited)'))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    return args


def main():
    args = get_arguments()

    if args.type == 'stardist':
        files = map(lambda fn: fn.split("/")[-1].split(".")[0],
                    glob.glob(os.path.join(args.in_dir, 'images/*.tif')))
    else:
        files = map(lambda fn: fn.split("/")[-1],
                    glob.glob(os.path.join(args.in_dir, '*')))

    if args.parallel > 1:
        Parallel(n_jobs=args.parallel, verbose=1) \
            (delayed(work)(args, f) for f in files)
    else:
        for f in files:
            work(args, f)


def work(args, sample):
    print("Processing {}, {}".format(args.in_dir, sample))
    out_fn = os.path.join(args.out_dir, sample)

    if args.type == 'stardist':
        raw_fn = os.path.join(args.in_dir, "images", sample + ".tif")
    else:
        raw_fn = os.path.join(args.in_dir, sample, "images", sample + ".png")
    raw = load_array(raw_fn).astype(np.float32)
    raw = preprocess(args, raw, sample)
    raw = normalize(args, raw, sample)
    raw = pad(args, raw, 'constant')

    if args.type == 'stardist':
        labels_fn = os.path.join(args.in_dir, "masks", sample + ".tif")
        gt_labels = load_array(labels_fn).astype(np.uint16)
    else:
        files = sorted(
            glob.glob(os.path.join(args.in_dir, sample, "masks", "*.png")))
        gt_labels = np.zeros_like(raw, dtype=np.uint16)
        for idx, f in enumerate(files, start=1):
            gt_label = load_array(f).astype(np.uint16)
            if np.any(gt_labels[gt_label != 0] > 0):
                overlap = np.count_nonzero(gt_labels[gt_label != 0] > 0)
                print("warning! multiple instances on same pixel ({},{},{})".
                      format(sample, f, overlap))
                if args.overlap_limit >= 0 and overlap > args.overlap_limit:
                    raise RuntimeError("error! overlap of {} pixels ({},{})".
                                       format(overlap, sample, f))
            gt_labels[gt_label != 0] = idx
    gt_labels = pad(args, gt_labels, 'constant')

    gt_threeclass = np.zeros(gt_labels.shape, dtype=np.uint8)
    struct = scipy.ndimage.generate_binary_structure(2, 2)
    for label in np.unique(gt_labels):
        if label == 0:
            continue

        label_mask = gt_labels==label
        eroded_label_mask = scipy.ndimage.binary_erosion(label_mask,
                                                         iterations=1,
                                                         structure=struct,
                                                         border_value=1)
        boundary = np.logical_xor(label_mask, eroded_label_mask)
        gt_threeclass[boundary] = 2
        gt_threeclass[eroded_label_mask] = 1

    gt_fgbg = np.zeros(gt_labels.shape, dtype=np.uint8)
    gt_fgbg[gt_labels > 0] = 1

    # if raw.shape[0] != 1:
    #     raw = np.expand_dims(raw, 0)

    labels = sorted(list(np.unique(gt_labels)))[1:]
    gt_cells = np.array(scipy.ndimage.measurements.center_of_mass(
        gt_labels > 0,
        gt_labels, labels))
    with open(out_fn + ".csv", 'w') as f:
        for cell, label in zip(gt_cells, labels):
            y = cell[0]
            x = cell[1]
            f.write("{}, {}, {}\n".format(y, x, label))


    if gt_labels.shape[0] != 1:
        gt_labels = np.expand_dims(gt_labels, 0)
        gt_fgbg = np.expand_dims(gt_fgbg, 0)
        gt_threeclass = np.expand_dims(gt_threeclass, 0)

    if args.out_format == "hdf":
        f = h5py.File(out_fn + '.hdf', 'w')
    elif args.out_format == "zarr":
        f = zarr.open(out_fn + '.zarr', 'w')

    f.create_dataset(
        'volumes/raw',
        data=raw,
        chunks=(256, 256),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_labels',
        data=gt_labels,
        chunks=(1, 256, 256),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_fgbg',
        data=gt_fgbg,
        chunks=(1, 256, 256),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_threeclass',
        data = gt_threeclass,
        chunks=(1, 256, 256),
        compression='gzip')


    for dataset in ['volumes/raw',
                    'volumes/gt_labels',
                    'volumes/gt_threeclass',
                    'volumes/gt_fgbg']:
        f[dataset].attrs['offset'] = (0, 0)
        f[dataset].attrs['resolution'] = (1, 1)

    if args.out_format == "hdf":
        f.close()

if __name__ == "__main__":
    main()
