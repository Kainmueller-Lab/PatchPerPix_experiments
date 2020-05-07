import argparse
import glob
import logging
import os

import h5py
from joblib import Parallel, delayed
from natsort import natsorted
import numpy as np
import scipy.ndimage
import scipy.stats
import skimage.io as io
import tifffile
import zarr

logger = logging.getLogger(__name__)

def load_array(filename):
    if filename.endswith(".tif") or \
       filename.endswith(".tiff") or \
       filename.endswith(".TIF") or \
       filename.endswith(".TIFF"):
        image = tifffile.imread(filename)
    elif filename.endswith(".png"):
        image = io.imread(filename, plugin="simpleitk")
    else:
        raise ValueError("invalid input file type", filename)

    logger.debug("%s shape %s", filename, image.shape)
    if len(image.shape) > 2 and image.shape[-1] > 1:
        image = rgb2gray(image,
                         np.iinfo(image.dtype).min, np.iinfo(image.dtype).max)
        logger.info("rgb2gray %s shape %s", filename, image.shape)
    return image


def rgb2gray(rgb, mi, mx):
    return np.clip(np.dot(rgb[..., :3], [1.0/3.0, 1.0/3.0, 1.0/3.0]), mi, mx)


# https://github.com/CSBDeep/CSBDeep/blob/master/csbdeep/utils/utils.py
def normalize_percentile(x, pmin=3, pmax=99.8, axis=None, clip=False,
                         eps=1e-20, dtype=np.float32):
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    logger.debug("min/max %s %s %s %s", mi, ma, np.min(x), np.max(x))
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


def normalize(args, raw, sample, mn, mx):
    logger.debug(
        "%s before norm %s: min %s, max %s, mean %s, std %s, median %s",
        sample, args.normalize, np.min(raw), np.max(raw), np.mean(raw),
        np.std(raw), np.median(raw))

    if args.normalize == "minmax":
        raw = normalize_min_max(raw, mn, mx)
    elif args.normalize == "percentile":
        raw = normalize_percentile(raw, mn, mx)

    logger.debug(
        "%s after norm %s:  min %s, max %s, mean %s, std %s, median %s",
        sample, args.normalize, np.min(raw), np.max(raw), np.mean(raw),
        np.std(raw), np.median(raw))
    return raw


def preprocess(args, raw, sample):
    logger.debug("%s before preproc %s: skew %s",
        sample, args.preprocess, scipy.stats.skew(raw.ravel()))
    if args.preprocess is None or args.preprocess == "no":
        pass
    elif args.preprocess == "square":
        raw = np.square(raw)
    elif args.preprocess == "cuberoot":
        raw = np.cbrt(raw)
    logger.debug("%s after preproc %s:  skew %s",
        sample, args.preprocess, scipy.stats.skew(raw.ravel()))
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
    parser.add_argument('--raw-gfp-min', dest='raw_gfp_min', type=int)
    parser.add_argument('--raw-gfp-max', dest='raw_gfp_max', type=int)
    parser.add_argument('--raw-bf-min', dest='raw_bf_min', type=int)
    parser.add_argument('--raw-bf-max', dest='raw_bf_max', type=int)

    parser.add_argument('--normalize', default='minmax',
                        choices=['minmax', 'percentile', 'meanstd'])
    parser.add_argument('--preprocess', default='no',
                        choices=['no', 'square', 'cuberoot'])
    parser.add_argument('--padding', default=0, type=int)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    return args


def main():
    logging.basicConfig(level='INFO')

    args = get_arguments()

    files = map(
        lambda fn: fn.split("/")[-1].split(".")[0].split("_")[6],
        glob.glob(os.path.join(args.in_dir, 'BBBC010_v2_images/*.tif')))
    files = sorted(list(set(files)))
    print(files)

    if args.parallel > 1:
        Parallel(n_jobs=args.parallel, verbose=1, backend='multiprocessing') \
            (delayed(work)(args, f) for f in files)
    else:
        for f in files:
            work(args, f)


def work(args, sample):
    logger.info("Processing %s, %s", args.in_dir, sample)
    out_fn = os.path.join(args.out_dir, sample)

    raw_fns = natsorted(glob.glob(
        os.path.join(args.in_dir,
                     "BBBC010_v2_images", "*_" + sample + "_*.tif")))
    # print(raw_fns)
    raw_gfp = load_array(raw_fns[0]).astype(np.float32)
    # print(raw_fns[0], np.min(raw_gfp), np.max(raw_gfp))
    raw_gfp = preprocess(args, raw_gfp, sample)
    raw_gfp = normalize(args, raw_gfp, sample,
                        args.raw_gfp_min, args.raw_gfp_max)
    raw_gfp = pad(args, raw_gfp, 'constant')
    raw_bf = load_array(raw_fns[1]).astype(np.float32)
    # print(raw_fns[1], np.min(raw_bf), np.max(raw_bf))
    raw_bf = preprocess(args, raw_bf, sample)
    raw_bf = normalize(args, raw_bf, sample,
                       args.raw_bf_min, args.raw_bf_max)
    raw_bf = pad(args, raw_bf, 'constant')

    files = natsorted(
        glob.glob(os.path.join(args.in_dir, "BBBC010_v1_foreground_eachworm",
                               sample + "*" + "_ground_truth.png")))
    # print(files)
    logger.info("number files: %s", len(files))
    gt_labels = np.zeros((len(files),) + raw_gfp.shape, dtype=np.uint16)
    gt_instances = np.zeros(raw_gfp.shape, dtype=np.uint16)
    for idx, f in enumerate(files):
        gt_label = load_array(f).astype(np.uint16)
        gt_labels[idx, ...] = 1*(gt_label!=0)
        gt_instances[gt_label != 0] = idx+1
    gt_numinst = np.sum(gt_labels, axis=0)
    tmp = np.sum(gt_labels, axis=0, keepdims=True)
    tmp[tmp==0] = 1
    gt_labels_norm = (gt_labels.astype(np.float32) /
                      tmp.astype(np.float32)).astype(np.float32)

    gt_fgbg = np.zeros(gt_labels.shape[1:], dtype=np.uint8)
    gt_fgbg[np.sum(gt_labels, axis=0) > 0] = 1

    # is slightly displaced +1,+0
    # gt_fgbg2 = load_array(
    #     os.path.join(args.in_dir, "BBBC010_v1_foreground",
    #                  sample + "_binary.png"))
    # gt_fgbg2[0:-1,1:] = gt_fgbg2[1:,1:]
    # print(np.count_nonzero(gt_fgbg2 != gt_fgbg))
    # gt_fgbg2 = pad(args, gt_fgbg2, 'constant')

    gt_labels = pad(args, gt_labels, 'constant')
    gt_instances = pad(args, gt_instances, 'constant')
    gt_labels_norm = pad(args, gt_labels_norm, 'constant')
    gt_numinst = pad(args, gt_numinst, 'constant')
    gt_fgbg = pad(args, gt_fgbg, 'constant')

    if raw_gfp.shape[0] != 1:
        raw_gfp = np.expand_dims(raw_gfp, 0)

    if raw_bf.shape[0] != 1:
        raw_bf = np.expand_dims(raw_bf, 0)
    raw = np.concatenate((raw_gfp, raw_bf), axis=0)

    if gt_instances.shape[0] != 1:
        gt_instances = np.expand_dims(gt_instances, 0)
        gt_numinst = np.expand_dims(gt_numinst, 0)
        gt_fgbg = np.expand_dims(gt_fgbg, 0)

    gt_labels_padded = np.pad(gt_labels, (0, 20-len(files)), 'constant')
    gt_labels_norm_padded = np.pad(gt_labels_norm, (0, 20-len(files)),
                                   'constant')

    if args.out_format == "hdf":
        f = h5py.File(out_fn + '.hdf', 'w')
    elif args.out_format == "zarr":
        f = zarr.open(out_fn + '.zarr', 'w')

    f.create_dataset(
        'volumes/raw',
        data=raw,
        chunks=(2, 256, 256),
        compression='gzip')
    f.create_dataset(
        'volumes/raw_gfp',
        data=raw_gfp,
        chunks=(1, 256, 256),
        compression='gzip')
    f.create_dataset(
        'volumes/raw_bf',
        data=raw_bf,
        chunks=(1, 256, 256),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_labels',
        data=gt_labels,
        chunks=(len(files), 256, 256),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_labels_norm',
        data=gt_labels_norm,
        chunks=(len(files), 256, 256),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_labels_padded',
        data=gt_labels_padded,
        chunks=(20, 256, 256),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_labels_norm_padded',
        data=gt_labels_norm_padded,
        chunks=(20, 256, 256),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_instances',
        data=gt_instances,
        chunks=(1, 256, 256),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_numinst',
        data=gt_numinst,
        chunks=(1, 256, 256),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_fgbg',
        data=gt_fgbg,
        chunks=(1, 256, 256),
        compression='gzip')

    for dataset in ['volumes/raw',
                    'volumes/raw_gfp',
                    'volumes/raw_bf',
                    'volumes/gt_labels',
                    'volumes/gt_labels_norm',
                    'volumes/gt_instances',
                    'volumes/gt_numinst',
                    'volumes/gt_fgbg']:
        f[dataset].attrs['offset'] = (0, 0)
        f[dataset].attrs['resolution'] = (1, 1)

    if args.out_format == "hdf":
        f.close()

if __name__ == "__main__":
    main()
