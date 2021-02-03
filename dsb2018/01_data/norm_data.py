import numpy as np
import zarr
import os
import sys
import glob

fns = glob.glob(os.path.join(sys.argv[1], "*/*.zarr"))

acc = []
for fn in fns:
    fl = zarr.open(fn, 'a')
    raw = fl['volumes/raw']
    shape = raw.shape
    pmi = np.percentile(raw, 3, axis=None, keepdims=True)
    pma = np.percentile(raw, 99.99, axis=None, keepdims=True)
    mi = np.min(raw)
    ma = np.max(raw)
    # print("{:.3f}, {:.3f}, {:.3f}, {:.3f}".format(float(pmi), float(pma), mi, ma))
    print("{} {:03d}, {:03d}, {:03d}, {:03d}".format(fn, int(pmi), int(pma), int(mi), int(ma)))


    raw = np.clip(raw, pmi, pma)
    pmi = np.min(raw)
    pma = np.max(raw)
    raw = (raw - pmi) / (pma - pmi)

    mi = np.min(raw)
    ma = np.max(raw)
    print("{} {}, {}, {}, {}".format(fn, pmi, pma, mi, ma))

    print(raw.dtype, np.min(raw), np.max(raw))
    raw_key = sys.argv[2]
    fl.create_dataset(raw_key,
                      data=raw.astype(np.float32),
                      shape=shape,
                      chunks=shape,
                      dtype=np.float32)
    fl[raw_key].attrs['offset'] = [0, 0]
    fl[raw_key].attrs['resolution'] = [1, 1]


for fn in fns:
    fl = zarr.open(fn, 'a')
    raw = fl[sys.argv[2]]
    shape = raw.shape
    pmi = np.percentile(raw, 3, axis=None, keepdims=True)
    pma = np.percentile(raw, 99.99, axis=None, keepdims=True)
    mi = np.min(raw)
    ma = np.max(raw)
    print("{} {}, {}, {}, {}".format(fn, pmi, pma, mi, ma))


# raw = np.array(acc)
# print(raw.shape)
# pmi = np.percentile(raw, 3, axis=None, keepdims=True)
# pma = np.percentile(raw, 99.8, axis=None, keepdims=True)
# mi = np.min(raw)
# ma = np.max(raw)
# # print("{:.3f}, {:.3f}, {:.3f}, {:.3f}".format(float(pmi), float(pma), mi, ma))
# print("{:03d}, {:03d}, {:03d}, {:03d}".format(int(pmi), int(pma), int(mi), int(ma)))
