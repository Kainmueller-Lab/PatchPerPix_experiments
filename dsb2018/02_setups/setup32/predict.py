import sys
import os
import numpy as np
import gunpowder as gp
import json
import logging
from datetime import datetime
import zarr
import h5py


def predict(**kwargs):
    name = kwargs['name']

    raw = gp.ArrayKey('RAW')
    pred_affs = gp.ArrayKey('PRED_AFFS')

    with open(os.path.join(kwargs['input_folder'],
                           name + '_config.json'), 'r') as f:
        net_config = json.load(f)
    with open(os.path.join(kwargs['input_folder'],
                           name + '_names.json'), 'r') as f:
        net_names = json.load(f)

    voxel_size = gp.Coordinate(kwargs['voxel_size'])
    input_shape_world = gp.Coordinate(net_config['input_shape']) * voxel_size
    output_shape_world = gp.Coordinate(net_config['output_shape']) * voxel_size
    context = (input_shape_world - output_shape_world) // 2
    # context = tuple([c+2 for c in context])
    # chunksize = list(np.asarray(output_shape_world) // 2)

    raw_key = kwargs.get('raw_key', 'volumes/raw')
    aff_key = kwargs.get('aff_key', 'volumes/pred_affs')

    # add ArrayKeys to batch request
    request = gp.BatchRequest()
    request.add(raw, input_shape_world, voxel_size=voxel_size)
    request.add(pred_affs, output_shape_world, voxel_size=voxel_size)

    if kwargs['input_format'] != "hdf" and kwargs['input_format'] != "zarr":
        raise NotImplementedError("predict node for %s not implemented yet",
                                  kwargs['input_format'])
    if kwargs['input_format'] == "hdf":
        sourceNode = gp.Hdf5Source
        with h5py.File(os.path.join(kwargs['data_folder'],
                                    kwargs['sample'] + ".hdf"), 'r') as f:
            shape = f[raw_key].shape
            if shape[0] == 1:
                shape = shape[1:]
    elif kwargs['input_format'] == "zarr":
        sourceNode = gp.ZarrSource
        f = zarr.open(os.path.join(kwargs['data_folder'],
                                   kwargs['sample'] + ".zarr"), 'r')
        shape = f[raw_key].shape
        if shape[0] == 1:
            shape = shape[1:]
    source = sourceNode(
        os.path.join(kwargs['data_folder'],
                     kwargs['sample'] + "." + kwargs['input_format']),
        datasets={
            raw: raw_key
        },
        array_specs={
        raw: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
        }
    )

    if kwargs['output_format'] != "zarr":
        raise NotImplementedError("Please use zarr as prediction output")

    # open zarr file
    zf = zarr.open(os.path.join(kwargs['output_folder'],
                                kwargs['sample'] + '.zarr'), mode='w')
    zf.create(aff_key,
              shape=[int(np.prod(kwargs['patchshape']))] + list(shape),
              chunks=[int(np.prod(kwargs['patchshape']))] + list(shape),
              dtype=np.float16)
    zf[aff_key].attrs['offset'] = [0, 0]
    zf[aff_key].attrs['resolution'] = kwargs['voxel_size']

    crop = []
    print(shape, net_config['output_shape'])
    for d in range(-2, 0):
        if shape[d] < net_config['output_shape'][d]:
            crop.append((net_config['output_shape'][d]-shape[d])//2)
        else:
            crop.append(0)
    print("cropping", crop)
    print(context)
    context += gp.Coordinate(crop)
    print(context)
    outputs = {
        net_names['pred_affs']: pred_affs,
    }
    outVolumes = {
        pred_affs: aff_key,
    }
    pipeline = (
        source +
        gp.Pad(raw, context) +
        # gp.Normalize(raw) +
        gp.IntensityScaleShift(raw, 2, -1) +

        # perform one training iteration for each passing batch (here we use
        # the tensor names earlier stored in train_net.config)
        gp.tensorflow.Predict(
            graph=os.path.join(kwargs['input_folder'], name + '.meta'),
            checkpoint=kwargs['checkpoint'],
            inputs={
                net_names['raw']: raw
            },
            outputs=outputs) +

        # store all passing batches in the same HDF5 file
        gp.ZarrWrite(
            outVolumes,
            output_dir=kwargs['output_folder'],
            output_filename=kwargs['sample'] + ".zarr",
            compression_type='gzip'
        ) +

        # show a summary of time spend in each node every 10 iterations
        gp.PrintProfilingStats(every=100) +

        # iterate over the whole dataset in a scanning fashion, emitting
        # requests that match the size of the network
        gp.Scan(reference=request)
    )

    with gp.build(pipeline):
        # request an empty batch from Scan to trigger scanning of the dataset
        # without keeping the complete dataset in memory
        pipeline.request_batch(gp.BatchRequest())
