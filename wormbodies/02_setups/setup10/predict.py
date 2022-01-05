import sys
import os
import numpy as np
import gunpowder as gp
import json
import logging
from datetime import datetime
import zarr


def predict(**kwargs):
    name = kwargs['name']

    raw = gp.ArrayKey('RAW')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    pred_numinst = gp.ArrayKey('PRED_NUMINST')

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
    chunksize = list(np.asarray(output_shape_world) // 2)
    chunksize = [int(c) for c in chunksize]

    raw_key = kwargs.get('raw_key', 'volumes/raw')

    # add ArrayKeys to batch request
    request = gp.BatchRequest()
    request.add(raw, input_shape_world, voxel_size=voxel_size)
    request.add(pred_affs, output_shape_world, voxel_size=voxel_size)
    if kwargs['overlapping_inst']:
        request.add(pred_numinst, output_shape_world, voxel_size=voxel_size)

    if kwargs['input_format'] != "hdf" and kwargs['input_format'] != "zarr":
        raise NotImplementedError("predict node for %s not implemented yet",
                                  kwargs['input_format'])
    if kwargs['input_format'] == "hdf":
        sourceNode = gp.Hdf5Source
        with h5py.File(os.path.join(kwargs['data_folder'],
                                    kwargs['sample'] + ".hdf"), 'r') as f:
            shape = f[raw_key].shape[1:]
            shape = [int(s) for s in shape]
    elif kwargs['input_format'] == "zarr":
        sourceNode = gp.ZarrSource
        f = zarr.open(os.path.join(kwargs['data_folder'],
                                   kwargs['sample'] + ".zarr"), 'r')
        shape = f[raw_key].shape[1:]
    source = sourceNode(
        os.path.join(kwargs['data_folder'],
                     kwargs['sample'] + "." + kwargs['input_format']),

        datasets={
            raw: raw_key
        })

    if kwargs['output_format'] != "zarr":
        raise NotImplementedError("Please use zarr as prediction output")

    # open zarr file
    zf = zarr.open(os.path.join(kwargs['output_folder'],
                                kwargs['sample'] + '.zarr'), mode='w')
    zf.create('volumes/pred_affs',
              shape=[int(np.prod(kwargs['patchshape']))] + list(shape),
              chunks=[int(np.prod(kwargs['patchshape']))] + list(chunksize),
              dtype=np.float16)
    zf['volumes/pred_affs'].attrs['offset'] = [0, 0]
    zf['volumes/pred_affs'].attrs['resolution'] = kwargs['voxel_size']

    if kwargs['overlapping_inst']:
        zf.create('volumes/pred_numinst',
                  shape=[int(kwargs['max_num_inst']) + 1] + list(shape),
                  chunks=[int(kwargs['max_num_inst']) + 1] + list(chunksize),
                  dtype=np.float16)
        zf['volumes/pred_numinst'].attrs['offset'] = [0, 0]
        zf['volumes/pred_numinst'].attrs['resolution'] = kwargs['voxel_size']

    outputs = {
        net_names['pred_affs']: pred_affs,
    }
    outVolumes = {
        pred_affs: '/volumes/pred_affs',
    }
    if kwargs['overlapping_inst']:
        outputs[net_names['pred_numinst']] = pred_numinst
        outVolumes[pred_numinst] = '/volumes/pred_numinst'

    pipeline = (
        source +
        gp.Pad(raw, context) +
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
