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
    pred_code = gp.ArrayKey('PRED_AFFS')
    if kwargs['overlapping_inst']:
        pred_numinst = gp.ArrayKey('PRED_NUMINST')
    else:
        pred_fgbg = gp.ArrayKey('PRED_FGBG')

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
    chunksize = [int(c) for c in np.asarray(output_shape_world) // 2]

    raw_key = kwargs.get('raw_key', 'volumes/raw')
    if kwargs['overlapping_inst']:
        numinst_key = kwargs.get('fg_key', 'volumes/pred_numinst')
    else:
        fgbg_key = kwargs.get('fg_key', 'volumes/pred_fgbg')
    code_key = kwargs.get('code_key', 'volumes/pred_code')

    # add ArrayKeys to batch request
    request = gp.BatchRequest()
    request.add(raw, input_shape_world, voxel_size=voxel_size)
    request.add(pred_code, output_shape_world, voxel_size=voxel_size)
    if kwargs['overlapping_inst']:
        request.add(pred_numinst, output_shape_world, voxel_size=voxel_size)
    else:
        request.add(pred_fgbg, output_shape_world, voxel_size=voxel_size)

    if kwargs['input_format'] != "hdf" and kwargs['input_format'] != "zarr":
        raise NotImplementedError("predict node for %s not implemented yet",
                                  kwargs['input_format'])
    if kwargs['input_format'] == "hdf":
        sourceNode = gp.Hdf5Source
        with h5py.File(os.path.join(kwargs['data_folder'],
                                    kwargs['sample'] + ".hdf"), 'r') as f:
            shape = f[raw_key].shape[1:]
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
    zf.create(code_key,
              shape=[int(kwargs['code_units'])] + list(shape),
              chunks=[int(kwargs['code_units'])] + list(chunksize),
              dtype=np.float16)
    zf[code_key].attrs['offset'] = [0, 0]
    zf[code_key].attrs['resolution'] = kwargs['voxel_size']

    if kwargs['overlapping_inst']:
        zf.create(numinst_key,
                  shape=[int(kwargs['max_num_inst']) + 1] + list(shape),
                  chunks=[int(kwargs['max_num_inst']) + 1] + list(chunksize),
                  dtype=np.float16)
        zf[numinst_key].attrs['offset'] = [0, 0]
        zf[numinst_key].attrs['resolution'] = kwargs['voxel_size']
    else:
        zf.create(fgbg_key,
                  shape=[1] + list(shape),
                  chunks=[1] + list(chunksize),
                  dtype=np.float16)
        zf[fgbg_key].attrs['offset'] = [0, 0]
        zf[fgbg_key].attrs['resolution'] = kwargs['voxel_size']

    outputs = {
        net_names['pred_code']: pred_code,
    }
    outVolumes = {
        pred_code: code_key,
    }
    if kwargs['overlapping_inst']:
        outputs[net_names['pred_numinst']] = pred_numinst
        outVolumes[pred_numinst] = numinst_key
    else:
        outputs[net_names['pred_fgbg']] = pred_fgbg
        outVolumes[pred_fgbg] = fgbg_key

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
