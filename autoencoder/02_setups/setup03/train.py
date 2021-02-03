import json
import logging
import os
import time

import numpy as np
import tensorflow as tf
import gunpowder as gp
import zarr
import h5py


logger = logging.getLogger(__name__)


def train_until(**kwargs):
    print("cuda visibile devices", os.environ["CUDA_VISIBLE_DEVICES"])
    if tf.train.latest_checkpoint(kwargs['output_folder']):
        trained_until = int(
            tf.train.latest_checkpoint(kwargs['output_folder']).split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= kwargs['max_iteration']:
        return

    anchor = gp.ArrayKey('ANCHOR')
    gt_labels = gp.ArrayKey('GT_LABELS')
    gt_fgbg = gp.ArrayKey('GT_FGBG')
    gt_affs = gp.ArrayKey('GT_AFFS')

    pred_affs = gp.ArrayKey('PRED_AFFS')
    pred_affs_gradients = gp.ArrayKey('PRED_AFFS_GRADIENTS')

    with open(os.path.join(kwargs['output_folder'],
                           kwargs['name'] + '_config.json'), 'r') as f:
        net_config = json.load(f)
    with open(os.path.join(kwargs['output_folder'],
                           kwargs['name'] + '_names.json'), 'r') as f:
        net_names = json.load(f)

    voxel_size = gp.Coordinate(kwargs['voxel_size'])
    input_shape_world = (1,) * len(voxel_size)
    output_shape_world = (1,) * len(voxel_size)

    # formulate the request for what a batch should (at least) contain
    request = gp.BatchRequest()
    request.add(gt_fgbg, input_shape_world)
    request.add(gt_labels, output_shape_world)
    request.add(gt_affs, output_shape_world)

    snapshot_request = gp.BatchRequest()
    snapshot_request.add(gt_affs, output_shape_world)
    snapshot_request.add(pred_affs, output_shape_world)

    if kwargs['input_format'] != "hdf" and kwargs['input_format'] != "zarr":
        raise NotImplementedError("train node for %s not implemented yet",
                                  kwargs['input_format'])

    fls = []
    locations = []
    for f in kwargs['data_files']:
        fls.append(os.path.splitext(f)[0])
        if kwargs['input_format'] == "hdf":
            fgbg = np.squeeze(np.array(h5py.File(f, 'r')['volumes/gt_fgbg']))
        elif kwargs['input_format'] == "zarr":
            fgbg = np.squeeze(np.array(zarr.open(f, 'r')['volumes/gt_fgbg']))
        locations.append(list(np.argwhere(fgbg > 0)))
    ln = len(fls)
    print("first 5 files: ", fls[0:4])
    print("first 5 locations: ", locations[0][:4])

    if kwargs['input_format'] == "hdf":
        sourceNode = gp.Hdf5Source
    elif kwargs['input_format'] == "zarr":
        sourceNode = gp.ZarrSource

    neighborhood = []
    psH = np.array(kwargs['patchshape']) // 2
    for i in range(-psH[1], psH[1] + 1, kwargs['patchstride'][1]):
        for j in range(-psH[2], psH[2] + 1, kwargs['patchstride'][2]):
            neighborhood.append([i, j])

    sources = tuple(
        sourceNode(
            fls[t] + "." + kwargs['input_format'],
            datasets={
                gt_fgbg: 'volumes/gt_fgbg',
                gt_labels: kwargs['gt_key'],
            },
            array_specs={
                gt_fgbg: gp.ArraySpec(voxel_size=voxel_size,
                                      interpolatable=False),
                gt_labels: gp.ArraySpec(voxel_size=voxel_size,
                                        interpolatable=False),
            }
        )
        + gp.Pad(gt_labels, None)
        + gp.Pad(gt_fgbg, None)

        + gp.SpecifiedLocation(locations[t], choose_randomly=True)

        for t in range(ln)
    )

    augmentation = kwargs['augmentation']
    pipeline = (
        sources +

        gp.RandomProvider() +

        gp.Reject(gt_fgbg) +

        # convert labels into affinities between voxels
        gp.AddAffinities(
            neighborhood,
            gt_labels,
            gt_affs,
            multiple_labels=kwargs['overlapping_inst']) +

        # pre-cache batches from the point upstream
        gp.PreCache(
            cache_size=kwargs['cache_size'],
            num_workers=kwargs['num_workers']) +

        gp.Stack(kwargs['batch_size']) +

        # perform one training iteration for each passing batch (here we use
        # the tensor names earlier stored in train_net.config)
        gp.tensorflow.Train(
            os.path.join(kwargs['output_folder'], kwargs['name']),
            optimizer=net_names['optimizer'],
            summary=net_names['summaries'],
            log_dir=kwargs['output_folder'],
            log_every=10,
            loss=net_names['loss'],
            inputs={
                net_names['gt_affs']: gt_affs,
            },
            outputs={
                net_names['pred_affs']: pred_affs,
            },
            gradients={
            },
            save_every=kwargs['checkpoints']) +

        # save the passing batch as an HDF5 file for inspection
        gp.Snapshot({
            pred_affs: '/volumes/pred_affs',
            gt_affs: '/volumes/gt_affs',
            },
            output_dir=os.path.join(kwargs['output_folder'], 'snapshots'),
            output_filename='batch_{iteration}.hdf',
            every=kwargs['snapshots'],
            additional_request=snapshot_request,
            compression_type='gzip') +

        # show a summary of time spend in each node every 10 iterations
        gp.PrintProfilingStats(every=kwargs['profiling'])
    )

    #########
    # TRAIN #
    #########
    print("Starting training...")
    with gp.build(pipeline):
        print(pipeline)
        for i in range(trained_until, kwargs['max_iteration']):
            start = time.time()
            pipeline.request_batch(request)
            time_of_iteration = time.time() - start

            logger.info(
                "Batch: iteration=%d, time=%f",
                i, time_of_iteration)
            # exit()
    print("Training finished")
