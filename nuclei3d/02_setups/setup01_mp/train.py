from __future__ import print_function
import json
import logging
import os

os.environ[
    'TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_ADD'] = (
        'Conv3D,Conv3DBackpropFilter,Conv3DBackpropFilterV2,'
        'Conv3DBackpropInput,Conv3DBackpropInputV2,'
        'Conv2D,Conv2DBackpropFilter,Conv2DBackpropFilterV2,'
        'Conv2DBackpropInput,Conv2DBackpropInputV2')

import sys
import time

import h5py
import numpy as np
import tensorflow as tf
import zarr

import gunpowder as gp

class NoOp(gp.BatchFilter):

    def __init__(self):
        pass

    def process(self, batch, request):
        pass


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
    raw = gp.ArrayKey('RAW')
    raw_cropped = gp.ArrayKey('RAW_CROPPED')
    gt_labels = gp.ArrayKey('GT_LABELS')
    gt_affs = gp.ArrayKey('GT_AFFS')

    pred_affs = gp.ArrayKey('PRED_AFFS')
    pred_affs_gradients = gp.ArrayKey('PRED_AFFS_GRADIENTS')

    with open(os.path.join(kwargs['output_folder'],
                           kwargs['name'] + '_config.json'), 'r') as f:
        net_config = json.load(f)
    with open(os.path.join(kwargs['output_folder'],
                           kwargs['name']  + '_names.json'), 'r') as f:
        net_names = json.load(f)

    voxel_size = gp.Coordinate(kwargs['voxel_size'])
    input_shape_world = gp.Coordinate(net_config['input_shape'])*voxel_size
    output_shape_world = gp.Coordinate(net_config['output_shape'])*voxel_size

    # formulate the request for what a batch should (at least) contain
    request = gp.BatchRequest()

    # when we make a snapshot for inspection (see below), we also want to
    # request the predicted affinities and gradients of the loss wrt the
    # affinities
    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw_cropped, output_shape_world)
    snapshot_request.add(pred_affs, output_shape_world)
    snapshot_request.add(gt_affs, output_shape_world)

    if kwargs['input_format'] != "hdf" and kwargs['input_format'] != "zarr":
        raise NotImplementedError("train node for %s not implemented yet",
                                  kwargs['input_format'])

    fls = []
    for f in kwargs['data_files']:
        fls.append(os.path.splitext(f)[0])
    ln = len(fls)
    print("first 5 files: ", fls[0:4])

    if kwargs['input_format'] == "hdf":
        sourceNode = gp.Hdf5Source
    elif kwargs['input_format'] == "zarr":
        sourceNode = gp.ZarrSource

    neighborhood = []
    psH = np.array(kwargs['patchshape'])//2
    for i in range(-psH[0], psH[0]+1, kwargs['patchstride'][0]):
        for j in range(-psH[1], psH[1]+1, kwargs['patchstride'][1]):
            for k in range(-psH[2], psH[2]+1, kwargs['patchstride'][2]):
                neighborhood.append([i,j, k])

    datasets = {
        raw: 'volumes/raw',
        gt_labels: 'volumes/gt_labels',
        anchor: 'volumes/gt_fgbg',
    }
    input_specs = {
        raw: gp.ArraySpec(roi=gp.Roi((0,)*len(input_shape_world),
                                     input_shape_world),
                          interpolatable=True, dtype=np.float32),
        gt_labels: gp.ArraySpec(roi=gp.Roi((0,)*len(output_shape_world),
                                     output_shape_world),
                                interpolatable=False, dtype=np.uint16),
        anchor: gp.ArraySpec(roi=gp.Roi((0,)*len(output_shape_world),
                                     output_shape_world),
                             interpolatable=False, dtype=np.uint8),
        gt_affs: gp.ArraySpec(roi=gp.Roi((0,)*len(output_shape_world),
                                         output_shape_world),
                              interpolatable=False, dtype=np.uint8)
    }
    inputs = {
        net_names['raw']: raw,
        net_names['gt_affs']: gt_affs,
        net_names['anchor']: anchor,
    }

    outputs = {
        net_names['pred_affs']: pred_affs,
        net_names['raw_cropped']: raw_cropped,
    }
    snapshot = {
        raw_cropped: 'volumes/raw_cropped',
        gt_affs: '/volumes/gt_affs',
        pred_affs: '/volumes/pred_affs',
    }

    augmentation = kwargs['augmentation']
    pipeline = (
        tuple(
            sourceNode(
                fls[t] + "." + kwargs['input_format'],
                datasets=datasets,
                # array_specs=array_specs
            )
            + gp.Pad(raw, None)
            + gp.Pad(gt_labels, None)

            # chose a random location for each requested batch
            + gp.RandomLocation()

            for t in range(ln)
        ) +

        # chose a random source (i.e., sample) from the above
        gp.RandomProvider() +

        # elastically deform the batch
        gp.ElasticAugment(
            augmentation['elastic']['control_point_spacing'],
            augmentation['elastic']['jitter_sigma'],
            [augmentation['elastic']['rotation_min']*np.pi/180.0,
             augmentation['elastic']['rotation_max']*np.pi/180.0],
            subsample=4) +

        # apply transpose and mirror augmentations
        gp.SimpleAugment(mirror_only=augmentation['simple'].get("mirror"),
                         transpose_only=augmentation['simple'].get("transpose")) +

        # scale and shift the intensity of the raw array
        gp.IntensityAugment(
            raw,
            scale_min=augmentation['intensity']['scale'][0],
            scale_max=augmentation['intensity']['scale'][1],
            shift_min=augmentation['intensity']['shift'][0],
            shift_max=augmentation['intensity']['shift'][1],
            z_section_wise=False) +

        # grow a boundary between labels
        gp.GrowBoundary(
            gt_labels,
            steps=1,
            only_xy=False) +

        # convert labels into affinities between voxels
        gp.AddAffinities(
            neighborhood,
            gt_labels,
            gt_affs) +

        # create a weight array that balances positive and negative samples in
        # the affinity array
        # gp.BalanceLabels(
        #     gt_affs,
        #     loss_weights_affs) +

        # pre-cache batches from the point upstream
        gp.PreCache(
            cache_size=kwargs['cache_size'],
            num_workers=kwargs['num_workers']) +

        # pre-fetch batches from the point upstream
        (gp.tensorflow.TFData() \
         if kwargs.get('use_tf_data') else NoOp()) +

        # perform one training iteration for each passing batch (here we use
        # the tensor names earlier stored in train_net.config)
        gp.tensorflow.Train(
            os.path.join(kwargs['output_folder'], kwargs['name']),
            optimizer=net_names['optimizer'],
            summary=net_names['summaries'],
            log_dir=kwargs['output_folder'],
            loss=net_names['loss'],
            inputs=inputs,
            outputs=outputs,
            array_specs=input_specs,
            gradients={
                net_names['pred_affs']: pred_affs_gradients,
            },
            auto_mixed_precision=kwargs['auto_mixed_precision'],
            learning_rate=kwargs['lr'],
            use_tf_data=kwargs['use_tf_data'],
            save_every=kwargs['checkpoints'],
            snapshot_every=kwargs['snapshots']) +

        # save the passing batch as an HDF5 file for inspection
        gp.Snapshot(
            snapshot,
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
    try:
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
    except KeyboardInterrupt:
        sys.exit()
    print("Training finished")
