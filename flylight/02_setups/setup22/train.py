from __future__ import print_function
import sys
import gunpowder as gp
import neurolight.gunpowder as nl
import os
import json
import tensorflow as tf
import h5py
import zarr
import logging
import numpy as np
import math


def train_until(**kwargs):
    output_folder = kwargs['output_folder']
    name = kwargs['name']
    data_files = kwargs['data_files']

    if tf.train.latest_checkpoint(output_folder):
        trained_until = int(
            tf.train.latest_checkpoint(output_folder).split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= kwargs['max_iteration']:
        return

    with open(os.path.join(output_folder, name + '_config.json'), 'r') as f:
        net_config = json.load(f)
    with open(os.path.join(output_folder, name + '_names.json'), 'r') as f:
        net_names = json.load(f)

    # define array keys
    raw = gp.ArrayKey('RAW')
    gt_instances = gp.ArrayKey('GT_INSTANCES')
    gt_numinst = gp.ArrayKey('GT_NUMINST')
    gt_affs = gp.ArrayKey('GT_AFFS')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    pred_numinst = gp.ArrayKey('PRED_NUMINST')
    gt_mask_points = gp.PointsKey('GT_MASK_POINTS')

    input_shape = gp.Coordinate(net_config['input_shape'])
    output_shape = gp.Coordinate(net_config['output_shape'])
    context = gp.Coordinate(input_shape - output_shape)/2
    voxel_size = gp.Coordinate(kwargs['voxel_size'])
    print('input shape: ', input_shape)
    print('output shape: ', output_shape)
    print("context: ", context)
    print("voxel_size: ", voxel_size)

    # add array keys to batch
    request = gp.BatchRequest()
    request.add(raw, input_shape)
    request.add(gt_instances, output_shape)
    request.add(gt_numinst, output_shape)
    request.add(gt_affs, output_shape)

    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw, input_shape)
    snapshot_request.add(pred_affs, output_shape)
    snapshot_request.add(pred_numinst, output_shape)
    snapshot_request.add(gt_instances, output_shape)
    snapshot_request.add(gt_numinst, output_shape)

    neighborhood = []
    psH = np.array(kwargs['patchshape'])//2
    for z in range(-psH[0], psH[0] + 1):
        for y in range(-psH[1], psH[1] + 1):
            for x in range(-psH[2], psH[2] + 1):
                neighborhood.append([z, y, x])

    if kwargs['input_format'] != "hdf" and kwargs['input_format'] != "zarr":
        raise NotImplementedError("train node for %s not implemented yet",
                kwargs['input_format'])
    
    if kwargs['input_format'] == "hdf":
        sourceNode = gp.Hdf5Source
    elif kwargs['input_format'] == "zarr":
        sourceNode = gp.ZarrSource

    datasets = {
            raw: kwargs.get("raw_key", "volumes/raw"),
            gt_instances: kwargs.get("gt_key", "volumes/gt"),
    }
    
    array_specs={
            raw: gp.ArraySpec(interpolatable=True, dtype=np.uint16,
                voxel_size=voxel_size),
            gt_instances: gp.ArraySpec(interpolatable=False,
                dtype=np.uint8,
                voxel_size=voxel_size),
    }

    fg_locations = {}
    overlap_locations = {}
    overlap_samples = []
    data_files = kwargs['data_files']
    print("in train: ", data_files)

    for data_file in data_files:
        print(data_file)
        if kwargs["input_format"] == "hdf":
            inf = h5py.File(data_file, "r")
        elif kwargs["input_format"] == "zarr":
            inf = zarr.open(data_file, mode="r")
        # get overlap locations
        mask = np.squeeze(np.array(inf["volumes/dist_15_to_next"]))
        if np.sum(mask > 0) > 0:
            locs = list(np.argwhere(mask > 0))
            locs = [gp.Coordinate(loc) for loc in locs]
            overlap_samples.append(data_file)
            overlap_locations[data_file] = locs
        # get fg locations
        mask = np.squeeze(np.array(inf["volumes/fg_rm_5"]))
        locs = list(np.argwhere(mask > 0))
        locs = [gp.Coordinate(loc) for loc in locs]
        fg_locations[data_file] = locs
        if kwargs["input_format"] == "hdf":
            inf.close()

    # (1) specify random location source
    data_sources_random = tuple(
        sourceNode(
            data_file,
            datasets=datasets,
            array_specs=array_specs,
        ) +
        gp.Pad(raw, context) +
        gp.Pad(gt_instances, context) +
        gp.RandomLocation()
        for data_file in data_files)
    data_sources_random += gp.RandomProvider()

    # (2) specify fg data source
    data_sources_fg = tuple(
        sourceNode(
            data_file,
            datasets=datasets,
            array_specs=array_specs,
        ) +
        gp.Pad(raw, context) +
        gp.Pad(gt_instances, context) +
        gp.SpecifiedLocation(
            fg_locations[data_file], choose_randomly=True
        )
        for data_file in data_files)
    data_sources_fg += gp.RandomProvider()

    # (3) specify overlap data source
    data_sources_overlap = tuple(
        sourceNode(
            data_file,
            datasets=datasets,
            array_specs=array_specs,
        ) +
        gp.Pad(raw, context) +
        gp.Pad(gt_instances, context) +
        gp.SpecifiedLocation(
            overlap_locations[data_file], choose_randomly=True
        )
        for data_file in data_files if data_file in overlap_samples)
    data_sources_overlap += gp.RandomProvider()

    sampling_probs = [
        float(kwargs['probability_random']),
        float(kwargs['probability_fg']),
        float(kwargs['probability_overlap'])
    ]
    data_sources = tuple(
        [data_sources_random, data_sources_fg, data_sources_overlap]
    )
    data_sources += gp.RandomProvider(
        probabilities=sampling_probs
    )

    train_pipeline = (
        data_sources + 
        nl.Clip(raw, 0, kwargs['clipmax']) +
        gp.Normalize(raw, factor=1.0/kwargs['clipmax']) +
        nl.PermuteChannel(raw) +
        nl.OverlayAugment(
            raw,
            gt_instances,
            apply_probability=kwargs['probability_fuse']
        ) +
        nl.RandomHue(
            raw, kwargs['hue_max_change'],
            kwargs['probability_hue']
        ) +
        gp.ElasticAugment(
            control_point_spacing=(20, 20, 20),
            jitter_sigma=(1, 1, 1),
            rotation_interval=[0, math.pi / 2.],
            subsample=4) +
        gp.SimpleAugment(mirror_only=[1, 2], transpose_only=[1, 2]) +
        gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1) +
        gp.IntensityScaleShift(raw, 2, -1) +

        nl.AddAffinities(
            neighborhood,
            gt_instances,
            gt_affs,
            multiple_labels=True,
        ) +
        nl.CountOverlap(gt_instances, gt_numinst, maxnuminst=2) +

        gp.PreCache(
            cache_size=kwargs['cache_size'],
            num_workers=kwargs['num_workers']) +
        gp.tensorflow.Train(
            os.path.join(output_folder, name),
            optimizer=net_names['optimizer'],
            summary=net_names['merged'],
            log_dir=output_folder,
            loss=net_names['loss'],
            inputs={
                net_names['raw']: raw,
                net_names['gt_affs']: gt_affs,
                net_names['gt_numinst']: gt_numinst,
            },
            gradients={
            },
            outputs={
                net_names['pred_affs']: pred_affs,
                net_names['pred_numinst']: pred_numinst,
            },
            save_every=kwargs['checkpoints']) +
        gp.Snapshot({
                raw: 'raw',
                gt_instances: 'gt_instances',
                gt_numinst: 'gt_numinst',
                gt_affs: 'gt_affs',
                pred_affs: 'pred_affs',
                pred_numinst: 'pred_numinst',
            },
            dataset_dtypes={
                gt_instances: np.uint16,
            },
            every=kwargs['snapshots'],
            output_filename=os.path.join(output_folder,
                                         'snapshots',
                                         'batch_{iteration}.hdf'
                                         ),
            additional_request=snapshot_request) +
        gp.PrintProfilingStats(every=kwargs['profiling'])
    )

    print("Starting training...")
    with gp.build(train_pipeline) as b:
        for i in range(kwargs['max_iteration'] - trained_until):
            b.request_batch(request)
    print("Training finished")


if __name__ == "__main__":

    # call with checkpoint, experiment id, clipmax, netname
    # and output folder base
    iteration = int(sys.argv[1])
    experiment = sys.argv[2]
    clipmax = int(sys.argv[3])
    netname = sys.argv[4]
    root = sys.argv[5]

    output_folder = os.path.join(root, experiment, 'train')
    try:
        os.makedirs(os.path.join(output_folder, 'snapshots'))
    except OSError:
        pass

    logging.basicConfig(level=logging.INFO)
    train_until(iteration, netname, output_folder, clipmax)
