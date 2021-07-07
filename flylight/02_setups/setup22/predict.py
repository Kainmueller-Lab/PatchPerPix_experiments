import sys
import os
import numpy as np
import gunpowder as gp
import neurolight.gunpowder as nl
import json
import logging
from datetime import datetime
import zarr


def predict(
        name,
        sample,
        checkpoint,
        data_folder,
        input_folder,
        output_folder='.',
        voxel_size=(1, 1, 1),
        input_format='hdf',
        train_net_name='train_net',
        test_net_name='test_net',
        num_workers=5,
        clipmax=1500,
        **kwargs
):

    test_net_conf = os.path.join(input_folder, test_net_name + '_config.json')
    test_net_tensors = os.path.join(input_folder, test_net_name + '_names.json')

    with open(test_net_conf, 'r') as f:
        net_config = json.load(f)

    with open(test_net_tensors, 'r') as f:
        net_names = json.load(f)

    # ArrayKeys
    raw = gp.ArrayKey('RAW')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    pred_numinst = gp.ArrayKey('PRED_NUMINST')

    input_shape = gp.Coordinate(net_config['input_shape'])
    output_shape = gp.Coordinate(net_config['output_shape'])

    voxel_size = gp.Coordinate(voxel_size)
    context = gp.Coordinate(input_shape - output_shape) / 2

    print('input shape: ', input_shape, ', output shape: ', output_shape,
          ', context: ', context)

    # add ArrayKeys to batch request
    request = gp.BatchRequest()
    request.add(raw, input_shape, voxel_size=voxel_size)
    request.add(pred_affs, output_shape, voxel_size=voxel_size)
    request.add(pred_numinst, output_shape, voxel_size=voxel_size)

    if input_format != "hdf" and input_format != "zarr":
        raise NotImplementedError("predict node for %s not implemented yet",
                input_format)

    if input_format == "hdf":
        sourceNode = gp.Hdf5Source
    elif input_format == "zarr":
        sourceNode = gp.ZarrSource
    
    print("in predict: ", data_folder, sample)
    if data_folder.endswith("." + input_format):
        infn = data_folder
    else:
        infn = os.path.join(data_folder, sample + "." + input_format)

    source = (
        sourceNode(
            infn,
            datasets={
                raw: "volumes/raw",
            },
            array_specs={
                raw: gp.ArraySpec(interpolatable=True, dtype=np.uint16,
                                  voxel_size=voxel_size),
            },
        ) +
        gp.Pad(raw, context) +
        nl.Clip(raw, 0, clipmax) +
        gp.Normalize(raw, factor=1.0/clipmax) +
        gp.IntensityScaleShift(raw, 2, -1)
    )

    with gp.build(source):
        raw_roi = source.spec[raw].roi
        print("raw_roi: %s" % raw_roi)
        sample_shape = raw_roi.grow(-context, -context).get_shape()

    # create zarr file with corresponding chunk size
    chunksize = list(np.asarray(output_shape) // 2)
    print('sample shape: ', sample_shape, ', chunksize: ', chunksize)

    num_patch_output = np.prod(kwargs['patchshape'])

    # open zarr file
    zf = zarr.open(os.path.join(output_folder, sample + '.zarr'), mode='w')

    # create pred affs dataset
    zf.create('volumes/pred_affs',
              shape=[num_patch_output] + list(sample_shape),
              chunks=[num_patch_output] + chunksize,
              dtype=np.float16)
    zf['volumes/pred_affs'].attrs['offset'] = [0, 0, 0]
    zf['volumes/pred_affs'].attrs['resolution'] = [1, 1, 1]

    # create numinst dataset
    num_inst_output = 3
    zf.create('volumes/pred_numinst',
              shape=[num_inst_output] + list(sample_shape),
              chunks=[num_inst_output] + chunksize, dtype=np.float16)
    zf['volumes/pred_numinst'].attrs['offset'] = [0, 0, 0]
    zf['volumes/pred_numinst'].attrs['resolution'] = [1, 1, 1]

    pipeline = (
        source +
        gp.tensorflow.Predict(
            graph=os.path.join(input_folder, test_net_name + '.meta'),
            checkpoint=checkpoint,
            inputs={
                net_names['raw']: raw,
            },
            outputs={
                net_names['pred_affs']: pred_affs,
                net_names['pred_numinst']: pred_numinst,
            },
            array_specs={
                pred_affs: gp.ArraySpec(
                    roi=raw_roi.grow(-context, -context),
                    voxel_size=voxel_size),
                pred_numinst: gp.ArraySpec(
                    roi=raw_roi.grow(-context, -context),
                    voxel_size=voxel_size),
            },
            max_shared_memory=1024*1024*1024*8
            ) +
        nl.Convert(pred_affs, np.float16) +
        nl.Threshold(pred_affs, 0.5) +

        gp.ZarrWrite(
            dataset_names={
                pred_affs: 'volumes/pred_affs',
                pred_numinst: 'volumes/pred_numinst'
            },
            output_dir=output_folder,
            output_filename=sample + '.zarr',
            compression_type='gzip',
        ) +

        # show a summary of time spend in each node every x iterations
        gp.PrintProfilingStats(every=100) +
        gp.Scan(reference=request, num_workers=num_workers, cache_size=50)
    )

    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())


if __name__ == "__main__":

    # call with experiment id, checkpoint, clipmax, sample, data dir,
    # and output folder base
    # experiment = sys.argv[1]
    # iteration = int(sys.argv[2])
    # clipmax = int(sys.argv[3])
    # sample = sys.argv[4]
    # data_dir = sys.argv[5]
    # root = sys.argv[6]
    # test_net_name = 'test_net'
    # train_net_name = 'train_net'
    #
    # output_dir = os.path.join(root, experiment, 'test', 'processed', '%d' % iteration)
    # train_dir = os.path.join(root, experiment, 'train')
    # try:
    #     os.makedirs(output_dir)
    # except:
    #     pass
    #
    # logging.basicConfig(level=logging.INFO)
    #
    # start = datetime.now()
    #
    # predict(
    #     sample, iteration, data_dir, train_dir,
    #     test_net_name=test_net_name,
    #     train_net_name=train_net_name,
    #     output_dir=output_dir,
    #     clipmax=clipmax,
    #     num_workers=5,
    # )
    #
    # print(datetime.now() - start)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    logging.basicConfig(level=logging.INFO)

    predict('test_net',
            sample='BJD_103F02_AE_01-20171222_63_E1',
            checkpoint='/home/maisl/workspace/ppp/flylight/setup03_190712_00/train/train_net_checkpoint_50000',
            data_folder='/home/maisl/data/flylight/flylight_v2'
                        '/flylight_val_sel_cropped.hdf',
            input_folder='/home/maisl/workspace/ppp/flylight/setup03_190712_00/test',
            output_folder='/home/maisl/workspace/ppp/flylight/setup03_190712_00/val/processed/50000',
            voxel_size=(1, 1, 1),
            input_format='hdf',
            clipmax=1500,
            num_workers=10,
            )
