import tensorflow as tf
import numpy as np
import json
# from PatchPerPix.models import unet, conv_pass, crop
from funlib.learn.tensorflow.models import unet, conv_pass, crop
import sys
import os


def mk_net(**kwargs):

    tf.reset_default_graph()

    input_shape = kwargs['input_shape']
    if not isinstance(input_shape, tuple):
        input_shape = tuple(input_shape)

    # create a placeholder for the 3D raw input tensor
    raw = tf.placeholder(tf.float32,
                         shape=input_shape,
                         name="raw")

    # create a U-Net
    raw_batched = tf.reshape(raw, (1, kwargs['num_channels']) + input_shape)
    model, _, _ = unet(raw_batched,
                 num_fmaps=kwargs['num_fmaps'],
                 fmap_inc_factors=kwargs['fmap_inc_factors'],
                 fmap_dec_factors=kwargs['fmap_dec_factors'],
                 downsample_factors=kwargs['downsample_factors'],
                 activation=kwargs['activation'],
                 padding=kwargs['padding'],
                 kernel_size=kwargs['kernel_size'],
                 num_repetitions=kwargs['num_repetitions'],
                 upsampling=kwargs['upsampling'],
                 crop_factor=kwargs.get('crop_factor', True))
    print(model)

    num_patch_fmaps = np.prod(kwargs['patchshape'])
    model, _ = conv_pass(
        model,
        kernel_sizes=[1],
        num_fmaps=num_patch_fmaps,
        padding=kwargs['padding'],
        activation=None,
        name="output")
    print(model)


    logits = tf.squeeze(model, axis=0)
    output_shape = logits.get_shape().as_list()[1:]
    logitspatch = logits

    pred_affs = tf.sigmoid(logitspatch)

    raw_cropped = crop(raw, output_shape)

    # placeholder for gt
    gt_affs_shape = pred_affs.get_shape().as_list()
    gt_affs = tf.placeholder(tf.uint8, shape=gt_affs_shape,
                             name="gt_affs")
    anchor = tf.placeholder(tf.float32, shape=[1] + output_shape,
                            name="anchor")


    # loss
    # loss_weights_affs = tf.placeholder(
    #     tf.float32,
    #     shape=pred_affs.get_shape(),
    #     name="loss_weights_affs")

    loss = tf.losses.sigmoid_cross_entropy(
        gt_affs,
        logitspatch)
        # loss_weights_affs)

    loss_sums = []
    loss_sums.append(tf.summary.scalar('loss_sum', loss))
    summaries = tf.summary.merge(loss_sums, name="summaries")

    # optimizer
    opt = getattr(tf.train, kwargs['optimizer'])(
                    *kwargs['args'].values(),
                    **kwargs['kwargs'])
    optimizer = opt.minimize(loss)

    tf.train.export_meta_graph(filename=os.path.join(kwargs['output_folder'],
                                                     kwargs['name'] +'.meta'))

    fn = os.path.join(kwargs['output_folder'], kwargs['name'])
    names = {
        'raw': raw.name,
        'raw_cropped': raw_cropped.name,
        'gt_affs': gt_affs.name,
        'pred_affs': pred_affs.name,
        # 'loss_weights_affs': loss_weights_affs.name,
        'anchor': anchor.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'summaries': summaries.name
    }
    with open(fn + '_names.json', 'w') as f:
        json.dump(names, f)

    config = {
        'input_shape': input_shape,
        'gt_affs_shape': gt_affs_shape,
        'output_shape': output_shape,
    }

    with open(fn + '_config.json', 'w') as f:
        json.dump(config, f)
