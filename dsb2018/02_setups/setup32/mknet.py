import tensorflow as tf
import numpy as np
import json
# from PatchPerPix.models import unet, conv_pass, crop, autoencoder
from funlib.learn.tensorflow.models import unet, conv_pass, crop
from PatchPerPix import util
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
                 upsampling=kwargs['upsampling'])
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

    raw_cropped = crop(raw, output_shape)

    # placeholder for gt
    gt_affs_shape = logits.get_shape().as_list()
    gt_affs = tf.placeholder(tf.float32, shape=gt_affs_shape,
                             name="gt_affs")
    # anchor = tf.placeholder(tf.float32, shape=[1] + output_shape,
    #                         name="anchor")
    gt_fgbg = tf.placeholder(tf.float32, shape=[1] + output_shape,
                             name="gt_fgbg")


    # loss_weights_affs = tf.placeholder(
    #     tf.float32,
    #     shape=[1] + output_shape,
    #     name="loss_weights_affs")

    loss, pred_affs, _ = \
       util.get_loss(gt_affs, logits,
                     kwargs['loss'], "affs", True)

    loss_sums = []
    loss_sums.append(tf.summary.scalar('loss_sum', loss))
    summaries = tf.summary.merge(loss_sums, name="summaries")

    global_step = tf.Variable(0, name='global_step', trainable=False,
                              dtype=tf.int64)

    # optimizer
    learning_rate = tf.placeholder_with_default(kwargs['lr'], shape=(),
                                                name="learning-rate")
    opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)

    tf.train.export_meta_graph(filename=os.path.join(kwargs['output_folder'],
                                                     kwargs['name'] +'.meta'))

    fn = os.path.join(kwargs['output_folder'], kwargs['name'])
    names = {
        'raw': raw.name,
        'raw_cropped': raw_cropped.name,
        'gt_affs': gt_affs.name,
        'pred_affs': pred_affs.name,
        # 'loss_weights_fgbg': loss_weights_fgbg.name,
        # 'anchor': anchor.name,
        'gt_fgbg': gt_fgbg.name,
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
