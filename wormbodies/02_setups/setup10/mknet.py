import tensorflow as tf
import numpy as np
import json
import os

from PatchPerPix.models import unet, conv_pass, crop


def mk_net(**kwargs):
    tf.reset_default_graph()

    input_shape = kwargs['input_shape']
    if not isinstance(input_shape, tuple):
        input_shape = tuple(input_shape)

    # create a placeholder for the 3D raw input tensor
    raw = tf.placeholder(tf.float32,
                         shape=(kwargs['num_channels'],) + input_shape,
                         name="raw")

    # create a U-Net
    raw_batched = tf.reshape(raw, (1, kwargs['num_channels']) + input_shape)
    model = unet(raw_batched,
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
    if not kwargs['overlapping_inst']:
        num_inst_fmaps = 0
    else:
        num_inst_fmaps = kwargs['max_num_inst'] + 1
    model = conv_pass(
        model,
        kernel_size=1,
        num_fmaps=num_patch_fmaps + num_inst_fmaps,
        num_repetitions=1,
        padding=kwargs['padding'],
        activation=None,
        name="output")
    print(model)

    logits = tf.squeeze(model, axis=0)
    output_shape = logits.get_shape().as_list()[1:]

    if kwargs['overlapping_inst']:
        logitspatch, logitsnuminst = tf.split(
            logits, [num_patch_fmaps, num_inst_fmaps], 0)
        pred_numinst = tf.nn.softmax(logitsnuminst, axis=0)
    else:
        logitspatch = logits

    pred_affs = tf.sigmoid(logitspatch)
    raw_cropped = crop(raw, [kwargs['num_channels']] + output_shape)

    # placeholder for gt
    gt_affs_shape = pred_affs.get_shape().as_list()
    gt_affs = tf.placeholder(tf.float32, shape=gt_affs_shape, name="gt_affs")

    if kwargs['overlapping_inst']:
        gt_numinst = tf.placeholder(tf.int32, shape=[1] + output_shape,
                                    name="gt_numinst")
        gt_numinstTmp = tf.squeeze(gt_numinst, 0)
        gt_numinst_clipped = tf.clip_by_value(gt_numinstTmp, 0,
                                              kwargs['max_num_inst'])

        # get loss
        losspatch = tf.losses.sigmoid_cross_entropy(
            gt_affs,
            logitspatch,
            weights=tf.reshape(
                tf.to_float(gt_numinst_clipped < 2),
                [1, ] + list(gt_numinst_clipped.shape))
        )
        lossnuminst = tf.losses.sparse_softmax_cross_entropy(
            gt_numinst_clipped,
            tf.transpose(logitsnuminst, [1, 2, 0]))
        loss = losspatch + lossnuminst
    else:
        loss = tf.losses.sigmoid_cross_entropy(gt_affs, logitspatch)

    # write loss to summary
    loss_sums = [tf.summary.scalar('loss_sum', loss)]
    if kwargs['overlapping_inst']:
        loss_sums.append(tf.summary.scalar('loss_patch_sum', losspatch))
        loss_sums.append(tf.summary.scalar('loss_numinst_sum', lossnuminst))
    summaries = tf.summary.merge(loss_sums, name="summaries")

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
                                                     kwargs['name'] + '.meta'))

    fn = os.path.join(kwargs['output_folder'], kwargs['name'])
    names = {
        'raw': raw.name,
        'raw_cropped': raw_cropped.name,
        'gt_affs': gt_affs.name,
        'pred_affs': pred_affs.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'summaries': summaries.name
    }
    if kwargs['overlapping_inst']:
        names['gt_numinst'] = gt_numinst.name
        names['pred_numinst'] = pred_numinst.name

    with open(fn + '_names.json', 'w') as f:
        json.dump(names, f)

    config = {
        'input_shape': input_shape,
        'gt_affs_shape': gt_affs_shape,
        'output_shape': output_shape,
    }

    with open(fn + '_config.json', 'w') as f:
        json.dump(config, f)
