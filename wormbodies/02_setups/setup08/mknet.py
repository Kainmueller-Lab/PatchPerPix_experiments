import tensorflow as tf
import numpy as np
import json
import os

import toml

from PatchPerPix.models import unet, conv_pass, crop, autoencoder


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

    num_patch_fmaps = kwargs['autoencoder']['code_units']
    if kwargs['overlapping_inst']:
        num_inst_fmaps = kwargs['max_num_inst'] + 1
    else:
        num_inst_fmaps = 1

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
        logitspatch, logits_fgbg = tf.split(
            logits, [num_patch_fmaps, 1], 0)

    pred_code = logitspatch
    if kwargs['autoencoder'].get('code_activation') is not None:
        activation = getattr(tf.nn,
                             kwargs['autoencoder'].get('code_activation'))
        pred_code = activation(pred_code)
    raw_cropped = crop(raw, [kwargs['num_channels']] + output_shape)

    patchshape_squeezed = tuple(p for p in kwargs['patchshape']
                                if p > 1)
    patchsize = int(np.prod(patchshape_squeezed))
    # placeholder for gt
    gt_affs_shape = [patchsize] + output_shape
    gt_affs = tf.placeholder(tf.float32, shape=gt_affs_shape, name="gt_affs")

    if kwargs['overlapping_inst']:
        gt_numinst = tf.placeholder(tf.int32, shape=[1] + output_shape,
                                    name="gt_numinst")
        gt_numinstTmp = tf.squeeze(gt_numinst, 0)
        gt_numinst_clipped = tf.clip_by_value(gt_numinstTmp, 0,
                                              kwargs['max_num_inst'])
        gt_fg = tf.equal(gt_numinstTmp, 1)
    else:
        gt_fgbg = tf.placeholder(tf.float32, shape=[1] + output_shape,
                                 name="gt_fgbg")
        gt_fg = tf.squeeze(gt_fgbg, 0)


    code = tf.transpose(tf.sigmoid(logitspatch), [1, 2, 0])
    sample_cnt = 1024
    gt_fg_loc = tf.where(gt_fg)
    samples_loc = tf.random.uniform(
        (tf.math.minimum(sample_cnt, tf.shape(gt_fg_loc)[0]),),
        minval=0,
        maxval=tf.shape(gt_fg_loc)[0],
        dtype=tf.int32)
    samples_loc = tf.gather(gt_fg_loc, samples_loc)
    code_samples = tf.gather_nd(code, samples_loc)
    gt_affs_samples = tf.gather_nd(tf.transpose(gt_affs, [1, 2, 0]),
                                   samples_loc)

    ae_config = kwargs['autoencoder']
    ae_config['only_decode'] = True
    ae_config['dummy_in'] = tf.placeholder(
        tf.float32, (None,) + patchshape_squeezed)
    ae_config['is_training'] = True
    ae_config['input_shape_squeezed'] = patchshape_squeezed
    net, sums, _ = autoencoder(code_samples, **ae_config)

    # get loss
    net = tf.reshape(net, (-1, patchsize), name="rs2")
    print(net.get_shape(), gt_affs_samples.get_shape())
    losspatch = tf.losses.sigmoid_cross_entropy(
        gt_affs_samples,
        net)

    if kwargs['overlapping_inst']:
        lossnuminst = tf.losses.sparse_softmax_cross_entropy(
            gt_numinst_clipped,
            tf.transpose(logitsnuminst, [1, 2, 0]))
        loss = losspatch + lossnuminst
    else:
        lossfgbg = tf.losses.sigmoid_cross_entropy(
            gt_fgbg,
            logits_fgbg)
        pred_fgbg = tf.sigmoid(logits_fgbg)
        loss = losspatch + lossfgbg

    # write loss to summary
    loss_sums = [tf.summary.scalar('loss_sum', loss),
                 tf.summary.scalar('loss_patch_sum', losspatch)]
    if kwargs['overlapping_inst']:
        loss_sums.append(tf.summary.scalar('loss_numinst_sum', lossnuminst))
    else:
        loss_sums.append(tf.summary.scalar('loss_fgbg', lossfgbg))
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
                                                     kwargs['name'] + '.meta'))

    fn = os.path.join(kwargs['output_folder'], kwargs['name'])
    names = {
        'raw': raw.name,
        'raw_cropped': raw_cropped.name,
        'gt_affs': gt_affs.name,
        'pred_code': pred_code.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'summaries': summaries.name
    }
    if kwargs['overlapping_inst']:
        names['gt_numinst'] = gt_numinst.name
        names['pred_numinst'] = pred_numinst.name
    else:
        names['gt_fgbg'] = gt_fgbg.name
        names['pred_fgbg'] = pred_fgbg.name

    with open(fn + '_names.json', 'w') as f:
        json.dump(names, f)

    config = {
        'input_shape': input_shape,
        'gt_affs_shape': gt_affs_shape,
        'output_shape': output_shape,
    }

    with open(fn + '_config.json', 'w') as f:
        json.dump(config, f)
