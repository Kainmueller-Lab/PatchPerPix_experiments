import tensorflow as tf
import numpy as np
import json
from PatchPerPix.models import autoencoder
import sys
import os


def mk_net(**kwargs):

    tf.reset_default_graph()

    input_shape = tuple(p for p in kwargs['input_shape'] if p > 1)
    gt_affs = tf.placeholder(tf.float32,
                             shape=(None,) +
                                   (np.prod(kwargs['input_shape']),) +
                                   (1,)*len(input_shape),
                             name="inputs")

    logits, sums, code = autoencoder(gt_affs,
                                     is_training=True,
                                     input_shape_squeezed=input_shape,
                                     **kwargs)
    print(logits)
    output_shape = logits.get_shape().as_list()[2:]

    pred_affs = tf.sigmoid(logits)

    anchor = tf.placeholder(tf.float32, shape=[None] + output_shape,
                            name="anchor")

    # loss
    if kwargs['loss_fn'] == "mse":
        lossFn = tf.losses.mean_squared_error
        # pred_cp = tf.sigmoid(pred_cp)
    elif kwargs['loss_fn'] == "ce":
        lossFn = tf.losses.sigmoid_cross_entropy

    # print_ops = None
    # print_ops = [tf.print("pred", pred_affs, gt_affs_patch)]
    # with tf.control_dependencies(print_ops):
    # loss = lossFn(gt_affs_patch, pred_affs)
                      # reduction=tf.losses.Reduction.MEAN)


    pred_affs = tf.reshape(pred_affs,
                           shape=(-1,) +
                                 (np.prod(kwargs['input_shape']),) +
                                 (1,)*len(input_shape),
                           name="affinities")
    l2_loss = tf.losses.get_regularization_loss()
    net_loss = lossFn(gt_affs, pred_affs)
    loss = net_loss + l2_loss
    loss_sums = []
    loss_sums.append(tf.summary.scalar('loss', net_loss))
    loss_sums.append(tf.summary.scalar('loss2', l2_loss))
    loss_sums.append(tf.summary.scalar('loss_sum', loss))
    summaries = tf.summary.merge(sums + loss_sums, name="summaries")

    # optimizer
    learning_rate = tf.placeholder_with_default(kwargs['lr'], shape=(),
                                                name="learning-rate")
    opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)

    global_step = tf.Variable(0, name="global_step", dtype=tf.int64)

    tf.train.export_meta_graph(filename=os.path.join(kwargs['output_folder'],
                                                     kwargs['name'] +'.meta'))

    fn = os.path.join(kwargs['output_folder'], kwargs['name'])
    names = {
        'pred_affs': pred_affs.name,
        'gt_affs': gt_affs.name,
        'anchor': anchor.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'summaries': summaries.name
    }

    with open(fn + '_names.json', 'w') as f:
        json.dump(names, f)

    config = {
        'input_shape': input_shape,
        'output_shape': output_shape,
    }

    with open(fn + '_config.json', 'w') as f:
        json.dump(config, f)

    encoder = {
        'inputs': [gt_affs.name],
        'outputs': [code.name],
    }

    with open(fn + '_encoder.json', 'w') as f:
        json.dump(encoder, f)

    decoder = {
        'inputs': [code.name],
        'outputs': [pred_affs.name],
    }

    with open(fn + '_decoder.json', 'w') as f:
        json.dump(decoder, f)
