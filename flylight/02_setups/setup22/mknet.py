import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np
import json
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
                         shape=(kwargs['num_channels'],) + input_shape,
                         name="raw")
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
    if not kwargs['overlapping_inst']:
        num_inst_fmaps = 0
    elif kwargs['regress_num_inst']:
        num_inst_fmaps = 1
    else:
        num_inst_fmaps = kwargs['max_num_inst'] + 1

    model, _ = conv_pass(
        model,
        kernel_sizes=[1],
        num_fmaps=num_patch_fmaps + num_inst_fmaps,
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

    # placeholder for gt
    gt_affs_shape = pred_affs.get_shape().as_list()
    gt_affs = tf.placeholder(tf.float32, shape=gt_affs_shape, name='gt_affs')
    gt_numinst = tf.placeholder(tf.int32, shape=output_shape)
    gt_numinst = tf.clip_by_value(gt_numinst, 0, kwargs['max_num_inst'])

    # loss
    losspatch = tf.losses.sigmoid_cross_entropy(
        gt_affs,
        logitspatch,
        weights=tf.reshape(
            tf.to_float(gt_numinst < 2),
            [1, ] + list(gt_numinst.shape))
    )
    lossnuminst = tf.losses.sparse_softmax_cross_entropy(
        gt_numinst,
        tf.transpose(logitsnuminst, [1, 2, 3, 0]))

    loss = losspatch + lossnuminst

    # losspatch = tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=gt_affs,
    #     logits=tf.multiply(tf.to_float(gt_numinst < 2), logitspatch))

    # logitsnuminst = tf.transpose(logitsnuminst, perm=[1, 2, 3, 0])
    #
    # lossnuminst = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     labels=gt_numinst,
    #     logits=logitsnuminst)
    #
    # lossnuminst = tf.reshape(lossnuminst, shape=[1, ] + output_shape)
    # print('losspatch: ', losspatch.shape)
    # print('lossnuminst: ', lossnuminst.shape)

    tf.summary.scalar('losspatch', losspatch)
    tf.summary.scalar('lossnuminst', lossnuminst)

    print('losspatch: ', losspatch)
    print('lossnuminst: ', lossnuminst)
    print('loss: ', loss.shape)
    tf.summary.scalar('loss', loss)

    merged = tf.summary.merge_all(name="merged")

    learning_rate = tf.placeholder_with_default(kwargs['lr'], shape=(),
                                                name="learning-rate")
    # optimizer
    opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)

    optimizer = opt.minimize(loss)

    print("raw %s logits %s gt_affs %s loss %s" % (
        input_shape, output_shape,
        gt_affs.get_shape().as_list(),
        loss.get_shape().as_list()))

    fn = os.path.join(kwargs['output_folder'], kwargs['name'])
    tf.train.export_meta_graph(filename=fn + '.meta')

    names = {
        'raw': raw.name,
        'gt_affs': gt_affs.name,
        'gt_numinst': gt_numinst.name,
        'pred_affs': pred_affs.name,
        'pred_numinst': pred_numinst.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'merged': merged.name,
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

    num_trainable_parameters = np.sum([
        np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()
    ])
    print('total number variables: ', num_trainable_parameters)
    with open(fn + '_trainable_variables_' + str(num_trainable_parameters) +
              '_.txt', 'w') as f:
        f.write(
            'total number variables: ' + str(num_trainable_parameters) + '\n')
        for tensor in tf.trainable_variables():
            f.write(str(tensor))


if __name__ == "__main__":

    # call with experiment id and output folder base
    experiment = sys.argv[1]
    root = sys.argv[2]

    output_folder = os.path.join(root, experiment, 'train')
    try:
        os.makedirs(output_folder)
    except OSError:
        pass

    mk_net((140, 140, 140), 'train_net', output_folder)
    mk_net((180, 180, 180), 'test_net', output_folder)
