import logging
try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)

import gunpowder.nodes.add_affinities as gp
import h5py
import numpy as np
import tensorflow as tf
import zarr

from PatchPerPix.models import autoencoder

logger = logging.getLogger(__name__)


class FastPredict:

    def __init__(self, estimator, input_fn, checkpoint_path, params):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.input_fn = input_fn
        self.checkpoint_path = checkpoint_path
        self.params = params

    def _create_generator(self):
        while not self.closed:
            yield self.next_features

    def predict(self, feature_batch):
        """ Runs a prediction on a set of features. Calling multiple times
            does *not* regenerate the graph which makes predict much faster.
            feature_batch a list of list of features.
            IMPORTANT: If you're only classifying 1 thing,
            you still need to make it a batch of 1 by wrapping it in a list
            (i.e. predict([my_feature]), not predict(my_feature)
        """
        self.next_features = feature_batch
        if self.first_run:
            self.batch_size = len(feature_batch)
            self.predictions = self.estimator.predict(
                input_fn=self.input_fn(
                    self._create_generator,
                    (-1, 1) + self.params['input_shape']),
                checkpoint_path=self.checkpoint_path)
            self.first_run = False
        elif self.batch_size != len(feature_batch):
            raise ValueError("All batches must be of the same size."
                             " First-batch:" + str(self.batch_size) +
                             " This-batch:" + str(len(feature_batch)))

        results = []
        for _ in range(self.batch_size):
            results.append(next(self.predictions))
        return results

    def close(self):
        self.closed = True
        try:
            next(self.predictions)
        except:
            print("Exception in fast_predict. This is probably OK")


def predict_input_fn(generator, input_shape):
    def _inner_input_fn():
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(tf.float32),
            output_shapes=(tf.TensorShape(input_shape))).batch(1)
        return dataset

    return _inner_input_fn


def autoencoder_input_fn(mode, params):
    print(mode, params)

    def get_sample(it):
        rnd_sample = np.random.randint(len(samples))
        arr = samples[rnd_sample]

        rnd_pos = np.random.randint(fg_coords[rnd_sample].shape[0])
        rnd_pos = fg_coords[rnd_sample][rnd_pos]

        if params['overlapping_inst'] and len(arr.shape) == 3:
            seg_to_affgraph_fun = gp.seg_to_affgraph_2d_multi
            arr = arr[:,
                      slice(rnd_pos[0]-rad[0], rnd_pos[0]+rad[0]+1),
                      slice(rnd_pos[1]-rad[1], rnd_pos[1]+rad[1]+1)]
        elif len(arr.shape) == 2:
            seg_to_affgraph_fun = gp.seg_to_affgraph_2d
            arr = arr[slice(rnd_pos[0]-rad[0], rnd_pos[0]+rad[0]+1),
                      slice(rnd_pos[1]-rad[1], rnd_pos[1]+rad[1]+1)]
        elif params['overlapping_inst'] and len(arr.shape) == 4:
            raise NotImplementedError
            seg_to_affgraph_fun = gp.seg_to_affgraph_multi
            arr = arr[:,
                      slice(rnd_pos[0]-rad[0], rnd_pos[0]+rad[0]+1),
                      slice(rnd_pos[1]-rad[1], rnd_pos[1]+rad[1]+1),
                      slice(rnd_pos[2]-rad[2], rnd_pos[2]+rad[2]+1)]
        else:
            seg_to_affgraph_fun = gp.seg_to_affgraph
            arr = arr[slice(rnd_pos[0]-rad[0], rnd_pos[0]+rad[0]+1),
                      slice(rnd_pos[1]-rad[1], rnd_pos[1]+rad[1]+1),
                      slice(rnd_pos[2]-rad[2], rnd_pos[2]+rad[2]+1)]
        affinities = seg_to_affgraph_fun(
            arr,
            np.array(neighborhood)
        ).astype(np.uint8)
        if len(rad) == 2:
            sample = affinities[:, rad[0], rad[1]]
        else:
            sample = affinities[:, rad[0], rad[1], rad[2]]
        sample.shape = patchshape
        return sample, sample

    def generator_fn():
        it = 0
        while True:
            sample, label = get_sample(it)
            it += 1

            if it >= params['max_samples']:
                raise StopIteration
            yield sample, label

    patchshape = params['patchshape']
    patchstride = params['patchstride']
    rad = np.array([p // 2 for p in patchshape])
    if rad[0] == 0:
        rad = rad[1:]
        patchshape = patchshape[1:]
        patchstride = patchstride[1:]

    neighborhood = []
    if len(rad) == 2:
        for i in range(-rad[0], rad[0]+1, patchstride[0]):
            for j in range(-rad[1], rad[1]+1, patchstride[1]):
                neighborhood.append([i, j])
    else:
        for i in range(-rad[0], rad[0]+1, patchstride[0]):
            for j in range(-rad[1], rad[1]+1, patchstride[1]):
                for k in range(-rad[2], rad[2]+1, patchstride[2]):
                    neighborhood.append([i, j, k])

    samples = []
    fg_coords = []
    for sample in params['samples']:
        if "zarr" in params['input_format']:
            arr = np.array(zarr.open(sample, 'r')[params['gt_key']])
        elif "hdf" in params['input_format']:
            with h5py.File(sample, 'r') as f:
                arr = np.array(f[params['gt_key']])
        else:
            raise NotImplementedError("invalid input format")

        fg = 1 * (np.sum(arr, axis=0))

        if arr.shape[0] == 1:
            arr.shape = arr.shape[1:]

        samples.append(arr)
        radslice = tuple([slice(rad[i], fg.shape[i] - rad[i])
                          for i in range(len(rad))])
        mask = np.ones(fg.shape, np.bool)
        mask[radslice] = 0
        fg[mask] = 0
        fg_coords.append(np.argwhere(fg > 0))

    input_shape = params['input_shape']
    logger.info("input shape: %s", input_shape)

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        (tf.float32, tf.float32),
        (tf.TensorShape(input_shape), tf.TensorShape(input_shape)))

    dataset = dataset.batch(params['batch_size']).prefetch(8)
    return dataset


def autoencoder_model_fn(features, labels, mode, params):
    if mode != tf.estimator.ModeKeys.EVAL and \
       mode != tf.estimator.ModeKeys.PREDICT:
        raise RuntimeError("invalid tf estimator mode %s", mode)

    logger.info("feature tensor: %s", features)
    logger.info("label tensor: %s", labels)

    is_training = False
    gt_affs = tf.reshape(features, (-1, 1) + params['input_shape'])
    input_shape = tuple(p for p in params['input_shape'] if p > 1)
    logits, _, _ = autoencoder(gt_affs, is_training=is_training,
                               input_shape_squeezed=input_shape, **params)
    pred_affs = tf.sigmoid(logits, name="affinities")

    predictions = {
        "affinities": pred_affs,
    }

    pred_affs = tf.layers.flatten(pred_affs)
    gt_affs = tf.layers.flatten(gt_affs)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    if params['loss_fn'] == "mse":
        lossFn = tf.losses.mean_squared_error
        # pred_cp = tf.sigmoid(pred_cp)
    elif params['loss_fn'] == "ce":
        lossFn = tf.losses.sigmoid_cross_entropy

    loss = lossFn(gt_affs, pred_affs,
                  reduction=tf.losses.Reduction.MEAN)
    eval_metric_ops = {
        "cos_dist": tf.metrics.mean_cosine_distance(
            labels=tf.math.l2_normalize(tf.layers.flatten(labels), axis=0),
            predictions=tf.math.l2_normalize(
                tf.layers.flatten(predictions["affinities"]), axis=0),
            dim=0)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def run(**kwargs):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(
        model_dir=kwargs['output_folder'],
        session_config=sess_config)

    autoencoder = tf.estimator.Estimator(
        model_fn=autoencoder_model_fn, params=kwargs, config=config)

    if kwargs['mode'] == tf.estimator.ModeKeys.PREDICT:
        autoencoder = FastPredict(autoencoder, predict_input_fn,
                                  kwargs['checkpoint_file'],
                                  kwargs)
        return autoencoder.predict
        # predictions = autoencoder.predict(
        #     input_fn=autoencoder_input_fn,
        #     checkpoint_path=kwargs['checkpoint_file'])
        # for p in predictions:
        #     print(p)
    elif kwargs['mode'] == tf.estimator.ModeKeys.EVAL:
        results = autoencoder.evaluate(
            input_fn=lambda mode, params: autoencoder_input_fn(
                mode, params),
            checkpoint_path=kwargs['checkpoint_file'])
        logger.info("Evaluation: %s", results)

        return results
