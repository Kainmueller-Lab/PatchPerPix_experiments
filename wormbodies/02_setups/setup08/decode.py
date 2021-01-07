import logging
try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)
import numpy as np
import tensorflow as tf
import h5py
import zarr
import os
import toml

from PatchPerPix.models import autoencoder, FastPredict
from PatchPerPix.visualize import visualize_patches

logger = logging.getLogger(__name__)


def predict_input_fn(generator, input_shape):
    def _inner_input_fn():
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=tf.float32,
            output_shapes=(tf.TensorShape(input_shape))).batch(1)
        return dataset

    return _inner_input_fn


def decode_sample(decoder, sample, **kwargs):
    batch_size = kwargs['decode_batch_size']
    code_units = kwargs['code_units']
    patchshape = kwargs['patchshape']
    if type(patchshape) != np.ndarray:
        patchshape = np.array(patchshape)
    patchshape = patchshape[patchshape > 1]

    # load data depending on prediction.output_format and prediction.aff_key
    if "zarr" in kwargs['output_format']:
        pred_code = np.array(zarr.open(sample, 'r')[kwargs['code_key']])
        pred_fg = np.array(zarr.open(sample, 'r')[kwargs['fg_key']])
    elif "hdf" in kwargs['output_format']:
        with h5py.File(sample, 'r') as f:
            pred_code = np.array(f[kwargs['code_key']])
            pred_fg = np.array(f[kwargs['fg_key']])
    else:
        raise NotImplementedError("invalid input format")

    # check if fg is numinst with one channel per number instances [0,1,..]
    # heads up: assuming probabilities for numinst [0, 1, 2] in this order!
    if pred_fg.shape[0] > 1:
        pred_fg = np.any(np.array([
            pred_fg[i] >= kwargs['fg_thresh']
            for i in range(1, pred_fg.shape[0])
        ]), axis=0).astype(np.uint8)
    else:
        pred_fg = np.squeeze((pred_fg >= kwargs['fg_thresh']).astype(np.uint8), axis=0)

    fg_coords = np.transpose(np.nonzero(pred_fg))
    num_batches = int(np.ceil(fg_coords.shape[0] / float(batch_size)))
    logger.info("processing %i batches", num_batches)

    output = np.zeros((np.prod(patchshape),) + pred_fg.shape)

    for b in range(0, len(fg_coords), batch_size):
        fg_coords_batched = fg_coords[b:b + batch_size]
        fg_coords_batched = [(slice(None),) + tuple(
            [slice(i, i + 1) for i in fg_coord])
                             for fg_coord in fg_coords_batched]
        pred_code_batched = [pred_code[fg_coord].reshape((1, code_units))
                             for fg_coord in fg_coords_batched]
        if len(pred_code_batched) < batch_size:
            pred_code_batched = pred_code_batched + ([np.zeros(
                (1, code_units))] * (batch_size - len(pred_code_batched)))
        print('in decode sample: ', pred_code_batched[0].shape)
        predictions = decoder.predict(pred_code_batched)

        for idx, fg_coord in enumerate(fg_coords_batched):
            prediction = predictions[idx]
            output[fg_coord] = np.reshape(
                prediction['affinities'],
                (np.prod(prediction['affinities'].shape), 1, 1)
            )
    return output


def decoder_model_fn(features, labels, mode, params):
    if mode != tf.estimator.ModeKeys.PREDICT:
        raise RuntimeError("invalid tf estimator mode %s", mode)

    logger.info("feature tensor: %s", features)
    logger.info("label tensor: %s", labels)

    ae_config = params['included_ae_config']

    is_training = False
    code = tf.reshape(features, (-1,) + params['input_shape'])
    dummy_in = tf.placeholder(
        tf.float32, [None, ] + ae_config['patchshape'])
    input_shape = tuple(p for p in ae_config['patchshape']
                        if p > 1)
    logits, _, _ = autoencoder(
        code,
        is_training=is_training,
        input_shape_squeezed=input_shape,
        only_decode=True,
        dummy_in=dummy_in,
        **ae_config
    )
    pred_affs = tf.sigmoid(logits, name="affinities")
    predictions = {
        "affinities": pred_affs,
    }
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


def decode(**kwargs):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(
        model_dir=kwargs['output_folder'],
        session_config=sess_config)

    decoder = tf.estimator.Estimator(model_fn=decoder_model_fn,
                                     params=kwargs, config=config)

    if kwargs['mode'] == tf.estimator.ModeKeys.PREDICT:
        decoder = FastPredict(decoder, predict_input_fn,
                              kwargs['checkpoint_file'], kwargs)

        for sample in kwargs['samples']:
            # decode each sample
            prediction = decode_sample(decoder, sample, **kwargs)

            # save prediction
            sample_name = os.path.basename(sample).split('.')[0]
            outfn = os.path.join(kwargs['output_folder'],
                                 sample_name + '.' + kwargs['output_format'])
            mode = 'a' if os.path.exists(outfn) else 'w'
            if kwargs['output_format'] == 'zarr':
                outf = zarr.open(outfn, mode=mode)
            elif kwargs['output_format'] == 'hdf':
                outf = h5py.File(outfn, mode)
            else:
                raise NotImplementedError
            outf.create_dataset(
                kwargs['aff_key'],
                data=prediction,
                compression='gzip'
            )
            if kwargs['output_format'] == 'hdf':
                outf.close()

            # visualize patches if given
            if kwargs.get('show_patches'):
                if sample_name in kwargs.get('samples_to_visualize', []):
                    outfn_patched = os.path.join(kwargs['output_folder'],
                                                 sample_name + '.hdf')
                    out_key = kwargs['aff_key'] + '_patched'
                    _ = visualize_patches(prediction, kwargs['patchshape'],
                                          out_file=outfn_patched,
                                          out_key=out_key)
