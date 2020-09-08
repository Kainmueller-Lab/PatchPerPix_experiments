import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

# os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_ADD'] = 'Conv3D,Conv3DBackpropFilter,Conv3DBackpropFilterV2,Conv3DBackpropInput,Conv3DBackpropInputV2,Conv2D,Conv2DBackpropFilter,Conv2DBackpropFilterV2,Conv2DBackpropInput,Conv2DBackpropInputV2'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_VMODULE'] = 'amp_optimizer=2'
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

import argparse
from copy import deepcopy
from datetime import datetime
from glob import glob
import fnmatch
import functools
import importlib
import itertools
import logging

try:
    import absl.logging

    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)
import multiprocessing
import operator
import os
import queue
import random
import runpy
import shutil
import sys
import time

import h5py
import zarr
from joblib import Parallel, delayed
from natsort import natsorted
import numpy as np
import toml
import json

from PatchPerPix import util
from PatchPerPix.evaluate import evaluate_patch, evaluate_numinst, evaluate_fg

from PatchPerPix import vote_instances as vi
from evaluateInstanceSegmentation import evaluate_file, summarize_metric_dict
from PatchPerPix.visualize import visualize_patches, visualize_instances


def merge_dicts(sink, source):
    if not isinstance(sink, dict) or not isinstance(source, dict):
        raise TypeError('Args to merge_dicts should be dicts')

    for k, v in source.items():
        if isinstance(source[k], dict) and isinstance(sink.get(k), dict):
            sink[k] = merge_dicts(sink[k], v)
        else:
            sink[k] = v

    return sink


def backup_and_copy_file(source, target, fn):
    target = os.path.join(target, fn)
    if os.path.exists(target):
        os.replace(target, target + "_backup" + str(int(time.time())))
    if source is not None:
        source = os.path.join(source, fn)
        shutil.copy2(source, target)

def check_file(fn, remove_on_error=False, key=None):
    if fn.endswith("zarr"):
        try:
            fl = zarr.open(fn, 'r')
            if key is not None:
                tmp = fl[key]
            return True
        except Exception as e:
            logger.info("%s", e)
            if remove_on_error:
                shutil.rmtree(fn, ignore_errors=True)
            return False
    elif fn.endswith("hdf"):
        try:
            with h5py.File(fn, 'r') as fl:
                if key is not None:
                    tmp = fl[key]
            return True
        except Exception as e:
            if remove_on_error:
                os.remove(fn)
            return False
    else:
        raise NotImplementedError("invalid file type")

def time_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = datetime.now()
        ret = func(*args, **kwargs)
        logger.info('time %s: %s', func.__name__, str(datetime.now() - t0))
        return ret

    return wrapper


def fork(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.info("forking %s", func)
            p = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
            p.start()
            p.join()
            if p.exitcode != 0:
                raise RuntimeError("child process died")
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            p.terminate()
            p.join()
            os._exit(-1)

    return wrapper


def fork_return(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.info("forking %s", func)
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=func,
                                        args=args + (q,), kwargs=kwargs)
            p.start()
            results = None
            while p.is_alive():
                try:
                    results = q.get_nowait()
                except queue.Empty:
                    time.sleep(2)
            if p.exitcode == 0 and results is None:
                results = q.get()
            p.join()
            if p.exitcode != 0:
                raise RuntimeError("child process died")
            return results
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            p.terminate()
            p.join()
            os._exit(-1)

    return wrapper


logger = logging.getLogger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', action='append',
                        help=('Configuration files to use. For defaults, '
                              'see `config/default.toml`.'))
    parser.add_argument('-a', '--app', dest='app', required=True,
                        help=('Application to use. Choose out of cityscapes, '
                              'flylight, kaggle, etc.'))
    parser.add_argument('-r', '--root', dest='root', default=None,
                        help='Experiment folder to store results.')
    parser.add_argument('-s', '--setup', dest='setup', default=None,
                        help='Setup for experiment.', required=True)
    parser.add_argument('-id', '--exp-id', dest='expid', default=None,
                        help='ID for experiment.')

    # action options
    parser.add_argument('-d', '--do', dest='do', default=[], nargs='+',
                        choices=['all',
                                 'mknet',
                                 'train',
                                 'predict',
                                 'decode',
                                 'label',
                                 'infer',
                                 'validate_checkpoints',
                                 'validate',
                                 'postprocess',
                                 'evaluate',
                                 'cross_validate',
                                 'visualize'
                                 ],
                        help='Task to do for experiment.')

    parser.add_argument('--test-checkpoint', dest='test_checkpoint',
                        default='last', choices=['last', 'best'],
                        help=('Specify which checkpoint to use for testing. '
                              'Either last or best (checkpoint validation).'))

    parser.add_argument('--checkpoint', dest='checkpoint', default=None,
                        type=int,
                        help='Specify which checkpoint to use.')

    parser.add_argument("--run_from_exp", action="store_true",
                        help='run from setup or from experiment folder')
    parser.add_argument("--validate_on_train", action="store_true",
                        help=('validate using training data'
                              '(to check for overfitting)'))

    # train / val / test datasets
    parser.add_argument('--input-format', dest='input_format',
                        choices=['hdf', 'zarr', 'n5', 'tif'],
                        help='File format of dataset.')
    parser.add_argument('--train-data', dest='train_data', default=None,
                        help='Train dataset to use.')
    parser.add_argument('--val-data', dest='val_data', default=None,
                        help='Validation dataset to use.')
    parser.add_argument('--test-data', dest='test_data', default=None,
                        help='Test dataset to use.')

    # parameters for vote instances
    parser.add_argument('--vote-instances-cuda', dest='vote_instances_cuda',
                        action='store_true',
                        help='Determines if CUDA should be used to process '
                             'vote instances.')
    parser.add_argument('--vote-instances-blockwise',
                        dest='vote_instances_blockwise',
                        action='store_true',
                        help='Determines if vote instances should be '
                             'processed blockwise.')

    parser.add_argument("--debug_args", action="store_true",
                        help=('Set some low values to certain'
                              ' args for debugging.'))

    parser.add_argument("--predict_single", action="store_true",
                        help=('predict a single sapmle, for testing'))
    parser.add_argument("--term_after_patch_graph", action="store_true",
                        help=('terminate after patch graph, to split into GPU and CPU parts'))
    parser.add_argument("--graph_to_inst", action="store_true",
                        help=('only do patch graph to inst part of vote_instances'))
    parser.add_argument('--sample', default=None,
                        help='Sample to process.')

    args = parser.parse_args()

    return args


def create_folders(args, filebase):
    # create experiment folder
    os.makedirs(filebase, exist_ok=True)

    if args.expid is None and args.run_from_exp:
        setup = os.path.join(args.app, '02_setups', args.setup)
        backup_and_copy_file(setup, filebase, 'train.py')
        backup_and_copy_file(setup, filebase, 'mknet.py')
        try:
            backup_and_copy_file(setup, filebase, 'predict.py')
        except FileNotFoundError:
            pass
        try:
            backup_and_copy_file(setup, filebase, 'label.py')
        except FileNotFoundError:
            pass
        try:
            backup_and_copy_file(setup, filebase, 'decode.py')
        except FileNotFoundError:
            pass

    # create train folders
    train_folder = os.path.join(filebase, 'train')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'snapshots'), exist_ok=True)

    # create val folders
    val_folder = os.path.join(filebase, 'val')
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'instanced'), exist_ok=True)

    # create test folders
    test_folder = os.path.join(filebase, 'test')
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'instanced'), exist_ok=True)

    return train_folder, val_folder, test_folder


def update_config(args, config):
    if args.train_data is not None:
        config['data']['train_data'] = args.train_data

    if args.val_data is not None:
        config['data']['val_data'] = args.val_data

    if args.test_data is not None:
        config['data']['test_data'] = args.test_data

    if args.input_format is not None:
        config['data']['input_format'] = args.input_format
    if 'input_format' not in config['data']:
        raise ValueError("Please provide data/input_format in cl or config")

    if args.validate_on_train:
        config['data']['validate_on_train'] = True
    else:
        config['data']['validate_on_train'] = False

    if args.vote_instances_cuda:
        config['vote_instances']['cuda'] = True

    if args.vote_instances_blockwise:
        config['vote_instances']['blockwise'] = True


def setDebugValuesForConfig(config):
    config['training']['max_iterations'] = 10
    config['training']['checkpoints'] = 10
    config['training']['snapshots'] = 10
    config['training']['profiling'] = 10
    config['training']['num_workers'] = 1
    config['training']['cache_size'] = 1


@fork
@time_func
def mknet(args, config, train_folder, test_folder):
    if args.run_from_exp:
        mk_net_fn = runpy.run_path(
            os.path.join(config['base'], 'mknet.py'))['mk_net']
    else:
        mk_net_fn = importlib.import_module(
            args.app + '.02_setups.' + args.setup + '.mknet').mk_net

    mk_net_fn(name=config['model']['train_net_name'],
              input_shape=config['model']['train_input_shape'],
              output_folder=train_folder,
              autoencoder=config.get('autoencoder'),
              **config['data'],
              **config['model'],
              **config['optimizer'],
              debug=config['general']['debug'])
    mk_net_fn(name=config['model']['test_net_name'],
              input_shape=config['model']['test_input_shape'],
              output_folder=test_folder,
              autoencoder=config.get('autoencoder'),
              **config['data'],
              **config['model'],
              **config['optimizer'],
              debug=config['general']['debug'])


@fork
@time_func
def train(args, config, train_folder):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError("no free GPU available!")

    data_files = get_list_train_files(config)
    if args.run_from_exp:
        train_fn = runpy.run_path(
            os.path.join(config['base'], 'train.py'))['train_until']
    else:
        train_fn = importlib.import_module(
            args.app + '.02_setups.' + args.setup + '.train').train_until

    train_fn(name=config['model']['train_net_name'],
             max_iteration=config['training']['max_iterations'],
             output_folder=train_folder,
             data_files=data_files,
             **config['data'],
             **config['model'],
             **config['training'],
             **config['optimizer'],
             **config.get('preprocessing', {}))


def get_list_train_files(config):
    data = config['data']['train_data']
    if os.path.isfile(data):
        files = [data]
    elif os.path.isdir(data):
        if 'folds' in config['training']:
            files = glob(os.path.join(
                data + "_folds" + config['training']['folds'],
                "*." + config['data']['input_format']))
        elif config['data'].get('sub_folders', False):
            files = glob(os.path.join(
                data, "*", "*." + config['data']['input_format']))
        else:
            files = glob(os.path.join(data,
                                      "*." + config['data']['input_format']))
    else:
        raise ValueError(
            "please provide file or directory for data/train_data", data)
    return files


def get_list_samples(config, data, file_format, filter=None):
    logger.info("reading data from %s", data)
    # read data
    if os.path.isfile(data):
        if file_format == ".hdf" or file_format == "hdf":
            with h5py.File(data, 'r') as f:
                samples = [k for k in f]
        else:
            NotImplementedError("Add reader for %s format",
                                os.path.splitext(data)[1])
    elif data.endswith('.zarr'):
        samples = [os.path.basename(data).split(".")[0]]
    elif os.path.isdir(data):
        samples = fnmatch.filter(os.listdir(data),
                                 '*.' + file_format)
        samples = [os.path.splitext(s)[0] for s in samples]
        if not samples:
            for d in os.listdir(data):
                tmp = fnmatch.filter(os.listdir(os.path.join(data, d)),
                                     '*.' + file_format)
                tmp = [os.path.join(d, os.path.splitext(s)[0]) for s in tmp]
                samples += tmp
    else:
        raise NotImplementedError("Data must be file or directory")
    print(samples)

    # read filter list
    if filter is not None:
        if os.path.isfile(filter):
            if filter.endswith(".hdf"):
                with h5py.File(filter, 'r') as f:
                    filter_list = [k for k in f]
            else:
                NotImplementedError("Add reader for %s format",
                                    os.path.splitext(data)[1])
        elif filter.endswith('.zarr'):
            filter_list = [os.path.basename(filter).split(".")[0]]
        elif os.path.isdir(filter):
            filter_list = fnmatch.filter(os.listdir(filter), '*')
            filter_list = [os.path.splitext(s)[0] for s in filter_list]
            if not filter_list:
                for d in os.listdir(filter):
                    tmp = fnmatch.filter(os.listdir(os.path.join(filter, d)),
                                         '*.' + file_format)
                    tmp = [os.path.join(d, os.path.splitext(s)[0]) for s in tmp]
                    filter_list += tmp
        else:
            raise NotImplementedError("Data must be file or directory")
        print(filter_list)
        samples = [s for s in samples if s in filter_list]
    print(samples)
    return samples


@fork
@time_func
def predict_sample(args, config, name, data, sample, checkpoint, input_folder,
                   output_folder):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError("no free GPU available!")

    if args.run_from_exp:
        predict_fn = runpy.run_path(
            os.path.join(config['base'], 'predict.py'))['predict']
    else:
        predict_fn = importlib.import_module(
            args.app + '.02_setups.' + args.setup + '.predict').predict

    logger.info('predicting %s!', sample)
    predict_fn(name=name, sample=sample, checkpoint=checkpoint,
               data_folder=data, input_folder=input_folder,
               output_folder=output_folder,
               **config['data'],
               **config['model'],
               **config.get('preprocessing', {}),
               **config['prediction'])


@fork
@time_func
def predict_autoencoder(args, config, data, checkpoint, train_folder,
                        output_folder):
    import tensorflow as tf

    if args.run_from_exp:
        eval_predict_fn = runpy.run_path(
            os.path.join(config['base'], 'eval_predict.py'))['run']
    else:
        eval_predict_fn = importlib.import_module(
            args.app + '.02_setups.' + args.setup + '.eval_predict').run

    input_shape = tuple(p for p in config['model']['patchshape'] if p > 1)
    checkpoint_file = get_checkpoint_file(checkpoint,
                                          config['model']['train_net_name'],
                                          train_folder)
    samples = get_list_samples(config, data, config['data']['input_format'])
    if not os.path.isfile(data):
        for idx, s in enumerate(samples):
            samples[idx] = os.path.join(data,
                                        s + "." + config['data']['input_format'])

    eval_predict_fn(mode=tf.estimator.ModeKeys.PREDICT,
                    input_shape=input_shape,
                    max_samples=32,
                    checkpoint_file=checkpoint_file,
                    output_folder=output_folder,
                    samples=samples,
                    **config['model'],
                    **config['data'],
                    **config['prediction'],
                    **config['visualize'])


@time_func
def predict(args, config, name, data, checkpoint, test_folder, output_folder):
    samples = get_list_samples(config, data, config['data']['input_format'])

    if args.sample is not None:
        samples = [s for s in samples if args.sample in s]

    for idx, sample in enumerate(samples):
        fl = os.path.join(output_folder,
                          sample + '.' + config['prediction']['output_format'])
        if not config['general']['overwrite'] and os.path.exists(fl):
            key = 'aff_key' if not config['training'].get('train_code') else 'code_key'
            if check_file(
                    fl, remove_on_error=False,
                    key=config['prediction'].get(key, "volumes/pred_affs")):
                logger.info('Skipping prediction for %s. Already exists!',
                            sample)
                if args.predict_single:
                    break
                else:
                    continue
            else:
                logger.info('prediction %s broken. recomputing..!',
                            sample)

        if args.debug_args and idx >= 2:
            break

        predict_sample(args, config, name, data, sample, checkpoint,
                       test_folder, output_folder)
        if args.predict_single:
            break


@fork
def decode(args, config, data, checkpoint, pred_folder, output_folder):
    in_format = config['prediction']['output_format']
    samples = get_list_samples(config, pred_folder, in_format, data)

    if args.sample is not None:
        samples = [s for s in samples if args.sample in s]

    to_be_skipped = []
    for sample in samples:
        pred_file = os.path.join(output_folder, sample + '.' + in_format)
        if not config['general']['overwrite'] and os.path.exists(pred_file):
            if check_file(pred_file, remove_on_error=False,
                          key=config['prediction'].get('aff_key',
                                                       "volumes/pred_affs")):
                logger.info('Skipping decoding for %s. Already exists!', sample)
                to_be_skipped.append(sample)
    for sample in to_be_skipped:
        samples.remove(sample)
    if len(samples) == 0:
        return

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError("no free GPU available!")
    import tensorflow as tf
    for idx, s in enumerate(samples):
        samples[idx] = os.path.join(pred_folder, s + "." + in_format)

    if args.run_from_exp:
        decode_fn = runpy.run_path(
            os.path.join(config['base'], 'decode.py'))['decode']
    else:
        decode_fn = importlib.import_module(
            args.app + '.02_setups.' + args.setup + '.decode').decode

    if config['model'].get('code_units'):
        input_shape = (config['model'].get('code_units'),)
    else:
        input_shape = None

    decode_fn(
        mode=tf.estimator.ModeKeys.PREDICT,
        input_shape=input_shape,
        checkpoint_file=checkpoint,
        output_folder=output_folder,
        samples=samples,
        included_ae_config=config.get('autoencoder'),
        **config['model'],
        **config['prediction'],
        **config['visualize'],
        **config['data']
    )


def get_checkpoint_file(iteration, name, train_folder):
    return os.path.join(train_folder, name + '_checkpoint_%d' % iteration)


def get_checkpoint_list(name, train_folder):
    checkpoints = natsorted(glob(
        os.path.join(train_folder, name + '_checkpoint_*.index')))
    return [int(os.path.splitext(os.path.basename(cp))[0].split("_")[-1])
            for cp in checkpoints]


def select_validation_data(config, train_folder, val_folder):
    if config['data'].get('validate_on_train'):
        if 'folds' in config['training']:
            data = config['data']['train_data'] + \
                   "_folds" + str(config['training']['folds'])
        else:
            data = config['data']['train_data']
        output_folder = train_folder
    else:
        if 'fold' in config['validation']:
            data = config['data']['val_data'] + \
                   "_fold" + str(config['validation']['fold'])
        else:
            data = config['data']['val_data']
        output_folder = val_folder
    return data, output_folder


@time_func
def validate_checkpoint(args, config, data, checkpoint, params, train_folder,
                        test_folder, output_folder):
    logger.info("validating checkpoint %d %s", checkpoint, params)

    # create test iteration folders
    params_str = [k + "_" + str(v).replace(".", "_")
                  for k, v in params.items()]
    pred_folder = os.path.join(output_folder, 'processed', str(checkpoint))

    inst_folder = os.path.join(output_folder, 'instanced', str(checkpoint),
                               *params_str)
    eval_folder = os.path.join(output_folder, 'evaluated', str(checkpoint),
                               *params_str)
    os.makedirs(pred_folder, exist_ok=True)
    os.makedirs(inst_folder, exist_ok=True)
    os.makedirs(eval_folder, exist_ok=True)

    # predict val data
    checkpoint_file = get_checkpoint_file(checkpoint,
                                          config['model']['train_net_name'],
                                          train_folder)
    logger.info("predicting checkpoint %d", checkpoint)
    # predict and evaluate autoencoder separately
    if args.app == "autoencoder":
        metric = evaluate_autoencoder(args, config,
                                      config['data']['test_data'],
                                      checkpoint, train_folder, eval_folder)
        logger.info("%s checkpoint %6d: %.4f (%s)",
                    config['evaluation']['metric'], checkpoint, metric, params)
        return metric
    # predict other apps
    predict(args, config, config['model']['test_net_name'], data,
            checkpoint_file, test_folder, pred_folder)

    # if ppp learns code
    if config['training'].get('train_code'):
        autoencoder_chkpt = config['model']['autoencoder_chkpt']
        if autoencoder_chkpt == "this":
            autoencoder_chkpt = checkpoint_file
        decode(args, config, data, autoencoder_chkpt, pred_folder, pred_folder)

    if config['evaluation'].get('prediction_only'):
        metric = evaluate_prediction(
            args, config, data, pred_folder, eval_folder)
        return metric

    # vote instances
    logger.info("vote_instances checkpoint %d %s", checkpoint, params)
    vote_instances(args, config, data, pred_folder, inst_folder)
    if args.term_after_patch_graph:
        exit(0)

    # evaluate
    logger.info("evaluating checkpoint %d %s", checkpoint, params)
    metric = evaluate(args, config, data, inst_folder, eval_folder)
    logger.info("%s checkpoint %6d: %.4f (%s)",
                config['evaluation']['metric'], checkpoint, metric, params)

    return metric


def get_postprocessing_params(config, params_list, test_config):
    params = {}
    for p in params_list:
        if config is None or config[p] == []:
            params[p] = [test_config[p]]
        else:
            params[p] = config[p]

    return params


def named_product(**items):
    if items:
        names = items.keys()
        vals = items.values()
        for res in itertools.product(*vals):
            yield dict(zip(names, res))
    else:
        yield {}

def named_zip(**items):
    if items:
        names = items.keys()
        vals = items.values()
        for res in zip(*vals):
            yield dict(zip(names, res))
    else:
        yield {}


def validate_checkpoints(args, config, data, checkpoints, train_folder,
                         test_folder, output_folder):
    # validate all checkpoints and return best one
    metrics = []
    ckpts = []
    params = []
    results = []
    param_sets = list(named_product(
        **get_postprocessing_params(
            config['validation'],
            config['validation'].get('params', []),
            config['vote_instances']
        )))

    for checkpoint in checkpoints:
        for param_set in param_sets:
            val_config = deepcopy(config)
            for k in param_set.keys():
                val_config['vote_instances'][k] = param_set[k]

            metric = validate_checkpoint(args, val_config, data, checkpoint,
                                         param_set, train_folder, test_folder,
                                         output_folder)
            metrics.append(metric)
            ckpts.append(checkpoint)
            params.append(param_set)
            results.append({'checkpoint': checkpoint,
                            'metric': str(metric),
                            'params': param_set})

    for ch, metric, p in zip(ckpts, metrics, params):
        logger.info("%s checkpoint %6d: %.4f (%s)",
                    config['evaluation']['metric'], ch, metric, p)

    if config['general']['debug'] and None in metrics:
        logger.error("None in checkpoint found: %s (continuing with last)",
                     tuple(metrics))
        best_checkpoint = ckpts[-1]
        best_params = params[-1]
    else:
        best_checkpoint = ckpts[np.argmax(metrics)]
        best_params = params[np.argmax(metrics)]
    logger.info('best checkpoint: %d', best_checkpoint)
    logger.info('best params: %s', best_params)
    with open(os.path.join(output_folder, "results.json"), 'w') as f:
        json.dump(results, f)
    return best_checkpoint, best_params


@time_func
def vote_instances(args, config, data, pred_folder, output_folder):
    samples = get_list_samples(config, pred_folder,
                               config['prediction']['output_format'],
                               data)

    if args.sample is not None:
        samples = [s for s in samples if args.sample in s]

    # set vote instance parameter
    config['vote_instances']['check_required'] = False

    num_workers = config['vote_instances'].get("num_parallel_samples", 1)
    if num_workers > 1:
        def init(l):
            global mutex
            mutex = l

        mutex = multiprocessing.Lock()
        pool = multiprocessing.Pool(processes=num_workers,
                                    initializer=init, initargs=(mutex,))
        pool.map(functools.partial(vote_instances_sample, args, config,
                                   data, pred_folder, output_folder), samples)
        pool.close()
        pool.join()
    else:
        for sample in samples:
            vote_instances_sample_seq(args, config, data, pred_folder,
                                      output_folder, sample)
            if args.predict_single:
                break


# only fork if pool is not used
@fork
def vote_instances_sample_seq(args, config, data, pred_folder, output_folder,
                              sample):
    vote_instances_sample(args, config, data, pred_folder, output_folder,
                          sample)


def vote_instances_sample(args, config, data, pred_folder, output_folder,
                          sample):
    if config['data'].get('sub_folders', False):
        config['vote_instances']['result_folder'] = \
            os.path.join(output_folder,
                         os.path.basename(os.path.dirname(sample)))
    else:
        config['vote_instances']['result_folder'] = output_folder

    if args.term_after_patch_graph:
        config['vote_instances']['termAfterPatchGraph'] = True
    if args.graph_to_inst:
        config['vote_instances']['graphToInst'] = True

    # check if instance file already exists
    output_fn = os.path.join(
        config['vote_instances']['result_folder'],
        os.path.basename(sample) + '.' + config['vote_instances']['output_format'])
    if not config['general']['overwrite'] and os.path.exists(output_fn):
        if check_file(output_fn, remove_on_error=False,
                      key=config['evaluation']['res_key']):
            logger.info('Skipping vote instances for %s. Already exists!',
                        sample)
            return
        else:
            logger.info('vote instances %s broken. recomputing..!', sample)

    if config['vote_instances']['cuda'] and \
       'CUDA_VISIBLE_DEVICES' not in os.environ and \
       not config['vote_instances'].get('graphToInst', False):
        raise RuntimeError("no free GPU available!")

    if config['vote_instances']['cuda'] and \
        not config['vote_instances']['blockwise']:
        if config['vote_instances'].get("num_parallel_samples", 1) == 1:
            config['vote_instances']['mutex'] = multiprocessing.Lock()
        else:
            config['vote_instances']['mutex'] = mutex

    if config['vote_instances']['blockwise']:
        pred_file = os.path.join(
            pred_folder, sample + '.' + config['prediction']['output_format'])
        if config['vote_instances']['blockwise_old_stitch_fn']:
            fn = vi.ref_vote_instances_blockwise.main
        else:
            fn = vi.stitch_patch_graph.main
        fn(pred_file,
           **config['vote_instances'],
           **config['model'],
           **config['visualize'],
           aff_key=config['prediction'].get('aff_key'),
           fg_key=config['prediction'].get('fg_key'),
           )
    else:
        config['vote_instances']['affinities'] = os.path.join(
            pred_folder, sample + '.' + config['prediction'][
                'output_format'])
        vi.vote_instances.main(
            **config['vote_instances'],
            **config['model'],
            debug=config['general']['debug'],
            aff_key=config['prediction'].get('aff_key'),
            fg_key=config['prediction'].get('fg_key'),
            )


def evaluate_sample(config, args, data, sample, inst_folder, output_folder,
                    file_format):
    if os.path.isfile(data):
        gt_path = data
        gt_key = sample + "/gt"
    elif data.endswith(".zarr"):
        gt_path = data
        gt_key = config['data']['gt_key']
    else:
        gt_path = os.path.join(
            data, sample + "." + config['data']['input_format'])
        gt_key = config['data']['gt_key']

    sample_path = os.path.join(inst_folder, sample + "." + file_format)
    if config['vote_instances'].get('one_instance_per_channel'):
        if config['data'].get('one_instance_per_channel_gt'):
            gt_key = config['data'].get('one_instance_per_channel_gt')
            logger.info('call evaluation with key %s', gt_key)

    if config['evaluation'].get('evaluate_skeleton_coverage'):
        eval_skeleton_fn = importlib.import_module(
            args.app + '.03_evaluate.evaluate').evaluate_file
        return eval_skeleton_fn(sample_path,
                                config['evaluation']['res_key'],
                                gt_path,
                                output_folder=output_folder,
                                remove_small_comps=config['evaluation'][
                                    'remove_small_comps'],
                                show_tp=config['evaluation'].get('show_tp'),
                                save_postprocessed=config['evaluation'].get(
                                    'save_postprocessed'),
                                )

    else:
        return evaluate_file(
            sample_path, gt_path, res_key=config['evaluation']['res_key'],
            gt_key=gt_key, out_dir=output_folder, suffix="",
            foreground_only=config['evaluation'].get('foreground_only', False),
            debug=config['general']['debug'],
            overlapping_inst=config['vote_instances'].get(
                'one_instance_per_channel', False),
            use_linear_sum_assignment=config['evaluation'].get(
                'use_linear_sum_assignment', False),
            metric=config['evaluation'].get('metric', None),
            filterSz=config['evaluation'].get('filterSz', None),
    )


@fork_return
@time_func
def evaluate_autoencoder(args, config, data, checkpoint,
                         train_folder, output_folder, queue):
    import tensorflow as tf

    if args.run_from_exp:
        eval_predict_fn = runpy.run_path(
            os.path.join(config['base'], 'eval_predict.py'))['run']
    else:
        eval_predict_fn = importlib.import_module(
            args.app + '.02_setups.' + args.setup + '.eval_predict').run

    input_shape = tuple(p for p in config['model']['train_input_shape'] if p > 1)
    checkpoint_file = get_checkpoint_file(checkpoint,
                                          config['model']['train_net_name'],
                                          train_folder)

    samples = get_list_samples(config, data, config['data']['input_format'])
    for idx, s in enumerate(samples):
        samples[idx] = os.path.join(data,
                                    s + "." + config['data']['input_format'])
    results = eval_predict_fn(mode=tf.estimator.ModeKeys.EVAL,
                              input_shape=input_shape,
                              batch_size=config['evaluation']['batch_size'],
                              max_samples=config['evaluation']['max_samples'],
                              checkpoint_file=checkpoint_file,
                              output_folder=output_folder,
                              samples=samples,
                              **config['model'],
                              **config['data'])

    queue.put(results[config['evaluation']['metric']])


def evaluate_prediction_sample(args, config, sample, data, pred_folder,
                               file_format, output_folder):
    pred_fn = os.path.join(pred_folder, sample + '.' + file_format)
    if data.endswith('zarr') or data.endswith('hdf'):
        label_fn = data
        sample_gt_key = sample + '/' + config['data']['gt_key']
    else:
        label_fn = os.path.join(data, sample + '.' + args.input_format)
        sample_gt_key = config['data']['gt_key']
    metric = {}

    if config['evaluation'].get('eval_numinst_prediction'):
        numinst_metric = evaluate_numinst(
            pred_fn, label_fn,
            sample_gt_key=sample_gt_key,
            output_folder=output_folder,
            evaluate_skeleton_coverage=config['evaluation'].get(
                'evaluate_skeleton_coverage', False),
            **config['model'],
            **config['prediction'],
            **config['data']
        )
        metric.update(numinst_metric)

    if config['evaluation'].get('eval_fg_prediction'):
        fg_metric = evaluate_fg(
            pred_fn, label_fn,
            threshs=config['evaluation']['patch_thresholds'],
            remove_small_comps=config['evaluation'].get(
                'remove_small_comps', []),
            sample_gt_key=sample_gt_key,
            output_folder=output_folder,
            evaluate_skeleton_coverage=config['evaluation'].get(
                'evaluate_skeleton_coverage', False),
            **config['model'],
            **config['prediction'],
            **config['data']
        )
        metric.update(fg_metric)

    if config['evaluation'].get('eval_patch_prediction'):
        patch_metric = evaluate_patch(
            pred_fn, label_fn,
            threshs=config['evaluation']['patch_thresholds'],
            sample_gt_key=sample_gt_key,
            **config['model'],
            **config['prediction'],
            **config['data']
        )
        metric.update(patch_metric)
    return metric


def evaluate_prediction(args, config, data, pred_folder, output_folder):
    file_format = config['prediction']['output_format']
    samples = natsorted(get_list_samples(config, pred_folder, file_format))

    num_workers = config['evaluation'].get("num_workers", 1)
    if num_workers > 1:
        metric_dicts = Parallel(n_jobs=num_workers, backend='multiprocessing',
                                verbose=0)(
            delayed(evaluate_prediction_sample)(args, config, s, data,
                                                pred_folder, file_format,
                                                output_folder)
            for s in samples)
    else:
        metric_dicts = []
        for sample in samples:
            metric = evaluate_prediction_sample(
                args, config, sample, data, pred_folder, file_format,
                output_folder)
            metric_dicts.append(metric)
            print(metric)

    # key_list = []
    # for th in metric_dicts[0].keys():
    #     for k in metric_dicts[0][th].keys():
    #         key_list.append(str(th) + '.' + str(k))
    # print(key_list)

    summarize_metric_dict(
        metric_dicts,
        samples,
        config['evaluation']['summary'],
        os.path.join(output_folder, 'summary_prediction.csv')
    )

    metrics = []
    for metric_dict, sample in zip(metric_dicts, samples):
        if metric_dict is None:
            continue
        for k in config['evaluation']['metric'].split('.'):
            if k in metric_dict:
                metric_dict = metric_dict[k]
        if type(metric_dict) == dict:
            logger.info("%s sample has no overlap", sample)
        else:
            logger.info("%s sample %-19s: %.4f",
                        config['evaluation']['metric'], sample,
                        float(metric_dict))
            metrics.append(float(metric_dict))

    print(metrics)
    return np.mean(metrics)


@time_func
def evaluate(args, config, data, inst_folder, output_folder, return_avg=True):
    file_format = config['postprocessing']['watershed']['output_format']
    samples = natsorted(get_list_samples(config, inst_folder, file_format, data))

    if args.sample is not None:
        samples = [s for s in samples if args.sample in s]

    num_workers = config['evaluation'].get("num_workers", 1)
    if num_workers > 1:
        metric_dicts = Parallel(n_jobs=num_workers, backend='multiprocessing',
                                verbose=0)(
            delayed(evaluate_sample)(config, args, data, s, inst_folder,
                                     output_folder, file_format)
            for s in samples)
    else:
        metric_dicts = []
        for sample in samples:
            metric_dict = evaluate_sample(config, args, data, sample,
                                          inst_folder, output_folder,
                                          file_format)
            metric_dicts.append(metric_dict)
            if args.predict_single:
                break

    metrics = {}
    metrics_full = {}
    num_gt = 0
    num_fscore = 0
    for metric_dict, sample in zip(metric_dicts, samples):
        if metric_dict is None:
            continue
        metrics_full[sample] = metric_dict
        if config['evaluation'].get('print_f_factor_perc_gt_0_8', False):
            num_gt += int(metric_dict['general']['Num GT'])
            num_fscore += int(
                metric_dict['confusion_matrix']['th_0_5']['Fscore_cnt'])
        for k in config['evaluation']['metric'].split('.'):
            metric_dict = metric_dict[k]
        logger.info("%s sample %-19s: %.4f",
                    config['evaluation']['metric'], sample, float(metric_dict))
        metrics[sample] = float(metric_dict)

    if 'summary' in config['evaluation'].keys():
        summarize_metric_dict(
            metric_dicts,
            samples,
            config['evaluation']['summary'],
            os.path.join(output_folder, 'summary.csv')
        )

    if config['evaluation'].get('print_f_factor_perc_gt_0_8', False):
        logger.info("fscore (at iou0.5) percent > 0.8: %.4f", num_fscore/num_gt)
    if return_avg:
        return np.mean(list(metrics.values()))
    else:
        return metrics, metrics_full


def visualize(args, config, pred_folder, inst_folder):
    if config['visualize'].get('show_patches'):
        samples = get_list_samples(config, pred_folder,
                                   config['prediction']['output_format'])
        aff_key = config['prediction'].get('aff_key', 'volumes/pred_affs')
        for sample in config['visualize'].get('samples_to_visualize', []):
            if sample in samples:
                infn = os.path.join(
                    pred_folder,
                    sample + '.' + config['prediction']['output_format'])
                outfn = os.path.join(
                    pred_folder,
                    sample + '.hdf')
                out_key = aff_key + '_patched'
                if not os.path.exists(outfn):
                    _ = visualize_patches(infn, config['model']['patchshape'],
                                          in_key=aff_key, out_file=outfn,
                                          out_key=out_key)
    if config['visualize'].get('show_instances'):
        param_sets = list(named_product(
            **get_postprocessing_params(
                config['validation'],
                config['validation'].get('params', []),
                config['vote_instances']
            )))
        print(param_sets)
        for param_set in param_sets:
            params_str = [k + "_" + str(v).replace(".", "_")
                          for k, v in param_set.items()]
            vis_config = deepcopy(config)
            for k in param_set.keys():
                vis_config['vote_instances'][k] = param_set[k]

            current_inst_folder = os.path.join(inst_folder, *params_str)
            if not os.path.exists(current_inst_folder):
                continue
            samples = get_list_samples(
                vis_config, current_inst_folder,
                vis_config['vote_instances']['output_format'])
            inst_key = 'vote_instances'

            if 'one_instance_per_channel' in param_set:
                max_axis = 0 if param_set['one_instance_per_channel'] else None
            else:
                if vis_config['vote_instances'].get('one_instance_per_channel'):
                    max_axis = 0
                else:
                    max_axis = None

            for sample in vis_config['visualize'].get('samples_to_visualize', []):
                if sample in samples:
                    infn = os.path.join(
                        current_inst_folder,
                        sample + '.' + vis_config['vote_instances']['output_format'])
                    outfn = os.path.join(
                        current_inst_folder,
                        sample + '.png')
                    if not os.path.exists(outfn):
                        visualize_instances(infn, inst_key, outfn,
                                            max_axis=max_axis)


@time_func
def cross_validate(args, config, data, train_folder, test_folder):
    # data = config['data']['test_data']
    num_variations = 0
    # print(config['cross_validate'])
    for k, v in config['cross_validate'].items():
        if k == 'checkpoints':
            continue
        elif num_variations == 0:
            num_variations = len(v)
        else:
            assert num_variations == len(v), \
                'number of values for parameters has to be fixed'
    for checkpoint in config['cross_validate']['checkpoints']:
        pred_folder = os.path.join(test_folder, 'processed', str(checkpoint))
        checkpoint_file = get_checkpoint_file(checkpoint,
                                              config['model']['train_net_name'],
                                              train_folder)
        predict(args, config, config['model']['test_net_name'], data,
                checkpoint_file, test_folder, pred_folder)
        # if ppp learns code
        if config['training'].get('train_code'):
            autoencoder_chkpt = config['model']['autoencoder_chkpt']
            if autoencoder_chkpt == "this":
                autoencoder_chkpt = checkpoint_file
            decode(args, config, data, autoencoder_chkpt, pred_folder,
                   pred_folder)

    params = config['validation'].get('params', [])
    results = {}
    for checkpoint in config['cross_validate']['checkpoints']:
        for i in range(num_variations):
            param_set = {}
            for p in params:
                param_set[p] = config['cross_validate'][p][i] \
                               if p in config['cross_validate'] \
                                  else config['vote_instances'][p]
            params_str = [k + "_" + str(v).replace(".", "_")
                          for k, v in param_set.items()]
            # change values in config
            print(param_set)
            val_config = deepcopy(config)
            for k in param_set.keys():
                val_config['vote_instances'][k] = param_set[k]
            pred_folder = os.path.join(test_folder, 'processed',
                                       str(checkpoint))
            inst_folder = os.path.join(test_folder, 'instanced',
                                       str(checkpoint), *params_str)
            os.makedirs(inst_folder, exist_ok=True)
            eval_folder = os.path.join(test_folder, 'evaluated',
                                       str(checkpoint), *params_str)
            os.makedirs(eval_folder, exist_ok=True)
            logger.info("vote instances: %s", param_set)
            print('call vi with ', pred_folder, inst_folder, param_set)
            vote_instances(args, val_config, data, pred_folder, inst_folder)
            metrics = evaluate(args, val_config, data, inst_folder, eval_folder,
                               return_avg=False)
            results[(checkpoint, *(param_set.values()))] = metrics

    samples = natsorted(get_list_samples(config, data, config['data']['input_format']))
    for k, v in results.items():
        assert len(v[0]) == len(samples)
        for s1, s2 in zip(v[0].keys(), samples):
            assert s1 == s2
    random.Random(42).shuffle(samples)
    samples_fold1 = set(samples[:len(samples)//2])
    samples_fold2 = set(samples[len(samples)//2:])
    results_fold1 = {}
    results_fold2 = {}
    for setup, result in results.items():
        acc = []
        for s in samples_fold1:
            acc.append(result[0][s])
        acc = np.mean(acc)
        results_fold1[setup] = acc

        acc = []
        for s in samples_fold2:
            acc.append(result[0][s])
        acc = np.mean(acc)
        results_fold2[setup] = acc

    best_setup_fold1 = max(results_fold1.items(), key=operator.itemgetter(1))[0]
    best_setup_fold2 = max(results_fold2.items(), key=operator.itemgetter(1))[0]

    acc1 = []
    for s in samples_fold2:
        acc1.append(results[best_setup_fold1][0][s])
    acc2 = []
    for s in samples_fold1:
        acc2.append(results[best_setup_fold2][0][s])

    acc = np.mean(acc1+acc2)
    acc1 = np.mean(acc1)
    acc2 = np.mean(acc2)

    logger.info("%s CROSS: %.4f [%.4f (%s), %.4f (%s)]",
                config['evaluation']['metric'], acc,
                acc1, best_setup_fold2,
                acc2, best_setup_fold1)
    print("%s CROSS: %.4f [%.4f (%s), %.4f (%s)]" % (
        config['evaluation']['metric'], acc,
        acc1, best_setup_fold2,
        acc2, best_setup_fold1))

    ap_ths = ["confusion_matrix.avAP",
              "confusion_matrix.th_0_5.AP",
              "confusion_matrix.th_0_6.AP",
              "confusion_matrix.th_0_7.AP",
              "confusion_matrix.th_0_8.AP",
              "confusion_matrix.th_0_9.AP"
              ]
    for ap_th in ap_ths:
        metric_dicts = results[best_setup_fold2][1]
        metrics = {}
        for sample, metric_dict in metric_dicts.items():
            if metric_dict is None:
                continue
            for k in ap_th.split('.'):
                metric_dict = metric_dict[k]
            metrics[sample] = float(metric_dict)
        metric = np.mean(list(metrics.values()))
        print("%s: %.4f" % (ap_th, metric))


def main():
    # parse command line arguments
    args = get_arguments()

    if not args.do:
        raise ValueError("Provide a task to do (-d/--do)")

    # get experiment name
    is_new_run = True
    if args.expid is not None:
        if os.path.isdir(args.expid):
            base = args.expid
            is_new_run = False
        else:
            base = os.path.join(args.root, args.expid)
    else:
        base = os.path.join(args.root,
                            args.app + '_' + args.setup + '_' + \
                            datetime.now().strftime('%y%m%d_%H%M%S'))

    # create folder structure for experiment
    if args.debug_args:
        base = base.replace("experiments", "experimentsTmp")
    train_folder, val_folder, test_folder = create_folders(args, base)

    # read config file
    if args.config is None and args.expid is None:
        raise RuntimeError("No config file provided (-c/--config)")
    elif args.config is None:
        args.config = [os.path.join(base, 'config.toml')]
    try:
        config = {}
        for conf in args.config:
            if not is_new_run:
                if os.path.dirname(os.path.abspath(conf)) != \
                   os.path.abspath(base):
                    raise RuntimeError("overwriting config with external config file (%s - %s)", conf, base)
            config = merge_dicts(config, toml.load(conf))
    except:
        raise IOError('Could not read config file: {}! Please check!'.format(
            conf))
    config['base'] = base

    # set logging level
    prefix = "" if args.sample is None else args.sample
    # prefix += str(config['validation']['patch_threshold'][0])
    logging.basicConfig(
        level=config['general']['logging'],
        handlers=[
            logging.FileHandler(os.path.join(base,
                                             prefix + "run.log"), mode='a'),
            logging.StreamHandler(sys.stdout)
        ])
    logger.info('attention: using config file %s', args.config)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        try:
            selectedGPU = util.selectGPU(
                quantity=config['training']['num_gpus'])
        except FileNotFoundError:
            selectedGPU = None
        if selectedGPU is None:
            logger.warning("no free GPU available!")
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(i) for i in selectedGPU])
        logger.info("setting CUDA_VISIBLE_DEVICES to device {}".format(
            selectedGPU))
    else:
        logger.info("CUDA_VISIBILE_DEVICES already set, device {}".format(
            os.environ["CUDA_VISIBLE_DEVICES"]))

    # update config with command line values
    update_config(args, config)
    backup_and_copy_file(None, base, 'config.toml')
    with open(os.path.join(base, "config.toml"), 'w') as f:
        toml.dump(config, f)
    if args.debug_args:
        setDebugValuesForConfig(config)
    logger.info('used config: %s', config)

    # create network
    if 'all' in args.do or 'mknet' in args.do:
        mknet(args, config, train_folder, test_folder)

    # train network
    if 'all' in args.do or 'train' in args.do:
        train(args, config, train_folder)

    # determine which checkpoint to use
    checkpoint = None
    if any(i in args.do for i in ['all', 'validate', 'predict', 'decode',
                                  'label', 'infer', 'evaluate', 'visualize',
                                  'postprocess']):
        if args.checkpoint is not None:
            checkpoint = int(args.checkpoint)
            checkpoint_path = os.path.join(
                train_folder, config['model']['train_net_name'], '_checkpoint_',
                str(checkpoint))

        elif args.test_checkpoint == 'last':
            with open(os.path.join(train_folder, 'checkpoint')) as f:
                d = dict(
                    x.rstrip().replace('"', '').replace(':', '').split(None, 1)
                    for x in f)
            checkpoint_path = d['model_checkpoint_path']
            try:
                checkpoint = int(checkpoint_path.split('_')[-1])
            except ValueError:
                logger.error('Could not convert checkpoint to int.')
                raise

        if checkpoint is None and \
           args.test_checkpoint != 'best' and \
           any(i in args.do for i in ['validate', 'predict', 'decode',
                                      'label', 'evaluate', 'infer']):
            raise ValueError(
                'Please provide a checkpoint (--checkpoint/--test-checkpoint)')

    params = None
    # validation:
    # validate all checkpoints
    if ([do for do in args.do if do in ['all', 'predict', 'label',
                                       'postprocess', 'evaluate']]\
        and args.test_checkpoint == 'best') \
        or 'validate_checkpoints' in args.do:
        data, output_folder = select_validation_data(config, train_folder,
                                                     val_folder)
        if args.checkpoint is not None:
            checkpoints = [int(args.checkpoint)]
        elif config['validation'].get('checkpoints'):
            checkpoints = config['validation']['checkpoints']
        else:
            checkpoints = get_checkpoint_list(config['model']['train_net_name'],
                                              train_folder)
        logger.info("validating all checkpoints")
        checkpoint, params = validate_checkpoints(args, config, data,
                                                  checkpoints,
                                                  train_folder, test_folder,
                                                  output_folder)
    # validate single checkpoint
    else:
        if 'validate' in args.do:
            if checkpoint is None:
                raise RuntimeError("checkpoint must be set but is None")
            data, output_folder = select_validation_data(config, train_folder,
                                                         val_folder)
            _ = validate_checkpoints(args, config, data, [checkpoint],
                                     train_folder, test_folder, output_folder)

    if [do for do in args.do if do in ['all', 'predict', 'decode', 'label',
                                       'postprocess', 'evaluate', 'infer']]:
        if checkpoint is None:
            raise RuntimeError("checkpoint must be set but is None")
        checkpoint_file = get_checkpoint_file(
            checkpoint, config['model']['train_net_name'], train_folder)

        if args.app == 'autoencoder':
            params = {}
        elif args.test_checkpoint != 'best':
            params = get_postprocessing_params(
                None,
                config['validation'].get('params', []),
                config['vote_instances'])
            for k,v in params.items():
                params[k] = v[0]
        else:
            # update config with "best" params
            for k in params.keys():
                config['vote_instances'][k] = params[k]

        params_str = [k + "_" + str(v).replace(".", "_")
                      for k, v in params.items()]
        pred_folder = os.path.join(test_folder, 'processed', str(checkpoint))
        inst_folder = os.path.join(test_folder, 'instanced', str(checkpoint),
                                   *params_str)
        eval_folder = os.path.join(test_folder, 'evaluated', str(checkpoint),
                                   *params_str)


    # predict test set
    if 'all' in args.do or 'predict' in args.do or 'infer' in args.do:

        # assume checkpoint has been determined already
        os.makedirs(pred_folder, exist_ok=True)

        logger.info("predicting checkpoint %d", checkpoint)
        if args.app == "autoencoder":
            predict_autoencoder(args, config, config['data']['test_data'],
                                checkpoint, train_folder, pred_folder)
        else:
            predict(args, config, config['model']['test_net_name'],
                    config['data']['test_data'], checkpoint_file,
                    test_folder, pred_folder)

    if 'all' in args.do or 'decode' in args.do or 'infer' in args.do:
        if config['training'].get('train_code'):
            autoencoder_chkpt = config['model'].get('autoencoder_chkpt')
            if autoencoder_chkpt == "this":
                autoencoder_chkpt = checkpoint_file
            decode(args, config, config['data']['test_data'],
                   autoencoder_chkpt, pred_folder, pred_folder)

    if 'all' in args.do or 'label' in args.do or 'infer' in args.do:
        os.makedirs(inst_folder, exist_ok=True)
        logger.info("vote_instances checkpoint %d", checkpoint)
        vote_instances(args, config, config['data']['test_data'], pred_folder,
                       inst_folder)

    if 'all' in args.do or 'postprocess' in args.do:
        # remove small components should go here?
        # what else?
        print('postprocess')
        if config['postprocessing'].get('process_fg_prediction', False):
            os.makedirs(inst_folder, exist_ok=True)
            samples = get_list_samples(config, pred_folder,
                                       config['prediction']['output_format'])
            samples = [os.path.join(pred_folder, s + '.' + config['prediction'][
                'output_format']) for s in samples]
            logger.info("postprocess checkpoint %d", checkpoint)
            util.postprocess_fg(
                samples, inst_folder,
                fg_key=config['prediction']['fg_key'],
                **config['postprocessing']
            )
        if config['postprocessing'].get('process_instances', False):
            samples = get_list_samples(
                config, inst_folder,
                config['vote_instances']['output_format'],
                config['data']['test_data']
            )
            samples = [os.path.join(inst_folder, s + '.' + config[
                'vote_instances']['output_format']) for s in samples]
            util.postprocess_instances(
                samples,
                inst_folder,
                res_key=config['evaluation']['res_key'],
                **config['postprocessing']
            )

    if 'all' in args.do or 'evaluate' in args.do or 'infer' in args.do:
        os.makedirs(eval_folder, exist_ok=True)
        logger.info("evaluating checkpoint %d", checkpoint)
        if args.app == "autoencoder":
            metric = evaluate_autoencoder(args, config,
                                          config['data']['test_data'],
                                          checkpoint,
                                          train_folder, eval_folder)
        else:
            metric = evaluate(args, config, config['data']['test_data'],
                              inst_folder, eval_folder)
        logger.info("%s TEST checkpoint %d: %.4f (%s)",
                    config['evaluation']['metric'], checkpoint, metric,
                    params)

    if 'visualize' in args.do:
        if checkpoint is None:
            raise RuntimeError("checkpoint must be set but is None")
        pred_folder = os.path.join(val_folder, 'processed', str(checkpoint))
        inst_folder = os.path.join(val_folder, 'instanced', str(checkpoint))

        visualize(args, config, pred_folder, inst_folder)

    if 'cross_validate' in args.do:
        cross_validate(args, config, config['data']['val_data'], train_folder, val_folder)


if __name__ == "__main__":
    main()
