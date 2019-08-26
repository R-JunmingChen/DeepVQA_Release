from __future__ import absolute_import, division, print_function

import os
import timeit
from importlib import import_module
import numpy as np
import theano
import theano.tensor as T
from .config_parser import config_parser, dump_config
from .data_load.data_loader_VQA import DataLoader
from .trainer import Trainer


def train_vqa(config_file, section, snap_path,
              output_path=None, snap_file=None, tr_te_file=None):
    """
    video-wise training of an VQA model using both reference and
    distorted videos.
    """
    db_config, model_config, train_config = config_parser(
        config_file, section)

    # Check snapshot file
    if snap_file is not None:
        assert os.path.isfile(snap_file), \
            'Not existing snap_file: %s' % snap_file

    # Initialize patch step
    init_patch_step(db_config, int(model_config.get('ign', 0)),
                    int(model_config.get('ign_scale', 1)))

    # Load data
    data_loader = DataLoader(db_config)
    train_data, test_data = data_loader.load_data_tr_te(tr_te_file)

    # Create model
    model = create_model(model_config,
                         train_data.patch_size, train_data.num_ch)
    if snap_file is not None:
        print('Loading weight file...')
        model.load_params_keys(['sens_map', 'reg_mos'], snap_file)

    # Create trainer
    trainer = Trainer(train_config, snap_path, output_path)

    # Store current configuration file
    dump_config(os.path.join(snap_path, 'config.yaml'),
                db_config, model_config, train_config)

    ###########################################################################
    # Train the model
    epochs = train_config.get('epochs', 100)
    batch_size = train_config.get('batch_size', 8)
    minibatch_size = train_config.get('minibatch_size', 8)

    score = []
    score = run_vqa_iw(
        train_data, test_data, model, trainer, epochs, batch_size, minibatch_size)
    print("Best SRCC: {:.4f}, PLCC: {:.4f} ({:d})".format(score[0][0], score[0][1], score[0][2]))
    print("Best PLCC: {:.4f}, PLCC: {:.4f} ({:d})".format(score[1][0], score[1][1], score[1][2]))
    print("mean of PLCC and SRCC best; PLCC: {:.4f}, PLCC: {:.4f} ({:d})".format(score[2][0], score[2][1], score[2][2]))

def run_vqa_iw(train_data, test_data, model, trainer, epochs, n_batch_vids, n_minibatch_frms,
               x_c=None, x=None, x_d_diff=None, x_r_diff=None, mos_set=None, bat2img_idx_set=None,
               prefix2='vqa_'):
    """
    @type model: .models.model_basis.ModelBasis
    @type train_data: .data_load.dataset.Dataset
    @type test_data: .data_load.dataset.Dataset
    """
    # Make dummy shared dataset
    max_num_frm = np.max(np.array(train_data.n_frms_idx))
    max_num_patch = np.max(np.array([np.array(l) for l in train_data.npat_vid_list])[1])
    n_pats_dummy = max_num_patch * n_batch_vids
    sh = model.input_shape
    np_set_r = np.zeros((max_num_frm, n_pats_dummy, sh[2], sh[3], sh[1]), dtype='float32')
    np_set_d = np.zeros((max_num_frm, n_pats_dummy, sh[2], sh[3], sh[1]), dtype='float32')
    np_set_r_diff = np.zeros((max_num_frm, n_pats_dummy, sh[2], sh[3], sh[1]), dtype='float32')
    np_set_d_diff = np.zeros((max_num_frm, n_pats_dummy, sh[2], sh[3], sh[1]), dtype='float32')

    shared_set_r = theano.shared(np_set_r, borrow=True)
    shared_set_d = theano.shared(np_set_d, borrow=True)
    shared_set_r_diff = theano.shared(np_set_r_diff, borrow=True)
    shared_set_d_diff = theano.shared(np_set_d_diff, borrow=True)

    train_data.set_imagewise()
    test_data.set_imagewise()

    print('\nCompile theano function: Regress on MOS', end='')
    print(' (videowise / low GPU memory)')
    start_time = timeit.default_timer()
    if x is None:
        x = T.ftensor5('x')
    if x_c is None:
        x_c = T.ftensor5('x_c')
    if x_d_diff is None:
        x_d_diff = T.ftensor5('x_d_diff')
    if x_r_diff is None:
        x_r_diff = T.ftensor5('x_r_diff')
    if mos_set is None:
        mos_set = T.vector('mos_set')
    if bat2img_idx_set is None:
        bat2img_idx_set = T.imatrix('bat2img_idx_set')

    print(' (Make training model)')
    model.set_training_mode(True)
    cost, updates, rec_train = model.cost_updates_vqa(
        x, x_c, x_d_diff, x_r_diff, mos_set, n_batch_vids, bat2img_idx_set)
    outputs = [cost] + rec_train.get_function_outputs(train=True)

    train_model = theano.function(
        inputs=[mos_set, bat2img_idx_set],
        outputs=[output for output in outputs],
        updates=updates,
        givens={
            x: shared_set_r,
            x_c: shared_set_d,
            x_d_diff: shared_set_d_diff,
            x_r_diff: shared_set_r_diff,
        },
        on_unused_input='warn'
    )

    print(' (Make testing model)')
    model.set_training_mode(False)
    cost, rec_test = model.cost_vqa(
        x, x_c, x_d_diff, x_r_diff, mos_set, n_img=n_batch_vids, bat2img_idx_set=bat2img_idx_set)
    outputs = [cost] + rec_test.get_function_outputs(train=False)

    test_model = theano.function(
        [mos_set, bat2img_idx_set],
        [output for output in outputs],
        givens={
            x: shared_set_r,
            x_c: shared_set_d,
            x_d_diff: shared_set_d_diff,
            x_r_diff: shared_set_r_diff,
        },
        on_unused_input='warn'
    )

    minutes, seconds = divmod(timeit.default_timer() - start_time, 60)
    print(' - Compilation took {:02.0f}:{:05.2f}'.format(minutes, seconds))

    def get_train_outputs():
        res = train_data.next_batch_video_diff(n_batch_vids, n_minibatch_frms)
        shared_set_r.set_value(res['ref_data'].transpose((0, 1, 4, 2, 3)))
        shared_set_d.set_value(res['dis_data'].transpose((0, 1, 4, 2, 3)))
        shared_set_d_diff.set_value(res['dis_diff_data'].transpose((0, 1, 4, 2, 3)))
        shared_set_r_diff.set_value(res['ref_diff_data'].transpose((0, 1, 4, 2, 3)))
        return train_model(res['score_set'], None)

    def get_test_outputs():
        res = test_data.next_batch_video_diff(n_batch_vids, n_minibatch_frms)
        shared_set_r.set_value(res['ref_data'].transpose((0, 1, 4, 2, 3)))
        shared_set_d.set_value(res['dis_data'].transpose((0, 1, 4, 2, 3)))
        shared_set_d_diff.set_value(res['dis_diff_data'].transpose((0, 1, 4, 2, 3)))
        shared_set_r_diff.set_value(res['ref_diff_data'].transpose((0, 1, 4, 2, 3)))
        return test_model(res['score_set'], None)

    # Main training routine
    return trainer.training_routine(
        model, get_train_outputs, rec_train, get_test_outputs, rec_test,
        n_batch_vids, n_batch_vids, train_data, test_data,
        epochs, prefix2, check_mos_corr=True)

def init_patch_step(db_config, ign_border, ign_scale=8):
    """
    Initialize patch_step:
    patch_step = patch_size - ign_border * ign_scale.
    """
    patch_size = db_config.get('patch_size', None)
    patch_step = db_config.get('patch_step', None)
    random_crops = int(db_config.get('random_crops', 0))

    if (patch_size is not None and patch_step is None and
            random_crops == 0):
        db_config['patch_step'] = (
            patch_size[0] - ign_border * ign_scale,
            patch_size[1] - ign_border * ign_scale)
        print(' - Set patch_step according to patch_size and ign: (%d, %d)' % (
            db_config['patch_step'][0], db_config['patch_step'][1]
        ))


def make_opt_config_list(model_config):
    opt_scheme = model_config.get('opt_scheme', 'adam')
    lr = model_config.get('lr', 1e-4)
    opt_scheme_list = []
    if isinstance(opt_scheme, str):
        opt_scheme_list.append(opt_scheme)
        opt_scheme_list.append(opt_scheme)
    elif isinstance(opt_scheme, (list, tuple)):
        for c_opt_scheme in opt_scheme:
            opt_scheme_list.append(c_opt_scheme)
    else:
        raise ValueError('Improper type of opt_scheme:', opt_scheme)
    lr_list = []
    if isinstance(lr, (list, tuple)):
        for c_lr in lr:
            lr_list.append(float(c_lr))
    else:
        lr_list.append(float(lr))
        lr_list.append(float(lr))

    model_config['opt_scheme'] = opt_scheme_list[0]
    model_config['lr'] = lr_list[0]

    return opt_scheme_list, lr_list


def create_model(model_config, patch_size=None, num_ch=None):
    """
    Create a model using a model_config.
    Set input_size and num_ch according to patch_size and num_ch.
    """
    model_module_name = model_config.get('model', None)
    assert model_module_name is not None
    model_module = import_module(model_module_name)

    # set input_size and num_ch according to dataset information
    if patch_size is not None:
        model_config['input_size'] = patch_size
    if num_ch is not None:
        model_config['num_ch'] = num_ch

    return model_module.Model(model_config)
