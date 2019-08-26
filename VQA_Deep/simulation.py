from __future__ import absolute_import, division, print_function

import os

import collections
import numbers
import theano.tensor as T

from .config_parser import config_parser, dump_config
from .data_load.data_loader_IQA import DataLoader
from .trainer import Trainer
from .train_iqa import (init_patch_step, make_opt_config_list, create_model,
                        run_err_map_iw, run_nr_iqa_iw, run_nr_iqa_ref_iw)


def train_all(config_file, section, snap_path,
              output_path=None, snap_file=None, tr_te_file=None,
              epoch_1st=0, epoch_3rd=0, sens_err=True):
    """
    Train IQA model of all stages.
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
    # train_data, test_data = data_loader.load_data_tr_te(tr_te_file)
    # train_data, test_data = data_loader.load_toy_data_tr_te()

    # Create model
    opt_scheme_list, lr_list = make_opt_config_list(model_config)
    model = create_model(model_config,
                         train_data.patch_size, train_data.num_ch)
    if snap_file is not None:
        model.load(snap_file)

    # Create trainer
    trainer = Trainer(train_config, snap_path, output_path)

    # Store current configuration file
    dump_config(os.path.join(snap_path, 'config.yaml'),
                db_config, model_config, train_config)

    ###########################################################################
    # Train the model
    x_c = T.ftensor4('x_c')
    x = T.ftensor4('x')
    mos_set = T.vector('mos_set')
    bat2img_idx_set = T.imatrix('bat2img_idx_set')

    ###
    if epoch_1st > 0:
        prefix2 = 'ERR_'
        batch_size = 5
        model.set_opt_configs(opt_scheme=opt_scheme_list[0], lr=lr_list[0])

        score = run_err_map_iw(
            train_data, test_data, model, trainer, epoch_1st, batch_size,
            x_c=x_c, x=x, bat2img_idx_set=bat2img_idx_set, prefix2=prefix2)

        # Show information after train
        print("Best score0: {:.3f}, score1: {:.3f}, epoch: {:d}".format(
            score[0], score[1], score[2]))
        # prefix = train_config.get('prefix', '')
        # model.load(os.path.join(
        #     snap_path, prefix + prefix2 + 'snapshot_best.npy'))

    ###
    # Train NR-IQA
    if epoch_3rd > 0:
        prefix2 = 'NR_'
        # batch_size = train_config.get('batch_size', 8)
        batch_size = 5
        model.set_opt_configs(opt_scheme=opt_scheme_list[1], lr=lr_list[1])

        if sens_err:
            score = run_nr_iqa_ref_iw(
                train_data, test_data, model, trainer, epoch_3rd, batch_size,
                x_c=x_c, x=x, mos_set=mos_set, bat2img_idx_set=bat2img_idx_set,
                prefix2=prefix2)
        else:
            score = run_nr_iqa_iw(
                train_data, test_data, model, trainer, epoch_3rd, batch_size,
                x_c=x_c, mos_set=mos_set, bat2img_idx_set=bat2img_idx_set,
                prefix2=prefix2)

        # Show information after train
        print("Best SRCC: {:.3f}, PLCC: {:.3f}, epoch: {:d}".format(
            score[0], score[1], score[2]))


def train_iter_hyper(config_file, section, snap_path,
                     output_path=None, snap_file=None,
                     tr_te_file=None, n_data=1250, hparam_iter={},
                     epoch_1st=0, epoch_3rd=0, sens_err=True):
    """
    Imagewise training of an IQA model using both reference and
    distorted images.
    """
    db_config, model_config, train_config = config_parser(
        config_file, section)

    # Check snapshot file
    if snap_file is not None:
        assert os.path.isfile(snap_file), \
            'Not existing snap_file: %s' % snap_file

    # Check hyper parameter dictionary
    hparam_iter = collections.OrderedDict(sorted(hparam_iter.items()))
    n_hyp = len(list(hparam_iter))
    hparam_list = []

    def recur_add(hparam_dict, key_idx, tmp_dict):
        if key_idx < n_hyp:
            cur_key = list(hparam_dict)[key_idx]
            cur_hparams = hparam_dict[cur_key]
            for cur_val in cur_hparams:
                tmp_dict[cur_key] = cur_val
                recur_add(hparam_dict, key_idx + 1, tmp_dict)
        else:
            hparam_list.append(tmp_dict.copy())
            return

    recur_add(hparam_iter, 0, {})

    hp_key_val_strs = []
    hp_key_val_strs_show = []
    hp_val_strs = []
    for hparam_dict in hparam_list:
        # Set hyper parameter and make each name string
        key_val_str_list = []
        val_str_list = []
        for key in list(hparam_dict):
            if isinstance(hparam_dict[key], numbers.Number):
                key_val_str_list.append(
                    "%s=%.2e" % (key, hparam_dict[key]))
                val_str_list.append("%.2e" % (hparam_dict[key]))
            elif isinstance(hparam_dict[key], bool):
                key_val_str_list.append(
                    "%s=%r" % (key, hparam_dict[key]))
                val_str_list.append("%r" % (hparam_dict[key]))
            else:
                key_val_str_list.append(
                    "%s=" % key + hparam_dict[key])
                val_str_list.append(hparam_dict[key])
        hp_key_val_strs.append('.'.join(key_val_str_list))
        hp_key_val_strs_show.append(', '.join(key_val_str_list))
        hp_val_strs.append(', '.join(val_str_list))

    print('Simulating hyper-parameter list:')
    for idx, hparam_str in enumerate(hp_key_val_strs_show):
        print('  %d: %s' % (idx + 1, hparam_str))
    print('')

    # Initialize patch step
    init_patch_step(db_config, int(model_config.get('ign', 0)),
                    int(model_config.get('ign_scale', 1)))

    # Load data
    data_loader = DataLoader(db_config)
    if tr_te_file is not None:
        train_data, test_data = data_loader.load_data_tr_te(tr_te_file)
    else:
        train_data, test_data = data_loader.load_data_rand(n_data)
    # train_data, test_data = data_loader.load_toy_data_tr_te()

    # Create model
    opt_scheme_list, lr_list = make_opt_config_list(model_config)
    model = create_model(model_config,
                         train_data.patch_size, train_data.num_ch)
    if snap_file is not None:
        model.load(snap_file)

    # Save initial parameters for the reset after each try
    if not os.path.isdir(snap_path):
        os.makedirs(snap_path)
    init_snap_file = os.path.join(snap_path, 'init_params.npy')
    model.save(init_snap_file)

    # Store current configuration file
    dump_config(os.path.join(snap_path, 'config.yaml'),
                db_config, model_config, train_config)

    # Create trainer
    trainer = Trainer(train_config, snap_path, output_path)

    ###########################################################################
    # Train the model
    x = T.ftensor4('x')
    x_c = T.ftensor4('x_c')
    mos_set = T.vector('mos_set')
    bat2img_idx_set = T.imatrix('bat2img_idx_set')

    with open(os.path.join(snap_path, 'simulation.txt'), 'a') as f_hp:
        data = ', '.join(list(hparam_iter))
        data += ', score0, score1, best_epoch\n'
        f_hp.write(data)

    for hp_idx, hparam_dict in enumerate(hparam_list):
        print("\n(ITER %d/%d): %s" % (
            hp_idx + 1, len(hparam_list), hp_key_val_strs_show[hp_idx]))

        # Set hyper parameter
        for key in list(hparam_dict):
            model_config[key] = hparam_dict[key]
        model.set_configs(model_config)

        hp_key_val_str = hp_key_val_strs[hp_idx]
        hp_val_str = hp_val_strs[hp_idx]

        # Restore model parameters to the un-trained state
        model.load(init_snap_file)

        # Set each output path for trainer
        trainer.set_path(os.path.join(snap_path, '%s/' % (hp_key_val_str)))

        # Train local metric scores
        if epoch_1st > 0:
            prefix2 = 'ERR_'
            batch_size = 5
            # model.set_opt_configs(opt_scheme=opt_scheme_list[0], lr=lr_list[0])

            score = run_err_map_iw(
                train_data, test_data, model, trainer, epoch_1st, batch_size,
                x_c=x_c, x=x, bat2img_idx_set=bat2img_idx_set, prefix2=prefix2)

        # Train NR-IQA
        if epoch_3rd > 0:
            prefix2 = 'NR_'
            # batch_size = train_config.get('batch_size', 8)
            batch_size = 5
            # model.set_opt_configs(opt_scheme=opt_scheme_list[1], lr=lr_list[1])

            if sens_err:
                score = run_nr_iqa_ref_iw(
                    train_data, test_data, model, trainer, epoch_3rd,
                    batch_size, x_c=x_c, x=x, mos_set=mos_set,
                    bat2img_idx_set=bat2img_idx_set, prefix2=prefix2)
            else:
                score = run_nr_iqa_iw(
                    train_data, test_data, model, trainer, epoch_3rd,
                    batch_size, x_c=x_c, mos_set=mos_set,
                    bat2img_idx_set=bat2img_idx_set, prefix2=prefix2)

        # Show information after train
        print("Best score0: {:.3f}, score1: {:.3f}, epoch: {:d}".format(
            score[0], score[1], score[2]))
        print("(%s)" % (hp_key_val_str))

        # Write log
        with open(os.path.join(snap_path, 'simulation.txt'), 'a') as f_hp:
            data = hp_val_str
            data += ', {:.4f}, {:.4f}, {:d}\n'.format(
                score[0], score[1], score[2])
            f_hp.write(data)


def train_over_snaps(config_file, section, snap_path,
                     output_path=None, snap_files=[],
                     tr_te_file=None, n_data=1250, hparam_iter={},
                     epoch_1st=0, epoch_3rd=0, sens_err=True):
    """
    Imagewise training of an IQA model using both reference and
    distorted images.
    """
    db_config, model_config, train_config = config_parser(
        config_file, section)

    # Check snapshot files
    print('Simulating snapshot list:')
    assert snap_files
    snap_has_none = False
    for idx, snap_file in enumerate(snap_files):
        if snap_file is None:
            snap_has_none = True
            print('  %d: %s' % (idx + 1, "None"))
            continue
        assert os.path.isfile(snap_file), \
            'Not existing snap_file: %s' % snap_file
        print('  %d: %s' % (idx + 1, snap_file))
    print('')

    if not os.path.isdir(snap_path):
        os.makedirs(snap_path)
    with open(os.path.join(snap_path, 'loaded_snapshots.txt'), 'a') as f_info:
        data = ''
        for snap_idx, snap_file in enumerate(snap_files):
            data += '%02d, %s\n' % (snap_idx + 1, snap_file)
        f_info.write(data)

    # Check hyper parameter dictionary
    hparam_iter = collections.OrderedDict(sorted(hparam_iter.items()))
    n_hyp = len(list(hparam_iter))
    hparam_list = []

    # Make all possible combinations of hyper parameters
    def recur_add(hparam_dict, key_idx, tmp_dict):
        if key_idx < n_hyp:
            cur_key = list(hparam_dict)[key_idx]
            cur_hparams = hparam_dict[cur_key]
            for cur_val in cur_hparams:
                tmp_dict[cur_key] = cur_val
                recur_add(hparam_dict, key_idx + 1, tmp_dict)
        else:
            hparam_list.append(tmp_dict.copy())
            return

    recur_add(hparam_iter, 0, {})

    hp_key_val_strs = []
    hp_key_val_strs_show = []
    hp_val_strs = []
    for hparam_dict in hparam_list:
        # Set hyper parameter and make each name string
        key_val_str_list = []
        val_str_list = []
        for key in list(hparam_dict):
            if isinstance(hparam_dict[key], numbers.Number):
                key_val_str_list.append(
                    "%s=%.2e" % (key, hparam_dict[key]))
                val_str_list.append("%.2e" % hparam_dict[key])
            elif isinstance(hparam_dict[key], bool):
                key_val_str_list.append(
                    "%s=%r" % (key, hparam_dict[key]))
                val_str_list.append("%r" % hparam_dict[key])
            elif isinstance(hparam_dict[key], (list, tuple)):
                key_val_str_list.append(
                    "%s=[%s]" % (key, ', '.join(hparam_dict[key])))
                val_str_list.append("[%s]" % ', '.join(hparam_dict[key]))
            else:
                key_val_str_list.append(
                    "%s=" % key + hparam_dict[key])
                val_str_list.append(hparam_dict[key])
        hp_key_val_strs.append('.'.join(key_val_str_list))
        hp_key_val_strs_show.append(', '.join(key_val_str_list))
        hp_val_strs.append(', '.join(val_str_list))

    print('Simulating hyper-parameter list:')
    for idx, hparam_str in enumerate(hp_key_val_strs_show):
        print('  %d: %s' % (idx + 1, hparam_str))
    print('')

    # Initialize patch step
    init_patch_step(db_config, int(model_config.get('ign', 0)),
                    int(model_config.get('ign_scale', 1)))

    # Load data
    data_loader = DataLoader(db_config)
    if tr_te_file is not None:
        train_data, test_data = data_loader.load_data_tr_te(tr_te_file)
    else:
        train_data, test_data = data_loader.load_data_rand(n_data)
    # train_data, test_data = data_loader.load_toy_data_tr_te()

    # Create model
    opt_scheme_list, lr_list = make_opt_config_list(model_config)
    model = create_model(model_config,
                         train_data.patch_size, train_data.num_ch)

    if snap_has_none:
        init_snap_file = os.path.join(snap_path, 'init_params.npy')
        model.save(init_snap_file)

    ###########################################################################
    # Start simulation over snap files
    for snap_idx, snap_file in enumerate(snap_files):
        print("\nSNAPSHOT %d/%d: %s" % (
            snap_idx + 1, len(snap_files), snap_file))

        # Save initial parameters for the reset after each try
        sub_snap_path = os.path.join(
            snap_path, 'snapshot_%02d/' % (snap_idx + 1))
        if not os.path.isdir(sub_snap_path):
            os.makedirs(sub_snap_path)

        # Store current configuration file
        dump_config(os.path.join(sub_snap_path, 'config.yaml'),
                    db_config, model_config, train_config)

        # Create trainer
        trainer = Trainer(train_config, sub_snap_path, output_path)

        #######################################################################
        # Train the model
        x_c = T.ftensor4('x_c')
        x = T.ftensor4('x')
        mos_set = T.vector('mos_set')
        bat2img_idx_set = T.imatrix('bat2img_idx_set')

        with open(os.path.join(snap_path, 'simul_snaps.txt'), 'a') as f_hp:
            if snap_file is not None:
                data = 'Snapshot %02d: ' % (snap_idx + 1) + snap_file + '\n'
            else:
                data = 'Snapshot %02d: ' % (snap_idx + 1) + 'None' + '\n'
            data += ', '.join(list(hparam_iter))
            data += ', score0, score1, best_epoch\n'
            f_hp.write(data)

        for hp_idx, hparam_dict in enumerate(hparam_list):
            print("\n(ITER %d/%d): %s" % (
                hp_idx + 1, len(hparam_list), hp_key_val_strs_show[hp_idx]))

            # Set hyper parameter
            for key in list(hparam_dict):
                model_config[key] = hparam_dict[key]
            model.set_configs(model_config)

            hp_key_val_str = hp_key_val_strs[hp_idx]
            hp_val_str = hp_val_strs[hp_idx]

            # Restore model parameters to the un-trained state
            if snap_file is None:
                model.load(init_snap_file)
            else:
                model.load(snap_file)

            # Set each output path for trainer
            trainer.set_path(
                os.path.join(sub_snap_path, '%s/' % (hp_key_val_str)))

            # Train local metric scores
            if epoch_1st > 0:
                prefix2 = 'ERR_'
                batch_size = 5
                # model.set_opt_configs(
                #     opt_scheme=opt_scheme_list[0], lr=lr_list[0])

                score = run_err_map_iw(
                    train_data, test_data, model, trainer, epoch_1st,
                    batch_size,
                    x_c=x_c, x=x, bat2img_idx_set=bat2img_idx_set,
                    prefix2=prefix2)

            # Train NR-IQA
            if epoch_3rd > 0:
                prefix2 = 'NR_'
                # batch_size = train_config.get('batch_size', 8)
                batch_size = 5
                # model.set_opt_configs(
                #     opt_scheme=opt_scheme_list[1], lr=lr_list[1])

                if sens_err:
                    score = run_nr_iqa_ref_iw(
                        train_data, test_data, model, trainer, epoch_3rd,
                        batch_size, x_c=x_c, x=x, mos_set=mos_set,
                        bat2img_idx_set=bat2img_idx_set, prefix2=prefix2)
                else:
                    score = run_nr_iqa_iw(
                        train_data, test_data, model, trainer, epoch_3rd,
                        batch_size, x_c=x_c, mos_set=mos_set,
                        bat2img_idx_set=bat2img_idx_set, prefix2=prefix2)

            # Show information after train
            print("Best score0: {:.3f}, score1: {:.3f}, epoch: {:d}".format(
                score[0], score[1], score[2]))
            print("(%s)" % (hp_key_val_str))

            # Write log
            with open(os.path.join(snap_path, 'simul_snaps.txt'), 'a') as f_hp:
                data = hp_val_str + ', {:.4f}, {:.4f}, {:d}\n'.format(
                    score[0], score[1], score[2])
                f_hp.write(data)

        with open(os.path.join(snap_path, 'simul_snaps.txt'), 'a') as f_hp:
            f_hp.write('\n')
