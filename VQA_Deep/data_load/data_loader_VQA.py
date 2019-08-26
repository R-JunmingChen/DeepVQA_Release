from __future__ import absolute_import, division, print_function

import os
import sys
import timeit
import numpy as np
from scipy.ndimage.filters import convolve
import math
from .dataset import Dataset

class DataLoader(object):
    """
    Arguments
        db_config: database configuration dictionary
    """
    def __init__(self, db_config):
        print('DataLoader')
        self.patch_size = None
        self.patch_step = None
        self.base_path = None
        self.n_images = 0
        self.n_videos = 0
        self.n_patches = 0
        self.npat_img_list = []
        self.npat_vid_list = []
        self.d_pat_set = []
        self.r_pat_set = []
        self.score_list = None
        self.fps_list = []
        self.n_frms_idx = []
        self.n_ref_frms_idx = []
        self.db_config = db_config
        self.n_frm_trte = 0
        if db_config is not None:
            self.set_config(db_config)

    def set_config(self, db_config):
        # Database
        self.sel_data = db_config.get('sel_data', None)
        self.scenes = db_config.get('scenes', 'all')
        self.dist_types = db_config.get('dist_types', 'all')
        self.select_database(self.sel_data, self.scenes, self.dist_types)

        # Initialize patch size
        self.patch_size = self.init_patch_size(
            db_config.get('patch_size', None))

        # Random crops
        self.random_crops = int(db_config.get('random_crops', 0))

        # If even patch division
        self.patch_step = self.init_patch_step(
            db_config.get('patch_step', None))
        self.patch_mode = db_config.get('patch_mode', None)
        assert self.patch_mode in ['both_side', 'shift_center', None]

        # Pre-processing
        self.color = db_config.get('color', 'gray')
        assert self.color in ['gray', 'rgb', 'ycbcr']
        self.num_ch = 1 if self.color == 'gray' else 3
        self.local_norm = db_config.get('local_norm', False)

        # etc.
        self.horz_ref = db_config.get('horz_ref', False)
        self.n_frm_trte = db_config.get('n_frm_trte', 50) ######new
        self.std_filt_r = db_config.get('std_filt_r', 1.0)
        self.train_size = db_config.get('train_size', 0.8)
        assert self.train_size > 0 and self.train_size < 1, \
            'train_size(%.2f) is not within 0 and 1' % (self.train_size)
        self.shuffle = db_config.get('shuffle', True)
        self.reverse_mos = db_config.get('reverse_mos', False)
        self.toy_example = db_config.get('toy_example', False)

    def init_patch_size(self, patch_size):
        # initialize patch size and step
        if patch_size:
            if isinstance(patch_size, int):
                patch_size_new = (patch_size, patch_size)
            elif isinstance(patch_size, (list, tuple)):
                assert len(patch_size) == 2
                patch_size_new = tuple(patch_size)
            else:
                raise ValueError('Wrong patch_size: {0}'.format(patch_size))
            self.use_original_size = False
        else:
            patch_size_new = None
            self.use_original_size = True

        return patch_size_new

    def init_patch_step(self, patch_step):
        # initialize patch step
        if not self.use_original_size:
            if isinstance(patch_step, int):
                patch_step_new = (patch_step, patch_step)
            elif isinstance(patch_step, (list, tuple)):
                assert len(patch_step) == 2
                patch_step_new = tuple(patch_step)
            elif patch_step is None:
                assert self.patch_size is not None
                patch_step_new = self.patch_size
            else:
                raise ValueError('Wrong patch_step:', patch_step)
        else:
            patch_step_new = (1, 1)
        return patch_step_new

    def select_database(self, sel_data, scenes, dist_types):
        """
        Select database to be loaded, and check scenes and dist_types.
        """
        if sel_data == 'LIVE_VQA':
            from . import LIVE_VQA
            self.DB_module = LIVE_VQA
        elif sel_data == 'CSIQ_VQA':
            from . import CSIQ_VQA
            self.DB_module = CSIQ_VQA
        else:
            raise ValueError('Improper sel_data: {0}'.format(sel_data))

        self.make_image_list_func = self.DB_module.make_image_list
        if scenes == 'all' or scenes is None:
            scenes = self.DB_module.ALL_SCENES
        if dist_types == 'all' or dist_types is None:
            dist_types = self.DB_module.ALL_DIST_TYPES

        self.scenes = scenes
        self.dist_types = dist_types
        self.sel_data = sel_data

        return scenes, dist_types

    def get_setting_dic(self):
        config_dict = {
            'sel_data': self.sel_data,
            'dist_types': self.dist_types,
            'patch_size': self.patch_size,
            'patch_step': self.patch_step,
            'random_crops': self.random_crops,
            'std_filt_r': self.std_filt_r,
            'horz_ref': self.horz_ref,
            'color': self.color,
            'local_norm': self.local_norm,
            'shuffle': self.shuffle,
            'train_size': self.train_size,
            'reverse_mos': self.reverse_mos,
            'n_images': self.n_images,
            'n_patches': self.n_patches,
        }
        return config_dict

    ###########################################################################

    def show_info(self):
        if self.patch_size is not None:
            print(' - Patch Size: (%d, %d)' % (
                self.patch_size[0], self.patch_size[1]))

        if self.random_crops > 0:
            print(' - Number of random crops: %d' % self.random_crops)
        else:
            print(' - Patch Step = (%d, %d)' % (
                self.patch_step[0], self.patch_step[1]), end='')
            print(' / mode: %s' % self.patch_mode)

        print(' - Color: %s' % self.color, end='')
        if self.local_norm:
            print(' (Local norm.)')
        else:
            print('')

        if self.std_filt_r < 1.0 and self.random_crops == 0:
            print(' - Patch sel. ratio (STD) =', self.std_filt_r)

        print(' - Augmentation :', end='')
        if self.horz_ref:
            print(' Horz. flip')
        else:
            print(' None')
        print(' - Reverse subj. score =', self.reverse_mos)

    ###########################################################################
    # Data loader interface
    def load_data_tr_te(self, tr_te_file=None, dataset_obj=False,
                        imagewise=True):
        """
        Load IQA database and divide into training and testing sets.
        """
        if self.toy_example:
            return self.load_toy_data_tr_te()
        print(' (Load training/testing data (shared ref.))')
        print(' - DB = %s' % (self.sel_data))
        train_scenes, test_scenes = self.divide_tr_te_wrt_ref(
            self.scenes, self.train_size, tr_te_file)

        self.show_info()

        # Get train set
        print('\n (Load training data)')
        data_dict = self.make_image_list_func(train_scenes, self.dist_types)
        self.load_ref_dis_images(data_dict)
        if self.horz_ref:
            self.data_augmentation_horz_refl()

        train_dataset = Dataset()
        train_dataset.put_data(
            self.d_pat_set,
            self.r_pat_set,
            self.d_pat_set_diff,
            self.r_pat_set_diff,
            self.dis2ref_idx,
            n_frms_idx=self.n_frms_idx,
            score_data=self.score_list,
            npat_img_list=self.npat_img_list,
            npat_vid_list=self.npat_vid_list,
            filt_idx_list=self.filt_idx_list,
            imagewise=imagewise, shuffle=self.shuffle)
        train_dataset.set_patch_config(
            self.patch_step, self.random_crops)

        # Get test set
        print('\n (Load testing data)')
        dist_type_test = self.dist_types
        data_dict = self.make_image_list_func(test_scenes, dist_type_test[1:])
        self.load_ref_dis_images(data_dict)

        test_dataset = Dataset()
        test_dataset.put_data(
            self.d_pat_set,
            self.r_pat_set,
            self.d_pat_set_diff,
            self.r_pat_set_diff,
            self.dis2ref_idx,
            n_frms_idx=self.n_frms_idx,
            score_data=self.score_list,
            npat_img_list=self.npat_img_list,
            npat_vid_list=self.npat_vid_list,
            filt_idx_list=self.filt_idx_list,
            imagewise=imagewise, shuffle=self.shuffle)
        test_dataset.set_patch_config(
            self.patch_step, self.random_crops)

        return train_dataset, test_dataset

    def load_data_rand(self, number, dataset_obj=False, imagewise=True):
        """
        Load IQA database (random images) and
        divide into training and testing sets.
        """
        print(' (Load training/testing data)', end='')
        print(' %d random videos (shared ref.)' % number)
        print(' - DB = %s' % (self.sel_data))
        scenes = self.DB_module.ALL_SCENES
        n_train_refs = int(np.ceil(number * self.train_size))
        n_test_refs = number - n_train_refs

        rand_seq = np.random.permutation(number)
        scenes_sh = [scenes[elem] for elem in rand_seq]
        train_scenes = sorted(scenes_sh[:n_train_refs])
        test_scenes = sorted(scenes_sh[n_train_refs:])

        print((' - Refs.: training  = %d / testing = %d (Ratio = %.2f)' %
               (n_train_refs, n_test_refs, self.train_size)))

        self.show_info()

        # Get train set
        data_dict = self.make_image_list_func(train_scenes, self.dist_types)
        self.load_ref_dis_images(data_dict)
        if self.horz_ref:
            self.data_augmentation_horz_refl()

        train_dataset = Dataset()
        train_dataset.put_data(
            self.d_pat_set, self.r_pat_set, self.dis2ref_idx,
            score_data=self.score_list,
            npat_img_list=self.npat_img_list,
            filt_idx_list=self.filt_idx_list,
            imagewise=imagewise, shuffle=self.shuffle)
        train_dataset.set_patch_config(
            self.patch_step, self.random_crops)

        # Get test set
        dist_type_test = self.dist_types
        data_dict = self.make_image_list_func(test_scenes, dist_type_test[1:])
        self.load_ref_dis_images(data_dict)

        test_dataset = Dataset()
        test_dataset.put_data(
            self.d_pat_set, self.r_pat_set, self.dis2ref_idx,
            score_data=self.score_list,
            npat_img_list=self.npat_img_list,
            filt_idx_list=self.filt_idx_list,
            imagewise=imagewise, shuffle=self.shuffle)
        test_dataset.set_patch_config(
            self.patch_step, self.random_crops)

        return train_dataset, test_dataset

    def load_data_for_test(self, tr_te_file, dataset_obj=False):
        """
        Load data with MOS and image data.
        There are no overlapping reference images between
        training and testing sets.
        """
        print(' - (Load testing data)')
        train_scenes, test_scenes = self.divide_tr_te_wrt_ref(
            self.scenes, self.train_size, tr_te_file)

        self.show_info()

        # Get test set
        # test set does not contain references for fair comparison
        dist_type_test = self.dist_types

        data_dict = self.make_image_list_func(test_scenes, dist_type_test[1:])
        self.load_ref_dis_images(data_dict)

        test_dataset = Dataset()
        test_dataset.put_data(
            self.d_pat_set, self.r_pat_set, self.dis2ref_idx,
            score_data=self.score_list,
            npat_img_list=self.npat_img_list,
            filt_idx_list=self.filt_idx_list,
            imagewise=True, shuffle=False)
        test_dataset.set_patch_config(
            self.patch_step, self.random_crops)

        return test_dataset

    def divide_tr_te_wrt_ref(self, scenes, train_size=0.8, tr_te_file=None):
        """
        Divdie data with respect to scene
        """
        tr_te_file_loaded = False
        if tr_te_file is not None and os.path.isfile(tr_te_file):
            # Load tr_te_file and divide scenes accordingly
            tr_te_file_loaded = True
            with open(tr_te_file, 'r') as f:
                train_scenes = f.readline().strip().split()
                train_scenes = [int(elem) for elem in train_scenes]
                test_scenes = f.readline().strip().split()
                test_scenes = [int(elem) for elem in test_scenes]

            n_train_refs = len(train_scenes)
            n_test_refs = len(test_scenes)
            train_size = (len(train_scenes) /
                          (len(train_scenes) + len(test_scenes)))
        else:
            # Divide scenes randomly
            # Get the numbers of training and testing scenes
            n_scenes = len(scenes)
            n_train_refs = int(np.ceil(n_scenes * train_size))
            n_test_refs = n_scenes - n_train_refs

            # Randomly divide scenes
            rand_seq = np.random.permutation(n_scenes)
            scenes_sh = [scenes[elem] for elem in rand_seq]
            train_scenes = sorted(scenes_sh[:n_train_refs])
            test_scenes = sorted(scenes_sh[n_train_refs:])

            # Write train-test idx list into file
            if tr_te_file is not None:
                fpath, fname = os.path.split(tr_te_file)
                if not os.path.isdir(fpath):
                    os.makedirs(fpath)
                with open(tr_te_file, 'w') as f:
                    for idx in range(n_train_refs):
                        f.write('%d ' % train_scenes[idx])
                    f.write('\n')
                    for idx in range(n_scenes - n_train_refs):
                        f.write('%d ' % test_scenes[idx])
                    f.write('\n')

        print(' - Refs.: training = %d / testing = %d (Ratio = %.2f)' %
              (n_train_refs, n_test_refs, train_size), end='')
        if tr_te_file_loaded:
            print(' (Loaded %s)' % (tr_te_file))
        else:
            print('')
        return train_scenes, test_scenes

    ###########################################################################

    def load_ref_dis_images(self, data_dict):
        self.score_list = data_dict['score_list']
        if self.reverse_mos and self.score_list is not None:
            self.score_list = 1.0 - self.score_list
        self.n_images = data_dict['n_images']
        self.n_videos = data_dict['n_videos']
        self.fps_list = data_dict['fps_list']
        base_path = data_dict['base_path']
        d_img_list = data_dict['d_img_list']
        r_img_list = data_dict['r_img_list']
        r_idx_list = data_dict['r_idx_list']
        scenes = data_dict['scenes']

        res = self.load_ref_frames(base_path, r_img_list, r_idx_list, scenes)
        ref_img2pat_idx, ref_top_left_set, ref_npat_img_list, inv_scenes, ref_total_frm2pat_idx = res
        self.load_dis_frames(
            base_path, d_img_list, r_idx_list, inv_scenes,
            ref_img2pat_idx, ref_top_left_set, ref_npat_img_list, ref_total_frm2pat_idx)

    def load_ref_frames(self, base_path, r_img_list, r_idx_list, scenes):
        """
        Actual routine of loading reference images.
        """
        self.n_ref_images = len(scenes)
        n_dis_images = len(r_img_list)

        # make a list of reference index
        ref_idx_idx_list = []
        for ref_idx in scenes:
            for idx, this_ref_idx in enumerate(r_idx_list):
                if ref_idx == this_ref_idx:
                    ref_idx_idx_list.append(idx)
                    break

                if idx == n_dis_images - 1:
                    raise ValueError('@ No %d index in r_idx_list' % ref_idx)

        new_r_img_list = []
        for idx in ref_idx_idx_list:
            new_r_img_list.append(r_img_list[idx])

        inv_scenes = np.ones(max(scenes) + 1, dtype='int32') * -1
        for idx, scn in enumerate(scenes):
            inv_scenes[scn] = idx

        patch_size = self.patch_size
        patch_step = self.patch_step
        n_frm_trte = self.n_frm_trte

        n_ref_patches = 0
        ref_npat_img_list = []
        ref_total_frm2pat_idx = []
        ref_top_left_set = []
        r_pat_set = []
        r_pat_set_diff = []
        n_ref_frms_idx = []
        #######################################################################
        # Read images
        start_time = timeit.default_timer()
        pass_list = []

        for vid_idx in range(self.n_ref_images):
            # Show progress
            show_progress(float(vid_idx) / self.n_ref_images)
        #######################################################################
        #video read
        #######################################################################
            ref_img2pat_idx = []
            start_frm = 5 # first 5 frms are ignored
            height = 432
            width = 768
            fwidth = (width + 31) // 32 * 32
            fheight = (height + 15) // 16 * 16

            end_frm = int(self.fps_list[ref_idx_idx_list[vid_idx]])
            n_ref_frms_idx.append(n_frm_trte)  #for same number of frms in video
            slide_frms = int(math.ceil((end_frm)/n_frm_trte)) # sliding window frm size
            vid_1 = []
            vid_2 = []
            stream = open('%s/%s' % (base_path, new_r_img_list[vid_idx]), 'rb')
            for i in range(start_frm, end_frm, slide_frms):
                # frame 1 read
                stream.seek(int(i * width * height * 1.5))
                Y = np.fromfile(stream, dtype=np.uint8, count=fwidth * fheight).reshape((fheight, fwidth))
                U = np.fromfile(stream, dtype=np.uint8, count=(fwidth// 2) * (fheight // 2))
                V = np.fromfile(stream, dtype=np.uint8, count=(fwidth// 2) * (fheight // 2))
                vid_1.append(Y)
                # frame 2 read
                Y = np.fromfile(stream, dtype=np.uint8, count=fwidth * fheight).reshape((fheight, fwidth))
                U = np.fromfile(stream, dtype=np.uint8, count=(fwidth// 2) * (fheight // 2))
                V = np.fromfile(stream, dtype=np.uint8, count=(fwidth// 2) * (fheight // 2))
                if end_frm > 300: # over 250frm +1frm
                    Y = np.fromfile(stream, dtype=np.uint8, count=fwidth * fheight).reshape((fheight, fwidth))
                    U = np.fromfile(stream, dtype=np.uint8, count=(fwidth// 2) * (fheight // 2))
                    V = np.fromfile(stream, dtype=np.uint8, count=(fwidth// 2) * (fheight // 2))
                    vid_2.append(Y)
                else:
                    vid_2.append(Y)
            vid_1 = np.array(vid_1)
            vid_2 = np.array(vid_2)

            for frm_no in range(n_frm_trte):
                r_frm_raw_1 = np.squeeze(vid_1[frm_no])
                r_frm_raw_1_norm = r_frm_raw_1.astype('float32') / 255.
                r_frm_raw_2 = np.squeeze(vid_2[frm_no])
                r_frm_raw_2_norm = r_frm_raw_2.astype('float32') / 255.
                #NOTE: abs is conducted in FR_SENS_VQA
                r_frm_diff = r_frm_raw_1_norm - r_frm_raw_2_norm
                cur_h = r_frm_raw_1.shape[0]
                cur_w = r_frm_raw_1.shape[1]

                if self.use_original_size:
                    patch_size = (cur_h, cur_w)
                # Gray or RGB
                if self.local_norm:
                    if self.color == 'gray':
                        # faster
                        r_img_norm = local_normalize_1ch(r_frm_raw_1)
                    else:
                        r_img_norm = local_normalize(r_frm_raw_1, self.num_ch)
                else:
                    r_img_norm = r_frm_raw_1.astype('float32') / 255.

                if self.color == 'gray':
                    r_img_norm = r_img_norm[:, :, None]
                    r_frm_diff = r_frm_diff[:, :, None]

                # numbers of patches along y and x axes
                ny = (cur_h - patch_size[0]) // patch_step[0] + 1
                nx = (cur_w - patch_size[1]) // patch_step[1] + 1
                npat = int(ny * nx)
                ref_npat_img_list.append((npat, ny, nx))
                ref_img2pat_idx.append(n_ref_patches + np.arange(npat))
                n_ref_patches += npat

                if npat == 0:
                    pass_list.append(vid_idx)
                    ref_top_left_set.append(False)
                    continue

                # get non-covered length along y and x axes
                cov_height = patch_step[0] * (ny - 1) + patch_size[0]
                cov_width = patch_step[1] * (nx - 1) + patch_size[1]
                nc_height = cur_h - cov_height
                nc_width = cur_w - cov_width

                # Shift center
                if self.patch_mode == 'shift_center':
                    shift = [(nc_height + 1) // 2, (nc_width + 1) // 2]
                    if shift[0] % 2 == 1:
                        shift[0] -= 1
                    if shift[1] % 2 == 1:
                        shift[1] -= 1
                    shift = tuple(shift)
                else:
                    shift = (0, 0)

                # generate top_left_set of patches
                top_left_set = np.zeros((nx * ny, 2), dtype=np.int)
                for yidx in range(ny):
                    for xidx in range(nx):
                        top = (yidx * patch_step[0] + shift[0])
                        left = (xidx * patch_step[1] + shift[1])
                        top_left_set[yidx * nx + xidx] = [top, left]
                ref_top_left_set.append(top_left_set)

                # Crop the images to patches
                for idx in range(ny * nx):
                    [top, left] = top_left_set[idx]

                    if top + patch_size[0] > cur_h:
                        print('\n@Error: imidx=%d, pat=%d' % (vid_idx, idx), end='')
                        print(' (%d > %d)' % (top + patch_size[0], cur_h))

                    if left + patch_size[1] > cur_w:
                        print('\n@Error: imidx=%d, pat=%d' % (vid_idx, idx), end='')
                        print(' (%d > %d)' % (left + patch_size[1], cur_w))

                    r_crop_norm = r_img_norm[top:top + patch_size[0],
                                             left:left + patch_size[1]]
                    r_crop_diff = r_frm_diff[top:top + patch_size[0],
                                             left:left + patch_size[1]]

                    r_pat_set.append(r_crop_norm)
                    r_pat_set_diff.append(r_crop_diff) #difference map
            ref_total_frm2pat_idx.append(ref_img2pat_idx) #################new

        # Show 100% progress bar
        show_progress(1.0)
        minutes, seconds = divmod(timeit.default_timer() - start_time, 60)
        print(' - It took {:02.0f}:{:05.2f}'.format(minutes, seconds))
        print(' - Loaded num of ref. patches: {:,}'.format(n_ref_patches))

        if len(pass_list) > 0:
            self.n_images -= len(pass_list)
            print(' - Ignored ref. videos due to small size: %s' %
                  ', '.join(str(i) for i in pass_list))

        self.n_ref_patches = n_ref_patches
        self.ref_npat_img_list = ref_npat_img_list
        self.ref_top_left_set = ref_top_left_set
        self.ref_img2pat_idx = ref_img2pat_idx
        self.n_ref_frms_idx = n_ref_frms_idx
        self.ref_total_frm2pat_idx = ref_total_frm2pat_idx #################new
        self.r_pat_set = r_pat_set
        self.r_pat_set_diff = r_pat_set_diff

        return ref_img2pat_idx, ref_top_left_set, ref_npat_img_list, inv_scenes, ref_total_frm2pat_idx

    def load_dis_frames(self, base_path, d_img_list, r_idx_list, inv_scenes,
                        ref_img2pat_idx, ref_top_left_set, ref_npat_img_list, ref_total_frm2pat_idx):
        """
        Actual routine of loading distorted images.
        """
        self.n_images = len(d_img_list)
        self.n_videos = len(d_img_list)

        n_frm_trte = self.n_frm_trte
        d_img_list = d_img_list

        assert self.n_images > 0, \
            'n_videos(%d) is not positive number' % (self.n_images)

        patch_size = self.patch_size
        n_patches = 0
        npat_vid_list =[]
        d_pat_set = []
        d_pat_set_diff = []
        filt_idx_list = []
        dis2ref_idx = []
        n_frms_idx = []   # number of frames of each videos

        #######################################################################
        # Read images
        start_time = timeit.default_timer()
        pat_idx = 0
        pass_list = []
        for vid_idx in range(self.n_images):

            npat_img_list = []

            ref_idx = inv_scenes[r_idx_list[vid_idx]]
            assert ref_idx >= 0
            if ref_top_left_set[ref_idx] is False:
                continue

            # Show progress
            show_progress(float(vid_idx) / self.n_images)
            #######################################################################
            # video read
            #######################################################################
            start_frm = 5  # first 5 frms are ignored
            height = 432
            width = 768
            fwidth = (width + 31) // 32 * 32
            fheight = (height + 15) // 16 * 16
            end_frm = int(self.fps_list[vid_idx])
            n_frms_idx.append(n_frm_trte)  # for same number of frms in video
            slide_frms = int(math.ceil(end_frm/n_frm_trte))  # sliding window frm size
            vid_1 = []
            vid_2 = []

            stream = open('%s/%s' % (base_path, d_img_list[vid_idx]), 'rb')
            for i in range(start_frm, end_frm, slide_frms):
                # frame 1 read
                stream.seek(int(i * width * height * 1.5))
                Y = np.fromfile(stream, dtype=np.uint8, count=fwidth * fheight).reshape((fheight, fwidth))
                U = np.fromfile(stream, dtype=np.uint8, count=(fwidth// 2) * (fheight // 2))
                V = np.fromfile(stream, dtype=np.uint8, count=(fwidth// 2) * (fheight // 2))
                vid_1.append(Y)
                # frame 2 read
                Y = np.fromfile(stream, dtype=np.uint8, count=fwidth * fheight).reshape((fheight, fwidth))
                U = np.fromfile(stream, dtype=np.uint8, count=(fwidth// 2) * (fheight // 2))
                V = np.fromfile(stream, dtype=np.uint8, count=(fwidth// 2) * (fheight // 2))
                if end_frm > 300: # over 250frm +1frm
                    Y = np.fromfile(stream, dtype=np.uint8, count=fwidth * fheight).reshape((fheight, fwidth))
                    U = np.fromfile(stream, dtype=np.uint8, count=(fwidth// 2) * (fheight // 2))
                    V = np.fromfile(stream, dtype=np.uint8, count=(fwidth// 2) * (fheight // 2))
                    vid_2.append(Y)
                else:
                    vid_2.append(Y)
            vid_1 = np.array(vid_1)
            vid_2 = np.array(vid_2)

            for frm_no in range(n_frm_trte):
                d_frm_raw_1 = np.squeeze(vid_1[frm_no])
                d_frm_raw_1_norm = d_frm_raw_1.astype('float32')/255.
                d_frm_raw_2 = np.squeeze(vid_2[frm_no])
                d_frm_raw_2_norm = d_frm_raw_2.astype('float32')/255.
                #NOTE: abs is conducted in FR_SENS_VQA
                d_frm_diff = d_frm_raw_1_norm - d_frm_raw_2_norm
                cur_h = d_frm_raw_1.shape[0]
                cur_w = d_frm_raw_1.shape[1]

                if self.use_original_size:
                    patch_size = (cur_h, cur_w)

                if cur_h < patch_size[0] or cur_w < patch_size[1]:
                    pass_list.append(vid_idx)
                    continue

                # Local normalization
                if self.local_norm:
                    if self.color == 'gray':
                        # faster
                        d_img_norm = local_normalize_1ch(d_frm_raw_1)
                    else:
                        d_img_norm = local_normalize(d_frm_raw_1, self.num_ch)
                else:
                    d_img_norm = d_frm_raw_1.astype('float32') / 255.

                if self.color == 'gray':
                    d_img_norm = d_img_norm[:, :, None]
                    d_frm_diff = d_frm_diff[:, :, None]

                top_left_set = ref_top_left_set[ref_idx]
                cur_n_patches = top_left_set.shape[0]

                if self.random_crops > 0: # NONE
                    if self.random_crops < cur_n_patches:
                        n_crops = self.random_crops
                        rand_perm = np.random.permutation(cur_n_patches)
                        sel_patch_idx = sorted(rand_perm[:n_crops])
                        top_left_set = top_left_set[sel_patch_idx].copy()
                    else:
                        n_crops = cur_n_patches
                        sel_patch_idx = np.arange(cur_n_patches)

                    npat_filt = n_crops
                    npat_img_list.append((npat_filt, 1, npat_filt))
                    n_patches += npat_filt

                    idx_set = list(range(npat_filt))
                    filt_idx_list.append(idx_set)

                else:
                    # numbers of patches along y and x axes
                    npat, ny, nx = ref_npat_img_list[ref_idx]
                    npat_filt = int(npat * self.std_filt_r)

                    npat_img_list.append((npat_filt, ny, nx))
                    n_patches += npat_filt

                    if self.std_filt_r < 1.0: # std_filt_r == 1 default
                        std_set = np.zeros((nx * ny))
                        for idx, top_left in enumerate(top_left_set):
                            top, left = top_left
                            std_set[idx] = np.std(
                                d_img_norm[top:top + patch_size[0],
                                      left:left + patch_size[1]])

                    # Filter the patches with low std
                    if self.std_filt_r < 1.0:
                        idx_set = sorted(list(range(len(std_set))),
                                         key=lambda x: std_set[x], reverse=True)
                        idx_set = sorted(idx_set[:npat_filt])
                    else: # std_filt_r == 1 default
                        idx_set = list(range(npat_filt))
                    filt_idx_list.append(idx_set)

                # Crop the images to patches
                for idx in idx_set:
                    [top, left] = top_left_set[idx]

                    if top + patch_size[0] > cur_h:
                        print('\n@Error: imidx=%d, pat=%d' % (vid_idx, idx), end='')
                        print(' (%d > %d)' % (top + patch_size[0], cur_h))

                    if left + patch_size[1] > cur_w:
                        print('\n@Error: imidx=%d, pat=%d' % (vid_idx, idx), end='')
                        print(' (%d > %d)' % (left + patch_size[1], cur_w))

                    d_crop_norm = d_img_norm[top:top + patch_size[0],
                                             left:left + patch_size[1]]
                    d_crop_diff = d_frm_diff[top:top + patch_size[0],
                                             left:left + patch_size[1]]

                    d_pat_set.append(d_crop_norm)
                    d_pat_set_diff.append(d_crop_diff)

                    if self.random_crops > 0:
                        dis2ref_idx.append(
                            ref_total_frm2pat_idx[ref_idx][sel_patch_idx][idx])
                    else:
                        # print(' ref_idx=%d, frm_no=%d, idx=%d \n' % (ref_idx, frm_no, idx), end='')
                        dis2ref_idx.append(ref_total_frm2pat_idx[ref_idx][frm_no][idx])

                    pat_idx += 1
            npat_vid_list.append(npat_img_list)  #################new
        # Show 100 % progress bar
        show_progress(1.0)
        minutes, seconds = divmod(timeit.default_timer() - start_time, 60)
        print(' - It took {:02.0f}:{:05.2f}'.format(minutes, seconds))
        print(' - Loaded num of patches: {:,}'.format(n_patches))

        if len(pass_list) > 0:
            self.n_images -= len(pass_list)
            print(' - Ignored frame list due to small size: %s' %
                  ', '.join(str(i) for i in pass_list))

        self.n_patches = n_patches
        self.npat_img_list = npat_img_list
        self.npat_vid_list = npat_vid_list
        self.n_frms_idx = n_frms_idx
        self.d_pat_set = d_pat_set
        self.d_pat_set_diff = d_pat_set_diff
        self.filt_idx_list = filt_idx_list
        self.dis2ref_idx = dis2ref_idx

    def make_toy_examples(self, patch_size=None, n_images=10):
        if patch_size is None:
            patch_size = self.patch_size
            if self.patch_size is None:
                patch_size = [64, 64]
                self.patch_size = patch_size
        n_ch = 1 if self.color == 'gray' else 3
        ny = 2
        nx = 2
        score_list = np.zeros(n_images, dtype='float32')
        npat_img_list = []
        filt_idx_list = []
        n_patches = 0
        for im_idx in range(n_images):
            npat = ny * nx
            npat_img_list.append((npat, ny, nx))
            n_patches += npat
            filt_idx_list.append(list(range(npat)))

        d_pat_set = []
        pat_shape = (patch_size[0], patch_size[1], n_ch)
        dummy_pat = np.ones(pat_shape, dtype='float32') * 0.5
        for idx in range(n_patches):
            d_pat_set.append(dummy_pat)
        print(' - Generated toy examples: %d x' % n_patches, pat_shape)

        dis2ref_idx = []
        for idx in range(n_patches):
            dis2ref_idx.append(idx)

        self.n_images = n_images
        self.score_list = score_list
        self.n_patches = n_patches
        self.npat_img_list = npat_img_list
        self.d_pat_set = d_pat_set
        self.r_pat_set = d_pat_set
        self.filt_idx_list = filt_idx_list
        self.dis2ref_idx = dis2ref_idx

    def data_augmentation_horz_refl(self):
        # Patches augmentation
        pat_idx = 0
        for pat_idx in range(self.n_patches):
            # Flip horizontal
            self.d_pat_set.append(self.d_pat_set[pat_idx][:, ::-1])
            self.dis2ref_idx.append(
                self.dis2ref_idx[pat_idx] + self.n_ref_patches)

        for pat_idx in range(self.n_ref_patches):
            # Flip horizontal
            self.r_pat_set.append(self.r_pat_set[pat_idx][:, ::-1])

        # Image data augmentation
        if self.score_list is not None:
            self.score_list = np.tile(self.score_list, 2)
        self.npat_img_list += self.npat_img_list
        self.filt_idx_list += self.filt_idx_list

        self.n_patches *= 2
        self.n_images *= 2
        self.n_ref_patches *= 2

        if self.score_list is not None:
            assert self.score_list.shape[0] == self.n_images, (
                'n_score_list: %d != n_videos: %d' %
                (self.score_list.shape[0], self.n_images))
        assert len(self.npat_img_list) == self.n_images, (
            'n_npat_img_list: %d != n_videos: %d' %
            (len(self.npat_img_list), self.n_images))
        assert len(self.filt_idx_list) == self.n_images, (
            'n_filt_idx_list: %d != n_videos: %d' %
            (len(self.filt_idx_list), self.n_images))

        print(' - Augmented patches: {0:,}'.format(self.n_patches), end=' ')
        print('(x2 horz. reflection)')


def show_progress(percent):
    hashes = '#' * int(round(percent * 20))
    spaces = ' ' * (20 - len(hashes))
    sys.stdout.write("\r - Load videos: [{0}] {1}%".format(
        hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()


def convert_color(img, color):
    """ Convert image into gray or RGB or YCbCr.
    """
    assert len(img.shape) in [2, 3]
    if color == 'gray':
        # if d_img_raw.shape[2] == 1:
        if len(img.shape) == 2:  # if gray
            img_ = img[:, :, None]
        elif len(img.shape) == 3:  # if RGB
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img_ = rgb2gray(img)[:, :, None]
    elif color == 'rgb':
        if len(img.shape) == 2:  # if gray
            img_ = gray2rgb(img)
        elif len(img.shape) == 3:  # if RGB
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img_ = img
    elif color == 'ycbcr':
        if len(img.shape) == 2:  # if gray
            img_ = rgb2ycbcr(gray2rgb(img))
        elif len(img.shape) == 3:  # if RGB
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img_ = rgb2ycbcr(img)
    else:
        raise ValueError("Improper color selection: %s" % color)
    return img_


def convert_color2(img, color):
    """ Convert image into gray or RGB or YCbCr.
    (In case of gray, dimension is not increased for
    the faster local normalization.)
    """
    assert len(img.shape) in [2, 3]
    if color == 'gray':
        # if d_img_raw.shape[2] == 1:
        if len(img.shape) == 3:  # if RGB
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img_ = rgb2gray(img)
    elif color == 'rgb':
        if len(img.shape) == 2:  # if gray
            img_ = gray2rgb(img)
        elif len(img.shape) == 3:  # if RGB
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img_ = img
    elif color == 'ycbcr':
        if len(img.shape) == 2:  # if gray
            img_ = rgb2ycbcr(gray2rgb(img))
        elif len(img.shape) == 3:  # if RGB
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img_ = rgb2ycbcr(img)
    else:
        raise ValueError("Improper color selection: %s" % color)

    return img_


def gray2rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, :] = im[:, :, np.newaxis]
    return ret


def rgb2gray(rgb):
    assert rgb.shape[2] == 3
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def rgb2ycbcr(rgb):
    xform = np.array([[.299, .587, .114],
                      [-.1687, -.3313, .5],
                      [.5, -.4187, -.0813]])
    ycbcr = np.dot(rgb[..., :3], xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return ycbcr


def ycbcr2rgb(ycbcr):
    xform = np.array([[1, 0, 1.402],
                      [1, -0.34414, -.71414],
                      [1, 1.772, 0]])
    rgb = ycbcr.astype('float32')
    rgb[:, :, [1, 2]] -= 128
    return rgb.dot(xform.T)


k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
kern = k / k.sum()


def local_normalize_1ch(img, const=127.0):
    mu = convolve(img, kern, mode='nearest')
    mu_sq = mu * mu
    im_sq = img * img
    tmp = convolve(im_sq, kern, mode='nearest') - mu_sq
    sigma = np.sqrt(np.abs(tmp))
    structdis = (img - mu) / (sigma + const)

    # Rescale within 0 and 1
    # structdis = (structdis + 3) / 6
    structdis = 2. * structdis / 3.
    return structdis


def local_normalize(img, num_ch=1, const=127.0):
    if num_ch == 1:
        mu = convolve(img[:, :, 0], kern, mode='nearest')
        mu_sq = mu * mu
        im_sq = img[:, :, 0] * img[:, :, 0]
        tmp = convolve(im_sq, kern, mode='nearest') - mu_sq
        sigma = np.sqrt(np.abs(tmp))
        structdis = (img[:, :, 0] - mu) / (sigma + const)

        # Rescale within 0 and 1
        # structdis = (structdis + 3) / 6
        structdis = 2. * structdis / 3.
        norm = structdis[:, :, None]
    elif num_ch > 1:
        norm = np.zeros(img.shape, dtype='float32')
        for ch in range(num_ch):
            mu = convolve(img[:, :, ch], kern, mode='nearest')
            mu_sq = mu * mu
            im_sq = img[:, :, ch] * img[:, :, ch]
            tmp = convolve(im_sq, kern, mode='nearest') - mu_sq
            sigma = np.sqrt(np.abs(tmp))
            structdis = (img[:, :, ch] - mu) / (sigma + const)

            # Rescale within 0 and 1
            # structdis = (structdis + 3) / 6
            structdis = 2. * structdis / 3.
            norm[:, :, ch] = structdis

    return norm
