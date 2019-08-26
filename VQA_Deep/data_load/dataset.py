import numpy as np

class Dataset(object):
    """
    Dataset class containing images, scores, and supplementary information
    for image quality assessment.
    Attributes
    ----------

        dis_data: 4D numpy array
            distorted image patches
        ref_data: 4D numpy array (optional)
            reference image patches
        dis2ref_idx: 1D numpy array (optional)
            index to ref. patcehs of dis. patches
        loc_data: 4D numpy array (optional)
            local quality scores
        score_data: 2D numpy array
            subjective score list
        npat_img_list: 2D numpy array
            number of patches of each image
        pat2img_idx_list: 2D numpy array
            start and end indices list of images
    """
    def __init__(self):
        # Data
        self.dis_data = None  # distorted image patches
        self.ref_data = None  # reference image patches
        self.dis_diff_data = None
        self.ref_diff_data = None
        self.dis2ref_idx = None  # index to ref. patches of dis. patches
        self.loc_data = None  # local scores
        self.score_data = None  # subjective score list
        self.n_patches = 0

        # Data for image-wise training
        self.npat_img_list = None  # number of patches of each image
        self.npat_vid_list = None  # number of patches of each video
        self.pat2img_idx_list = None  # start and end indices of each image
        self.pat2vid_idx_list = None # start and end indices of each video
        self.filt_idx_list = None  # filtered indices list of each image
        self.n_images = 0
        self.n_videos = 0
        self.n_frms_idx =None
        self.n_ref_frms_idx = None
        # Configurations
        self.shuffle = False
        self.imagewise = False

        self.exist_ref = False
        self.exist_loc = False
        self.exist_score = False
        self.exist_npat = False
        self.exist_filt_idx = False

        # Variables
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.rand_imidx_list = None

        # Data configuration
        self.num_ch = None
        self.patch_size = None
        self.patch_step = None
        self.random_crops = None
        self.loc_size = None

    def set_patch_config(self, patch_step=None, random_crops=None):
        assert patch_step is not None and random_crops is not None
        if patch_step is not None:
            assert isinstance(patch_step, (list, tuple))
        self.patch_step = patch_step
        self.random_crops = random_crops

    def put_data(self, dis_data,
                 ref_data=[],
                 dis_diff_data=[],
                 ref_diff_data=[],
                 dis2ref_idx=None,
                 loc_data=[], n_frms_idx=None,
                 score_data=None, npat_img_list=None, npat_vid_list=None, filt_idx_list=None,
                 imagewise=True, shuffle=False):
        # dis_data
        self.dis_data = dis_data
        if isinstance(self.dis_data, list):
            self.dis_data = np.asarray(self.dis_data, 'float32')  #data float np
        if len(self.dis_data[0].shape) < 3:
            self.dis_data = np.expand_dims(self.dis_data, 3)

        self.n_patches = self.dis_data.shape[0]
        self.patch_size = (
            self.dis_data.shape[1], self.dis_data.shape[2])  #patch size 112 * 112
        self.num_ch = self.dis_data.shape[3]

        # ref_data
        if ref_data:
            self.exist_ref = True
            self.ref_data = ref_data
            if isinstance(self.ref_data, list):
                self.ref_data = np.asarray(self.ref_data, 'float32') #data float np
            assert len(self.dis_data[0].shape) == len(self.ref_data[0].shape)
            if len(self.ref_data[0].shape) < 3:
                self.ref_data = np.expand_dims(self.ref_data, 3)

            assert dis2ref_idx is not None
            self.dis2ref_idx = np.asarray(dis2ref_idx, 'int32')
        else:
            self.exist_ref = False

        # ref_diff_data
        if ref_diff_data:
            self.ref_diff_data = ref_diff_data
            if isinstance(self.ref_diff_data, list):
                self.ref_diff_data = np.asarray(self.ref_diff_data, 'float32') #data float np
            assert len(self.dis_data[0].shape) == len(self.ref_diff_data[0].shape)
            if len(self.ref_diff_data[0].shape) < 3:
                self.ref_diff_data = np.expand_dims(self.ref_diff_data, 3)

        if dis_diff_data:
            self.dis_diff_data = dis_diff_data
            if isinstance(self.dis_diff_data, list):
                self.dis_diff_data = np.asarray(self.dis_diff_data, 'float32')  # data float np
            assert len(self.dis_data[0].shape) == len(self.dis_diff_data[0].shape)
            if len(self.dis_diff_data[0].shape) < 3:
                    self.dis_diff_data = np.expand_dims(self.dis_diff_data, 3)

        # loc_data
        if loc_data:  #no usage
            self.exist_loc = True
            self.loc_data = loc_data
            if isinstance(self.loc_data, list):
                self.loc_data = np.asarray(self.loc_data, 'float32')
            if len(self.loc_data[0].shape) < 3:
                self.loc_data = np.expand_dims(self.loc_data, 3)

            self.loc_size = (
                self.loc_data.shape[1], self.loc_data.shape[2])
        else:
            self.exist_loc = False

        # score_data
        if score_data is not None:
            self.exist_score = True
            self.score_data_org = np.asarray(score_data, 'float32')  #data float np score
            self.score_data = self.score_data_org.copy()
            self.n_images = self.score_data.shape[0]
        else:
            self.exist_score = False

        self.imagewise = imagewise

        # npat_img_list
        if npat_vid_list is not None:
            self.exist_npat = True
            self.npat_vid_list = npat_vid_list
            self.n_frms_idx = np.asarray(n_frms_idx, dtype='int32')
            if self.n_images == 0:
                self.n_images = int(len(npat_vid_list))
            self.pat2vid_idx_list = self.gen_pat2frm_idx_list()
            # self.pat2frm_idx_list = self.gen_pat2frm_idx_list()
            if not self.imagewise and self.exist_score:
                self.score_data = self.gen_patchwise_scores()
        else:
            self.exist_npat = False

        # filt_idx_list
        if filt_idx_list is not None:
            self.exist_filt_idx = True
            self.filt_idx_list = filt_idx_list
        else:
            self.exist_filt_idx = False

        self.n_data = self.n_images if self.imagewise else self.n_patches

        self.shuffle = shuffle
        if self.shuffle:
            self.rand_imidx_list = np.random.permutation(self.n_data)
        else:
            self.rand_imidx_list = np.arange(self.n_data)

        self.validate_datasize()

    def set_imagewise(self):
        """Set this Dataset for imagewise training and testing
        """
        if self.imagewise is False:
            self.imagewise = True
            self.n_data = self.n_images
            if self.exist_score and self.exist_npat:
                self.score_data = self.score_data_org.copy()
        # Reset batch to generate prpoer rand_imidx_list
        self.reset_batch()

    def set_patchwise(self):
        """Set this Dataset for patchwise training and testing
        """
        if self.imagewise is True:
            self.imagewise = False
            self.n_data = self.n_patches
            if self.exist_score and self.exist_npat:
                self.score_data = self.gen_patchwise_scores()

        # Reset batch to generate prpoer rand_imidx_list
        self.reset_batch()

    def validate_datasize(self):

        if self.exist_loc:
            assert self.n_patches == self.loc_data.shape[0], (
                'dis_data.shape: %s loc_data.shape: %s' % (
                    self.dis_data.shape, self.loc_data.shape))

        if self.exist_npat:
            # assert self.exist_score
            assert self.n_images == int(len(self.npat_vid_list)), (
                'n_score_data: %d != n_npat_img_list: %d' % (
                    self.n_images, int(len(self.npat_vid_list))))

    def gen_pat2img_idx_list(self):
        """
        Generate pat2img_idx_list from npat_img_list
        """
        pat2img_idx_list = np.zeros((self.n_patches, 2), dtype='int32')
        n_patches = 0
        for im_idx in range(self.n_images):
            (cur_npat, ny, nx) = self.npat_img_list[im_idx]
            pat2img_idx_list[im_idx] = [n_patches, n_patches + cur_npat]
            n_patches += cur_npat
        assert n_patches == self.n_patches, (
            'obtained n_patches(%d) ~= n_patches(%d)' % (
                n_patches, self.n_patches))

        return pat2img_idx_list

    def gen_pat2frm_idx_list(self):
        """
        Generate pat2frm_idx_list from npat_vid_list
        """
        pat2vid_idx_list = []
        n_patches = 0

        for vi_idx in range(self.n_images):
            pat2frm_idx_list = []
            for frm_idx in range(self.n_frms_idx[vi_idx]):
                (cur_npat, ny, nx) = self.npat_vid_list[vi_idx][frm_idx]
                pat2frm_idx_list.append([n_patches, n_patches + cur_npat])
                n_patches += cur_npat
            pat2vid_idx_list.append(pat2frm_idx_list)
        assert n_patches == self.n_patches, (
            'obtained n_patches(%d) ~= n_patches(%d)' % (
                n_patches, self.n_patches))
        return pat2vid_idx_list

    def gen_patchwise_scores(self):
        """
        Generate patch-wise training targets by expanding
        image-wise score_data using pat2img_idx_list
        """
        new_scores = np.zeros(self.n_patches, dtype='float32')
        for im_idx in range(self.n_images):
            cur_idx_from, cur_idx_to = self.pat2img_idx_list[im_idx]
            new_scores[cur_idx_from:cur_idx_to] = self.score_data[im_idx]

        return new_scores

    def next_batch_video(self, batch_size, minibatch_size):
        """
        Return the next `batch_size` examples from this dataset.
        Parameters
        ----------
            batch_size: integer
                number of images (imagewise) or patches (patchwise) of
                current batch
        Returns
        -------
        A dictionary containing:

            'dis_data': 5D numpy array
                distorted image patches
            'ref_data': 5D numpy array (optional)
                reference image patches
            'loc_data': 5D numpy array (optional)
                local quality scores
            'score_data': 2D numpy array
                subjective score list
            'bat2img_idx_set': 2D numpy array (optional - imagewise)
                from and to indices of each image in the current batch
            'n_data': integer (optional - imagewise)
                number of patches in the current batch
        """
        assert batch_size <= self.n_data

        # print('%d',minibatch_size)
        start = self.index_in_epoch  #initial 0
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.n_data:
            # Finished epoch
            self.epochs_completed += 1

            # Shuffle the data
            if self.shuffle:
                self.rand_imidx_list = np.random.permutation(self.n_data)

            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
        end = self.index_in_epoch
        self.im_idx_list = self.rand_imidx_list[start:end]

        if self.imagewise:
            # If image-wise training
            # Get bat2img_idx_set and idx_set
            bat2frm_idx_set_temp3 = []
            score_set = np.zeros(batch_size, dtype='float32')
            idx_set_list = []
            idx_set_list_vid = []
            cur_inb_from = 0
            frm_no = 0
            for in_bat_idx, im_idx in enumerate(self.im_idx_list):
                minibatch_range = int(round(self.n_frms_idx[im_idx]/minibatch_size))
                for minibatch_idx in range(minibatch_range):
                    idx_set_list_tmp = []
                    for frm_idx in range(minibatch_idx*minibatch_size,
                                          minibatch_size*(1+minibatch_idx), 1):
                        cur_idx_from, cur_idx_to = self.pat2vid_idx_list[im_idx][frm_no]
                        idx_set_list_tmp.append(
                            np.arange(cur_idx_from, cur_idx_to, dtype='int32'))
                        frm_no = frm_no + 1
                    idx_set_list_tmp = np.array(idx_set_list_tmp).reshape((-1))
                    idx_set_list_vid.append(idx_set_list_tmp)
                    if self.exist_score:
                        score_set[in_bat_idx] = self.score_data[im_idx]
            idx_set_list_vid = np.asarray(idx_set_list_vid)
            # idx_set = np.concatenate(idx_set_list)


            # self.bat2img_idx_set = bat2img_idx_set

        if self.imagewise:
            # If image-wise training
            # Get bat2img_idx_set and idx_set
            bat2frm_idx_set_temp3 = []
            score_set = np.zeros(batch_size, dtype='float32')
            # idx_set_list = []
            cur_inb_from = 0
            for in_bat_idx, im_idx in enumerate(self.im_idx_list):
                bat2frm_idx_set_temp2 = []
                for frm_idx in range(self.n_frms_idx[im_idx]):
                    # cur_idx_from, cur_idx_to = self.pat2img_idx_list[im_idx]
                    cur_idx_from, cur_idx_to = self.pat2vid_idx_list[im_idx][frm_idx]
                    # current patch start, curren_patch_last_
                    cur_inb_to = cur_inb_from + (cur_idx_to - cur_idx_from)
                    bat2frm_idx_set_temp = [cur_inb_from, cur_inb_to]
                    # bat2img_idx_set[in_bat_idx] = [cur_inb_from, cur_inb_to]
                    cur_inb_from = cur_inb_to
                    bat2frm_idx_set_temp2.append(bat2frm_idx_set_temp)
                bat2frm_idx_set_temp3.append(bat2frm_idx_set_temp2)
                if self.exist_score:
                    score_set[in_bat_idx] = self.score_data[im_idx]

            bat2frm_idx_set = np.asarray(bat2frm_idx_set_temp3, dtype='int32')
            self.bat2frm_idx_set = bat2frm_idx_set

            idx_set = idx_set_list_vid


            # self.bat2img_idx_set = bat2img_idx_set

            res = {
                'dis_data': self.dis_data[idx_set],
                'bat2frm_idx_set': bat2frm_idx_set,
                # 'bat2img_idx_set': bat2img_idx_set,
                'n_data': cur_inb_to
            }
            if self.exist_score:
                res['score_set'] = score_set
            if self.exist_ref:
                res_ref_data = []
                for minibatch_idx in range(int(idx_set.shape[0])):
                    res_ref_data_tmp = self.ref_data[self.dis2ref_idx[idx_set[minibatch_idx]]]
                    res_ref_data.append(res_ref_data_tmp)
                res_ref_data = np.array(res_ref_data)
                res['ref_data'] = res_ref_data
            if self.exist_loc:
                res['loc_data'] = self.loc_data[idx_set]
        else:
            res = {'dis_data': self.dis_data[self.im_idx_list]}
            if self.exist_score:
                res['score_set'] = self.score_data[self.im_idx_list]
            if self.exist_ref:
                res['ref_data'] = self.ref_data[
                    self.dis2ref_idx[self.im_idx_list]]
            if self.exist_loc:
                res['loc_data'] = self.loc_data[self.im_idx_list]
        return res

    def next_batch_video_diff(self, batch_size, minibatch_size):
        """
        Return the next `batch_size` examples from this dataset.
        Parameters
        ----------
            batch_size: integer
                number of images (imagewise) or patches (patchwise) of
                current batch
        Returns
        -------
        A dictionary containing:

            'dis_data': 5D numpy array
                distorted image patches
            'ref_data': 5D numpy array (optional)
                reference image patches
            'loc_data': 5D numpy array (optional)
                local quality scores
            'score_data': 2D numpy array
                subjective score list
            'bat2img_idx_set': 2D numpy array (optional - imagewise)
                from and to indices of each image in the current batch
            'n_data': integer (optional - imagewise)
                number of patches in the current batch
        """
        assert batch_size <= self.n_data

        # print('%d',minibatch_size)
        start = self.index_in_epoch  #initial 0
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.n_data:
            # Finished epoch
            self.epochs_completed += 1

            # Shuffle the data
            if self.shuffle:
                self.rand_imidx_list = np.random.permutation(self.n_data)

            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
        end = self.index_in_epoch
        self.im_idx_list = self.rand_imidx_list[start:end]

        if self.imagewise:
            # If image-wise training
            # Get bat2img_idx_set and idx_set
            bat2frm_idx_set_temp3 = []
            score_set = np.zeros(batch_size, dtype='float32')
            idx_set_list = []
            idx_set_list_vid = []
            cur_inb_from = 0
            frm_no = 0
            for in_bat_idx, im_idx in enumerate(self.im_idx_list):
                minibatch_range = int(round(self.n_frms_idx[im_idx]/minibatch_size))
                for minibatch_idx in range(minibatch_range):
                    idx_set_list_tmp = []
                    for frm_idx in range(minibatch_idx*minibatch_size,
                                          minibatch_size*(1+minibatch_idx), 1):
                        cur_idx_from, cur_idx_to = self.pat2vid_idx_list[im_idx][frm_no]
                        idx_set_list_tmp.append(
                            np.arange(cur_idx_from, cur_idx_to, dtype='int32'))
                        frm_no = frm_no + 1
                    idx_set_list_tmp = np.array(idx_set_list_tmp).reshape((-1))
                    idx_set_list_vid.append(idx_set_list_tmp)
                    if self.exist_score:
                        score_set[in_bat_idx] = self.score_data[im_idx]
            idx_set_list_vid = np.asarray(idx_set_list_vid)

        if self.imagewise:
            # If image-wise training
            # Get bat2img_idx_set and idx_set
            bat2frm_idx_set_temp3 = []
            score_set = np.zeros(batch_size, dtype='float32')
            # idx_set_list = []
            cur_inb_from = 0
            for in_bat_idx, im_idx in enumerate(self.im_idx_list):
                bat2frm_idx_set_temp2 = []
                for frm_idx in range(self.n_frms_idx[im_idx]):
                    # cur_idx_from, cur_idx_to = self.pat2img_idx_list[im_idx]
                    cur_idx_from, cur_idx_to = self.pat2vid_idx_list[im_idx][frm_idx]
                    # current patch start, curren_patch_last_
                    cur_inb_to = cur_inb_from + (cur_idx_to - cur_idx_from)
                    bat2frm_idx_set_temp = [cur_inb_from, cur_inb_to]
                    # bat2img_idx_set[in_bat_idx] = [cur_inb_from, cur_inb_to]
                    cur_inb_from = cur_inb_to
                    bat2frm_idx_set_temp2.append(bat2frm_idx_set_temp)
                bat2frm_idx_set_temp3.append(bat2frm_idx_set_temp2)
                if self.exist_score:
                    score_set[in_bat_idx] = self.score_data[im_idx]

            bat2frm_idx_set = np.asarray(bat2frm_idx_set_temp3, dtype='int32')
            self.bat2frm_idx_set = bat2frm_idx_set

            idx_set = idx_set_list_vid


            # self.bat2img_idx_set = bat2img_idx_set

            res = {
                'dis_data': self.dis_data[idx_set],
                'dis_diff_data': self.dis_diff_data[idx_set],
                'bat2frm_idx_set': bat2frm_idx_set,
                # 'bat2img_idx_set': bat2img_idx_set,
                'n_data': cur_inb_to
            }
            if self.exist_score:
                res['score_set'] = score_set
            if self.exist_ref:
                res_ref_data = []
                res_ref_diff_data = []
                for minibatch_idx in range(int(idx_set.shape[0])):
                    res_ref_data_tmp = self.ref_data[self.dis2ref_idx[idx_set[minibatch_idx]]]
                    res_ref_data.append(res_ref_data_tmp)

                    res_ref_diff_data_tmp = self.ref_diff_data[self.dis2ref_idx[idx_set[minibatch_idx]]]
                    res_ref_diff_data.append(res_ref_diff_data_tmp)

                res_ref_data = np.array(res_ref_data)
                res['ref_data'] = res_ref_data

                res_ref_diff_data = np.array(res_ref_diff_data)
                res['ref_diff_data'] = res_ref_diff_data

            if self.exist_loc:
                res['loc_data'] = self.loc_data[idx_set]
        else:
            res = {'dis_data': self.dis_data[self.im_idx_list]}
            if self.exist_score:
                res['score_set'] = self.score_data[self.im_idx_list]
            if self.exist_ref:
                res['ref_data'] = self.ref_data[
                    self.dis2ref_idx[self.im_idx_list]]
            if self.exist_ref:
                res['ref_diff_data'] = self.self.ref_diff_data[
                    self.dis2ref_idx[self.im_idx_list]]

            if self.exist_loc:
                res['loc_data'] = self.loc_data[self.im_idx_list]
        return res

    def reset_batch(self):
        """
        Make batch index in epoch 0, and shuffle data.
        """
        self.epochs_completed = 0
        self.index_in_epoch = 0
        if self.shuffle:
            self.rand_imidx_list = np.random.permutation(self.n_data)
        else:
            self.rand_imidx_list = np.arange(self.n_data)

    def get_current_recon_info(self):
        """
        Get information to reconstruct patches into an image.

        Returns
        -------
        A dictionary containing:

            'npat_img_list': (N, 3) matrix
                where N is the number of images, and each row
                indicate each image.
                [number of patches, num of patches along y-axis,
                num of patches along x-axis].
            'bat2img_idx_set': (N, 2) matrix
                where each row indicate each image.
                [from-index in current batch, to-index in current batch]
            'filt_idx_list' (optional): (N, 1) list
                where where each element has indices list of existing
                patches.
        """
        assert self.imagewise
        assert self.index_in_epoch != 0

        res = {
            'npat_img_list': self.npat_img_list[self.im_idx_list],
            'bat2img_idx_set': self.bat2img_idx_set
        }
        if self.exist_filt_idx:
            res['filt_idx_list'] = [
                self.filt_idx_list[idx] for idx in self.im_idx_list]
        else:
            res['filt_idx_list'] = None

        return res
