import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from VQA_Deep import train_vqa as tm
########################################################################################################################
# DeepVQA
# ver2 (training prosess update)

# Kim, Woojae, et al.
# "Deep video quality assessor: From spatio-temporal visual sensitivity to a convolutional neural aggregation network."
# Proceedings of the European Conference on Computer Vision (ECCV). 2018.
# This code was developed and tested with Theano 1.0.2, CUDA 9.0, and Windows python.
########################################################################################################################
tm.train_vqa(
    # config parser
    config_file='VQA_Deep/configs/FR_sens_VQA.yaml',

    # section on configuration parser 'FR_sens_VQA.yaml'
    ############# LIVE_VQA ##############
    section='fr_sens_LIVE_VQA',
    ############# CSIQ_VQA ##############
    # section='fr_sens_CSIQ_VQA',

    # training/testing reference setting on each database
    ############# LIVE_VQA ##############
    tr_te_file='outputs/tr_va_live_VQA.txt',
    ############# CSIQ_VQA ##############
    # tr_te_file='outputs/tr_va_csiq_VQA.txt',

    # path for saving snap files of best model
    ############# LIVE_VQA ##############
    snap_path='outputs/FR/FR_sens_LIVE_VQA/',
    ############# CSIQ_VQA ##############
    # snap_path='outputs/FR/FR_sens_CSIQ_VQA/',

    ############# snap file load ##############
    # if user tests STAGE 2, Load below snap_file which is fully trained in STAGE 1.
    # and also set model path in 'FR_sens_VQA.yaml' as model: VQA_Deep.models.FR_sens_VQA_stage2
    # snap_file = 'outputs/FR/FR_sens_LIVE_VQA/USERS_BEST_RESULT_IN_STAGE1.npy'
)
