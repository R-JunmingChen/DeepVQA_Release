from __future__ import absolute_import, division, print_function
import numpy as np

# Define DB information
BASE_PATH = 'D:\DB\VQA\CSIQ VQA DB'
# BASE_PATH = 'Your DB path'
LIST_FILE_NAME = 'CSIQ_VQA.txt'
ALL_SCENES = list(range(12))
ALL_DIST_TYPES = list(range(7)) #including reference


def make_image_list(scenes, dist_types=None, show_info=True):
    """
    Make image list from CSIQ VIDEO database
    CSIQ: 12 reference images x 7 types (including REFERENCE)
    """
    # Get reference / distorted image file lists:
    # d_img_list and score_list
    d_vid_list, r_vid_list, r_idx_list, score_list, fps_list = [], [], [], [], []
    # list_file_name = os.path.join(BASE_PATH, LIST_FILE_NAME)
    list_file_name = LIST_FILE_NAME
    with open(list_file_name, 'r') as listFile:
        for line in listFile:
            # ref_idx ref_name dist_name dist_types, DMOS, width, height
            scn_idx, dis_idx, ref, dis, score, width, height, fps= line.split()
            scn_idx = int(scn_idx)
            dis_idx = int(dis_idx)
            if scn_idx in scenes and dis_idx in dist_types:
                d_vid_list.append(dis)
                r_vid_list.append(ref)
                r_idx_list.append(scn_idx)
                score_list.append(float(score))
                fps_list.append(fps)

    score_list = np.array(score_list, dtype='float32')
    # DMOS -> reverse subjecive scores by default
    score_list = 1 - score_list

    n_videos = len(d_vid_list)

    dist_names = ['ref', 'AVC', 'Packet', 'MJPEG', 'Wavelet', 'WN', 'HEVC']
    # H.264 / AVC compression
    # H.264 video with packet loss rate
    # MJPEG compression
    # Wavelet compression(snow codec)
    # White noise
    # HEVC compression

    if show_info:
        scenes.sort()
        print(' - Scenes: %s' % ', '.join([str(i) for i in scenes]))
        print(' - Distortion types: %s' % ', '.join(
            [dist_names[idx] for idx in dist_types]))
        print(' - Number of videos: {:,}'.format(n_videos))
        print(' - DMOS range: [{:.2f}, {:.2f}]'.format(
            np.min(score_list), np.max(score_list)), end='')
        print(' (Scale reversed)')

    return {
        'scenes': scenes,
        'dist_types': dist_types,
        'base_path': BASE_PATH,
        'n_images': n_videos,
        'n_videos': n_videos,
        'd_img_list': d_vid_list,
        'r_img_list': r_vid_list,
        'r_idx_list': r_idx_list,
        'score_list': score_list,
        'fps_list': fps_list
    }
