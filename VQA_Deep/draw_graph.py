from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import glob
import os
import fnmatch
import sys


def draw_log(log_file, out_img_file=None):
    """
    Read log_file and draw.
    """
    #---------------------------------------------------------
    # Read log file
    print('Load data: %s' % log_file)
    with open(log_file, 'r') as l_file:
        lines = l_file.readlines()

    # Find the last starting line
    last_start = -1
    for idx, line in enumerate(lines):
        match_date = re.search(r'(\d+/\d+/\d+)', line)
        if match_date:
            last_start = idx
    if last_start < 0:
        print("Not proper file: %s." % log_file)
        print("Starting line must contain date 00/00/00.")
        return

    # Get time and date
    match_time = re.search(r'(\d+:\d+:\d+)', lines[last_start])
    time_str = match_time.group() if match_time else ""
    match_date = re.search(r'(\d+/\d+/\d+)', lines[last_start])
    date_str = match_date.group() if match_date else ""
    gen_title = log_file + ' - ' + time_str + ' ' + date_str

    # Get labels of data
    labels = lines[last_start + 1].replace(', ', ' ').split()
    labels = ["epoch"] + labels

    # Load data
    n_label = len(labels)
    n_data = len(lines) - last_start - 2
    data = np.zeros((n_data, n_label), dtype=float)
    for row in range(n_data):
        line_idx = row + last_start + 2
        cur = lines[line_idx].replace(', ', ' ').split()
        if len(cur) != n_label:
            print("Not proper file: %s." % log_file)
            print("Data dimension (in line %d)" % (line_idx + 1), end=' ')
            print("doesn't match to the number of labels.")
            return
        for col in range(n_label):
            data[row, col] = float(cur[col])

    #---------------------------------------------------------
    # Get tile shape ~ sqrt(number of figures)
    n_figure = n_label - 1
    draw_cc = labels[-1] == 'PLCC' and labels[-2] == 'SRCC'
    if draw_cc:  # if draw_cc, the last figure contains both SRCC and PLCC
        n_figure = n_figure - 1
    tile_sh = int(np.ceil(np.sqrt(n_figure)))
    tile_sh = str(tile_sh) + str(tile_sh)

    #---------------------------------------------------------
    # Draw graph
    matplotlib.rcParams.update({'font.size': 8})
    plt.figure()
    plt.suptitle(gen_title)
    for fig_idx in range(n_figure):
        if fig_idx == n_figure - 1 and draw_cc:
            # if draw_cc, the last figure contains both SRCC and PLCC
            break
        plt.subplot(tile_sh + str(fig_idx + 1))
        lab_idx = fig_idx + 1
        plt.plot(data[:, 0], data[:, lab_idx])
        plt.title(labels[lab_idx])
        plt.grid(True)
        plt.xlim(1, data[-1, 0])

    if draw_cc:
        plt.subplot(tile_sh + str(n_figure))
        plt.plot(data[:, 0], data[:, -1], 'b-x', label='PLCC')
        plt.plot(data[:, 0], data[:, -2], 'r-.', label='SRCC')
        plt.legend(loc=0)
        plt.title('CC')
        plt.grid(True)
        plt.xlim(1, data[-1, 0])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if out_img_file:
        print(' - Save to image: %s' % out_img_file)
        plt.savefig(out_img_file, dpi=100)


def draw_all_logs(root_path, keywords=['log', 'test'], show_figs=False):
    log_file_list = []
    if sys.version_info >= (3, 5):
        for filename in glob.iglob(root_path + "**/*.txt", recursive=True):
            if any(word in filename for word in keywords):
                log_file_list.append(filename)
    else:
        for root, dirnames, filenames in os.walk(root_path):
            for filename in fnmatch.filter(filenames, '*.txt'):
                if any(word in filename for word in keywords):
                    log_file_list.append(os.path.join(root, filename))

    for log_file in log_file_list:
        head, tail = os.path.split(log_file)
        prefix = head[len(root_path):]
        prefix = prefix.replace('\\', '_').replace('/', '_')
        draw_log(log_file,
                 os.path.join(root_path, prefix + tail[:-4] + '.png'))

    if show_figs:
        plt.show()
