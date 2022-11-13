
# -*- coding: utf-8 -*-

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from melodyExtraction_JDC.featureExtraction import *
from melodyExtraction_JDC.model import *
from scipy.signal import medfilt

root_path = os.path.dirname(os.path.abspath(__file__))

class Options(object):
    def __init__(self):
        self.num_spec = 513
        self.input_size = 31#115
        self.batch_size = 32#64
        self.resolution = 16
        self.figureON = False

options = Options()

def main(filepath,output_dir,gpu_index):
    # print(filepath, output_dir, gpu_index)
    if gpu_index is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    

    pitch_range = np.arange(38, 83 + 1.0/options.resolution, 1.0/options.resolution)
    pitch_range = np.concatenate([np.zeros(1), pitch_range])

    '''  Features extraction'''
    X_test, X_spec = spec_extraction(file_name=filepath, win_size=options.input_size)

    '''  melody predict'''
    model = melody_ResNet_joint_add(options)
    # model = melody_ResNet_joint_add2(options)
    model.load_weights(os.path.join(root_path, 'weights/ResNet_joint_add_L(CE_G).hdf5'))
#     model.load_weights('./weights/ResNet_joint_add_L(CE_G)_r16_t3_singleGPU.hdf5')
    y_predict = model.predict(X_test, batch_size=options.batch_size, verbose=1)

    num_total = y_predict[0].shape[0] * y_predict[0].shape[1]
    est_pitch = np.zeros(num_total)
    y_predict = np.reshape(y_predict[0], (num_total, y_predict[0].shape[2]))  # origin
    
    for i in range(y_predict.shape[0]):
        index_predict = np.argmax(y_predict[i, :])
        pitch_MIDI = pitch_range[np.int32(index_predict)]
        if pitch_MIDI >= 38 and pitch_MIDI <= 83:
            est_pitch[i] = 2 ** ((pitch_MIDI - 69) / 12.) * 440

    est_pitch = medfilt(est_pitch, 5)

    ''' save results '''

    PATH_est_pitch = os.path.join(output_dir, '%s.txt' % os.path.basename(filepath).replace(".mp3", ""))
    # PATH_est_pitch = output_dit+'/pitch_'+filepath.split('/')[-1]

    if not os.path.exists(os.path.dirname(PATH_est_pitch)):
        os.makedirs(os.path.dirname(PATH_est_pitch))
    # print(os.path.join(output_dir, '%s.txt' % os.path.basename(filepath).replace(".mp3", "")))
    f = open(PATH_est_pitch, 'w')
    for j in range(len(est_pitch)):
        est = "%.2f %.4f\n" % (0.01 * j, est_pitch[j])
        f.write(est)
    f.close()

    ''' Plot '''
    if options.figureON == True:
        start = 2000
        end = 7000
        fig = plt.figure()
        plt.imshow(X_spec[:,start:end], origin='lower')
        plt.plot(est_pitch[start:end],'r',linewidth=0.5)
        fig.tight_layout()
        plt.show()

# def parser():
#     p = argparse.ArgumentParser()
#     p.add_argument('-p', '--filepath',
#                    help='Path to input audio (default: %(default)s',
#                    type=str, default='test_audio_file.mp3')
#     p.add_argument('-gpu', '--gpu_index',
#                    help='Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s',
#                    type=int, default=None)
#     p.add_argument('-o', '--output_dir',
#                    help='Path to output folder (default: %(default)s',
#                    type=str, default='./results/')
#     return p.parse_args()


# def run():
    # options = Options()
    # args = parser()
    # main(args.filepath, args.output_dir, args.gpu_index)


if __name__ == '__main__':
    run()
    # ft_plt.ftt_plt()

# def JDC():    
#     AudioPATH = './'  # ex) AudioPATH = './dataset/*.mp3'
#     filePath = glob.glob(AudioPATH)
#     for fileName in filePath:
#         string = "python melodyExtraction_JDC.py "
#         string += fileName
#         os.system(string)
