'''
File    : train.py
Author  : Jay Santokhi (jks1g15)
Brief   : Carries out training using either Spectrogram or MFCC inputs
Usage   : python train.py -p <'Spectrogram' or 'MFCC'> -c <'BoVW' or 'NN'>
'''
from Data_Utils import wav_to_spectrogram
from Data_Utils import load_wav_files
from Data_Utils import wav_to_MFCC
from models import NN_classifier
from models import BoVW_classifier

import argparse
import sys


def define_args():
    ''' Defines the script arguments.
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--PreProcessing", required=True,
                    help=" 'Spectrogram' or 'MFCC' ")
    ap.add_argument("-c", "--Classifier", required=True,
                    help=" 'BoVW' or 'NN' ")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = define_args()
    pp = args["PreProcessing"]
    classifier = args["Classifier"]

    # train_path = 'Dataset/train/'
    # valid_path = 'Dataset/valid/'

    train_path = 'Dataset_edit/train/'
    # valid_path = 'Dataset_edit/valid/'
    valid_path = 'Dataset_edit/test/'

    # train_path = 'NO_ASTHMA_LRTI_TEST_Cycle_based_Train_Test_Val_Split/train'
    # valid_path = 'NO_ASTHMA_LRTI_TEST_Cycle_based_Train_Test_Val_Split/valid'

    n_wav_files_t, wav_files_t, class_labels_t = load_wav_files(train_path)
    n_wav_files_v, wav_files_v, class_labels_v = load_wav_files(valid_path)

    if pp == 'MFCC':
        print('Using MFCC')
        print('Loading training set')
        # trainX,trainY = wav_to_MFCC(n_wav_files_t, wav_files_t, class_labels_t, 15000, 2966)
        trainX,trainY = wav_to_MFCC(n_wav_files_t, wav_files_t, class_labels_t, 15000, 3894)
        print('Loading validation set')
        # validX, validY = wav_to_MFCC(n_wav_files_v, wav_files_v, class_labels_v, 15000, 642)
        validX, validY = wav_to_MFCC(n_wav_files_v, wav_files_v, class_labels_v, 15000, 2788) #5234 #2788
    elif pp == 'Spectrogram':
        print('Using Spectrogram')
        print('Loading training set')
        # trainX,trainY = wav_to_spectrogram(n_wav_files_t, wav_files_t, class_labels_t, 15000, 2966)
        trainX, trainY = wav_to_spectrogram(n_wav_files_t, wav_files_t, class_labels_t, 15000, 3894)
        print('Loading validation set')
        # validX, validY = wav_to_spectrogram(n_wav_files_v, wav_files_v, class_labels_v, 15000, 642)
        validX, validY = wav_to_spectrogram(n_wav_files_v, wav_files_v, class_labels_v, 15000, 5234)

    print(trainX.shape)
    print(trainY.shape)

    print(validX.shape)
    print(validY.shape)

    if classifier == 'BoVW':
        BoVW_classifier(trainX, trainY, validX, validY)
    elif classifier == 'NN'
        NN_classifier(trainX, trainY, validX, validY, pp)
