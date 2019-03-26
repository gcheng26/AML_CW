'''
This script trains a model to classify features (crackles and wheezes) in .wav files.
'''

import pandas as pd
import numpy as np
import os, glob, csv
import matplotlib.pyplot as plt
from pydub import AudioSegment
import IPython
import warnings
warnings.filterwarnings('ignore')

# DATA PREPROCESSING
data_path = '../../Data/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
# Load all .wav file names
wav_files = []
txt_files = []
for file in os.listdir(data_path):
    if file.endswith('.wav'):
        wav_files.append(file)
    elif file.endswith('.txt'):
        txt_files.append(file)
wav_files.sort()
txt_files.sort()
''' To export as csv
with open('wav_filenames.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in wav_files:
        writer.writerow([val])
'''
col_names = ['Beginning_of_respiratory_cycle', 'End_of_respiratory_cycle', 'Presence/absence_of_crackles', 'Presence/absence_of_wheezes']
wav_breakdown = pd.read_csv(data_path+txt_files[0], sep="\t", header=None, names=col_names)
wav_breakdown.head(5)