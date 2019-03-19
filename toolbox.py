import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from pydub import AudioSegment
import os


class AudioAugmentation:

    def read_audio_file(self, file_path):
        input_length = 16000
        data = librosa.core.load(file_path)[0]
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

    def add_noise(self, data):
        noise = np.random.randn(len(data))
        data_noise = data + 0.005 * noise
        return data_noise

    def shift(self, data):
        return np.roll(data, 1600)

    def stretch(self, data, rate=1):
        input_length = 16000
        data = librosa.effects.time_stretch(data, rate)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

    def write_audio_file(self, file, data, sample_rate=16000):
        librosa.output.write_wav(file, data, sample_rate)

    def plot_time_series(self, data):
        fig = plt.figure(figsize=(14, 8))
        plt.title('Raw wave ')
        plt.ylabel('Amplitude')
        plt.plot(np.linspace(0, 1, len(data)), data)
        plt.show()


def check_dir(directory):
    '''

    Args:
        directory: Path of directory

    Returns:

    This function checks whether the input directory exists and proceeds to create it if it does not
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass


def fromfolder(datadir_in, headers):
    '''

    Args:
        datadir_in: The directory of the raw data files
        headers: column headers for Pandas dataframe

    Returns:
        nameList_out: The names of all the files found
        data_frames: The list of all dataframes made from each TXT file
        A concatenation of all the dataframes to preserve pandas series datatype
    '''
    nameList_out = []
    data_frames = []
    for filename in os.listdir(datadir_in):
        if filename.endswith(".txt"):
            name = os.path.splitext(filename)[0]
            nameList_out.append(filename)
            df = pd.read_csv(datadir_in + '/' + filename, sep="\t", header=None, names=headers)
            df.rename(columns={0: name}, inplace=True)
            data_frames.append(df)
    return nameList_out, data_frames, pd.concat(data_frames, keys=nameList_out, axis=1)


def sort_segment(times, c, w, path, filename, dir):
    '''
    Args:
        times: beginning-end of each breathing cycle
        c: list of crackles associated with each breathing cycle
        w: list of wheezes associated with each breathing cycle
        path: The path to the wav file
        filename: The name of the audio file (with .wav extension)
        dir: The directories for sorting the segmented audio files

    Returns:

    This function takes in the path to each audio file and segments it based on the breathing cycles
    provided in the text file. Each segment is then renamed based on it's index in the array of
    breathing cycles. Finally each segment is exported to it's according location based on the
    presence/absence of a crackle/wheeze
    '''
    for i in range(len(times)-1):
        cycle = '_cycle_' + str(i)
        start = 1000 * times[i]
        end = 1000 * times[i+1]
        newAudio = AudioSegment.from_wav(path)  # path is defined above
        newAudio = newAudio[start:end]
        if not c[i] and not w[i]:
            newAudio.export(dir[0] + '/' + filename + cycle + '.wav', format="wav")

        elif not c[i] and w[i]:
            newAudio.export(dir[1] + '/' + filename + cycle + '.wav', format="wav")

        elif c[i] and not w[i]:
            newAudio.export(dir[2] + '/' + filename + cycle + '.wav', format="wav")

        elif c[i] and w[i]:
            newAudio.export(dir[3] + '/' + filename + cycle + '.wav', format="wav")

        else:
            print('crackle is {} and wheezes is {}'.format(c[i], w[i]))

        print('file_exported')


def get_max_cycle(cycles):
    '''

    Args:
        cycles: breathing cycles

    Returns:

    This simply finds the longest breathing cycle in a patient's recording
    '''

    max_dt = 0
    for i in range(len(cycles)-1):
        delta_t = cycles[i+1] - cycles[i]
        if delta_t > max_dt:
            max_dt = delta_t
    return max_dt
