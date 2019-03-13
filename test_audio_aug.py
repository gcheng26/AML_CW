from toolbox import AudioAugmentation

file_path = 'C:/Users/geoff/OneDrive/Desktop/AML/Respiratory_Sound_Database/audio_and_txt_files/101_1b1_Al_sc_Meditron.wav'

aa = AudioAugmentation()

data = aa.read_audio_file(file_path)

# aa.plot_time_series(data)

data_noise = aa.add_noise(data)

# aa.plot_time_series(data_noise)