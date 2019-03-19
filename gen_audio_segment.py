from toolbox import *


'''Change the following to your according directories then run the code
'''
path = 'C:/Users/geoff/OneDrive/Desktop/AML/Respiratory_Sound_Database/audio_and_txt_files'

directories = [R'C:\Users\geoff\OneDrive\Desktop\AML\Respiratory_Sound_Database\data\C0W0',
               R'C:\Users\geoff\OneDrive\Desktop\AML\Respiratory_Sound_Database\data\C0W1',
               R'C:\Users\geoff\OneDrive\Desktop\AML\Respiratory_Sound_Database\data\C1W0',
               R'C:\Users\geoff\OneDrive\Desktop\AML\Respiratory_Sound_Database\data\C1W1']
col_names = ['Beginning_of_respiratory_cycle', 'End_of_respiratory_cycle', 'Presence/absence_of_crackles',
             'Presence/absence_of_wheezes']


audio_files, audio_dataframe_list, audio_dataframe = fromfolder(path, col_names)


for i in directories:
    check_dir(i)

count = 1
max_cycles = 0
for filename, audio_file in zip(audio_files, audio_dataframe_list):
    cycles = audio_file[col_names[0]].to_numpy()
    crackles = audio_file[col_names[2]].to_numpy()
    wheezes = audio_file[col_names[3]].to_numpy()
    filename = filename[:-3] + 'wav'
    full_path = path + '/' + filename
    # x = get_max_cycle(cycles)
    # if x > max_cycles:
    #     max_cycles = x
    #     print('new max: {} from {}'.format(x, filename))
    # else:
    #     pass
    sort_segment(cycles, crackles, wheezes, full_path, filename, directories)
    count = count + 1
    print(count)

