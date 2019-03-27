% Do a frequency analysis of the wav files with normalised frequency, save the power spectra into freq_components.mat. 
load('wavfilenames.mat');
cd('/home/hans/Documents/Year 4/Advanced Machine Learning/Assignment/Data/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files');
num_of_wav_files = length(wavfilenames);
frequency_points = 4096;
freq_components = zeros(num_of_wav_files, frequency_points);
for i = 1:num_of_wav_files
    disp(i);
    signal = audioread(wavfilenames(i));
    power_spectra = pspectrum(signal);
    power_spectra = log(power_spectra);
    freq_components(i,:) = transpose(power_spectra);
end
csvwrite('freq_components.csv', freq_components);
cd('/home/hans/Documents/Year 4/Advanced Machine Learning/Assignment/MATLAB');