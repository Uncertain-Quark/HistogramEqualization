# Code to make mel spectrograms from WAV files and store them in .txt files
import os, sys, glob

folderWav = sys.argv[1]
wavFiles = glob.glob(folderWav + '/*.wav')

for wav in wavFiles:
    # for each wavfile, make the .pt file and the corresponding text file
    print('Processing {}'.format(wav))
    
