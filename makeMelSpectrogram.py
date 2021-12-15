# Code to make mel spectrograms from WAV files and store them in .txt files
import os, sys, glob

folderWav = sys.argv[1]
outputPt = sys.argv[2]
wavFiles = glob.glob(folderWav + '/*.wav')

# Getting .pt files from WAV files
os.system('ls {}/*.wav > train_files.txt'.format(folderWav))
os.system('ls {}/*.wav > test_files.txt'.format(folderWav))
os.system('python mel2samp.py -f test_files.txt -o {} -c config.json'.format(outputPt))
