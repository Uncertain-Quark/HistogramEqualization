# Main python file for performing histogram equalization
import glob, os, sys, torch, numpy
from makeHistogram import HistEq

sourceWavFolder = sys.argv[1]
targetWavFolder = sys.argv[2]

sourcePtFolder = './Source'
targetPtFolder = './Target'
os.system('mkdir -p {}'.format(sourcePtFolder))
os.system('mkdir -p {}'.format(targetPtFolder))

# generating the .pt files which is the mel spectrogram
os.system('python3 makeMelSpectrogram.py {} {}'.format(sourceWavFolder,sourcePtFolder))
os.system('python3 makeMelSpectrogram.py {} {}'.format(targetWavFolder,targetPtFolder))

sourcePtFiles = glob.glob(sourcePtFolder + '/*.pt')
targetPtFiles = glob.glob(targetPtFolder + '/*.pt')

# reading the .pt files and enabling the histogram range creation
sourceMelSpec = torch.load(sourcePtFiles[0]).numpy()
for file in sourcePtFiles[1:]:
    sourceMelSpec = numpy.concatenate((sourceMelSpec,torch.load(file).numpy()), axis=1)
print(sourceMelSpec.shape)

targetMelSpec = torch.load(targetPtFiles[0]).numpy()
for file in targetPtFiles[1:]:
    targetMelSpec = numpy.concatenate((targetMelSpec,torch.load(file).numpy()), axis=1)
print(targetMelSpec.shape)

hist_eq_class = HistEq(80,80)
hist_eq_class.train(sourceMelSpec,targetMelSpec)
