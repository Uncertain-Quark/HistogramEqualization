# Main python file for performing histogram equalization
import glob, os, sys, torch, numpy
from makeHistogram import HistEq

sourceWavFolder = sys.argv[1]
targetWavFolder = sys.argv[2]
sourceInferenceWavFolder = sys.argv[3]
targetInferenceWavFolder = sys.argv[4]

sourcePtFolder = './Source'
targetPtFolder = './Target'
sourceInferencePtFolder = './SourceInf'
targetInferencePtFolder = './TargetInf'

os.system('mkdir -p {}'.format(sourcePtFolder))
os.system('mkdir -p {}'.format(targetPtFolder))
os.system('mkdir -p {}'.format(sourceInferencePtFolder))
os.system('mkdir -p {}'.format(targetInferencePtFolder))

# generating the .pt files which is the mel spectrogram
os.system('python3 makeMelSpectrogram.py {} {}'.format(sourceWavFolder,sourcePtFolder))
os.system('python3 makeMelSpectrogram.py {} {}'.format(targetWavFolder,targetPtFolder))
os.system('python3 makeMelSpectrogram.py {} {}'.format(sourceInferenceWavFolder,sourceInferencePtFolder))

sourcePtFiles = glob.glob(sourcePtFolder + '/*.pt')
targetPtFiles = glob.glob(targetPtFolder + '/*.pt')
sourceInferencePtFiles = glob.glob(sourceInferencePtFolder + '/*.pt')

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
hist_eq_class.train(sourceMelSpec.T,targetMelSpec.T)

for file in sourceInferencePtFiles :
    melInf = torch.load(file).numpy()
    targetMelInf = hist_eq_class.inference(melInf.T)
    targetMelInfPath = targetInferencePtFolder + '/' + file,rstrip('/').split('/')[-1]
    torch.save(targetMelInf.T, targetMelInfPath)

targetInferencePtFiles = glob.glob(targetInferencePtFolder + '/*.pt')
os.system('ls {}/*.pt > mel_files.txt'.format(targetInferencePtFiles))
os.system('python3 inference.py -f mel_files.txt -w checkpoints/waveglow_10000 -o {} --is_fp16 -s 0.6'.format(targetInferenceWavFolder))

#hist_eq_class.inference(sourceMelSpec.T)
