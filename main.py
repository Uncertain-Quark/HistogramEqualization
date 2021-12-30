# Main python file for performing histogram equalization
import glob, os, sys, torch, numpy
from makeHistogram import HistEq

sampleRate = 22050

def convertRate(folder, destFolder, sampleRate):
    files = glob.glob(folder + '/*.wav')
    for f in files : 
        destfile = destFolder + '/' + f.split('/')[-1]
        os.system('ffmpeg -i {} -ac 1 -ar {} {}'.format(f, sampleRate, destfile))

sourceFolder = sys.argv[1]
targetFolder = sys.argv[2]
sourceInfFolder = sys.argv[3]
targetInfFolder = sys.argv[4]

sourceWavFolder = 'data/Source'
targetWavFolder = 'data/Target'
sourceInferenceWavFolder = 'data/SourceInf'
targetInferenceWavFolder = 'data/TargetInf'

os.system('mkdir -p data')
os.system('mkdir -p {}'.format(sourceWavFolder))
os.system('mkdir -p {}'.format(targetWavFolder))
os.system('mkdir -p {}'.format(sourceInferenceWavFolder))
os.system('mkdir -p {}'.format(targetInferenceWavFolder))

convertRate(sourceFolder, sourceWavFolder, sampleRate)
convertRate(targetFolder, targetWavFolder, sampleRate)
convertRate(sourceInfFolder, sourceInferenceWavFolder, sampleRate)
#convertRate(targetInfFolder, targetInferenceWavFolder, sampleRate)

waveglowCheckpoint = '/tts/anusha/journal_phrase_TTS/waveglow/Hindi/male/mono/waveglow_v1_5hrs_using_pretrained_ljspeech_model/checkpoints/waveglow_15000'

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

print(sourceInferencePtFiles)
for file in sourceInferencePtFiles :
    melInf = torch.load(file).numpy()
    targetMelInf = hist_eq_class.inference(melInf.T)
    targetMelInfPath = targetInferencePtFolder + '/' + file.rstrip('/').split('/')[-1]
    torch.save(torch.from_numpy(targetMelInf.T), targetMelInfPath)

targetInferencePtFiles = glob.glob(targetInferencePtFolder + '/*.pt')
os.system('ls {}/*.pt > mel_files.txt'.format(targetInferencePtFolder))
os.system('python3 waveglow/inference.py -f mel_files.txt -w {} -o {} --is_fp16 -s 0.6'.format(waveglowCheckpoint, targetInferenceWavFolder))

#hist_eq_class.inference(sourceMelSpec.T)
