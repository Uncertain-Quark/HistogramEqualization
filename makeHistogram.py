# Code to make histograms given a list of audio text files
import numpy as np

num_bins = 80 # emperical can be any number
num_columns = 80 # number of mel features in spectrogram

class HistEq :
    def __init__(self,num_bins,num_columns):
        self.num_bins = num_bins
        self.num_columns = num_columns
        self.rangesHistogramSource, self.rangesHistogramTarget = np.zeros((self.num_columns,self.num_bins,2)), np.zeros((self.num_columns,self.num_bins,2))

    def oneColumnGenerate(melVectorOneColumn):
        # Function to return the ranges of the histogram bins for a specific column of Mel Spectrum
        rangesOneColumn = np.zeros((num_bins,2))

        # Sort the column of mel vectors
        sortedMelVectorOneColumn = np.sort(melVectorOneColumn)

        # Storing the ranges of histogram bins
        for i in range(num_bins):
            lowerRange, higherRange = i*int(melVectorOneColumn.shape[0]/num_bins), (i+1)*int(melVectorOneColumn.shape[0]/num_bins)
            rangesOneColumn[i,0], rangesOneColumn[i,1] = sortedMelVectorOneColumn[lowerRange], sortedMelVectorOneColumn[higherRange]

        return rangesOneColumn

    def train(self,sourceMelSpectrograms, targetMelSpectrograms):
        # generate histogram ranges for every column
        rangesHistogramSource, rangesHistogramTarget = np.zeros((self.num_columns,self.num_bins,2)), np.zeros((self.num_columns,self.num_bins,2))

        for i in range(self.num_columns):
            rangesHistogramSource[i,:,:] = self.oneColumnGenerate(sourceMelSpectrograms[:,i])
            rangesHistogramTarget[i,:,:] = self.oneColumnGenerate(targetMelSpectrograms[:,i])

        self.rangesHistogramSource = rangesHistogramSource
        self.rangesHistogramTarget = rangesHistogramTarget

    def storeRangeArray(self,arr,filename):
        # storing a given numpy array in a text file
        np.savetxt(filename,arr)

    def inference(self,sourceMelSpectrgram):
        # Code to convert a mel spectrogram given source mel spectrogram
        convertedMelSpectrogram = np.zeros(sourceMelSpectrgram.shape)
        for i in range(sourceMelSpectrgram.shape[0]):
            for i in range(sourceMelSpectrgram.shape[1]):
                # convert each and every element of the source mel spectrogram to the target
        return convertedMelSpectrogram
