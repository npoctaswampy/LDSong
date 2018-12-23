'''
pitch.py
Nathan William Pocta
Research Assistant at Brigham-Young University
Student at Clemson University
This module is used to implement a pitch detection algorithm. This algorithm is outlined in [1], as defined in "references.txt". The specifics are described
in the comments below. This algorithm is not perfect in the sense that it would be able to transcribe a song's notes (real-world
audio signals are very complex and can rarely be described without some error), but it does give a good indication for the range 
of pitches detected in a signal, and the strength of those pitches. 
'''
import librosa
import matplotlib.pyplot
import numpy as np
from scipy.signal import butter, lfilter, argrelextrema
import scipy
import math
import sklearn.preprocessing

class pitch:

	#Init describes the main flow of the algorithm
	def __init__(self,y,sr):

		song_duration = librosa.core.get_duration(y=y, sr=sr)
		
		self.computeSumEnvelope(y,sr)
		self.computeAutocorrelation()
		self.computePitchHistograms(song_duration)
		self.normalizeHistograms()
		self.computeFeatures()

	#This function filters out frequencies below and above 1000 hz. It then 
	#extracts the envelopes from each of those frequency bands. Finally, it sums
	#those two envelopes to give the sum envelope.
	def computeSumEnvelope(self,y,sr):
		new_fs = 22050
		EnvelopeDecimated = 250

		subBand1 = self.butter_filter(y,1000,sr,btype='low')
		subBand2 = self.butter_filter(y,1000,sr,btype='high')

		DecimateValue = math.ceil(sr/new_fs)

		envelope1 = self.Envelope(subBand1,DecimateValue,new_fs,sr,1000)
		envelope2 = self.Envelope(subBand2,DecimateValue,new_fs,sr,1000)

		self.sum_envelope = envelope1 + envelope2

	#(Probably could be named better) This function is used to calculate a filter
	#to be run on a signal. The particular filter is a butter filter. the caller of
	#this function is able to specify the filter type (low,high,bandpass).
	def butter_filt(self,cut,fs,order=5,btype='low'):
		nyq=0.5*fs
		cut=cut/nyq
		b,a=butter(order,cut,btype=btype)
		return b,a

	#This fucntion is used to both calculate a butter filter, and to filter the 
	#signal passed in the "data" signal. 
	def butter_filter(self,data,cut,fs,order=5,btype='low'):
		b,a = self.butter_filt(cut,fs,order=order,btype=btype)
		y=lfilter(b,a,data)
		return y

	#This function is used to extract the envelope for use in this algorithm. The oper-
	#ations performed include half-wave rectification and low-pass filtering.
	def Envelope(self,SubBand,DecimateValue,new_fs,Fs,cut):
		SubBand[SubBand<0] = 0
		LowPassSubBand = self.butter_filter(SubBand,cut,Fs,btype='low')
		return LowPassSubBand

	#For each window, the autocorrelation is computed. This reveals spikes corresponding to
	#frequencies that are present. (the further the spike is in the window, the lower the 
	#frequency)
	def computeAutocorrelation(self):
		signal = np.split(self.sum_envelope,range(0,len(self.sum_envelope),512))
		for x in range(0,len(signal)):
			signal[x] = librosa.core.autocorrelate(signal[x])
			signal[x] -= scipy.ndimage.filters.median_filter(signal[x], size=20)
			signal[x] = self.enhanceAuto(signal[x])
		self.autocorrelation = signal

	#This enhancement of the autocorrelation is suggested in [1] and is described in [8]. The 
	#autocorrelation has its zeros removed, is time-scaled by a factor of 2, and then is subtracted
	#from the zero-removed signal, and has its zeros removed again. 
	def enhanceAuto(self,signal):
		signal[signal<0] = 0
		scaled = signal[np.sort(np.append(np.arange(len(signal)),np.arange(len(signal))))]
		signal -= scaled[0:len(signal)]
		signal[signal<0] = 0
		return signal

	#This function is used to compute both the folded and unfolded histograms of pitch. The unfolded 
	#version contains values for each MIDI note number, revealing which pitches are detected and the
	#general octave of the song. The folded histogram gives an indication of the pitch classes of the 
	#song- containing a value in 12 bins- each indicating a pitch class.
	def computePitchHistograms(self,song_duration):
		frame_len = float(1.0/float(len(self.sum_envelope)))*float(song_duration) #time len of each frame
		self.full_hist = np.zeros(128) 						  #allocate mem for full hist(127 notes)
		self.folded = np.zeros(12)						  #allocate mem for folded hist(127 notes)
		for x in range(0,len(self.autocorrelation)):
			peaks = argrelextrema(self.autocorrelation[x],np.greater)[0]      #Gather local max in autocor window
			peaks = np.trim_zeros(peaks)					  #If peak occurs at 0, it will yield inf freq
			for y in range(0,min(len(peaks),3)):
				m = np.argmax(self.autocorrelation[x][peaks])		  #We find three max values in each window
				freq = int(1.0/float(peaks[m]*frame_len))		  #convert to frequency
				midi_note = (12*math.log((freq/440.0),2))+69.0            #Find MIDI representation
				self.full_hist[int(midi_note)] += self.autocorrelation[x][peaks[m]] #add to histograms
				self.folded[(7*(int(midi_note)%12))%12] += self.autocorrelation[x][peaks[m]]
				peaks = np.delete(peaks,m)

	#For more consistent features, histograms should be normalized.
	def normalizeHistograms(self):
		self.full_hist = (self.full_hist)/np.linalg.norm(self.full_hist)
		self.folded = (self.folded)/np.linalg.norm(self.folded)
		
	#Features that are extracted include the strength of max bins in both full and folded histograms,
	#the notes those maximums represent, and the sum of all the values in the full histogram (to indicate
	#the strength of the estimation algorithm.
	def computeFeatures(self):
		self.f0 = np.max(self.folded)
		self.u0 = np.max(self.full_hist)
		self.up0 = np.argmax(self.full_hist)
		self.fp0 = np.argmax(self.folded)
		self.hist_sum = np.sum(self.full_hist)
		
	#Plots both full and folded histograms
	def plotHistograms(self):
		matplotlib.pyplot.subplot(2,1,1)
		matplotlib.pyplot.plot(self.full_hist)
		matplotlib.pyplot.title('Full Histogram')
		matplotlib.pyplot.xlabel('MIDI Note Number')
		matplotlib.pyplot.ylabel('Strength of Note (across song)')
		matplotlib.pyplot.annotate('Strongest Note (up0) = ' + str(self.up0) + '\nStrongest Note Strength (u0): %.2f' % self.u0
					    +'\nSum of Note Strengths (hist_sum): %.2f' % self.hist_sum, 
					    xy=(float(float(5.0/8.0)*float(len(self.full_hist))),
					    float(float(6.0/8.0)*float(np.max(self.full_hist)))))
		matplotlib.pyplot.subplot(2,1,2)
		matplotlib.pyplot.plot(self.folded)
		matplotlib.pyplot.title('Folded Histogram')
		matplotlib.pyplot.xlabel('Pitch Class')
		matplotlib.pyplot.ylabel('Pitch Strength (across song)')
		matplotlib.pyplot.annotate('Strongest Pitch Class (fp0) = ' + str(self.fp0) + '\nStrongest Pitch Class Strength (f0): %.2f' % self.f0, 
					    xy=(float(float(5.0/8.0)*float(len(self.folded))),
					    float(float(6.0/8.0)*float(np.max(self.folded)))))
		matplotlib.pyplot.show()
		






















