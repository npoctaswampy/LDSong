'''
beat.py
Nathan William Pocta
Research Assistant at Brigham-Young University
Student at Clemson University
This module is used to implement a beat tracking algorithm. This algorithm is outlined in [1], as defined in "references.txt". The specifics are described
in the comments below. This algorithm uses autocorrelation to gather information about a song's regularity. Using this information, it is able to guess the
beat regularity in beats per minute and the strength of the beats. Some songs are more resistant than others to this algorithm, though most songs work rather
well. 
'''
import librosa
import matplotlib.pyplot
import numpy as np
from scipy.signal import butter, lfilter, argrelextrema
import scipy
import math


class beat:

	#Init describes the main flow of the algorithm
	def __init__(self,y,sr):

		song_duration = librosa.core.get_duration(y=y, sr=sr)
		
		self.computeAutocorrelation(y,sr)
		self.computeBeatHistogram(song_duration)
		self.computeFeatures()

	#This function combines subband separation, envelope extraction, envelope summing, and
	#autocorrelation into a single step. 
	def computeAutocorrelation(self,y,Fs):
		new_fs = 22050
		EnvelopeDecimated = 250

		#Break signal into several subbands
		SubBand1 = self.subBandDWT(y,Fs,2000,3000)
		SubBand2 = self.subBandDWT(y,Fs,3000,4000)
		SubBand3 = self.subBandDWT(y,Fs,4000,5000)
		SubBand4 = self.subBandDWT(y,Fs,5000,6000)
		SubBand5 = self.subBandDWT(y,Fs,6000,8000)
		SubBand6 = self.subBandDWT(y,Fs,8000,10000)

		DecimateValue = math.ceil(Fs/new_fs)

		#Extract envelopes from each
		Envelope1 = self.Envelope(SubBand1,DecimateValue,new_fs,Fs,2000,3000)
		Envelope2 = self.Envelope(SubBand2,DecimateValue,new_fs,Fs,3000,4000)
		Envelope3 = self.Envelope(SubBand3,DecimateValue,new_fs,Fs,4000,5000)
		Envelope4 = self.Envelope(SubBand4,DecimateValue,new_fs,Fs,5000,6000)
		Envelope5 = self.Envelope(SubBand5,DecimateValue,new_fs,Fs,6000,8000)
		Envelope6 = self.Envelope(SubBand6,DecimateValue,new_fs,Fs,8000,10000)

		#Downsample each subband for quicker processing
		EnvelopeDecimated1 = Envelope1[range(0,len(Envelope1),int(math.floor((Fs/DecimateValue)/EnvelopeDecimated)))]
		EnvelopeDecimated2 = Envelope2[range(0,len(Envelope2),int(math.floor((Fs/DecimateValue)/EnvelopeDecimated)))]
		EnvelopeDecimated3 = Envelope3[range(0,len(Envelope3),int(math.floor((Fs/DecimateValue)/EnvelopeDecimated)))]
		EnvelopeDecimated4 = Envelope4[range(0,len(Envelope4),int(math.floor((Fs/DecimateValue)/EnvelopeDecimated)))]
		EnvelopeDecimated5 = Envelope5[range(0,len(Envelope5),int(math.floor((Fs/DecimateValue)/EnvelopeDecimated)))]
		EnvelopeDecimated6 = Envelope6[range(0,len(Envelope6),int(math.floor((Fs/DecimateValue)/EnvelopeDecimated)))]

		#Add all envelopes together
		SumEnvelope = Envelope1 + Envelope2 + Envelope3 + Envelope4 + Envelope5 + Envelope6

		#Take autocorrelation
		CorrelationEnvelope = librosa.core.autocorrelate(SumEnvelope)

		CorrelationEnvelope = CorrelationEnvelope[range(0,len(CorrelationEnvelope),200)]

		#Find the smoothed version and subtract so that all values reside around 0.
		avg = scipy.ndimage.filters.median_filter(CorrelationEnvelope, size=200)
		self.correlation = CorrelationEnvelope-avg

		self.correlation[self.correlation<0] = 0

	#(Probably could be named better) This function is used to calculate a filter
	#to be run on a signal. The particular filter is a butter filter. the caller of
	#this function is able to specify the filter type (low,high,bandpass).
	def butter_filt(self,lowcut,highcut,fs,order=5,btype='band'):
		nyq=0.5*fs
		low=lowcut/nyq
		high=highcut/nyq
		b,a=butter(order,[low,high],btype=btype)
		return b,a

	#This fucntion is used to both calculate a butter filter, and to filter the 
	#signal passed in the "data" signal. 
	def butter_filter(self,data,lowcut,highcut,fs,order=5,btype='band'):
		b,a = self.butter_filt(lowcut,highcut,fs,order=order,btype=btype)
		y=lfilter(b,a,data)
		return y

	#Filter the signal to obtain a subband between L and H frequencies (in Hz)
	def subBandDWT(self,Signal,Fs,L,H):
		return self.butter_filter(Signal,L,H,Fs,btype='band')

	#Envelope extraction- full wave rectification, low pass fitering, decimation, and mean removal
	def Envelope(self,SubBand,DecimateValue,new_fs,Fs,L,H):
		SubBand = abs(SubBand)
		LowPassSubBand = self.butter_filter(SubBand,L,H,Fs,btype='low')
		bands = scipy.signal.decimate(LowPassSubBand, int(DecimateValue))
		MeanRemoval = bands-np.mean(bands)
		return MeanRemoval

	#Function computes the distance between beats and the strength of each beat, and adds the value to bins in
	#the beat histogram.
	def computeBeatHistogram(self,song_duration):
		#Compute the local maxima in the autocorrelation-corresponding to 
		#rhythmic events.
		peaks = argrelextrema(self.correlation,np.greater)[0]

		#Allocate memory for beat histogram
		self.beat_hist = np.zeros(201-40)

		#Compute distances between five rhythmic events at a time
		peaks_sections = np.split(peaks,range(0,len(peaks),5))

		#This step calculates the highest value in the set of (5) peaks and finds the distances between it
		#and the other peaks, adding the values of those peaks to the histogram. This emphasizes main beats,
		#and takes into account the subbeats.
		for x in range(1,len(peaks_sections)):
			peak_vals = self.correlation[peaks_sections[x]]
			max_loc = np.argmax(peak_vals)
			for y in range(0,len(peaks_sections[x])):
				if y!=max_loc:
					dist = float(float(abs(peaks_sections[x][max_loc]-peaks_sections[x][y]))/float(len(self.correlation)))
					den = dist * song_duration
					index = int(60/den)-40
					if index <= 160 and index >= 0:
						self.beat_hist[index] += peak_vals[y]

	#Computes the features that can be gathered from the beat histogram, as suggested in [1]
	def computeFeatures(self):
		#Find peaks in the beat histogram
		peaks = argrelextrema(self.beat_hist,np.greater)[0]

		#Calculate the period of the highest peak and the relative amplitude of that period
		if len(peaks) > 1:
			self.period0 = peaks[np.argmax(self.beat_hist[peaks])] + 40
			self.amplitude0 = np.max(self.beat_hist[self.period0-40])/np.sum(self.beat_hist)
		
			peaks = np.delete(peaks,np.argmax(self.beat_hist[peaks]))
		else:
			self.period0 = 0
			self.amplitude0 = 0
		#Calculate the period of the second highest peak, the relative amplitude of that period,
		#and the ratio of that period to the strongest
		if len(peaks) > 1:
			self.period1 = peaks[np.argmax(self.beat_hist[peaks])] + 40
			self.amplitude1 = np.max(self.beat_hist[self.period1-40])/np.sum(self.beat_hist)
			self.ratio_period1 = float(self.period1)/float(self.period0)		

			peaks = np.delete(peaks,np.argmax(self.beat_hist[peaks]))
		else:
			self.period1 = 0
			self.amplitude1 = 0
			self.ratio_period1 = 0
		#Calculate the period of the third highest peak and the relative amplitude of that period,
		#and the ratio of that period to the second strongest
		if len(peaks) > 1:
			self.period2 = peaks[np.argmax(self.beat_hist[peaks])] + 40
			self.amplitude2 = np.max(self.beat_hist[self.period2-40])/np.sum(self.beat_hist)
			self.ratio_period2 = float(self.period2)/float(self.period1)		

			peaks = np.delete(peaks,np.argmax(self.beat_hist[peaks]))
		else:
			self.period2 = 0
			self.amplitude2 = 0
			self.ratio_period2 = 0

	#Plots the beat histogram of the song.
	def plotBeatHistogram(self):
		matplotlib.pyplot.plot(self.beat_hist)
		matplotlib.pyplot.title('Beat Histogram')
		matplotlib.pyplot.annotate('Period 0 = ' + str(self.period0)+'\nPeriod 1 = ' + str(self.period1)
					    + '\nPeriod 2 = ' + str(self.period2) + '\nAmplitude 0: %.2f' % self.amplitude0 
					    + '\nAmplitude 1: %.2f' % self.amplitude1 + '\nAmplitude 2: %.2f' % self.amplitude2
					    + '\nRatio Period 1: %.2f' % self.ratio_period1 + '\nRatio Period 2: %.2f' % self.ratio_period2, 
					    xy=(int(float(5.0/8.0)*float(len(self.beat_hist))), 
					    int(float(6.0/8.0)*float(np.max(self.beat_hist)))))
		matplotlib.pyplot.xlabel('Beats Per Minute')
		matplotlib.pyplot.xticks(np.arange(0,len(self.beat_hist),10),40+np.arange(0,len(self.beat_hist),10))
		matplotlib.pyplot.ylabel('Beat Strength')
		matplotlib.pyplot.show()

		
	
		
