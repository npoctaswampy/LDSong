'''
time_feat.py
Nathan William Pocta
Research Assistant at Brigham-Young University
Student at Clemson University
This class is used to compute and calculate properties of features that can be extracted from the time domain. The windows in the time
domain are calculated by cutting the signal into equal-sized segments and finding properties of each window. The features
that were chosen to be extracted were chosen based on frequency of appearance in "Music Information Retrieval" literature, and speed 
of computation. The papers that were taken into account for this module are in [0], [1], [2], [3], [4], [5], [6], and [7], as defined in
the 'references.txt' file included with this code.
'''
import numpy as np
import librosa.feature as lf
import matplotlib.pyplot
import librosa

class time_feat:

	def __init__(self,y,sr,nfft=2048):
		self.y=y
		self.sr=sr
		self.nfft = nfft
		self.song_duration = librosa.core.get_duration(y=self.y, sr=self.sr)
		
		self.converting_leny = len(self.y)

		#Section y into 'windows' of time that match the resolution used in
		#the stft for spectral features
		self.winlen = self.nfft/4
		self.computeWindowed()

		self.computeFeatures()


	#Compute the sections of the song that will serve as "windows"
	def computeWindowed(self):
		if len(self.y)%self.winlen != 0:
			self.y = np.append(self.y,np.zeros(self.winlen-(len(self.y)%self.winlen)))

		num_wins = len(self.y)/self.winlen

		self.windowed_signal = np.vstack(np.split(self.y,num_wins)).T

	#Compute features
	def computeFeatures(self):
		self.zero_crossing = lf.zero_crossing_rate(self.y, frame_length=self.nfft, hop_length=self.nfft/4)[0]
		self.computeRMS()
		self.computeLowEnergy()
		self.computeAmplitudeEnvelope()
		self.computeLoudness()

	#Computation of the root mean square of each window. This feature was computed based
	#on the definition in [2] and [5] of references listed in "references.txt".
	def computeRMS(self):
		self.rms = np.sqrt(np.mean(self.windowed_signal,axis=0)**2)

	#Computation of the Low Energy Rate of each window. This feature was computed based
	#on the definition in [2] and [5] of references listed in "references.txt".
	def computeLowEnergy(self):
		self.low_energy = (self.windowed_signal<np.mean(self.windowed_signal,axis=0)).sum(axis=0)

	#Computation of the amplitude envelope (max value in each window). This feature was computed based
	#on the definition in [2] and [5] of references listed in "references.txt".
	def computeAmplitudeEnvelope(self):
		self.amplitude_envelope = np.max(self.windowed_signal,axis=0)

	#Computation of the loudness of the signal in each window. This feature was computed based
	#on the definition in [2] and [5] of references listed in "references.txt".
	def computeLoudness(self):
		self.loudness = self.rms**.23

	#Plot all time-domain features together
	def plotFeatures(self):

		matplotlib.pyplot.subplot(2,3,1)
		matplotlib.pyplot.plot(self.zero_crossing)
		matplotlib.pyplot.title('Zero Crossing',size = 12)
		matplotlib.pyplot.ylabel('Times Signal Crossed 0 per Frame',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)
		matplotlib.pyplot.subplot(2,3,2)
		matplotlib.pyplot.plot(self.rms)
		matplotlib.pyplot.title('Root Mean Squared',size = 12)
		matplotlib.pyplot.ylabel('RMS Value',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)
		matplotlib.pyplot.subplot(2,3,3)
		matplotlib.pyplot.plot(self.low_energy)
		matplotlib.pyplot.title('Low Energy',size = 12)
		matplotlib.pyplot.ylabel('Percentage of Signal Below Avg',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)
		matplotlib.pyplot.subplot(2,3,4)
		matplotlib.pyplot.plot(self.amplitude_envelope)
		matplotlib.pyplot.title('Amplitude Envelope',size = 12)
		matplotlib.pyplot.ylabel('Highest Value in Frame',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)
		matplotlib.pyplot.subplot(2,3,5)
		matplotlib.pyplot.plot(self.loudness)
		matplotlib.pyplot.title('Loudness',size = 12)
		matplotlib.pyplot.ylabel('Loudness Value',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)

		matplotlib.pyplot.suptitle('Time Domain Features',size = 16)

		matplotlib.pyplot.show()


	#Finds statistics in a section of the song. The "borders" variable determines where
	#each section begins and ends. For example, if the "borders" variable contains only the start and
	#the end of the song, this function gathers statistics across the entire song.
	def findStats(self,signal,borders,normalize = True):
		seg_stats_v = np.array([])
		seg_stats_m = np.array([])
		segmented_signal = np.split(signal,borders)
		for x in range(1,len(segmented_signal)-1):
			seg_stats_v = np.append(seg_stats_v, np.var(segmented_signal[x]))
			seg_stats_m = np.append(seg_stats_m, np.mean(segmented_signal[x]))

		if normalize == True:
			seg_stats_v = (seg_stats_v/np.linalg.norm(seg_stats_v))
			seg_stats_m = (seg_stats_m/np.linalg.norm(seg_stats_m))

		return np.vstack([seg_stats_v,seg_stats_m])

	#Function is used to compute statistics of each feature for each second of the
	#song.
	def featuresPerSecond(self):

		#This step creates the borders that are used to calculate statistics
		#in findStats.
		step = int(np.shape(self.windowed_signal)[1]/self.song_duration)
		borders = range(0,np.shape(self.windowed_signal)[1],step)

		self.zero_crossing_sec = self.findStats(self.zero_crossing,borders)
		self.rms_sec = self.findStats(self.rms,borders)
		self.low_energy_sec = self.findStats(self.low_energy,borders)
		self.amplitude_envelope_sec = self.findStats(self.amplitude_envelope,borders)
		self.loudness_sec = self.findStats(self.loudness,borders)

	#This function is called by the program's top level to provide a set of features (statistics of each feature by the second)
	#that is used to segment the song into sections. 
	def buildSegmentingFeatures(self):
		self.featuresPerSecond()
		return np.vstack([self.zero_crossing_sec,self.rms_sec,self.low_energy_sec,self.amplitude_envelope_sec,self.loudness_sec])

	#This function is called from the program's top level to provide a set of features (statistics of each feature by the segment
	#as defined in the segment class algorithm) that are ultimately to be used in the user's application.
	def segment(self,borders):
		borders = np.floor((np.shape(self.windowed_signal)[1]*borders)/self.converting_leny)

		self.zero_crossing_seg = self.findStats(self.zero_crossing,borders,normalize = False)
		self.rms_seg = self.findStats(self.rms,borders,normalize = False)
		self.low_energy_seg = self.findStats(self.low_energy,borders,normalize = False)
		self.amplitude_envelope_seg = self.findStats(self.amplitude_envelope,borders,normalize = False)
		self.loudness_seg = self.findStats(self.loudness,borders,normalize = False)
















