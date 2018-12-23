'''
alternate_scales.py
Nathan William Pocta
Research Assistant at Brigham-Young University
Student at Clemson University
This class is used to calculate the Mel Frequency Cepstral Coefficients of a song. the Mel Frequency Cepstral
Coefficients are used to describe the shape of a frequency domain representation of a signal in a logarithmic
scale that attempts to replicate the human perception of sound. More information on MFCCs can be found in [9],
but for this calculation, the librosa python library is used.
'''
import librosa
import numpy as np

class alternate_scales:

	def __init__(self,y,sr,num_mfcc = 6, nfft = 2048):

		self.y = y
		self.sr = sr

		self.song_duration = librosa.core.get_duration(y=self.y, sr=self.sr)
		
		self.converting_leny = len(self.y)

		self.computeMFCC(num_mfcc,nfft)

	#Computation of Mel Frequency Cepstral Coefficients
	def computeMFCC(self,num_mfcc, mfcc_nfft):
		self.mfcc = librosa.feature.mfcc(y=self.y,sr=self.sr,n_mfcc=num_mfcc,n_fft=mfcc_nfft, hop_length=mfcc_nfft/4)

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
		step = int(np.shape(self.mfcc)[1]/self.song_duration)
		borders = range(0,np.shape(self.mfcc)[1],step)

		self.mfcc_sec = self.findStats(self.mfcc[0],borders)

		for x in range(1,len(self.mfcc)):
			self.mfcc_sec = np.vstack([self.mfcc_sec,self.findStats(self.mfcc[x],borders)])

	#This function is called by the program's top level to provide a set of features (statistics of each feature by the second)
	#that is used to segment the song into sections. 
	def buildSegmentingFeatures(self):
		self.featuresPerSecond()
		ret = self.mfcc_sec[0]
		for x in range(1,len(self.mfcc)):
			ret = np.vstack([ret,self.mfcc_sec[x]])

		return ret

	#This function is called from the program's top level to provide a set of features (statistics of each feature by the segment
	#as defined in the segment class algorithm) that are ultimately to be used in the user's application.
	def segment(self,borders):
		borders = np.floor((np.shape(self.mfcc)[1]*borders)/self.converting_leny)

		self.mfcc_seg = self.findStats(self.mfcc[0],borders)

		for x in range(1,len(self.mfcc)):
			self.mfcc_seg = np.vstack([self.mfcc_seg,self.findStats(self.mfcc[x],borders)])

