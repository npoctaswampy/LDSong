'''
spectral_feat.py
Nathan William Pocta
Research Assistant at Brigham-Young University
Student at Clemson University
This class is used to compute and calculate properties of features that can be extracted from the frequency domain. The frequency
domain is calculated using the short-time Fourier transform, as implemented by the incredibly useful librosa library. The features
that were chosen to be extracted were chosen based on frequency of appearance in "Music Information Retrieval" literature, and speed 
of computation. The papers that were taken into account for this module are in [0], [1], [2], [3], [4], [5], [6], and [7], as defined in
the 'references.txt' file included with this code.
'''
import numpy as np
import librosa
import matplotlib.pyplot
import scipy.stats as ss
import scipy.signal
import math

class spectral_feat:
	
	def __init__(self, y, sr, nfft = 2048):
		self.y = y
		self.sr = sr
		self.song_duration = librosa.core.get_duration(y=self.y, sr=self.sr)

		self.converting_leny = len(self.y)
		
		self.nfft = nfft

		#compute short time fourier transform
		self.stft = librosa.core.stft(y,n_fft=nfft)

		#to simplify, we only want magnitude
		self.stft = librosa.core.magphase(self.stft)[0]

		#compute spectral features
		self.computeFeatures()

	#Computation of all spectral features. The features selected 
	def computeFeatures(self):
		self.computeMeanStft()
		self.computePk()
		self.computeFk()
		self.computeCentroid()
		self.computeSpread()
		self.computeSkewness()
		self.computeKurtosis()
		self.computeFrameEnergy()
		self.computeRolloff()
		self.computeSlope()
		self.computeDecrease()
		self.computeFlatness()
		self.computeCrest()
		self.computeFlux()

	#Computation of the mean amplitude in each stft window
	def computeMeanStft(self):
		self.stft_mean = np.mean(self.stft,axis=0)
		self.stft_mean[self.stft_mean==0] = np.min(self.stft_mean[self.stft_mean > 0.0])

	#Computation of the normalized form of the amplitude of STFT. This computation is 
	#outlined in [0] in the 'references.txt' file included with this code.
	def computePk(self):
		sum_stft = np.sum(self.stft,axis=0)
		sum_stft[sum_stft == 0] = np.min(sum_stft[sum_stft > 0])
		self.pk = self.stft.astype(float)/np.array([sum_stft])

	#Computation of matrix that holds values of frequencies in columnsThis computation is 
	#outlined in [0] in the 'references.txt' file included with this code.
	def computeFk(self):
		#nyquist is the highest represenatable freq
		nyquist = float(self.sr)/2
		self.fk = ((np.array(range(1,len(self.stft)+1)))*nyquist)/len(self.stft)
		self.fk = np.tile(self.fk,(np.shape(self.stft)[1],1)).T

	#Computation of the centroid, or center of gravity of the spectral shape. This computation is 
	#outlined in [0] in the 'references.txt' file included with this code.
	def computeCentroid(self):
		self.centroid = np.sum(self.pk*self.fk,axis=0)

	#Computation of spread, or std dev about centroid. This computation is 
	#outlined in [0] in the 'references.txt' file included with this code.
	def computeSpread(self):
		self.spread = np.sum(((self.fk-np.array([self.centroid]))**2)*self.pk,axis=0)**0.5
		self.spread[self.spread==0] = np.min(self.spread[self.spread != 0.0])

	#Computation of skewness, or asymmetry of the spectrum around its mean value. This computation is 
	#outlined in [0] in the 'references.txt' file included with this code.
	def computeSkewness(self):
		self.skewness = np.sum(((self.fk-np.array([self.centroid]))**3)*self.pk,axis=0)/(self.spread**3)

	#Computation of kurtosis, or a measure of the flatness of the spectrum around its mean value. This computation is 
	#outlined in [0] in the 'references.txt' file included with this code.
	def computeKurtosis(self):
		self.kurtosis = np.sum(((self.fk-np.array([self.centroid]))**4)*self.pk,axis=0)/(self.spread**4)
	
	#Computation of the total energy in each stft frame
	def computeFrameEnergy(self):
		self.frame_energy = np.sum(self.stft,axis=0)
		self.frame_energy[self.frame_energy==0] = np.min(self.frame_energy[self.frame_energy > 0.0])

	#Computation of spectral rolloff: value aove 95% of all energy in frame. his computation is 
	#outlined in [1] in the 'references.txt' file included with this code. 
	def computeRolloff(self):
		self.rolloff = 0.95*np.sum(self.stft,axis=0)

	#Computation of spectral slope, indicating overall slope of spectrum. This computation is 
	#outlined in [0] in the 'references.txt' file included with this code.
	def computeSlope(self):
		slope_m = 1.0/self.frame_energy.astype(float)
		slope_n1 = (len(self.stft)*np.sum(self.fk*self.stft,axis=0))
		slope_n2 = np.sum(self.fk,axis=0)*np.sum(self.stft,axis=0)
		slope_d = (len(self.stft)*np.sum(self.fk**2,axis=0))-(np.sum(self.fk,axis=0)**2)
		self.slope = slope_m*((slope_n1-slope_n2)/slope_d)

	#Computation of spectral decrease, averages the set of slopes between frequency fk and f1. This computation is 
	#outlined in [0] in the 'references.txt' file included with this code.
	def computeDecrease(self):
		decrease_m1 = 1.0/self.frame_energy.astype(float)
		decrease_m2_d = np.insert(np.array(range(1,len(self.stft))),0,1)
		decrease_m2_d = np.tile(decrease_m2_d,(np.shape(self.stft)[1],1)).T
		decrease_m2_n = self.stft-self.stft[0]
		self.decrease = decrease_m1*np.sum(decrease_m2_n/decrease_m2_d,axis=0)	

	#Computation of the flatness, found as the geometrical mean divided by arithmetic mean. This computation is 
	#outlined in [0] in the 'references.txt' file included with this code.
	def computeFlatness(self):
		self.flatness =	ss.mstats.gmean(self.stft,axis=0).data/self.stft_mean

	#Computation of spectral crest, also a measure of flatness. This computation is 
	#outlined in [0] in the 'references.txt' file included with this code.
	def computeCrest(self):
		self.crest = np.max(self.stft,axis=0)/self.stft_mean

	#Computation of the spectral flux, or local changes in the stft. 
	def computeFlux(self):
		N_current = self.stft[:,1:np.shape(self.stft)[1]]
		N_minus = self.stft[:,0:np.shape(self.stft)[1]-1]
		self.flux = np.sum(((N_current-N_minus)**2),axis=0)
		self.flux = np.insert(self.flux,0,0)

	#Plot spectral features together
	def plotFeatures(self):

		matplotlib.pyplot.subplot(2,3,1)
		matplotlib.pyplot.plot(self.centroid)
		matplotlib.pyplot.title('Centroid',size = 12)
		matplotlib.pyplot.ylabel('Centroid Value',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)
		matplotlib.pyplot.subplot(2,3,2)
		matplotlib.pyplot.plot(self.spread)
		matplotlib.pyplot.title('Spread',size = 12)
		matplotlib.pyplot.ylabel('Spread Value',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)
		matplotlib.pyplot.subplot(2,3,3)
		matplotlib.pyplot.plot(self.skewness)
		matplotlib.pyplot.title('Skewness',size = 12)
		matplotlib.pyplot.ylabel('Skewness Value',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)
		matplotlib.pyplot.subplot(2,3,4)
		matplotlib.pyplot.plot(self.kurtosis)
		matplotlib.pyplot.title('Kurtosis',size = 12)
		matplotlib.pyplot.ylabel('Kurtosis Value',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)
		matplotlib.pyplot.subplot(2,3,5)
		matplotlib.pyplot.plot(self.frame_energy)
		matplotlib.pyplot.title('Frame Energy',size = 12)
		matplotlib.pyplot.ylabel('Energy in Frame',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)
		matplotlib.pyplot.subplot(2,3,6)
		matplotlib.pyplot.plot(self.rolloff)
		matplotlib.pyplot.title('Rolloff',size = 12)
		matplotlib.pyplot.ylabel('Rolloff Value',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)

		matplotlib.pyplot.suptitle('Frequency Domain Features Part 1',size = 16)

		matplotlib.pyplot.show()

		matplotlib.pyplot.subplot(2,3,1)
		matplotlib.pyplot.plot(self.slope)
		matplotlib.pyplot.title('Slope',size = 12)
		matplotlib.pyplot.ylabel('Slope Value',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)
		matplotlib.pyplot.subplot(2,3,2)
		matplotlib.pyplot.plot(self.decrease)
		matplotlib.pyplot.title('Decrease',size = 12)
		matplotlib.pyplot.ylabel('Decrease Value',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)
		matplotlib.pyplot.subplot(2,3,3)
		matplotlib.pyplot.plot(self.flatness)
		matplotlib.pyplot.title('Flatness',size = 12)
		matplotlib.pyplot.ylabel('Flatness Value',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)
		matplotlib.pyplot.subplot(2,3,4)
		matplotlib.pyplot.plot(self.crest)
		matplotlib.pyplot.title('Crest',size = 12)
		matplotlib.pyplot.ylabel('Crest Value',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)
		matplotlib.pyplot.subplot(2,3,5)
		matplotlib.pyplot.plot(self.flux)
		matplotlib.pyplot.title('Spectral Flux',size = 12)
		matplotlib.pyplot.ylabel('Flux Value',size = 10)
		matplotlib.pyplot.xlabel('Frame Number',size = 10)

		matplotlib.pyplot.suptitle('Frequency Domain Features Part 2',size = 16)

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
		step = int(np.shape(self.stft)[1]/self.song_duration)
		borders = range(0,np.shape(self.stft)[1],step)

		self.centroid_sec = self.findStats(self.centroid,borders)
		self.spread_sec = self.findStats(self.spread,borders)
		self.skewness_sec = self.findStats(self.skewness,borders)
		self.kurtosis_sec = self.findStats(self.kurtosis,borders)
		self.frame_energy_sec = self.findStats(self.frame_energy,borders)
		self.rolloff_sec = self.findStats(self.rolloff,borders)
		self.slope_sec = self.findStats(self.slope,borders)
		self.decrease_sec = self.findStats(self.decrease,borders)
		self.flatness_sec = self.findStats(self.flatness,borders)
		self.crest_sec = self.findStats(self.crest,borders)
		self.flux_sec = self.findStats(self.flux,borders)

	#This function is called by the program's top level to provide a set of features (statistics of each feature by the second)
	#that is used to segment the song into sections. 
	def buildSegmentingFeatures(self):
		self.featuresPerSecond()
		return np.vstack([self.centroid_sec,self.spread_sec,self.skewness_sec,self.kurtosis_sec,self.frame_energy_sec,
					   self.rolloff_sec,self.slope_sec,self.decrease_sec,self.flatness_sec,self.crest_sec,self.flux_sec])

	#This function is called from the program's top level to provide a set of features (statistics of each feature by the segment
	#as defined in the segment class algorithm) that are ultimately to be used in the user's application.
	def segment(self,borders):
		borders = np.floor((np.shape(self.stft)[1]*borders)/self.converting_leny)
		
		self.centroid_seg = self.findStats(self.centroid,borders,normalize = False)
		self.spread_seg = self.findStats(self.spread,borders,normalize = False)
		self.skewness_seg = self.findStats(self.skewness,borders,normalize = False)
		self.kurtosis_seg = self.findStats(self.kurtosis,borders,normalize = False)
		self.frame_energy_seg = self.findStats(self.frame_energy,borders,normalize = False)
		self.rolloff_seg = self.findStats(self.rolloff,borders,normalize = False)
		self.slope_seg = self.findStats(self.slope,borders,normalize = False)
		self.decrease_seg = self.findStats(self.decrease,borders,normalize = False)
		self.flatness_seg = self.findStats(self.flatness,borders,normalize = False)
		self.crest_seg = self.findStats(self.crest,borders,normalize = False)
		self.flux_seg = self.findStats(self.flux,borders,normalize = False)













