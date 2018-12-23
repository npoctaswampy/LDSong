'''
LDSong.py
Nathan William Pocta
Research Assistant at Brigham-Young University
Student at Clemson University
This is the top level of the program- enabling a user to instantiate a "song" object and gather
features from it. The list of references used to develop this work is in "references.txt". Most of
the information in this literature lended itself to be used for feature extraction and segmentation.
'''
import time_feat
import spectral_feat
import librosa
import segment
import beat
import pitch
import alternate_scales
import numpy as np

class LDSong:

	#Calling init on a song calculates features in several areas for that song.
	#By default, the song is segmented into similar sections. Disabling segmenting
	#will cause statistics to be calculated over the entire song.
	def __init__(self,name,segment_song = True,verbose = True):

		if verbose:
			print 'Loading song...'
		self.y, self.sr = librosa.load(name)

		if verbose:
			print 'Finding spectral features...'
		self.specfeats = spectral_feat.spectral_feat(self.y,self.sr)

		if verbose:
			print 'Finding temporal features...'
		self.timefeats = time_feat.time_feat(self.y,self.sr)

		if verbose:
			print 'Finding coefficients of alternate scales...'
		self.alternatescalecoef = alternate_scales.alternate_scales(self.y,self.sr)

		if segment_song == True:
			if verbose:
				print 'Finding song segments...'
			#Gathers segmenting features from previously computed features (stats by the second)
			self.segments = segment.segment(self.y,self.sr,np.vstack([self.specfeats.buildSegmentingFeatures(),
										  self.timefeats.buildSegmentingFeatures(),
										  self.alternatescalecoef.buildSegmentingFeatures()]))

			self.borders = self.segments.borders
		else: #If segmentation is not needed, stats for the entire song are calculated.
			self.borders = np.array([0,len(self.y)])

		if verbose:
			print 'Implementing beat tracking...'
		self.beats = beat.beat(self.y,self.sr)

		if verbose:
			print 'Implementing pitch estimation...'
		self.pitches = pitch.pitch(self.y,self.sr)

		#Find stats for signal features between segment borders
		self.specfeats.segment(self.borders)
		self.timefeats.segment(self.borders)
		self.alternatescalecoef.segment(self.borders)

		#Dictionary provides method for user to add features to final
		#matrix of features to be used for an application.
		self.buildDictionary()

		#The feature matrix is made up of two arrays- temporal holds values that are 
		#calculated by segment and global holds values that are calculated across the 
		#entire song.
		self.temporal_feature_matrix = np.array([])
		self.global_feature_matrix = np.array([])

		self.features = [self.temporal_feature_matrix, self.global_feature_matrix]

	def buildDictionary(self):
		
		self.temporal_feature_dictionary = { 'centroid' : self.specfeats.centroid_seg,
					    	     'spread' : self.specfeats.spread_seg,
					    	     'skewness' : self.specfeats.skewness_seg,
					    	     'kurtosis' : self.specfeats.kurtosis_seg,
					    	     'frame energy' : self.specfeats.frame_energy_seg,
					    	     'rolloff' : self.specfeats.rolloff_seg,
					    	     'slope' : self.specfeats.slope_seg,
					    	     'decrease' : self.specfeats.decrease_seg,
					    	     'flatness' : self.specfeats.flatness_seg,
					    	     'crest' : self.specfeats.crest_seg,
					    	     'flux' : self.specfeats.flux_seg,
					    	     'zero crossing' : self.timefeats.zero_crossing_seg,
					    	     'rms' : self.timefeats.rms_seg,
					    	     'low energy' : self.timefeats.low_energy_seg,
					    	     'amplitude envelope' : self.timefeats.amplitude_envelope_seg,
					    	     'loudness' : self.timefeats.loudness_seg,
						     'mfcc' : self.alternatescalecoef.mfcc_seg }

		self.global_feature_dictionary = { 'note 0' : self.pitches.up0,
						   'note strength 0' : self.pitches.u0,
						   'notes sum' : self.pitches.hist_sum,
						   'pitch class 0' : self.pitches.fp0,
						   'pitch class strength 0' : self.pitches.f0,
						   'period 0' : self.beats.period0,
						   'period 1' : self.beats.period1,
						   'period 2' : self.beats.period2,
						   'amplitude 0' : self.beats.amplitude0,
						   'amplitude 1' : self.beats.amplitude1,
						   'amplitude 2' : self.beats.amplitude2,
						   'ratio period 1' : self.beats.ratio_period1,
						   'ratio period 2' : self.beats.ratio_period2,
						   'duration' : librosa.core.get_duration(y=self.y, sr=self.sr) }
 
	#The method used to add features to the matrix that is to be used by the user for some application.
	#By default, both the mean and variance are included. "feature" is a string that indicates which feature
	#from the dictionary defined above is to be added. 
	def buildFeatureMatrix(self,feature,mean=True,var=True):

		if mean and var:
			start = 0
			step = 1
		if not mean and var:
			start = 0
			step = 2
		if mean and not var:
			start = 1
			step = 2
		if not mean and not var:
			start = 2
			step = 1
		
		if feature in self.temporal_feature_dictionary:
			entry = self.temporal_feature_dictionary[feature]
			if len(self.temporal_feature_matrix) == 0:
				self.temporal_feature_matrix = entry[range(start,len(entry),step)]
			else:
				self.temporal_feature_matrix = np.vstack([self.temporal_feature_matrix,entry[range(start,len(entry),step)]])

		elif feature in self.global_feature_dictionary:
			self.global_feature_matrix = np.append(self.global_feature_matrix, self.global_feature_dictionary[feature])

		else:
			print 'No such feature: \"'+feature+'\". Please try again.'

		self.features = [[self.temporal_feature_matrix, self.global_feature_matrix]]





			

		

