'''
segment.py
Nathan William Pocta
Research Assistant at Brigham-Young University
Student at Clemson University
This class finds borders in an audio file where the sound significantly changes. The full algorithm
is a product of experimentation across a wide range of songs, but has roots in [10] and [11]
Specific steps of the algorithm can be seen below.  
'''
import librosa
import sys
import numpy as np
import matplotlib.pyplot
import sklearn
import math
import scipy
import scipy.signal
from scipy import ndimage
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM


class segment:

	def __init__(self,y,sr,feature_vector):

		self.y = y
		self.sr = sr
		self.feat_vects = feature_vector
		
		self.gatherClusteringFeatures()
		self.HMMClusters()
		self.computeSimilarity()
		self.findBorders()

	#In order to cluster frames, a set of features must be gathered
	def gatherClusteringFeatures(self, nfft=32000):
		self.clust_feats = self.feat_vects

	#This function clusters the features gathered in several different ways to give us a
	#more discrete feature set.
	def HMMClusters(self, iterations = 20):

		hidden_states = np.zeros([2*iterations,np.shape(self.clust_feats)[1]])

		for x in range(0,iterations):
			model = GaussianHMM(x+2, covariance_type="diag", n_iter=1000)

			model.fit([self.clust_feats.T])

			hidden_states[x] = model.predict(self.clust_feats.T)

			kmeans = KMeans(init='k-means++', n_clusters=x+2)
			kmeans.fit(self.clust_feats.T)
			hidden_states[x+iterations] = kmeans.predict(self.clust_feats.T)

			hidden_states[x] = (hidden_states[x]-np.mean(hidden_states[x]))/math.sqrt(np.var(hidden_states[x]))
			hidden_states[x+iterations] = (hidden_states[x+iterations]-np.mean(hidden_states[x+iterations]))/math.sqrt(np.var(hidden_states[x+iterations]))

		self.segment_feats = hidden_states

	#Wrapper for the specific similarity function used to 
	#calculate distance between two arrays
	def calculateCosineSim(self,x,y):
		if(np.linalg.norm(x)!=0 and np.linalg.norm(y)!=0):
			return scipy.spatial.distance.cosine(x,y)
		else:
			return 0

	#Computation of the similarity matrix using the cosine
	#distance function
	def cosineMatrix(self,features):

		similarity = np.dot(features.T, features)

		square_mag = np.diag(similarity)

		inv_square_mag = 1 / square_mag

		inv_square_mag[np.isinf(inv_square_mag)] = 0

		inv_mag = np.sqrt(inv_square_mag)

		cosine = similarity * inv_mag
		cosine = cosine.T * inv_mag
	
		return cosine

	#Computes similarity by finding a similarity matrix, and then iterating
	#through a procedure that calculates several similarity matrices of those 
	#similarity matrices- this is useful for sharpening the matrices
	def computeSimilarity(self, sharpen_iterations = 5):

		S = self.cosineMatrix(self.segment_feats)

		self.similarity_matrix = S

		for x in range(len(S)):
			S[x,:] = scipy.ndimage.filters.median_filter(S[x],15)
			S[:,x] = scipy.ndimage.filters.median_filter(S[x],15)

		for x in range(0,sharpen_iterations):
			S = self.cosineMatrix(S)

		self.sharp_similarity_matrix = S

	#Function calculates the borders between sections of the song
	def findBorders(self):
		self.computeAdjacentSimilarities()
		self.identifyMajorBorders()
		self.cleanUpBorders()
		self.borders = (self.borders*len(self.y))/len(self.sharp_similarity_matrix)

	#When one row in the similarity matrix is drastically different from
	#another row in the similarity matrix, the adjacent similarities matrix experiences
	#a spike
	def computeAdjacentSimilarities(self):
		if len(self.sharp_similarity_matrix) == 0:
			return np.array([])
		x = 0
		self.adjacent_sim = np.array([])
		while x < len(self.sharp_similarity_matrix):
			if x != (len(self.sharp_similarity_matrix)-1):
				sim=self.calculateCosineSim(self.sharp_similarity_matrix[x],self.sharp_similarity_matrix[x+1])
				self.adjacent_sim = np.append(self.adjacent_sim,sim)
			x+=1

	#This function is able to find the spikes in the adjacent similarity array
	def identifyMajorBorders(self):
		self.borders = np.where(self.adjacent_sim>np.mean(self.adjacent_sim))[0]

	#Used to enforce certain rules on how the borders should appear, e.g. not right next to
	#each other, and starts with 0, ends with the end of the song
	def cleanUpBorders(self):
		self.borders = self.removeSequentials(self.borders)

		if len(self.borders)>0 and self.borders[0] != 0:
			self.borders = np.insert(self.borders,0,0)

		if self.borders[len(self.borders)-1] != len(self.sharp_similarity_matrix):
			self.borders = np.append(self.borders,len(self.sharp_similarity_matrix))

	#Removes borders that are very close together
	def removeSequentials(self,borders):
		new_borders = np.array([])
		x=0
		while x < (len(borders)):
			if x+1 < len(borders) and borders[x+1] <= (borders[x]+10):
				y = x
				while y<(len(borders)-1) and borders[y+1]<=(borders[y]+10):
					y+=1
				new_borders=np.append(new_borders,borders[int((x+y)/2)])
				x=y+1
			else:
				new_borders=np.append(new_borders,borders[x])
				x+=1
		return new_borders

	#Simple function used to print time from seconds
	def printTimeFromSeconds(self,seconds):
		m, s = divmod(int(seconds), 60)
		h, m = divmod(m, 60)
		print "%d:%02d:%02d" % (h, m, s)

	#Prints the times identified as being borders
	def printTimes(self):
		song_duration = librosa.core.get_duration(y=self.y, sr=self.sr)
		for x in range(0,len(self.borders)):
			t = self.borders[x]*(float(song_duration)/float(self.borders[len(self.borders)-1]))
			print x 
			self.printTimeFromSeconds(t)

	#Plots the similarity matrix
	def plotSimilarityMatrix(self):
		matplotlib.pyplot.imshow(self.sharp_similarity_matrix, cmap = 'Greys',shape = np.shape(self.similarity_matrix))
		matplotlib.pyplot.title('Similarity Matrix')
		matplotlib.pyplot.xlabel('Time (s)')
		matplotlib.pyplot.ylabel('Time (s)')
		matplotlib.pyplot.show()



















		
		

		
		
