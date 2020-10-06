"""
Main SOM class implementation
"""

import numpy as np
import netCDF4

class SOM(object):

	def __init__(self, *args):
		"""
		Create SOM instance with the given 2D shape (nodes x nodes)

		>>> som = SOM(5,4)
		>>> print(som.shape)
		(5, 4)
		"""
		self.shape = tuple(args)
		self.nodes = None


	def randinit(self, samples):
		"""
		Initialize the SOM node vectors with random values.  Input is an ndarray of shape:(samples, variables) and SOM node
		vectors will have shape (variables)

		>>> som = SOM(5,4)
		>>> samples = np.random.randn(1000,2)
		>>> samples *= (10 * np.random.randn(2))
		>>> samples += 10*np.random.randn(2)
		>>> som.randinit(samples)
		"""
		self.samples_shape = tuple(list(samples.shape[1:]))
		self.dtype = samples.dtype
	
		mean = samples.mean(axis=0)
		std = samples.std(axis=0)

		self.nodes = np.random.randn(*(self.shape + self.samples_shape))

		self.nodes += mean
		self.nodes *= std


	def closest(self, sample):
		"""
		Find the closest node to a give sample, return the node coordinates and the distance

		>>> som = SOM(5,4)
		>>> samples = np.random.randn(1000,20,30)
		>>> samples *= (10 * np.random.randn(20,30))
		>>> samples += 10*np.random.randn(20,30)
		>>> som.randinit(samples)
		>>> closest = som.closest(samples[42])

		"""

		if not isinstance(self.nodes, np.ndarray):
			raise Exception('SOM nodes not initialized, try running SOM.randinit first')

		min_distance = np.finfo(self.dtype).max
		node = None

		# Scan all the node vectors
		for index in np.ndindex(self.shape):

			# Calculate the distance (no need to apply square root as we go)
			d = np.sum(np.power(self.nodes[index] - sample, 2))

			# Update minimimum
			if d < min_distance:
				node = index
				min_distance = d

		return tuple(node), np.sqrt(min_distance)


	def _influence(self, node, othernode, radius):
		"""
		Calculate the influence factor for a node (othernode) give the current node (node)
		and the current influence radius
		"""

		# Euclidian distance in node coordinate space not sample space
		d = np.sum(np.power(np.array(node) - np.array(othernode), 2))

		radius2 = radius**2

		if d < radius2:
			return np.exp(-d / (2*(radius2)))
		else:
			return 0.0


	def fit(self, samples, iterations, radius, rate, progress=False):
		"""
		Update the node vectors for a certain number of iterations (iterations) using randmon
		samples from the data (data), an influence radius (radius), and a training rate (rate)

		Both radius and rate can be single values or tuples.  If single values then they will
		be constant through the training.  If tuples then they will progressively change from
		the first value to the second value through the training

		if progress is True then stats are reported as fitting occurs.  This is computationally
		expensive because it reports mean error across the full sample set.

		>>> som = SOM(5,4)
		>>> samples = np.random.randn(1000,2)
		>>> som.randinit(samples)
		>>> #samples *= 10
		>>> #samples += 20
		>>> som.fit(samples, 1000, 4, 1.0)
		using radius=(4, 0.0) and rate=(1.0, 0.0)
		>>> som.save('test.nc')

		"""

		if not isinstance(self.nodes, np.ndarray):
			raise Exception('SOM nodes not initialized, try running SOM.randinit first')

		# Force radius and rate to be tuples
		if not isinstance(radius, tuple):
			radius = (radius, 0.0,)

		if not isinstance(rate, tuple):
			rate = (rate, 0.0,)

		print("using radius={} and rate={}".format(radius, rate))

		# Keep track of total sample/node distances
		total_distance = 0.0

		for i in range(0,iterations):

			# Calculate radius and rate for this iteration
			this_radius = (float(iterations - i)/iterations)*radius[0] + (float(i)/iterations)*radius[1]
			this_rate = (float(iterations - i)/iterations)*rate[0] + (float(i)/iterations)*rate[1]

			# Report stats 10 times
			if progress and not i % int(iterations / 10):

				error, counts = self.error(samples)
				
				if progress:
					print("{} of {}, radius = {:.2f}, rate = {:.2f}, mean error = {:}\r".format(i, iterations, this_radius, this_rate, error.mean()))
					print("node counts:", counts)

			# Get a random sample
			sample_index = np.random.randint(0,samples.shape[0])
			sample = samples[sample_index]

			# Find closest node
			node, distance = self.closest(sample)

			# Accumulate distance
			total_distance += distance

			# Step through all nodes and update
			for index in np.ndindex(self.shape):

				influence = self._influence(index, node, this_radius)

				self.nodes[index] += influence * this_rate * (sample - self.nodes[index])


		return


	def error(self, samples):
		"""
		Calculate the mean sample error per node given the samples in data
		"""

		if not isinstance(self.nodes, np.ndarray):
			raise Exception('SOM nodes not initialized, try running SOM.randinit first')

		distances = np.zeros(self.shape, dtype=np.float32)
		counts = np.zeros(self.shape, dtype=np.int32)

		for i in range(0, samples.shape[0]):

#			print('error', self.nodes, samples[i])
			node, d = self.closest(samples[i])
			distances[node] += d
			counts[node] += 1

		distances[counts>0] = distances[counts>0] / counts[counts>0]

		return distances, counts

	def map(self, samples):
		"""
		Map samples to nodes and return associated node indices and mapping errors
		
		>>> som = SOM.fromfile('test.nc')
		>>> samples = np.random.randn(1000,2)
		>>> nodes, errors = som.map(samples)
		"""

		nodes = []
		error = []

		for i in range(0, samples.shape[0]):

			closest = self.closest(samples[i])
			
			nodes.append(closest[0])
			error.append(closest[1])

		return nodes, error

	def node_distances(self):
		"""
		Calculate the distances between the node vectors, could be used
		to produce a sammon map
		"""

		if not isinstance(self.nodes, np.ndarray):
			raise Exception('SOM nodes not initialized, try running SOM.randinit first')

		dmean = np.zeros(self.shape, dtype=np.float32)
		counts = np.zeros(self.shape, dtype=np.int32)
		dmax = np.zeros(self.shape, dtype=np.float32)

		for index in np.ndindex(self.shape):

			for i in range(0, len(index)):

				for offset in [-1, 1]:

					tmp = list(index)
					tmp[i] = index[i] + offset

					if tmp[i] >= 0 and tmp[i] < self.shape[i]:

						d = np.sqrt(np.sum(np.power(self.nodes[index] - self.nodes[tuple(tmp)], 2)))/2.0

						if d > dmax[index]:
							dmax[index] = d

						#print(i, tmp, self.nodes[tuple(tmp)][0], d, dmax[index])

						dmean[index] += d
						counts[index] += 1

		dmean = dmean/counts

		return dmean, dmax, counts


	def save(self, filename):
		
		with netCDF4.Dataset(filename, 'w') as ds:

			som_dim_names = []
			for i in range(len(self.shape)):
				name = 'som_dim{}'.format(i)
				ds.createDimension(name, size=self.shape[i])
				som_dim_names.append(name)

			sample_dim_names = []
			for i in range(len(self.samples_shape)):
				name = 'samples_dim{}'.format(i)
				ds.createDimension(name, size=self.samples_shape[i])
				sample_dim_names.append(name)

			nodes = ds.createVariable('nodes', self.dtype, tuple(som_dim_names) + tuple(sample_dim_names))

			nodes[:] = self.nodes[:]			


	@classmethod
	def fromfile(cls, filename):
		"""
		Create a SOM instance from a saved netcdf file

		>>> som = SOM.fromfile('test.nc')
		>>> samples = np.random.randn(1000,2)
		"""

		with netCDF4.Dataset(filename, 'r') as ds:

			shape = []
			for i in range(0,10):
				name = 'som_dim{}'.format(i)
				if name in ds.dimensions.keys():
					shape.append(ds.dimensions[name].size)

			shape = tuple(shape)
			
			som = cls(*shape)
			
			nodes_var = ds.variables['nodes']

			som.nodes = np.ndarray(nodes_var.shape, dtype=nodes_var.dtype)
			som.dtype = nodes_var.dtype

			som.nodes[:] = nodes_var[:]
			
			return som




			












