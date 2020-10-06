import netCDF4
import numpy as np
#matplotlib.use('cairo')
import time
import sys

import argparse

from som import SOM


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", choices=['train', 'map'],
	help="Select whether to train a new SOM or map data onto an existing SOM", required=True)
parser.add_argument("--shape", help="The nodes shape of the SOM eg. 4x5 used when training a new SOM", required=False)
parser.add_argument("-i", "--iterations", help="Number of training iterations", type=int, default=10000, required=False)
parser.add_argument("-r", "--rate", help="Initial learning rate, this decays to zero through the training", type=float, default=0.1)
parser.add_argument("data", nargs='*', help="NetCDF data files containing training or mapping data")
args = parser.parse_args()


# Parse out the SOM shape for training
if args.mode in ['train']:

	if not args.shape:
		print('ERROR, we need a shape argument for training!')
		sys.exit(1)

	som_shape = tuple([int(d) for d in args.shape.split('x')])


# If training or mapping we need to process all the data files and variables
if args.mode in ['train', 'map']:

	sources = []

	for file in args.data:

		filename, varstring = file.split(':')
		varnames = varstring.split(',')

		sources.append((filename, varnames))


	som = SOM(som_shape, sources)

print('SOM nodes shape ', som.nodes.shape)
print('SOM std and mean ', som.std.shape, som.mean.shape)

radius = (float(max(som.shape)), 0.0)

now = time.time()
som.train(args.iterations, radius, args.rate)
print("\nThat took {:.3f} seconds".format(time.time()-now))

#som.plot('result.png', 0)

#plt.figure(figsize=(8,8))

#i = 1
#for index in np.ndindex(som.shape):

#	print(index, i)

#	node_full = np.empty(tuple(variables[0].shape[1:])).flatten()
#	mask = np.ma.getmaskarray(variables[0][0,:].flatten())

#	print(node_full.shape, mask.shape)

	#node_full[~mask] = som.nodes[index][:mask.shape[0]]*som.std[:mask.shape[0]] + som.mean[:mask.shape[0]]
#	node_full[~mask] = som.nodes[index][:mask.shape[0]]
#	node_full = node_full.reshape(tuple(variables[0].shape[1:]))
#	node_full = np.ma.masked_greater(node_full, 1e9)

#	print node_full.shape
#	plt.subplot(som.shape[1], som.shape[0], i)
#	plt.pcolormesh(np.squeeze(node_full))

#	i += 1

#plt.savefig('result.png')

som.save('som.nc')

