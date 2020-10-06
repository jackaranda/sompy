# Python SOM clustering

A simple implementation of the Self Organizing Map clustering algorithm.  The interface is modelled after the scikit-learn API.

## Dependencies
* numpy: https://numpy.org/  
* Python netCDF4 package: https://unidata.github.io/netcdf4-python/netCDF4/index.html

```
pip install numpy
pip install netCDF4
```

### Example usage

```
# Import module
from sompy import SOM

# Create a SOM with 5 x 4 nodes
som = SOM(5,4)

# Just create a random dataset with 1000 samples and 2 variables/columns
samples = np.random.randn(1000,2)

# Randomly initialize the SOM using the mean and variance of the samples
som.randinit(samples)

# Fit the some using 1000 iterations, and initial learning radius of 4 and initial learning rate of 1.0
# The radius and learning rate decay to zero by default but you can also pass a tuple like (1.0, 0.5)
# to decay between 1.0 and 0.5
som.fit(samples, 1000, 4, 1.0)

# Save the SOM node states to a netCDF4 file
som.save('test.nc')

# Map samples to nodes, reading SOM state from previously saved file
som = SOM.fromfile('test.nc')
samples = np.random.randn(1000,2)
nodes, errors = som.map(samples)

```
