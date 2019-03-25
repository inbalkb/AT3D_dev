"""
Spherical Harmonic Discrete Ordinate Method for 3D Atmospheric Radiative Transfer

This is a python wrapper for SHDOM created by Aviad Levis and Amit Aides, Technion Inst. of Technology.
The purpose of this wrapper is to develop 3D remote sensing metoglogies. 

The documentation of the source Fortran code by Frank Evans can be found at
http://nit.colorado.edu/shdom/shdomdoc

Information about the source code taken from the documentation page:
This program computes unpolarized monochromatic or spectral band radiative transfer in a one, two,
or three-dimensional medium for either collimated solar and/or thermal emission sources of radiation.
The properties of the medium can be specified completely generally, i.e. the extinction, single 
scattering albedo, Legendre coefficients of the scattering phase function, and temperature for
the particular wavelength or spectral band may be specified at each input grid point. SHDOM is
superior to Monte Carlo radiative transfer methods when many radiative quantities are desired,
e.g. the radiance field across the domain top or the 3D distribution of heating. Radiances at
any angle, hemispheric fluxes, net fluxes, mean radiances, and net flux convergence (related
to heating rates) may be output anywhere in the domain. For highly peaked phase functions the 
delta-M method may be chosen, in which case the radiance is computed with an untruncated phase
function single scattering correction. A correlated k-distribution approach is used for the
integration over a spectral band. There may be uniform or spatially variable Lambertian
reflection and emission from the ground surface. Several types of bidirectional reflection
distribution functions (BRDF) for the surface are implemented, and more may be added easily.
SHDOM may be run on a single processor or on multiple processors (e.g. an SMP machine or a
cluster) using the Message Passing Interface (MPI).
"""

from scipy.interpolate import interp1d, RegularGridInterpolator


class Grid(object):
    """ 
    A Grid object defining the 3D or 1D grid of the atmopshere. 
    
    A 3D Grid can be defined with:
      1. x, y, z grids.
      2. A BoundingBox and grid resolution (nx, ny, nz).
    
    A 1D grid is defined with a z grid.
    
    Parameters
    ----------
    bounding_box: BoundingBox, optional
        A BoundingBox object for 3D grid. If specified nx, ny and nz must be specified as well.
    nx: integer, optional
        Number of grid point in x axis. Must be specified with a BoundingBox object.
    ny: integer, optional
        Number of grid point in y axis. Must be specified with a BoundingBox object.
    nz: integer, optional
        Number of grid point in z axis. Must be specified with a BoundingBox object.
    x: np.array(dtype=float, shape=(nx,)), optional
        Grid along x axis. Must be specified with y,z grids.
    y: np.array(dtype=float, shape=(ny,)), optional
        Grid along y axis. Must be specified with x,z grids. 
    z: np.array(dtype=float, shape=(nz,)), optional
        Grid along z axis. Either specified with x,y grids (3D grid) or by itself (1D grid).
    """    
    
    def __init__(self, **kwargs):
        
        # 3D grid with grids
        if kwargs.has_key('x') and kwargs.has_key('y') and kwargs.has_key('z'):
            self._type = '3D'
            self.x = kwargs['x']
            self.y = kwargs['y']
            self.z = kwargs['z']
            self._bounding_box = BoundingBox(self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax)
            
        # 3D grid with bounding box
        elif kwargs.has_key('bounding_box') and kwargs.has_key('nx') and kwargs.has_key('ny'):
            self._type = '3D'
            bb = kwargs['bounding_box']
            self._bounding_box = bb
            nx, ny = kwargs['nx'], kwargs['ny']
            self.x = np.linspace(bb.xmin, bb.xmax, nx)
            self.y = np.linspace(bb.ymin, bb.ymax, ny)
            if kwargs.has_key('z'):
                self.z = kwargs['z']
            elif kwargs.has_key('nz'):
                nz = kwargs['nx']
                self.z = np.linspace(bb.zmin, bb.zmax, nz)
            else:
                raise AttributeError('z or nz are missing')
            
        # 1D grid 
        elif kwargs.has_key('z'):
            self._type = '1D'
            self.z = kwargs['z']
            self._nx = self._ny = 1
            self._x = self._y = self._bounding_box = None
         
        else:
            raise AttributeError('kwargs in Grid initialization are not defined')
    
    
    def get_common_x_grid(self, other):
        """
        Find the common x grid which maintains a the minimum dx (distance between two grid points).
        
        Parameters
        ----------
        other: Grid object
           The other grid for which to find a common x grid.
           
        Returns
        -------
        x: np.array(dtype=np.float32)
            The common x Grid.
        """
        if self.type == '1D' and other.type == '1D':
            return None       
        if self.type == '3D' and other.type == '1D':
            return self.x
        if self.type == '1D' and other.type == '3D':
            return other.x
        
        if np.array_equiv(self.x, other.x):
            return self.x

        xmax = max(self.xmax, other.xmax)
        xmin = min(self.xmin, other.xmin) 
        x_size = xmax - xmin      
        dx = min(self.dx, other.dx)
        nx = int(y_size / dx)        
        return np.linspace(xmin, xmax, nx, dtype=np.float32)  
    
    
    def get_common_y_grid(self, other):
        """
        Find the common y grid which maintains a the minimum dy (distance between two grid points).
        
        Parameters
        ----------
        other: Grid object
           The other grid for which to find a common y grid.
           
        Returns
        -------
        y: np.array(dtype=np.float32)
            The common y Grid.
        """
        if self.type == '1D' and other.type == '1D':
            return None       
        if self.type == '3D' and other.type == '1D':
            return self.y
        if self.type == '1D' and other.type == '3D':
            return other.y
        
        if np.array_equiv(self.y, other.y):
            return self.y
        
        ymax = max(self.ymax, other.ymax)
        ymin = min(self.ymin, other.ymin) 
        y_size = ymax - ymin      
        dy = min(self.dy, other.dy)
        ny = int(y_size / dy)        
        return np.linspace(ymin, ymax, ny, dtype=np.float32)              
       
        
    def get_common_z_grid(self, other):
        """
        Find the common z grid which maintains a the high resolution z grid.
        
        Parameters
        ----------
        other: Grid object
           The other grid for which to find a common y grid.
           
        Returns
        -------
        z: np.array(dtype=np.float32)
            The common z Grid.
        """        
        
        if np.array_equiv(self.z, other.z):
            return self.z
        
        # Bottom part of the atmosphere (no grid intersection)
        if self.zmin < other.zmin:
            z_bottom = self.z[self.z < other.zmin]
        else:
            z_bottom = other.z[other.z < self.zmin]
    
        # Top part of the atmosphere (no grid intersection)
        if self.zmax < other.zmax:
            z_top = other.z[other.z > self.zmax]
        else:
            z_top = self.z[self.z > other.zmax]
    
        # Middle part of the atmosphere (grids intersect)
        z_middle_self = self.z
        z_middle_other = other.z
        if z_bottom.any():
            z_middle_self = self.z[self.z > z_bottom[-1]]
            z_middle_other = other.z[other.z > z_bottom[-1]]
        if z_top.any():
            z_middle_self = self.z[self.z < z_top[0]]
            z_middle_other = other.z[other.z < z_top[0]]
    
        z_middle = z_middle_self if len(z_middle_self) > len(z_middle_other) else z_middle_other
    
        # Check if an extra point is necessary at the bottom 
        if z_bottom.any() & len(z_middle)>2:
            extra_zlevel = 2*z_middle[0] - z_middle[1]
            if extra_zlevel > z_bottom[-1]:
                z_middle = np.append(extra_zlevel, z_middle)
    
        # Check if an extra point is necessary at the top 
        if z_top.any() & len(z_middle)>2:
            extra_zlevel = 2*z_middle[-1] - z_middle[-2]
            if extra_zlevel < z_top[0]:
                z_middle = np.append(z_middle, extra_zlevel)
    
        return np.concatenate((z_bottom, z_middle, z_top))      

    def __add__(self, other):
        """
        Add two grids by finding the common grid which maintains the higher resolution grid.
        """
        x_grid = self.get_common_x_grid(other)
        y_grid = self.get_common_y_grid(other)
        z_grid = self.get_common_z_grid(other)
        if x_grid is not None and y_grid is not None:
            grid = Grid(x=x_grid, y=y_grid, z=z_grid)
        else:
            grid = Grid(z=z_grid)
        return grid
    
    
    def __eq__(self, other) : 
        for item1, item2 in zip(self.__dict__.itervalues(), other.__dict__.itervalues()):
            if not np.array_equiv(np.nan_to_num(item1), np.nan_to_num(item2)):
                return False
        return True      

    @property
    def type(self):
        return self._type

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, val):
        val = np.array(val, dtype=np.float32)
        spacing = np.diff(val)
        assert np.all(np.isclose(spacing, spacing[0])), 'x grid supoprt equally spacing only'
        self._x = val
        self._dx = spacing[0]  
        self._nx = len(val)
        self._xmin, self._xmax = val[0], val[-1]
    
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, val):
        val = np.array(val, dtype=np.float32)
        spacing = np.diff(val)
        assert np.all(np.isclose(spacing, spacing[0])), 'y grid supoprt equally spacing only'
        self._y = val
        self._dy = spacing[0] 
        self._ny = len(val)
        self._ymin, self._ymax = val[0], val[-1]

    @property
    def z(self):
        return self._z                 
    
    @z.setter
    def z(self, val):
        val = np.array(val, dtype=np.float32)
        self._z = val
        self._nz = len(val)
        self._zmin, self._zmax = val[0], val[-1]

    @property 
    def nx(self):
        return self._nx
    
    @property 
    def ny(self):
        return self._ny

    @property 
    def nz(self):
        return self._nz    

    @property
    def shape(self):
        if self.type == '1D':
            return (self.nz,)
        else:
            return (self.nx, self.ny, self.nz)
    
    @property
    def num_points(self):
        return self.nx * self.ny * self.nz
    
    @property 
    def dx(self):
        return self._dx
    
    @property 
    def dy(self):
        return self._dy    
    
    @property 
    def xmin(self):
        return self._xmin
    
    @property 
    def ymin(self):
        return self._ymin

    @property 
    def zmin(self):
        return self._zmin
    
    @property 
    def xmax(self):
        return self._xmax
    
    @property 
    def ymax(self):
        return self._ymax    

    @property 
    def zmax(self):
        return self._zmax    

    @property
    def bounding_box(self):
        return self._bounding_box
    
    
class GridData(object):
    """ 
    A container for scalar fields which are defined on a Grid. 
    
    Parameters
    ----------
    grid: Grid object
        A Grid object of type '1D' or '3D'.
    data: np.array
        data contains the scalar field.
    """    
    def __init__(self, grid, data):
        self._type = grid.type
        self._grid = grid
        self._data = data
        self._shape = self._data.shape[:3]
        self._ndim = self._data.ndim        
        if self.type == '1D' and self.ndim is not 1:
            raise AttributeError('Grid is 1D but data dimension is:{}'.format(self.ndim))
        if self.type == '3D' and self.ndim < 3:
            raise AttributeError('Grid is 3D but data dimension is:{}'.format(self.ndim))
        
        assert self.shape == grid.shape, 'Data shape is {}, grid shape is {}'.format(self.shape, grid.shape)

        self._linear_interpolator1d = interp1d(grid.z, self.data, assume_sorted=True, copy=False, bounds_error=False, fill_value=0.0) 
        self._nearest_interpolator1d = interp1d(grid.z, self.data, assume_sorted=True, kind='nearest', copy=False, bounds_error=False, fill_value=0)
        if self.type == '3D':
            self._linear_interpolator3d = RegularGridInterpolator((grid.x, grid.y, grid.z), self.data, bounds_error=False, fill_value=0.0)
            self._nearest_interpolator3d = RegularGridInterpolator((grid.x, grid.y, grid.z), self.data, method='nearest', bounds_error=False, fill_value=0)
    
    
    def __add__(self, other):
        """Add two GridData objects by resampling to a common grid."""
        grid = self.grid + other.grid
        data = self.resample(grid) + other.resample(grid)
        return GridData(grid, data)
    
    
    def __sub__(self, other):
        """Subtract two GridData objects by resampling to a common grid."""
        grid = self.grid + other.grid
        data = self.resample(grid) - other.resample(grid)
        return GridData(grid, data)       
    
    
    def __mul__(self, other):
        """ 
        Two multiplication options:
          1. Multiply two GridData objects by resampling to a common grid. 
          2. Multiply a GridData object by a GridPhase object.
        """
        if other.__class__ is GridPhase:
            result = other * self
        else:
            grid = self.grid + other.grid
            data = self.resample(grid) * other.resample(grid)
            result = GridData(grid, data)  
        return result
    
    
    def __div__(self, other):
        """Divide two GridData objects by resampling to a common grid."""
        grid = self.grid + other.grid
        data = self.resample(grid) / other.resample(grid)
        result = GridData(grid, data) 
        return result
    
    
    def resample(self, grid, method='linear'):
        """Resample data to a new Grid."""
        if self.type == '1D':
            if np.array_equiv(self.grid.z, grid.z):
                return self.data
            if method == 'linear':
                data = self._linear_interpolator1d(grid.z)
            elif method == 'nearest':
                data = self._nearest_interpolator1d(grid.z)
        else:
            if self.grid == grid:
                return self.data
            if method == 'linear':
                data = self._linear_interpolator3d(np.stack(np.meshgrid(grid.x, grid.y, grid.z, indexing='ij'), axis=-1))
            elif method == 'nearest':
                data = self._nearest_interpolator3d(np.stack(np.meshgrid(grid.x, grid.y, grid.z, indexing='ij'), axis=-1)) 
        return data
    
    @property
    def grid(self):
        return self._grid
    
    @property
    def data(self):
        return self._data
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def ndim(self):
        return self._ndim      
    
    @property
    def max_value(self):
        return self.data.max()
    
    @property
    def min_value(self):
        return self.data.min()    
    
    @property
    def type(self):
        return self._type    


class BoundingBox(object):
    """ 
    A bounding box object.
    
    Parameters
    ----------
    xmin: float
         Minimum x (North).
    ymin: float
         Minimum y (East)
    zmin: float
         Minimum z (Up).
    xmax: float
         Maximum x (North).
    ymax: float
         Maximum y (East).
    zmax: float
         Maximum z (Up).

    
    Notes
    -----
    All values are in [km] units
    """ 
    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax):
        assert xmin < xmax, 'Zero area bounding_box along x axis.'  
        assert ymin < ymax, 'Zero area bounding_box along y axis.' 
        assert zmin < zmax, 'Zero area bounding_box along z axis.'       
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax
        
    def __eq__(self, other) : 
        return self.__dict__ == other.__dict__    
    
    
    def __add__(self, other):
        xmin = self.xmin
        if np.isfinite(other.xmin):
            if np.isfinite(xmin):
                xmin = min(xmin, other.xmin)
            else:
                xmin = other.xmin
            
        ymin = self.ymin
        if np.isfinite(other.ymin):
            if np.isfinite(ymin):
                ymin = min(ymin, other.ymin)
            else:
                ymin = other.ymin
                
        zmin = self.zmin
        if np.isfinite(other.zmin):
            if np.isfinite(zmin):
                zmin = min(zmin, other.zmin)
            else:
                zmin = other.zmin
        
        xmax = self.xmax
        if np.isfinite(other.xmax):
            if np.isfinite(xmax):
                xmax = max(xmax, other.xmax)
            else:
                xmax = other.xmax
            
        ymax = self.ymax
        if np.isfinite(other.ymax):
            if np.isfinite(ymax):
                ymax = max(ymax, other.ymax)
            else:
                ymax = other.ymax
                
        zmax = self.zmax
        if np.isfinite(other.zmax):
            if np.isfinite(zmax):
                zmax = min(zmax, other.zmax)
            else:
                zmax = other.zmax
                
        return BoundingBox(xmin, ymin, zmin, xmax, ymax, zmax)

 
from phase import *
from medium import *
from sensor import *
from rte_solver import *
from optimize import *
import generate as Generate
import parameters as Parameters


def save_forward_model(directory, medium, solver, measurements):
    """
    Save the forward model parameters for reconstruction.
    
    Parameters
    ----------
    directory: str
        Directory path where the forward modeling parameters are saved. 
        If the folder doesnt exist it will be created.
    medium: shdom.Medium object
        The atmospheric medium. This ground-truth medium will be used to 
    solver: shdom.RteSolver object
        The solver and the parameters used. This includes the scene parameters (such as solar and surface parameters)
        and the numerical parameters.
    measurements: shdom.Measurements
        Contains the sensor used to image the mediu and the radiance measurements. 
        
    Notes
    -----
    The ground-truth medium is later used for evaulation of the recovery.
    """  
    if not os.path.isdir(directory):
        os.makedirs(directory)  
    measurements.save(os.path.join(directory, 'measurements'))
    medium.save(os.path.join(directory, 'ground_truth_medium'))
    solver.save_params(os.path.join(directory, 'solver_parameters'))   


def load_forward_model(directory):
    """
    Save the forward model parameters for reconstruction.
    
    Parameters
    ----------
    directory: str
        Directory path where the forward modeling parameters are saved. 
    
    Returns
    -------
    medium: shdom.Medium object
        The atmospheric medium. This ground-truth medium will be used to 
    solver: shdom.RteSolver object
        The solver and the parameters used. This includes the scene parameters (such as solar and surface parameters)
        and the numerical parameters.
    measurements: shdom.Measurements
        Contains the sensor used to image the mediu and the radiance measurements. 
        
    Notes
    -----
    The ground-truth medium is used for evaulation of the recovery.
    """  
    
    # Load the ground truth medium for error analysis and ground-truth known phase and albedo
    medium_path = os.path.join(directory, 'ground_truth_medium')
    if os.path.exists(medium_path):
        medium = shdom.Medium()
        medium.load(path=medium_path)   
    else: 
        medium = None
        
    # Load shdom.Measurements object (sensor geometry and radiances)
    measurements = shdom.Measurements()
    measurements_path = os.path.join(directory, 'measurements')
    assert os.path.exists(measurements_path), 'No measurements file in directory: {}'.format(directory)
    measurements.load(path=measurements_path)
    
    # Load RteSolver according to numerical and scene parameters
    solver_path = os.path.join(directory, 'solver_parameters')
    solver = shdom.RteSolver()
    if os.path.exists(solver_path):
        solver.load_params(path=os.path.join(directory, 'solver_parameters'))   
    else:
        numerical_params = shdom.NumericalParameters()
        solver.set_numerics(numerical_params)

    return medium, solver, measurements