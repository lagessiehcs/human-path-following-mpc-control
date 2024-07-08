import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.integrate import quad
from scipy import io
import os


def gen_path(shape, f=1):
    """
    Generate path based on the specified shape.
    
    Parameters:
    shape (str): Shape of the path. Options are 'Sinus', 'Straight', 'Sample', '8'.
    f (float): Frequency for the sine function. Default is 1.
    
    Returns:
    np.ndarray: Generated path with shape (n, 2).
    """

    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    if shape == "Sinus":
        x = np.linspace(0, 1, 200)
        y = np.sin(2 * np.pi * f * x)
    elif shape == "Straight":
        x = np.linspace(0, 10, 200)
        y = np.zeros_like(x)
    elif shape == "Sample":
        mat_file_path = os.path.join(current_script_dir, 'route.mat')
        route = io.loadmat(mat_file_path)['route']
        return route
    elif shape == "8":
        mat_file_path = os.path.join(current_script_dir, 'eight.mat')
        route = io.loadmat(mat_file_path)['route']
        return route
    else:
        raise ValueError("Unsupported shape: {}".format(shape))
    
    return np.column_stack((x, y))


def arclength(px, py, *args):
    """
    Compute arc length of a space curve or any curve represented as a sequence of points.

    Parameters:
        px, py (array-like): Vectors defining points along the curve.
        *args: Optional arguments which could include pz for 3D curves and method.

    Returns:
        arclen (float): Total arc length of the curve.
        seglen (numpy array): Arc length of each independent curve segment.
    """
    
    # Ensure px and py are numpy arrays and have the same length
    px = np.asarray(px)
    py = np.asarray(py)
    
    if len(px) != len(py):
        raise ValueError("px and py must be vectors of the same length")
    if len(px) < 2:
        raise ValueError("px and py must have length at least 2")
    
    # Compile the curve into one array
    data = np.column_stack((px, py))
    
    # Defaults for method
    method = 'linear'
    
    # Check for additional arguments
    for arg in args:
        if isinstance(arg, str):
            # It must be the method
            valid_methods = ['linear', 'pchip', 'spline']
            if arg.lower() not in valid_methods:
                raise ValueError("Invalid method indicated. Only 'linear', 'pchip', 'spline' allowed.")
            method = arg.lower()
        else:
            # It must be pz, defining a space curve in higher dimensions
            arg = np.asarray(arg)
            if len(arg) != len(px):
                raise ValueError("pz was supplied, but is inconsistent in size with px and py")
            data = np.column_stack((data, arg))
    
    # Compute the chordal linear arc lengths
    diff_data = np.diff(data, axis=0)
    seglen = np.sqrt(np.sum(diff_data**2, axis=1))
    arclen = np.sum(seglen)
    
    # We can quit if the method was 'linear'
    if method == 'linear':
        return arclen, seglen
    
    # 'spline' or 'pchip' must have been indicated, so we will be doing an integration
    chordlen = seglen
    cum_chordlen = np.insert(np.cumsum(chordlen), 0, 0)
    
    # Compute the splines or pchip interpolators
    splines = []
    for i in range(data.shape[1]):
        if method == 'pchip':
            splines.append(PchipInterpolator(cum_chordlen, data[:, i]))
        elif method == 'spline':
            splines.append(CubicSpline(cum_chordlen, data[:, i]))
    
    def integrand(t, splines):
        """Helper function for integration"""
        result = 0
        for spline in splines:
            result += spline(t, 1)**2
        return np.sqrt(result)
    
    # Numerical integration along the curve
    seglen = np.zeros(len(chordlen))
    for i in range(len(chordlen)):
        seglen[i], _ = quad(integrand, cum_chordlen[i], cum_chordlen[i+1], args=(splines,))
    
    # Sum the segments to get the total arc length
    arclen = np.sum(seglen)
    
    return arclen, seglen

def main(args=None):    
    # Example usage:
    theta = np.linspace(0, 2*np.pi, 10)
    x = np.cos(theta)
    y = np.sin(theta)

    path = gen_path("Sinus")
    print(f'Path: {path[:,0]}')

    # Integrated pchip curve fit
    arclen, seglen = arclength(x, y)
    print(f'Default method (linear): Total arc length = {arclen}, Segment lengths = {seglen}')

    # Linear chord lengths
    arclen, seglen = arclength(x, y, 'linear')
    print(f'Linear method: Total arc length = {arclen}, Segment lengths = {seglen}')

    # Integrated pchip curve fit
    arclen, seglen = arclength(x, y, 'pchip')
    print(f'Pchip method: Total arc length = {arclen}, Segment lengths = {seglen}')

    # Integrated spline fit
    arclen, seglen = arclength(x, y, 'spline')
    print(f'Spline method: Total arc length = {arclen}, Segment lengths = {seglen}')

if __name__ == '__main__':
    main()