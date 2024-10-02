import numpy as np
import pandas as pd
    
def calculate_strip_centroid_z(strip_df,centroid_x,centroid_y):
    """
    Find z0 such that the point (centroid_x, centroid_y, z0) is closest to the
    line defined by the strip
    
    Derived in a Mathematica Notebook
    
    Parameters:
    -----------
    strip_df : pd.DataFrame
        The DataFrame containing the strip (one row)
    centroid_x,y : float
        The centroid_(x,y) coordinates found by the intersecting/adjacent crossing lines
    
    Returns:
    --------
    float
        The z0 from the function definition
    
    """
    x0 = centroid_x
    y0 = centroid_y
    
    # Extract values directly for x1, x2, y1, y2, z1, z2
    x1 = strip_df['xo'].iloc[0]
    x2 = strip_df['xe'].iloc[0]
    y1 = strip_df['yo'].iloc[0]
    y2 = strip_df['ye'].iloc[0]
    z1 = strip_df['zo'].iloc[0]
    z2 = strip_df['ze'].iloc[0]
    
    # If the strip has the same defined z1,z2, avoid inf
    if z1==z2:
        return z1
    
    numerator = ((1 * x0 * x1 - 1 * x0 * x2 - 1 * x1 * x2 + 1 * x2**2 + 
                 1 * y0 * y1 - 1 * y0 * y2 - 1 * y1 * y2 + 1 * y2**2) * z1**3 +
                (-3 * x0 * x1 + 1 * x1**2 + 3 * x0 * x2 + 1 * x1 * x2 - 2 * x2**2 - 
                 3 * y0 * y1 + 1 * y1**2 + 3 * y0 * y2 + 1 * y1 * y2 - 2 * y2**2) * z1**2 * z2 +
                (3 * x0 * x1 - 2 * x1**2 - 3 * x0 * x2 + 1 * x1 * x2 + 1 * x2**2 + 
                 3 * y0 * y1 - 2 * y1**2 - 3 * y0 * y2 + 1 * y1 * y2 + 1 * y2**2) * z1 * z2**2 +
                (-1 * x0 * x1 + 1 * x1**2 + 1 * x0 * x2 - 1 * x1 * x2 - 
                 1 * y0 * y1 + 1 * y1**2 + 1 * y0 * y2 - 1 * y1 * y2) * z2**3)
    
    denominator = ((1 * x1**2 - 2 * x1 * x2 + 1 * x2**2 + 1 * y1**2 - 
                   2 * y1 * y2 + 1 * y2**2) * z1**2 +
                  (-2 * x1**2 + 4 * x1 * x2 - 2 * x2**2 - 2 * y1**2 + 
                   4 * y1 * y2 - 2 * y2**2) * z1 * z2 +
                  (1 * x1**2 - 2 * x1 * x2 + 1 * x2**2 + 1 * y1**2 - 
                   2 * y1 * y2 + 1 * y2**2) * z2**2)
    
    z0 = numerator / denominator
    return z0
    
def closest_point_between_lines(a1, b1, a2, b2):
    """
    Calculate the closest point between two lines in 2D space.
    
    Parameters:
    -----------
    a1, b1 : numpy.ndarray
        The starting point and direction vector for the first line.
    a2, b2 : numpy.ndarray
        The starting point and direction vector for the second line.
        
    Returns:
    --------
    numpy.ndarray
        The closest point between the two lines.
    """
    b1 = b1 / np.linalg.norm(b1)
    b2 = b2 / np.linalg.norm(b2)
    a_diff = a2 - a1
    det = b1[0] * b2[1] - b1[1] * b2[0]
    
    if np.isclose(det, 0):
        lambda_1 = np.dot(a_diff, b1)
        closest_point_1 = a1 + lambda_1 * b1
        return closest_point_1
    else:
        lambda_ = (a_diff[0] * b2[1] - a_diff[1] * b2[0]) / det
        closest_point = a1 + lambda_ * b1
        return closest_point


def line_intersection(a1, b1, a2, b2):
    """
    Calculate the intersection point of two line segments in 2D space.
    
    Parameters:
    -----------
    a1, b1 : numpy.ndarray
        The starting point and direction vector for the first line.
    a2, b2 : numpy.ndarray
        The starting point and direction vector for the second line.
        
    Returns:
    --------
    numpy.ndarray or None
        The intersection point of the two line segments if they intersect, otherwise None.
    """
    
    xmin = np.amin([a1[0], a1[0] + b1[0], a2[0], a2[0] + b2[0]])
    xmax = np.amax([a1[0], a1[0] + b1[0], a2[0], a2[0] + b2[0]])
    ymin = np.amin([a1[1], a1[1] + b1[1], a2[1], a2[1] + b2[1]])
    ymax = np.amax([a1[1], a1[1] + b1[1], a2[1], a2[1] + b2[1]])
    
    b1 = b1 / np.linalg.norm(b1)
    b2 = b2 / np.linalg.norm(b2)
    a_diff = a2 - a1
    det = b1[0] * b2[1] - b1[1] * b2[0]
    
    if np.isclose(det, 0):  # Lines are parallel or coincident
        return None
    else:
        lambda_ = (a_diff[0] * b2[1] - a_diff[1] * b2[0]) / det
        intersection = a1 + lambda_ * b1
        # Check if the intersection point lies within the bounding box of both segments
        if (
            xmin <= intersection[0] <= xmax and
            ymin <= intersection[1] <= ymax
        ):
            return intersection
        else:
            return None