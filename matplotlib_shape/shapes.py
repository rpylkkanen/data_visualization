# Convert between [x], [y] and [(x, y)]
def points_to_xy(points):
  points = np.array(points)
  return points[:,0], points[:,1]

def xy_to_points(x, y):
  x, y = np.array(x), np.array(y)
  return np.dstack((x, y))[0]

# Operations

def rotate(x0, y0, x, y, angle):
  x, y = np.array(x), np.array(y)
  angle = np.deg2rad(angle)
  qx = x0 + np.cos(angle) * (x - x0) - np.sin(angle) * (y - y0)
  qy = y0 + np.sin(angle) * (x - x0) + np.cos(angle) * (y - y0)
  return qx, qy

# Analytics

def distance_between(x0, y0, x1, y1):
  return np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

def angle_between(x0, y0, x1, y1):
  return np.rad2deg(np.arctan2(y1 - y0, x1 - x0))