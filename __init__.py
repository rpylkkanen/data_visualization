### Dependencies

import sys
import subprocess
import pkg_resources

required = {'matplotlib', 'numpy', 'pandas', 'scipy', 'BaselineRemoval', 'matplotlib_scalebar'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing])

from .matplotlib_exact import Alignment
from .matplotlib_style import colors, color_abbreviations
from .matplotlib_extension import *

## Image adjustments

import cv2
from math import factorial

def adjust_contrast(file, limit=1.0, grid_size=(8, 8)):

  # Load the image in greyscale
  img = cv2.imread(file,0)

  # create a CLAHE object (Arguments are optional).
  clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid_size)
  out = clahe.apply(img)

  return out

def savitzky_golay(y, window_size=21, order=2, deriv=0, rate=1):
  window_size = np.abs(np.int(window_size))
  order = np.abs(np.int(order))
  if window_size % 2 != 1 or window_size < 1:
      raise TypeError("window_size size must be a positive odd number")
  if window_size < order + 2:
      raise TypeError("window_size is too small for the polynomials order")
  order_range = range(order+1)
  half_window = (window_size -1) // 2
  # precompute coefficients
  b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
  m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
  # pad the signal at the extremes with
  # values taken from the signal itself
  firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
  lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
  y = np.concatenate((firstvals, y, lastvals))
  return np.convolve( m[::-1], y, mode='valid')

def smooth(y, kwargs={}):
  return savitzky_golay(y, **kwargs)

### Color shortcuts

color_index = 7

red = colors['red'][color_index]
pink = colors['pink'][color_index]
purple = colors['purple'][color_index]
deep_purple = colors['deep purple'][color_index]
indigo = colors['indigo'][color_index]
blue = colors['blue'][color_index]
light_blue = colors['light blue'][color_index]
cyan = colors['cyan'][color_index]
teal = colors['teal'][color_index]
green = colors['green'][color_index]
light_green = colors['light green'][color_index]
lime = colors['lime'][color_index]
yellow = colors['yellow'][color_index]
amber = colors['amber'][color_index]
orange = colors['orange'][color_index]
deep_orange = colors['deep orange'][color_index]
brown = colors['brown'][color_index]
grey = colors['grey'][color_index]
blue_grey = colors['blue grey'][color_index]

def get_colors(color_abbrevs_str, shade_index):
    color_abbrevs_list = [color_abbrevs_str[i:i+2] for i in range(0, len(color_abbrevs_str), 2)]
    codes = [color_abbreviations[color] for color in color_abbrevs_list]
    result = [colors[code][shade_index] for code in codes]
    if len(result) == 1:
      result = result[0]
    elif len(result) == 0:
      result = None
    return result

### Fitting

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

import numpy as np

from BaselineRemoval import BaselineRemoval

def read_maldi_dx(path):

  with open(path) as f:
    data = f.readlines()[22].split(';')
    data.remove('\n')
    frame = pd.DataFrame(data)
    l = frame[0].str.split(',')
    x = []
    y = []
    for i in l:
      x.append(float(i[0]))
      y.append(float(i[1]))

    x = np.array(x)
    y = np.array(y)
    
    return x, y


def normalize(y, remove_baseline=False):
  
  if remove_baseline:
    y = BaselineRemoval(y).ZhangFit()
  values, counts = np.unique(y, return_counts=True)
  index = np.argmax(counts)
  mode = values[index]
  y = (y - mode)/np.ptp(y)

  return y

def filter_xy(x, y, xmin=None, xmax=None):
  # xlims
  if xmin is not None:
    idx = (x >= xmin)
    x, y = x[idx], y[idx]
  if xmax is not None:
    idx = (x <= xmax)
    x, y = x[idx], y[idx]
  return x, y

def gompertz(t, a, b, c):
  return a * np.exp(-b * np.exp(-c * t))

def gompertz_error(x, popt, pcov):
  sigma = np.sqrt(np.diag(pcov))
  values = np.array([
    gompertz(x, popt[0] + sigma[0], popt[1] + sigma[1], popt[2] + sigma[2]), 
    gompertz(x, popt[0] + sigma[0], popt[1] - sigma[1], popt[2] + sigma[2]),   
    gompertz(x, popt[0] + sigma[0], popt[1] + sigma[1], popt[2] - sigma[2]), 
    gompertz(x, popt[0] + sigma[0], popt[1] - sigma[1], popt[2] - sigma[2]), 
    gompertz(x, popt[0] - sigma[0], popt[1] + sigma[1], popt[2] + sigma[2]), 
    gompertz(x, popt[0] - sigma[0], popt[1] - sigma[1], popt[2] + sigma[2]),
    gompertz(x, popt[0] - sigma[0], popt[1] + sigma[1], popt[2] - sigma[2]), 
    gompertz(x, popt[0] - sigma[0], popt[1] - sigma[1], popt[2] - sigma[2]) 
  ])
  errfit = np.std(values, axis=0)
  return errfit

def r2(x, y, func, params):
  residuals = y - func(x, *params)
  residual_sum_of_squares = np.sum(residuals**2)
  total_sum_of_squares = np.sum((y - np.mean(y))**2)
  r_squared = 1 - (residual_sum_of_squares/total_sum_of_squares)
  return r_squared

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def make_multimodal_gauss(num):
  params = []
  ret = []
  for i in range(num):
    params += [f'mu{i}', f'sigma{i}', f'A{i}']
    ret += [f'gauss(x, mu{i}, sigma{i}, A{i})']
  definition = f'def multimodal(x, {", ".join(params)}): return {" + ".join(ret)}'
  exec(definition)
  return locals()['multimodal']

def gaussian_fit(number, x, y, guess=None):
  func = make_multimodal_gauss(number)
  params, covariances = curve_fit(func, x, y, p0=guess, bounds=[0, np.inf])
  stdevs = np.sqrt(np.diag(covariances))
  rsq = r2(x, y, func, params)
  return func, params, stdevs, rsq

def bigauss(x, center=0.0, w1=1.0, w2=1.0, height=1.0):
  y = x.copy()
  y[x < center] = height*np.exp(-0.5*((x[x < center] - center)/w1)**2)
  y[x >= center] = height*np.exp(-0.5*((x[x >= center] - center)/w2)**2)
  return y

def make_multimodal_bigauss(num):
  params = []
  ret = []
  for i in range(num):
    params += [f'center_{i}', f'w1_{i}', f'w2_{i}', f'height_{i}']
    ret += [f'bigauss(x, center_{i}, w1_{i}, w2_{i}, height_{i})']
  definition = f'def multimodal(x, {", ".join(params)}): return {" + ".join(ret)}'
  exec(definition)
  return locals()['multimodal']

def bigaussian_fit(number, x, y, guess=None):
  func = make_multimodal_bigauss(number)
  params, covariances = curve_fit(func, x, y, p0=guess, bounds=[0, np.inf])
  stdevs = np.sqrt(np.diag(covariances))
  rsq = r2(x, y, func, params)
  return func, params, stdevs, rsq

### Template

def template(box='h', n=1, dpi=200):
	fig = matplotlib.pyplot.figure(dpi=dpi)
	if box == 'h':
		box = fig.hbox(n)
	else:
		box = fig.vbox(n)
	axes = box.children
	ax = axes[0]
	return fig, box, axes, ax


### Processing

import pandas as pd
from enum import Enum

def read_sec_ars(f, labels):

  with open(f, 'r', encoding="ISO-8859-1") as file:
    lines = file.readlines()
    lines = [line for line in lines if 'Page' not in line]

  def format_line(line):
      line = line.split('"')
      for symbol in ['', '\t', '\n']:
        if line:
          while symbol in line:
            line.remove(symbol)
      return line

  start = False
  label = None
  label_idx = 0
  hold_data = False
  data = {}
  for i, line in enumerate(lines):
    if 'Mp' in line:
      label = labels[label_idx]
      label_idx += 1
      data[label] = {
          '#': [],
          'Slice Log MW': [],
          'dwt/d(logM)': [],
      }
    if label != None:
      if 'GPC' not in line and ':' not in line and '#' not in line:
        line = line.rstrip()
        line = line.replace('"', '')
        line = line.replace(',', '.')
        line = line.split('\t')
        if len(line) != 3:
          hold_data = True
          result = line
        elif hold_data and len(line) == 2:
          result = result + line
          hold_data = False
        else:
          result = line
        if len(result) == 3:
          for key, value in zip(data[label].keys(), result):
            data[label][key].append(float(value))

  data = dict(sorted(data.items()))

  class GPC(Enum):
    DEFAULT = 0
    RESULT = 1
    DISTRIBUTION_TABLE = 2

  mode = GPC.DEFAULT
  result = {
      '#': [],
      'label': [],
      'Mn (Daltons)': [], 
      'Mw (Daltons)': [],
      'Mz (Daltons)': [],
      'MP (Daltons)': [],
      'Polydispersity': [],
      '(µV*sec)': [],
      'Processing Method': [], 
      'Result Id': [],
      }

  with open(f, 'r', encoding="ISO-8859-1") as file:
    lines = file.readlines()
    for line in lines:
      line = line.rstrip().replace('"', '').replace(',', '.')
      if 'Page' in line:
        continue
      if line == 'GPC Results':
        mode = GPC.RESULT
        continue
      elif line == 'GPC Distribution Table':
        mode = GPC.DISTRIBUTION_TABLE
        continue
      else:
        if mode == GPC.RESULT:
          line = line.split('\t')
          if len(line) == 10 and line[0] != '#':
            if line[0] == '#':
              result.append([])
            else:
              for key, value in zip(result.keys(), line):
                result[key].append(value)
        elif mode == GPC.DISTRIBUTION_TABLE:
          if 'Mp' in line:
            mp = int(line.split(' ')[1])
          line = line.split('\t')
          
  table = pd.DataFrame(result)
  table = table.astype({'#': int, 'Mn (Daltons)': int, 'Mw (Daltons)': int, 'Mz (Daltons)': int, 'MP (Daltons)': int, 'Polydispersity': float, '(µV*sec)': int, 'Result Id': int})
  table = table.sort_values(by=['label'])

  return table, data
    
