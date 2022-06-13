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
from .matplotlib_style import colors
from .matplotlib_extension import *


### Color shortcuts

color_index = 5

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


### Fitting

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

import numpy as np

def r2(x, y, func, params):
  residuals = y - func(x, *params)
  residual_sum_of_squares = np.sum(residuals**2)
  total_sum_of_squares = np.sum((y - np.mean(y))**2)
  r_squared = 1 - (residual_sum_of_squares/total_sum_of_squares)
  return r_squared

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def make_multimodal(num):
  params = []
  ret = []
  for i in range(num):
    params += [f'mu{i}', f'sigma{i}', f'A{i}']
    ret += [f'gauss(x, mu{i}, sigma{i}, A{i})']
  definition = f'def multimodal(x, {", ".join(params)}): return {" + ".join(ret)}'
  exec(definition)
  return locals()['multimodal']

def gaussian_fit(number, x, y, guess=None, bounds=None):
  func = make_multimodal(number)
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
            print(repr(line), mp)
          line = line.split('\t')
          
  table = pd.DataFrame(result)
  table = table.astype({'#': int, 'Mn (Daltons)': int, 'Mw (Daltons)': int, 'Mz (Daltons)': int, 'MP (Daltons)': int, 'Polydispersity': float, '(µV*sec)': int, 'Result Id': int})
  table = table.sort_values(by=['label'])

  return table, data
    
