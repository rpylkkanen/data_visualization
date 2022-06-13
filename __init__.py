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