# matplotlib_shape

A wrapper for drawing different shapes using matplotlib. Examples are drawn using higher level data_visualization module.


# Example 1

The simplest example is drawing an exact sized single plot. Size adjusting is done using `matplotlib_exact.Alignment` class, which in this case is initialized with 1 row and 1 column. `Alignment` stores `matplotlib_exact.Axes` in an N-dimensional array, which can be configured individually. For data plotting, the common matplotlib fig, ax objects can be obtained using `ax = matplotlib_exact.Axes.matplotlib()` and `fig = matplotlib_exact.Alignment.figure()`. `matplotlib_exact.Axes` class has two helper methods to display the spacing and size of the plot, `matplotlib_exact.Axes.annotate_text()` and `matplotlib_exact.Axes.annotate_rect()`.

![Example 1](https://github.com/rpylkkanen/exact-size-matplotlib/blob/main/examples/example_1.png)

```python
from matplotlib_exact import Alignment

nrows, ncols = 1, 1
alignment = Alignment(nrows, ncols)

# Configure spacing and sizes.
a = alignment[0][0]
axis_spacing = 0.1
axis_width = 0.8
axis_height = 0.8
a.set_spacing(every=axis_spacing)
a.set_size(width=axis_width, height=axis_height)

# Plot something.
ax = a.matplotlib()
ax.set_xticks([])
ax.set_yticks([])	
a.annotate_rect()
a.annotate_text()

# Save
fig = alignment.figure()
fig.savefig('examples/example_1.png', dpi=300)
```


## Example 2

A bit more complex is a grid, but follows similar logic. 

![Example 1](https://github.com/rpylkkanen/exact-size-matplotlib/blob/main/examples/example_2.png)

```python
# Example 2

from matplotlib_exact import Alignment

nrows, ncols = 4, 4
alignment = Alignment(nrows, ncols)

# Configure spacing and sizes.
fig_width = nrows
fig_height = nrows
space = 0.1

for a in alignment.flatten():
	a.set_spacing(every=space)
	axis_width = fig_width/alignment.ncols() - a.spacing_width()
	axis_height = fig_height/alignment.nrows() - a.spacing_height()
	a.set_width(axis_width)
	a.set_height(axis_height)

# Plot something.
for a in alignment.flatten():
	ax = a.matplotlib()
	if ax:
		ax.set_xticks([])
		ax.set_yticks([])	
	a.annotate_rect()
	a.annotate_text()

# Save
fig = alignment.figure()
fig.savefig('examples/example_2.png', dpi=300)
```

## Example 3

It is possible to make very strange alignments, and also possible to treat the whole Axes as empty space if the `ax = matplotlib_exact.Axes.matplotlib()` is not called. However, the alignments are calculated based on rows, but it is possible to have different left, right, top, and bottom spacings for same row members:

![Example 3](https://github.com/rpylkkanen/exact-size-matplotlib/blob/main/examples/example_3.png)

```python
# Example 3

from matplotlib_exact import Alignment

nrows, ncols = 4, 4
alignment = Alignment(nrows, ncols)

# Configure spacing and sizes
fig_width = nrows 
fig_height = ncols * 3 / 4

for r, row in enumerate(alignment):
	for c, a in enumerate(row):

		pad = 0.025
		space = 0.15
		increment = 0.05
		a.set_left(space - pad + increment * r)
		a.set_right(space + pad - increment * r)
		a.set_top(space - pad + increment * c)
		a.set_bottom(space + pad - increment * c)
		if c == 1:
			a.set_right(space + pad + increment * r)
		if c == 3:
			a.set_left(a.left()/2)
			a.set_top(a.top()/2)
			a.set_bottom(space)

		axis_width = fig_width/alignment.ncols() - a.spacing_width()
		axis_height = fig_height/alignment.nrows() - a.spacing_height()
		a.set_width(axis_width)
		a.set_height(axis_height)

# Plot something.
for r, row in enumerate(alignment):
	for c, a in enumerate(row):

		if c == 0 and r != 0:
			ax = a.matplotlib()	
			ax.set_xticks([])
			ax.set_yticks([])
			ax.plot([c for c in range(10)], 'o-', mfc='white', color='rgbk'[r], alpha=0.1)
			ax.set_xlim(-r - c, r + c)
			a.annotate_rect()
			a.annotate_text()

		if c == 1 and r != 1:
			ax = a.matplotlib()	
			ax.set_xticks([])
			ax.set_yticks([])
			ax.plot([c for c in range(10)], 'o-', mfc='white', color='rgbk'[r], alpha=0.1)
			ax.set_xlim(r - c, r + c)
			a.annotate_rect()
			a.annotate_text()

		if c == 2 and r != 2:
			ax = a.matplotlib()	
			ax.set_xticks([])
			ax.set_yticks([])
			ax.plot([c for c in range(10)], 'o-', mfc='white', color='rgbk'[r], alpha=0.1)
			ax.set_ylim(r + c, r + c * 2)
			a.annotate_rect()
			a.annotate_text()

		if c == 3 and r != 3:
			ax = a.matplotlib()	
			ax.set_xticks([])
			ax.set_yticks([])
			ax.plot([c for c in range(10)], 'o-', mfc='white', color='rgbk'[r], alpha=0.1)
			ax.set_xlim(- r - c * 2, c)
			a.annotate_rect()
			a.annotate_text()

# Add lines to figure
from matplotlib.lines import Line2D

fig = alignment.figure()
fig.add_artist(Line2D([0.5, 0.5], [0.0, 1.0], ls=':', lw=1, color='k', alpha=0.5))
fig.add_artist(Line2D([0.0, 1.0], [0.5, 0.5], ls=':', lw=1, color='k', alpha=0.5))
fig_w, fig_h = fig.get_size_inches()
fig.text(0.5, 0.5, f'Figure:\n{fig_w:.1f}x{fig_h:.1f}"', ha='center', va='center', fontsize='xx-small', bbox=dict(fc='white', alpha=0.5))

# Save
fig.savefig('examples/example_3.png', dpi=300)
```