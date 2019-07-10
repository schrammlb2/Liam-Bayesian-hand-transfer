import glob
import matplotlib.pyplot as plt
import numpy as np
import pdb

split_string = '--------------------------------------------'

for filename in glob.glob('*_diagnostics'):
	file = open(filename)
	lines = file.read().split('\n')
	# blocks = file.read().split(split_string)
	# lines_in_blocks = [block.split('\n') for block in blocks]
	grad_norms = []
	for line in lines:
		line_start = 'Grad norm: '
		if line_start in line:
			norm_val = float(line[len(line_start):])
			grad_norms.append(norm_val)

	if len(grad_norms) >=16:
		grad_norms = np.array(grad_norms)

		# pdb.set_trace()
		outfilename = 'model_data/' + filename[:-len('_diagnostics')] + '.png'

		plt.figure(1)
		plt.plot(grad_norms[:], color='blue', label='Gradient norm', marker='.')
		# plt.axis('scaled')
		plt.legend()
		plt.savefig(outfilename)
		plt.close()


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


all_grad_norms = []
name_list = []
for filename in glob.glob('*constrained_restart*'):
	file = open(filename)
	lines = file.read().split('\n')
	# blocks = file.read().split(split_string)
	# lines_in_blocks = [block.split('\n') for block in blocks]
	grad_norms = []
	for line in lines:
		line_start = 'Grad norm: '
		if line_start in line:
			norm_val = float(line[len(line_start):])
			grad_norms.append(norm_val)

	grad_norms = np.array(grad_norms)
	all_grad_norms.append(grad_norms)
	name_list.append(filename)
	# pdb.set_trace()
outfilename = 'model_data/all_grad_norms.png'

cmap = get_cmap(len(all_grad_norms))

plt.figure(1)
for i, grad_norms in enumerate(all_grad_norms):
	label=name_list[i][len('transferA2B_constrained_restart_l2.'):-len('_diagnostics')]#0.0_held_out.0.1
	plt.plot(grad_norms[:], c=cmap(i), label=label, marker='.')
# plt.axis('scaled')
plt.legend()
plt.savefig(outfilename)
plt.close()