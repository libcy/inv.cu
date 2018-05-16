#!/usr/bin/env python

import sys
from os.path import exists

import numpy as np
import pylab


def read_su(filename):
	data = []
	with open(filename, 'rb') as f:
		f.seek(114)
		nt = np.fromfile(f, dtype='int16', count=1)[0]

		nb = nt * 4 + 240
		i = 0
		while True:
			f.seek(i * nb + 240)
			trace = np.fromfile(f, dtype='float32', count=nt)
			if len(trace):
				i = i + 1
				data.append(trace)
			else:
				break

	return data

if __name__ == '__main__':
	""" Plots su(Seismic Unix) data

	  SYNTAX
		  plot_trace.py  folder_name  component_name||file_name  (source id)
		  e.g. ./plot_trace.py output vx
			   ./plot_trace.py output vx 0
			   ./plot_trace.py output vx_000000.su
	"""
	istr = ''
	if len(sys.argv) > 3:
		istr = str(sys.argv[3])
		while len(istr) < 6:
			istr = '0' + istr
	else:
		istr = '000000'

	path = "%s/%s" % (sys.argv[1], sys.argv[2])
	if not exists(path):
		path = "%s/%s.su" % (sys.argv[1], sys.argv[2])
	if not exists(path):
		path = '%s/%s_%s.su' % (sys.argv[1], sys.argv[2], istr)

	assert exists(path)

	data = read_su(path)
	am = 0
	t = np.arange(len(data[0]))
	for i in range(len(data)):
		am = max(am, np.amax(data[i]))

	for i in range(len(data)):
		pylab.plot(t, data[i] + i * am, 'b')

	pylab.gca().yaxis.set_visible(False)
	pylab.show()
