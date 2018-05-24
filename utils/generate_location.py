#!/usr/bin/env python

import sys
from os import makedirs
from os.path import exists

if __name__ == '__main__':
	""" Generates evenly distributed sources / stations
	  SYNTAX
		  generate_location nx nz xmax zmax (source configuration)
		  or generate_location nx 0 xmax z0 (source configuration)
	  	  e.g. ./generate_location.py 5 3 9100 3500
		       ./generate_location.py 5 0 50 3500
	"""

	if not exists('output'):
		makedirs('output')

	argc = len(sys.argv)
	if argc > 5:
		filename = 'sources.dat'
	else:
		filename = 'stations.dat'

	x = []
	z = []
	nx = int(sys.argv[1])
	nz = int(sys.argv[2])
	xmax = float(sys.argv[3])
	zmax = float(sys.argv[4])
	lenx = len(str(int(xmax)))
	lenz = len(str(int(zmax)))

	cfg = ''
	for i in range(5, argc):
		cfg = cfg + ' ' + sys.argv[i]

	for i in range (1, nx + 1):
		x.append(i * xmax / (nx + 1))

	if nz > 0:
		for i in range(1, nz + 1):
			z.append(i * zmax / (nz + 1))
	else:
		z.append(zmax)

	with open('output/' + filename, 'w') as f:
		for i in range(len(z)):
			for j in range(nx):
				if xmax > 99999:
					xj = '%.5e' % x[j]
				else:
					xj = '%.6f' % x[j]
					for k in range(len(xj), lenx + 7):
						xj = ' ' + xj

				if zmax > 99999:
					zi = '%.5e' % z[i]
				else:
					zi = '%.6f' % z[i]
					for k in range(len(zi), lenz + 7):
						zi = ' ' + zi

				f.write(xj + ' ' + zi + cfg + '\n')
