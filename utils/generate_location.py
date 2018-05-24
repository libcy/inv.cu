#!/usr/bin/env python

import sys
from os import makedirs
from os.path import exists

if __name__ == '__main__':
	""" Generates evenly distributed sources / stations
	  SYNTAX
		  generate_location nx nz xmax zmax (source configuration)
		  or generate_location nx 0 x0 zmax (source configuration)
	  	  e.g. ./generate_location.py 5 3 9100 3500
		       ./generate_location.py 5 0 50 3500
	"""
