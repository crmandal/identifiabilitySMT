from __future__ import print_function
import json
import csv
from pprint import pprint
import os
import subprocess
import re
import sys, getopt
#from util.parseOutput import *
#from util.smtOutput import *

def main(argv):
	with open('testData.csv', 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=',', quotechar='\'', quoting=csv.QUOTE_MINIMAL)
		#writer.writerow(['time', 'K', 'radius', 'rho', 'g', 'x', 'v', 'mode'])
		with open('test9.json') as f:
			data = json.load(f)
			traces = data['traces']
			#pprint(traces)
			k = 0
			radius = 0
			rho = 0
			g = 0
			times = []
			xencls = []
			vencls = []
			modes = []
			for states in traces:
				for state in states:
					#pprint(state)
					key = state['key']
					mode = state['mode']
					step = state['step']
					values = state['values']
					
					value = values[0]
					time = value['time']
					encl = value['enclosure']
					#print(key)
					if(key.startswith('K_')):	
						k = encl	
					elif(key.startswith('g_')):
						g = encl
					elif(key.startswith('radius_')):
						radius = encl
					elif(key.startswith('rho_')):
						rho = encl
					
					for value in values:					
						time = value['time']
						encl = value['enclosure']					
						if(key.startswith('x_')): 
							xencls.append(encl)
						elif(key.startswith('v_')):
							vencls.append(encl)
						elif(key.startswith('tm_')): 
							times.append(encl)
						modes.append(mode)
					
			print(len(times),len(xencls), len(vencls))
			if(len(xencls) == len(times) and  len(vencls) == len(times)):
				for i in range(len(times)):
					writer.writerow([left(times[i]), left(k), left(radius), left(rho), left(g), left(xencls[i]), left(vencls[i]), modes[i]])
					writer.writerow([right(times[i]), right(k), right(radius), right(rho), right(g), right(xencls[i]), right(vencls[i]), modes[i]])
					#writer.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
		#	pprint(data)
		
def left(value):
	range1 = value[0]
	#print(value, range1)
	return range1
	
def right(value):
	range2 = value[1]
	#print(value, range2)
	return range2

if __name__ == "__main__": 	
   main(sys.argv[1:])
	
#def main(argv):
#	inputfile = sys.argv[1]
#	satInstance = parseInstance(inputfile)
	#ha = getModel(inputfile)
	#pha = convertHA2PHA(ha, params)
	
	
	
