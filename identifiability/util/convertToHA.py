from __future__ import print_function
import os
import subprocess
import re
import sys, getopt
import os.path
import collections
from collections import OrderedDict
from parser.parseModel import getModel, Goal
from parser.parseReactions import getReactions, Reaction

tempfile = "temp.drh"
folder = ''
tempfolder = 'temp'
SpeciesIndex = {}
RevSpeciesIndex = {}

def main(argv):
	try:
		opts, args = getopt.getopt(argv,"hi:o",["infile=","outfile="])
	except getopt.GetoptError:
			print("convertToHA.py -i <infile> -o <outfile>")
			sys.exit(2)
	infile  = ""
	outfile = ""

	global tempfile, tempfolder
	#tempfile = TNAME
	
	for opt, arg in opts:
		if opt == '-h':
			print("\t -i <infile> \n\t-o <outfile>")
			sys.exit()
		elif opt in ("-i", "--infile"):
			infile = arg
		elif opt in ("-o", "--outfile"):
			outfile = arg
		
	print("Reaction file is :"+infile)

	print("out file is :"+outfile)
	
	fname = os.path.join(folder, infile)
	reactions = getReactions(fname)

	ode = getODEsfromReactions(reactions)

def getODEsfromReactions(reactions):
	# create stoichiometric matrix
	#stoichiometricMatrix = [[]]
	matrix = {}
	j = 0
	val = 0
	for react in reactions:
		#stoichiometricMatrix.append([])
		for r in react.reactants:
			val = 0
			(i, new) = getIndex(r.name)
			if((i,j) in matrix):
				val = matrix[(i,j)]
			val+= -1 * r.stoichiometric
			matrix.update({(i,j):val})
			#stoichiometricMatrix[j][i] += -1 * r.stoichiometric
			
		for r in react.products:
			val = 0
			(i, new) = getIndex(r.name)
			if((i,j) in matrix):
				val = matrix[(i,j)]
			val+= r.stoichiometric
			matrix.update({(i,j):val})
			#stoichiometricMatrix[j][i] += r.stoichiometric		
		j= j+1
	print(SpeciesIndex)
	print(matrix)
	m = j
	n = len(SpeciesIndex)
	stoichiometricMatrix = [[0 for i in xrange(m)] for i in xrange(n)]
	for i in range(n):
		for j in range(m):
			if((i,j) in matrix):
				stoichiometricMatrix[i][j] = matrix[(i,j)]
			
	print(stoichiometricMatrix)	
	
	# create rate laws
	rateLaw = []
	#j = 0
	for react in reactions:
		law = ''
		rate = react.reactionRate
		law += str(rate)
		for r in react.reactants:
			if(r.stoichiometric > 1):
				for i in range(r.stoichiometric):
					law += '*'+ str(r.name)
			else:
				law += '*'+ str(r.name)
			
		rateLaw.append(law)
		#j++
	print(rateLaw)
		
	# create rate equation
	ode = multiply(rateLaw, stoichiometricMatrix)
	print(ode)

	with open(tempfile, 'w') as of:
		for r in ode:
			of.write('d/dt['+RevSpeciesIndex[r]+'] = '+ ode[r] + ';\n')


	
def getIndex(species):
	l = len(SpeciesIndex)
	#print(species, l)
	#print(SpeciesIndex)
	if(species in SpeciesIndex):
		return (SpeciesIndex[species],0)
	else:
		SpeciesIndex.update({species:l})
		RevSpeciesIndex.update({l:species})
	#	print(SpeciesIndex)
		#l = len(SpeciesIndex)
		#print(species, l)
		return (SpeciesIndex[species], 1)

def multiply(v, m):
	result = {}
	for i in range(len(m[0])): #this loops through columns of the matrix
		total = ''
		for j in range(len(v)): #this loops through vector coordinates & rows of matrix
			if(m[i][j] > 0):
				if(total != ''):
					total += '+'+ str(m[i][j]) + '*' + v[j]
				else:
					total += str(m[i][j]) + '*' + v[j]
			elif(m[i][j] < 0):
				total +=  str(m[i][j]) + '*' + v[j]
		result.update({i: total})
	return result
    
if __name__ == "__main__": 
   main(sys.argv[1:])
