from __future__ import print_function
import os
import subprocess
import re
import sys, getopt

import collections
from collections import OrderedDict
from decimal import Decimal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from model.property import *
from model.haModel import *
from parser.parseSTL import *
from model.node import *
#from model.condition import *
from util.reach import *
from util.graph import *
from ha2smt.dRealSMT import *
from ha2smt.utilFunc import *
from model.node_factory import *

timeRange = None

def main(argv):
	inputfile = sys.argv[1]
	propertyfile = sys.argv[2]
	outputfile = sys.argv[3]
	try:
		opts, args = getopt.getopt(argv,"hi:p:",["ifile=","pfile="])
	except getopt.GetoptError:
			print("smtEncoder.py -i <inputfile> -p <propertyfile>")
			sys.exit(2)
			
	for opt, arg in opts:
		if opt == '-h':
			print("smtEncoder.py -i <inputfile> -p <propertyfile>")
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-p", "--pfile"):
			propertyfile = arg
	
	print("Input file is :"+ inputfile)
	print("Property file is :"+ propertyfile)
	
	ha = getModel(inputfile)
	print('model parsed')
	#print(str(ha))
	
	goal = getSTLfromfile(propertyfile)
	print('property parsed')
	prop = goal[0]
	print(prop.to_infix())
	neg = prop.negate().delta_perturb(delta)#prop.negateMin()
	print(neg.to_infix())
	smts = encode2SMT(ha, neg, 0.0001, 2)
	
	outfile = inputfile.split('.')[0]
	print(outfile)
	i = 0
	for (smt, path) in smts:
		with open(outfile+'_'+str(i)+'.smt2', 'w') as of:
			of.write(smt)
		i +=1
   
		
def encode2SMT(model, goal, delta, k):
	#model = addTimeVar(model, 'tm')
		
	g = model.getGraph();
	st = model.init.mode
	tgt = model.goals[0].mode
	
	smts = []
		
	#~ for path in g.getPathsofLength(st, tgt, k):
		#~ print(path)
		#~ #smt = ''
		#~ smt = generateSMTforPath(model, path, delta)
		#~ smts.append(smt)
		
	for path in g.getKPaths(st, k):
		#print(path)
		#smt = ''
		smt = addGoalTopath(model, path, delta, goal)
		smts.append((smt,path))
	return smts

def addGoalTopath(model, path, delta, goal = None):
	
	smtEncode = generateSMTforPath(model, path, delta)	
	
	string = generateGoalCondition(smtEncode, goal)
	smtEncode.addGoal(string)
	
	smt = str(smtEncode)
#	print(smt)
	return smt
	
def generateSMTforPath(model, path, delta):
	smtEncode = SMT(model, path, 'QF_NRA_ODE', delta)
		
	generateVARdeclaration(smtEncode)
	
	generateODEdeclaration(smtEncode)
	
	#smt += generateVARbounds(model, path)
		
	#smt += smtEncode.generatePATHencoding()
	
	return smtEncode

def generateVARdeclaration(smt):
	#print('generateVARdeclaration')
	#smt = ''
	model = smt.getModel()
	path = smt.getPath()
	
	modes = []
	for st in model.states:
		modes.append(float(st.mode))
	m1 = min(modes)
	m2 = max(modes)
	#("{:.6f}".format(min(modes)), "{:.6f}".format(max(modes)))
	
	for var in model.variables.keys():
		if not var == 'time': 
			smt.addVariable(var, model.variables[var])
		
	m = len(path)
	for i in range(m):
		for var in model.variables.keys():
			if not var == 'time': 
				smt.addVariable(getVar_0_index(var, i), model.variables[var])
				smt.addVariable(getVar_t_index(var, i), model.variables[var])
				
		
		smt.addVariable(getVar_at_depth('time', i), model.variables['time'])
		smt.addVariable(getVar_at_depth('mode', i), Range(Node(m1), Node(m2)))
	
def generateODEdeclaration(smt):
	#print('generateODEdeclaration')
	model = smt.getModel()
	path = smt.getPath()
	
	odeAdded = {}
	m = len(path)
	for i in range(m):
		for loc in path:
			state = findMode(model, loc)
			mode = state.mode
			#print('Adding for mode ', mode)
			#odes = []
			#[todo:] for each unique mode in path
			if(mode not in odeAdded or odeAdded[mode] == 0 ):
				for ode in state.flow:
					smt.addODE(mode, ode)
					#odes.append(ode.clone())					
				odeAdded.update({mode:1})
			#smt.addODEs(mode, odes)
	return smt
	
def generateVARbounds(smt):
	model = smt.getModel()
	path = smt.getPath()
	
	m = len(path)
	for i in range(m):
		for var in model.variables.keys():
			if not var == 'time': 
				smt.addAssert('>=', getVar_0_index(var, i) , model.variables[var].getleft().evaluate());
				smt.addAssert('<=', getVar_0_index(var, i) , model.variables[var].getrightt().evaluate());
			else:
				
				smt.addAssert('>=', getVar_at_depth(var, i) , model.variables[var].getleft().evaluate());
				smt.addAssert('<=', getVar_at_depth(var, i) , model.variables[var].getrightt().evaluate());
	return smt

def generateGoalCondition(smtEncode, goal = None):	
	#print('generateGoalCondition')
	model = smtEncode.getModel()
	path = smtEncode.getPath()
	
	m = len(path)
	smt = '\n ; goal condition \n'
	if(goal == None):
		smt += '(= ' + getVar_at_depth('mode', m-1) +' '+ model.goals[0].mode+') '
		for condition in model.goals[0].condition:
			index = var_t_index(m-1)
			smt += '('+ condition.to_prefix(index) +') '
	else:
		smt += generateSMTfromSTL(goal, smtEncode)
	smt += '\n'
	#print(smt)
	return smt
	
def generateSMTfromSTL(goal, smtEncode):

	path = smtEncode.getPath()
	delta = smtEncode.getPrecision()
	depth = len(path)-1
	
	#smt ="(assert ( and ( = "+ getVar_at_depth('mode', depth) + " "+ path[depth-1]+")"
		
	smt2 = '(and '
	(smt1, time) = to_SMT(goal, smtEncode, 0, '0', 0, 0)
	smt2 += smt1+ '\n)'
	#c1 = ' ( >= '+ getVar_t_index('tm', depth)+' '+str(time)+') '
	#c2 = ' ( >= '+ getVar_at_depth('time', depth)+' '+str(time)+') '
	 
	smt ='(assert '
	smt +=  smt2 
	
	#c1 = ' ( >= '+ getVar_t_index('tm', depth)+' '+str(time)+') '
	#'\n\t(ite '+smt2 +' '+ c1 + ' false)'	
	smt += ')'
	#print('goal\n', smt)
	
	return smt
		
if __name__ == "__main__": 	
   main(sys.argv[1:])
