from __future__ import print_function
import os
import subprocess
import re
import sys, getopt

import collections
from collections import OrderedDict
from decimal import Decimal
from paramUtil.Point import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Variable:
	def __init__(self, name, initValue, value):
		self.name = name
		self.initValue = initValue # range variable
		self.endValue = value		# range variable
	
	def __str__(self):
		#mode_0 : [1, 2] = [1, 1]
		s = ''
		s += self.name + ' : '
		#if(self.initValue.getleft 
		s += str(self.initValue) + ' = '
		s += str(self.endValue)
		return s
		
class SATInstance:
	def __init__(self, variables = []):		
		self.variables = variables
		self.model = None
		self.valuemap = {}
		self.depth = 0
	
	def __str__(self):
		#mode_0 : [1, 2] = [1, 1]
		s = ''
		for var in self.variables:
			s += str(var) + '\n'
		return s
	
	def addDepth(self, i):
		self.depth = i
	
	def getDepth(self):
		return self.depth
		
	def addModel(self, model):
		self.model = model
		varLst = []
		#print('In addModel')
		oldvname = ''
		for var in self.variables:
			vname = self.getVarName(var)
			#print(vname,var)
			if (oldvname == vname):
				varLst.append(var)
			else:
				varLst = []
				varLst.append(var)
			self.valuemap.update({vname:varLst})	
			oldvname = vname
	
		# for var in self.valuemap:
		# 	print('\n'+var+' : ')
		# 	for v in self.valuemap[var]:
		# 		print(v)
	def getModel(self):
		return self.model
		
	def getSATPoint(self):
		i = self.depth
		values = {}
		for param in self.model.parameters:
			var = getVarInstance_t(self, param, i)[0]
			#print(param, var)			
			values.update({param:var.endValue.mid()})
		point = Point(values)
		# print('sat point : ', point)
		return point
		
	def getSATBox(self):
		i = self.depth
		# print()
		values = {}
		for param in self.model.parameters:
			var = getVarInstance_t(self, param, i)[0]
			#print(param, var)			
			values.update({param:PyInterval(var.endValue.leftVal(), var.endValue.rightVal())})
		box = Box(values)
		# print('sat box : ', box)
		return box
		
	def getVarName(self, var):
		model = self.model
		varName = var.name
		lst = varName.split('_')
		if(len(lst) > 2 and (lst[-1] == 't' or lst[-1] == '0')):
			vname = '_'.join(lst[:-2])
			if(vname in model.parameters or vname in model.variables):
				varName = vname	
		elif(len(lst) > 1 and  bool(re.match(r'[\d]+', lst[-1]))):
			vname = '_'.join(lst[:-1])
			if(vname in model.parameters or vname in model.variables):
				varName = vname	
			#variable.append(v)
		return varName
	
def getVarInstance(sat, varName,  depth = None):
	# print('getVarInstance', sat)
	variable = []
	if depth is None:
		depth = sat.getDepth()
	if varName == 'mode':
		varName = 'mode_'+str(depth)
	#else:	
	var = sat.valuemap[varName]
	#print('In getVarInstance', varName)
	for v in var:
		#print(v)
		lst = v.name.split('_')
		if(len(lst) > 2 and (lst[-1] == 't' or lst[-1] == '0')):
			if(lst[-2] == str(depth)):
				variable.append(v)
		elif(len(lst) > 1):
			if(lst[-1] == str(depth)):
				variable.append(v)
		else:
			variable.append(v)
	return variable
	
def getVarInstance_0(sat, varName,  depth = None):
	variable = []
	if depth is None:
		depth = sat.getDepth()
	var = getVarInstance(sat, varName, depth)
	#print('In getVarInstance_0')
	for v in var:
		#print(v)
		lst = v.name.split('_')
		if(len(lst) > 2 and lst[-1] == '0' and not lst[-1] == 't'):
			variable.append(v)
		if(len(lst) > 1 and len(lst) <= 2 and lst[-1] == str(depth)):
			variable.append(v)
	return variable
		
def getVarInstance_t(sat, varName, depth = None):
	variable = []
	if depth is None:
		depth = sat.getDepth()
	var = getVarInstance(sat, varName, depth)
	# print('getVarInstance_t', var)
	for v in var:
		lst = v.name.split('_')
		#print(lst)
		if(len(lst) > 2 and lst[-1] == 't' and not lst[-1] == '0'):
			variable.append(v)
		if (len(lst) > 0 and lst[0] == 'mode' and len(lst) == 2 and lst[-1] == str(depth)):
			variable.append(v)
			#print(str(v))
		elif(len(lst) > 1 and len(lst) <= 2 and lst[-1] == str(depth)):
			variable.append(v)
		

	return variable




