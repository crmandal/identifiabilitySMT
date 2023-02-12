import getopt
from collections import OrderedDict
#from ha.util.exprEval import *

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from util.exprEval import *
from util.graph import *
from model.range import *
from model.condition import *
from model.node import *
from model.state import *

class HA:
	def __init__(self, a, b, c, d, e):
		self.macros = a
		self.variables = b		
		self.states= c
		self.init = d
		self.goals = e
		self.constraints = []
		
	def __str__(self):
		ha = ""
		for key in self.macros.keys():
			if(key is None):
				continue
			ha += "#define "+ key+ " "+ str(self.macros[key]) +"\n"
		ha += "\n"
		for var in self.variables.keys():
			ha += str(self.variables[var]) + " " + var + ";\n"
		ha += "\n"			
		for state in self.states:
			ha+= "\n"+ str(state)
		ha+= "\n"+ str(self.init)
		ha += "\n goal:\n"
		for goal in self.goals:
			ha+= "\n"+ str(goal)
		ha+= '\n'
		if(len(self.constraints) > 0):
			ha+= '\nConstraints: \n'
			for const in self.constraints:
				ha += str(const) +  ';\n'
		return ha

	def deleteGoal(self):
		self.goals = []

	def addGoal(self, mode, cond):
		goal = Goal(mode, cond)
		self.goals = [goal]
		# self.goals.append(goal)
		# print('goal added: ', str(goal), str(self.goals))

	def addGoals(self, mode, conds, goals = []):
		# goals = []
		for cond in conds:
			goal = Goal(mode, cond)
			goals.append(goal)
		self.goals = goals
		# self.goals.append(goal)
		# print('--------- goal added: ', str(goal), str(self.goals))
		sys.stdout.flush()

	def addInitMode(self, mode = -1):
		if mode >= 0:
			self.init.mode = str(mode)

	def addInit(self, var, val):
		n1 = Node(var)
		if (isinstance(val, int) or isinstance(val, float)):
			n4 = Node(str(val))
		else:
			n2 = Node('0.5')
			n3 = Node('+', [val.getleft(), val.getright()])
			n4 = Node('*', [n2, n3])
		# print(n1, n2, n3, n4)
		# node = Node('=', [n1, n3])
		cond = Condition(n1, '=', n4)
		self.init.addCond(cond)
		# print('Init added: ', str(cond), str(self.init))

	def getGraph(self):
		num =  len(self.states)
		g = Graph(num+1)
		for state in self.states:
			src = state.mode
			for jump in state.jumps:
				tgt = jump.toMode
				#print(src, tgt)
				g = g.addEdge(src, tgt)
		return g
	
	def saveModel(self, outfile):	
		with open(outfile, 'w') as of:
			of.write(str(self))
		
	def clone(self):
		macro = OrderedDict()
		for key in self.macros.keys():			
			if(key is None):
				continue
			var = self.macros[key]
			macro.update({key : var.clone()})

		variable = OrderedDict()
		for var in self.variables.keys():
			val = self.variables[var]
			variable.update({var : val.clone()})	

		states = []
		for state in self.states:
			states.append(state.clone())
		init = self.init.clone()
		goals = []
		for goal in self.goals:
			goals.append(goal.clone())

		model = HA(macro, variable, states, init, goals)
		return model

	def updateVariable(self, var, value):
		#if(not self.getVariable(var) = None):
		self.variables.update({var:value})
		
	def addConstraints(self, conditions):
		#for cond in conditions:
		self.constraints = conditions
	
	def getConstraints(self):
		return self.constraints

	def getMacroValues(self, names):
		macro = OrderedDict()
		for key in self.macros: #reversed(self.macros):			
			if(key is None):
				continue
			var = self.macros[key].clone()
			macro.update({key : var})
			#print('haModel -- getMacroValues', key, var.to_infix(), type(var))
		macros_updated = OrderedDict()
		for key in (macro.keys()):
			var = macro[key]
			for key1 in macro.keys():	
				#if key1 == 'pi':
				var = var.replace(key1, macro[key1])
				# print('-- replaced', key, var.to_infix())
				# if not isinstance(var, float) and not isinstance(var, int):				
				# 	# if var.find(str(key), 0) != -1 :
				# 	for key1 in macro:	
				# 		var = var.replace(key, macro[key1])
				# 		# print('1 update', key, var)
				# 	macros_updated.update({key : var})
				# else:
				# 	# print('2 update', key, var)
				macro.update({key : var})	
			macros_updated.update({key : macro[key]})	
			#print('1, haModel -- getMacroValues', key, macros_updated[key].to_infix())

		# for key in macro.keys():
		# 	var = macro[key]
		# 	for key1 in macro.keys():	
		# 			var = var.replace(key1, macro[key1])
		# 	# if not isinstance(var, float) and not isinstance(var, int):				
		# 	# 	# if var.find(str(key), 0) != -1 :
		# 	# 	for key1 in macro:	
		# 	# 		var = var.replace(key, macro[key1])
		# 	# 		# print('1 update', key, var)
		# 	# 	macros_updated.update({key : var})
		# 	# else:
		# 	# 	# print('2 update', key, var)
		# 	macros_updated.update({key : var})	
			#print('2, haModel -- getMacroValues', key, var.to_infix())


		# print('getMacroValues', macros_updated)
		ret_vals = []
		for nm in names:
			#print('getMacroValues', nm, macros_updated[nm])
			ret_vals.append(macros_updated[nm].evaluate().to_infix())
		print('getMacroValues', ret_vals)
		return ret_vals

	def simplify(self, ONLY = True, skip= [], nrplc = []):
		'''macro = OrderedDict()
		for key in reversed(self.macros.keys()):			
			if(key is None):
				continue
			var = self.macros[key]
			macro.update({key : var})

		macros_updated = OrderedDict()
		for key in self.macros.keys():	
			var = macro[key]
			if not isinstance(var, float) and not isinstance(var, int):				
				# if var.find(str(key), 0) != -1 :
				for key1 in macro:	
					var = var.replace(key1, macro[key1])
				macros_updated.update({key : var})
			else:
				macros_updated.update({key : var})'''

		macro = OrderedDict()
		for key in self.macros: #reversed(self.macros):			
			if(key is None ):
				continue
			if(key in skip or key in nrplc):
				var = self.macros[key].clone()
			else:				
				var = self.macros[key].clone().evaluate()
			macro.update({key : var})
		macros_updated = OrderedDict()
		for key in (macro.keys()):
			var = macro[key]
			for key1 in macro.keys():	
				if(key1 not in nrplc): # not replacing some macros
					var = var.replace(key1, macro[key1])
				macro.update({key : var})
			if(key in skip):
				macros_updated.update({key : macro[key]})
			else:	
				macros_updated.update({key : macro[key].evaluate()})
		
		variable = OrderedDict()
		for var in self.variables.keys():
			val = self.variables[var]
			# print('in simplify', val)
			l = val.left #.to_infix()
			r = val.right #.to_infix()
			# print('in simplify', l, r)
			# sys.stdout.flush()
			# if not isinstance(expr, float) and not isinstance(expr, int):
			for key in macro:
				# if l.find(str(key), 0) != -1 :
				l = l.replace(key, macro[key])
				#if r.find(str(key), 0) != -1 :
				r = r.replace(key, macro[key])
			rng = Range(l, r)
			variable.update({var : rng})	
		# print('In simplify: ', macros_updated)		

		if ONLY:
			# variable = self.variables
			states = self.states
			init = self.init
			goals = self.goals
		else:
			states = []
			for state in self.states:
				# print('States 1', str(state))
				state1 = state.replace(macros_updated)
				# print('States 2', str(state1))
				states.append(state1)

			init = self.init.replace(macros_updated)
			goals = []
			for goal in self.goals:
				goals.append(goal.replace(macros_updated))

		model = HA(macros_updated, variable, states, init, goals)
		return model

	def updateMacros(self, var, value):
		if var in self.macros.keys():			
			self.macros.update({var : value})
			
	# def updateInits(self, var, value):

def main(argv):
	# ((+(- (*(*(* 2.0 1) trh) trhr))(* 2 1))))
	n0 = Node('0')
	n1 = Node('2.0')
	n2 = Node('1')
	n3 = Node('trh')
	n4 = Node('trhr')
	n5 = Node('2')
	n6 = Node('1')
	
	n12 = Node('*', [n1,n2])
	n123 = Node('*', [n12, n3])
	n1234 = Node('*', [n123, n4])
	n15 = Node('-', [n1234])
	n56 = Node('*', [n5,n6])
	n16 = Node('+', [n15, n56])
	
	print(n16.to_prefix()+'\n'+ n16.to_infix())
	
	print(n16.evaluate().to_prefix())
	
	n1 = Node('2.0')
	n2 = Node('1')
	n3 = Node('trh')
	n12 = Node('+', [n1,n2])
	n123 = Node('*', [n12, n3])
	print(n123.evaluate().to_prefix())
	
if __name__ == "__main__": 	
   main(sys.argv[1:])	
