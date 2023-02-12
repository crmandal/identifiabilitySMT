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
from model.haModel import *

class PHA:
	def __init__(self, macro, param, var, states, init, goal):
		self.macros = macro
		self.parameters = param
		self.variables = var	
		self.states= states
		self.init = init
		self.goals = goal
		self.constraints = []
		
	def __str__(self):
		ha = ""
		for key in self.macros.keys():
			ha += "#define "+ key+ " "+ str(self.macros[key]) +"\n"
		ha += "\n"
		for var in self.parameters.keys():
			ha += str(self.parameters[var]) + " " + var + ";\n" 
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
		# sys.stdout.flush()
	
	def addGoal_Mode(self, mode, conds, goals = []):
		# goals = []
		cond = conds[0]
		goal = Goal(mode, cond)
		goals.append(goal)
		self.goals = goals

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

		params = self.parameters

		model = PHA(macro, params, variable, states, init, goals)
		return model #.convertHA2PHA(model, params)

	def updateVariable(self, var, value):
		#if(not self.getVariable(var) = None):
		self.variables.update({var:value})
	
	def updateParameter(self, var, value):
		#if(not self.getVariable(var) = None):
		self.parameters.update({var:value})
		
	def addConstraints(self, conditions):
		#for cond in conditions:
		self.constraints = conditions
	
	def getConstraints(self):
		return self.constraints

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
