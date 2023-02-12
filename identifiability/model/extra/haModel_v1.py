import getopt
from collections import OrderedDict
#from ha.util.exprEval import *

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.exprEval import *
from util.graph import *


class HA:
	def __init__(self, a, b, c, d, e):
		self.macros = a
		self.variables = b		
		self.states= c
		self.init = d
		self.goals = e
		
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
		ha += "\n"
		for goal in self.goals:
			ha+= "\n"+ str(goal)
			
		return ha

	def deleteGoal(self):
		self.goals = []

	def addGoal(self, mode, cond):
		goal = Goal(mode, cond)
		self.goals.append(goal)
		print('goal added: ', str(goal))
	
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
			macro.update({key : var})

		variable = OrderedDict()
		for var in self.variables.keys():
			val = self.variables[var]
			variable.update({var : val})	

		states = []
		for state in self.states:
			states.append(state.clone())
		init = self.init.clone()
		goals = []
		for goal in self.goals:
			goals.append(goal.clone())

		model = HA(macro, variable, states, init, goals)
		return model

class State:
	def __init__(self, a, b, c, d):
		self.mode = a
		self.invariants = b		
		self.flow = c
		self.jumps = d
		
	def __str__(self):
		state = "{ mode "+self.mode + ";"
		state+= "\ninvt:"
		for invt in self.invariants:
			state+= "\n\t"+ str(invt) + ";"
		state+= "\nflow:"
		for ode in self.flow:
			state+= "\n\t"+ str(ode)  + ";"	
		state+= "\njump:"
		for jump in self.jumps:
		#	print(str(jump))
			state+= "\n\t"+ str(jump) + ";"	
		state += "\n}"
		#	print(state)
		return state

	def clone(self):
		mode = self.mode
		invariants = []
		for invt in self.invariants:
			invariants.append(invt)

		odes = []
		for ode in self.flow:
			odes.append(ode.clone())

		jumps = []
		for jump in self.jumps:
			jumps.append(jump.clone())

		state = State(mode, invariants, odes, jumps)
		return state

class ODE :
	def __init__(self, a, b):
		self.var = a
		self.expr = b	
	
	def __str__(self):
		ode = "d/dt["+self.var+"] = "
		for e in self.expr:
			ode += str(e)
		#ode += ")"		
		return(ode)	
	
	def toPrefix(self):
		print('ode', str(self), self.expr)
		expr = infix2prefix(self.expr)
		ode = "= d/dt["+self.var+"] ("
		#for e in expr:
		ode += toString(expr)
		ode += ")"		
		return(ode)

	def clone(self):
		var = self.var
		expr = self.expr
		return ODE(var, expr)
							
class Jump:
	def __init__(self, a, b, c):
		self.guard = a
		self.toMode = b	
		self.reset = c
	def __str__(self):
		jump = "(and ";
		#print(self.guard)
		for guard in self.guard:
			jump += str(guard)
			#.replace("'", '\\\'')
		jump += ")"
		jump += " ==> @"+str(self.toMode)
		jump += "(and ";
		for reset in self.reset:
			jump += str(reset)
			#.replace("'", '\\\'')
		jump += ")"
		#print(jump)
		return jump

	def clone(self):
		guards = []
		for guard in self.guard:
			guards.append(guard.clone())
		mode = self.toMode
		resets = []
		for reset in self.reset:
			resets.append(reset)
		jump = Jump(guards, mode, resets)
		return jump

class Guard:
	def __init__(self, a, b):
		self.binop = a
		self.conditions = b	

	def __str__(self):
		guard = "("+ self.binop;
		for cond in self.conditions:
			guard += str(cond)
		guard += ")"
		return guard
	
	def toPrefix(self):
		guard = self.binop + " ";
		for cond in self.conditions:
			guard += cond.toPrefix()
		guard += " "
		return guard

	def clone(self):
		conditions = []
		for cond in self.conditions:
			conditions.append(cond.clone())
		binop = self.binop
		guard = Guard(binop, conditions)
		return guard

class Condition:
	def __init__(self, a, b, c):
		self.literal1 = a
		self.binop = b	
		self.literal2 = c

	def __str__(self):
		cond = "(" 
		for l in self.literal1:
			cond += l
		cond += " "+  self.binop 
		for l in self.literal2:
			cond += l
		cond += ")"
		return cond

	def toPrefix(self):
		cond = "(" 
		cond +=   self.binop + " "
		#for l in :
		cond += toString(infix2prefix(self.literal1))+ " "	
		#for l in infix2prefix(self.literal2):
		#	cond += l
		cond += toString(infix2prefix(self.literal2))	
		cond += ")"
		return cond
		
	def clone(self):
		l1 = self.literal1
		bop = self.binop
		l2 = self.literal2
		return Condition(l1, bop, l2)
		
class Reset:
	def __init__(self, a):
		self.var = None
		self.expr = a
		
	def __init__(self, a, b):
		self.var = a
		self.expr = b	

	def __str__(self):
		reset = ""
		if(self.var == None):
			for e in self.expr:
				reset += str(e)
		else:
			reset = "(" + self.var + "' = "
			#for e in self.expr:
			reset += toString(self.expr)
			reset += ")"
		return reset
	
	def toPrefix(self):
		expr = infix2prefix(self.expr)
		reset = ""
		if(self.var == None):			
			reset += toString(expr)
			#for e in expr:
			#	reset += str(e)
		else:
			reset = "( = " + self.var + "("
				
			reset += toString(expr)
			#for e in expr:
			#	reset += str(e)
			reset += "))"
		return reset

	def clone(self):
		l1 = self.var
		l2 = self.expr
		return Reset(l1, l2)

class Range:
	def __init__(self, a, b):
		self.left = a
		self.right = b

	def __str__(self):
		ran = "["
		for e in self.left:
			ran += str(e) + ' '
		ran += ","
		for e in self.right:
			ran += str(e)+ ' '
		ran += "]"
		return ran
	
	def getleftPre(self):
		left = toString(infix2prefix(self.left))
		return left
	
	def getrighttPre(self):
		right = toString(infix2prefix(self.right))
		return right
	
	def toPrefix(self):
		range1 = self.clone()
		left1 = range1.left
		right1 = range1.right
		left = "{:.6f}".format(postfixEval(infix2postfix(left1)))
		right = "{:.6f}".format(postfixEval(infix2postfix(right1)))
		ran = "["+ str(left)+ ","+ str(right)+ "]"
		#for e in self.left:
			#ran += str(e)+ ' '
		#ran += ","
		#for e in self.right:
			#ran += str(e)+ ' '
		#ran += "]"
		return ran

	def clone(self):
		l = self.left
		r = self.right
		return Range(l, r)


class Init:
	def __init__(self, a, b):
		self.mode = a
		self.condition = b

	def __str__(self):
		init = "init : @"+ self.mode+" (and ";
		for cond in self.condition:
			init += str(cond)
		init +=");"
		return init

	def clone(self):
		mode = self.mode
		condition = []
		for cond in self.condition:
			condition.append(cond.clone())
		return Init(mode, condition)
		
class Goal:
	def __init__(self, a, b):
		self.mode = a
		self.condition = b	

	def __str__(self):
		goal = "goal : @"+ self.mode+" (and ";
		for cond in self.condition:
			goal += str(cond)
		goal +=");"
		return goal

	def clone(self):
		mode = self.mode
		condition = []
		for cond in self.condition:
			condition.append(cond)
		return Goal(mode, condition)

