		
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

	def replace(self, macros):
		mode = self.mode
		invariants = []
		for invt in self.invariants:
			invariants.append(invt.replace(macros))

		odes = []
		for ode in self.flow:
			odes.append(ode.replace(macros))

		jumps = []
		for jump in self.jumps:
			jumps.append(jump.replace(macros))

		state = State(mode, invariants, odes, jumps)
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
		#for e in self.expr:
		ode += self.expr.to_infix()
		#ode += self.expr.evaluate().to_infix()
		#ode += ")"		
		return(ode)	
	
	def to_prefix(self, index = ""):
		#print('ode', str(self), self.expr)
		#expr = self.expr.evaluate().to_prefix(index)
		expr = self.expr.to_prefix(index)
		#infix2prefix(self.expr)
		#expr = self.expr
		ode = "( = d/dt["+self.var+"] ("
		#for e in expr:
		ode += expr
		ode += "))"		
		return(ode)

	def clone(self):
		var = self.var
		expr = self.expr
		return ODE(var, expr)

	def replace(self, macros):
		var = self.var
		expr = self.expr #.evaluate().to_infix()
		# print('ODE replace 1', str(expr))
		if not isinstance(expr, float) and not isinstance(expr, int):
			for key in reversed(macros.keys()):
				# if expr.find(str(key), 0) != -1 :
				#print('ODE replace 1.5', str(expr), key)
				expr = expr.replace(key, macros[key])
			for key in macros.keys():
				# if expr.find(str(key), 0) != -1 :
				#print('ODE replace 1.5', str(expr), key)
				expr = expr.replace(key, macros[key])

		expr1 = expr #.evaluate()			
		# print('ODE replace 2', str(expr))
		return ODE(var, expr1)
							
class Jump:
	def __init__(self, a, b, c):
		self.guard = a
		self.toMode = b	
		self.reset = c
	def __str__(self):
		jump = "(and ";
		#print(self.guard.toString())
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
			resets.append(reset.clone())
		jump = Jump(guards, mode, resets)
		return jump

	def replace(self, macros):
		guards = []
		for guard in self.guard:
			guards.append(guard.replace(macros))
		mode = self.toMode
		resets = []
		for reset in self.reset:
			resets.append(reset.replace(macros))
		jump = Jump(guards, mode, resets)
		
		return jump

#~ class Guard:
	#~ def __init__(self, a, b):
		#~ self.binop = a
		#~ self.conditions = b	

	#~ def __str__(self):
		#~ guard = "("+ self.binop;
		#~ for cond in self.conditions:
			#~ guard += str(cond)
		#~ guard += ")"
		#~ return guard
	
	#~ def to_prefix(self):
		#~ guard = "("+self.binop + " ";
		#~ for cond in self.conditions:
			#~ guard += cond.to_prefix()
		#~ guard += " ) "
		#~ return guard

	#~ def clone(self):
		#~ conditions = []
		#~ for cond in self.conditions:
			#~ conditions.append(cond.clone())
		#~ binop = self.binop
		#~ guard = Guard(binop, conditions)
		#~ return guard
		
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
			reset = self.expr.to_infix()
			#for e in self.expr:
			#	reset += str(e)
		else:
			reset = "(" + self.var + "' = "
			#for e in self.expr:
			reset += self.expr.to_infix() #toString(self.expr)
			reset += ")"
		return reset
	
	def to_prefix(self, vindex="", index=""):
		#expr = self.expr.evaluate().to_prefix(index)
		expr = self.expr.to_prefix(index)
		#infix2prefix(self.expr)
		reset = ""
		if(self.var == None):			
			reset += expr #toString(expr)
			#for e in expr:
			#	reset += str(e)
		else:
			reset = "( = " + self.var+ vindex + "("
				
			reset +=  expr #toString(expr)
			#for e in expr:
			#	reset += str(e)
			reset += "))"
		return reset

	def clone(self):
		l1 = self.var
		l2 = self.expr
		return Reset(l1, l2)

	def replace(self, macros):
		l1 = self.var
		l2 = self.expr #.evaluate().to_infix()
		for key in macros:
			# if l2.find(str(key), 0) != -1 :	
			l2 = l2.replace(key,macros[key])
		return Reset(l1, l2)

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

	def addCond(self, cond):
		self.condition.append(cond)

	def replace(self, macros):
		mode = self.mode
		condition = []
		for cond in self.condition:
			condition.append(cond.replace(macros))
		return Init(mode, condition)
		
class Goal:
	def __init__(self, a, b):
		self.mode = a
		self.condition = b	

	def __str__(self):
		goal = "@"+ self.mode+" (and ";
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

	def replace(self, macros):
		mode = self.mode
		condition = []
		for cond in self.condition:
			condition.append(cond.replace(macros))
		return Goal(mode, condition)

class Expression:
	def __init__(self , value):
		self.value = value
		self.left = None
		self.right = None
		self.dtype = 0
	
	def addleft(self, value):
		if type(value) is str:
			exp = Expression(value)
			self.left = exp
		elif type(value) is Expression:
			self.left = value
	
	def addright(self, value):
		if type(value) is str:
			exp = Expression(value)
			self.right = exp
		elif type(value) is Expression:
			self.right = value
	
	def infix(self, out):
		if self is not None:
			inorder(self.left)
			out += self.value,
			inorder(self.right)
			
	def prefix(self, out):
		if self is not None:
			out += self.value,
			inorder(self.left)
			inorder(self.right)

