import re

class Node:	
	def __init__(self , value, operands=[]):
		val = value
		if isinstance(value, int) or isinstance(value, float):
			# print(value)
			val = "{}".format(value)
		self.value = val
		self.operands = operands
		self.ftype = 0
	
	def addOperand(self, ops):
		self.operands.append(ops)
		
	def setType(self, ftype):
		self.ftype = ftype
	
	def __str__(self):
		return self.to_infix("")
	
	def to_infix(self, index=""):
		if(self is None):
			return ""
		s = ""
		#checking whether n is an operation node
		if len(self.operands) > 1:
			s += "("
			for i in range(len(self.operands) - 1):
				s += self.operands[i].to_infix(index)
				s += self.value
				#checking if current node is a number
				if(not self.is_number() and self.is_terminal() and not self.isLit()):
					s += index
			s += self.operands[-1].to_infix(index) + ")"
		
		elif len(self.operands) == 1:
			if self.value == "-":
				s += "(" + self.value
				#checking if current node is a number
				if(not self.is_number() and self.is_terminal() and not self.isLit()):
					s += index
				s += self.operands[0].to_infix(index) + ")"
			else:
				s += self.value
				if(not self.is_number() and self.is_terminal() and not self.isLit()):
					s += index
				s += "(" + self.operands[0].to_infix(index) + ")"
		
		else:
			s += str(self.value)
			if(not self.is_number() and self.is_terminal() and not self.isLit()):
				s += index
		return str(s)
			
	def to_prefix(self, index=""):
		s = ""
		#checking whether n is an operation node
		if not self.is_terminal() :
			# print('not is_terminal', str(self.value))
			s += "(" + str(self.value);
			if(not self.is_number() and self.is_terminal() and not self.isLit()):
				s += index
			for n in self.operands:
				s += n.to_prefix(index)
			s += ")";	
		else:
			# print('is_terminal', str(self.value))
			s  += " " + str(self.value)
			if(not self.is_number() and self.is_terminal() and not self.isLit()):
				s += index
		# print('to_prefix:', str(self), str(s))
		return str(s)
	
	def replace(self, val1, val2):
		node = self.clone()
		if node.is_terminal():
			# if val1 == 'pi':
			# 	print('Node, replace -- terminal', type(val1), val1, type(self.value), self.value, val2.to_infix())
			if node.value == val1:
				# if val1 == 'pi':
				# 	print('node -- match', val1, node.value, val2.to_infix())
				#node = val #.clone()
				node = val2.clone()
				#return node
		elif not node.is_empty():
			ops = []
			for op in node.operands:
				op1 = op.replace(val1, val2)
				# if val1 == 'pi':
				# 	print('node -- adding operand', val1, node.value, val2.to_infix(), op1.to_infix(), op.to_infix())
				ops.append(op1)
			node.operands = ops
			# if val1 == 'pi':
			# 	print('node -- operand', node.to_infix(), node.value)
			#return node
		# if str(key) == 'off_t4':
		# 	print('in replace', key, val, node.to_infix())
		return node
				
		#return self.clone()
	
	def evaluate(self):
		val = self._eval()
		#print('in evaluate:', val.to_infix())
		if val.is_number():
			return Node(val.value)
		else:
			return val
			
	def _eval(self):		
		#print('in _eval:', self.value)
		if self.is_empty():
			return self
		elif self.is_number():
			return self
		elif self.isOperator(): #not self.is_terminal() and not self.isVariable() and not self.isLit():
			val = self.doMath()
			#print('calculated', val.to_infix())
			return val
		else:
			val = self.clone()
			operands = []
			for op in self.operands:
				operands.append(op._eval())
			val.operands = operands
			return val
			
	def doMath(self):
		val = 0
		op = self.value
		operands = self.operands		
		#print('in doMath:', op)
		if op == "+":
			op1 = operands[0]._eval()
			op2 = operands[1]._eval()
			if op1.is_number() and op2.is_number() :
				val = float(op1.value) + float(op2.value)
				return Node(str(val))
			#elif op1 
			else:
				operands1 = [op1, op2]
				#~ for op in self.operands:
					#~ operands.append(op._eval())
				return Node(self.value, operands1)
				#return self
		elif op == "-":
			op1 = operands[0]._eval()
			if op1.is_number():
				if len(operands) > 1:
					op2 = operands[1]._eval()
					if op2.is_number() :
						val = float(op1.value) - float(op2.value)
						return Node(str(val))
					else:
						operands1 = [op1, op2]
						#~ for op in self.operands:
						#~ operands.append(op._eval())
						return Node(self.value, operands1)
						#return self
				else:
					val = -1 * float(op1.value) 
					return Node(str(val))
			else:
				operands1 = []
				for op in self.operands:
					operands1.append(op._eval())
				return Node(self.value, operands1)
				#return self
		elif op == "*":
			op1 = operands[0]._eval()
			op2 = operands[1]._eval()
			if op1.is_number() and op2.is_number() :
				val = float(op1.value) * float(op2.value)
				return Node(str(val))
			else:
				operands1 = [op1, op2]
				#~ for op in self.operands:
					#~ operands.append(op._eval())
				return Node(self.value, operands1)
				
		elif op == "/": 
			op1 = operands[0]._eval()
			op2 = operands[1]._eval()
			if op1.is_number() and op2.is_number() :
				val = float(op1.value) / float(op2.value)
				return Node(str(val))
			else:
				operands1 = [op1, op2]
				#~ for op in self.operands:
					#~ operands.append(op._eval())
				return Node(self.value, operands1)
				#return self
		elif op == "^":
			op1 = operands[0]._eval()
			op2 = operands[1]._eval()
			if op1.is_number() and op2.is_number() :
				val = float(op1.value) ** float(op2.value)
				return Node(str(val))
			else:
				operands1 = [op1, op2]
				#~ for op in self.operands:
					#~ operands.append(op._eval())
				return Node(self.value, operands1)
				#return self
		else:
			operands1 = [op1, op2]
			#~ for op in self.operands:
			#~ operands.append(op._eval())
			return Node(self.value, operands1)
			#return self
	
	def negate(self):
		if self.is_terminal() :
			if(self.is_number() or (self.isVariable() and not self.isLit())):
				n = self.clone()
				return n
			elif(self.isLit()):
				n = self.clone()
				n.value = negateLit(self.value)
				return n
				
			ops = []
			for n in self.operands:
				s = n.negate()
				ops.append(s)
			return Node(self.value, ops)
		else:
			#print(','+self.value+',')
			if(self.value.startswith('X')):
				ops = []
				for n in self.operands:
					s = n.negate()
					ops.append(s)
				return Node(self.value, ops)
			elif(self.value.startswith('U')):
				v = self.value
				v = v.replace('U','R')
				ops = []
				for n in self.operands:
					s = n.negate()
					ops.append(s)
				node = Node(v, ops)
				#print('...'+self.value+'...'+v+ node.to_infix())
				return node
			elif(self.value.startswith('R')):
				v = self.value
				v = v.replace('R','U')
				ops = []
				for n in self.operands:
					s = n.negate()
					ops.append(s)
				node = Node(v, ops)
				#print('...'+self.value+'...'+v+ node.to_infix())
				return node
			elif(self.value.startswith('F')):
				v = self.value.replace('F','G')				
				ops = []
				for n in self.operands:
					s = n.negate()
					ops.append(s)
				return Node(v, ops)
			elif(self.value.startswith('G')):
				v = self.value.replace('G','F')				
				ops = []
				for n in self.operands:
					s = n.negate()
					ops.append(s)
				return Node(v, ops)
			elif(self.value  == '&'):
				v = '|'			
				ops = []
				for n in self.operands:
					s = n.negate()
					ops.append(s)
				return Node(v, ops)
			elif(self.value  == '|'):
				v = '&'			
				ops = []
				for n in self.operands:
					s = n.negate()
					ops.append(s)
				return Node(v, ops)
			elif(self.value  == '->'):
				v = '&'			
				n1 = self.operands[0].clone()
				n2 = self.operands[1].negate()
				ops = [n1, n2]
				return Node(v, ops)
			elif(self.value  == '!'):
				n= self.operands[0].clone()
				return n
			elif(self.value  == '='):
				v = '!'			
				node = self.clone()
				return Node(v,[node])
			elif(self.value  == '!='):
				v = '='			
				ops = []
				for n in self.operands:
					ops.append(n.clone())
				return Node(v,ops)
			elif(self.value  == '<'):
				v = '>='			
				ops = []
				for n in self.operands:
					ops.append(n.clone())
				return Node(v, ops)
			elif(self.value  == '>'):
				v = '<='			
				ops = []
				for n in self.operands:
					ops.append(n.clone())
				return Node(v, ops)
			elif(self.value  == '<='):
				v = '>'			
				ops = []
				for n in self.operands:
					ops.append(n.clone())
				return Node(v, ops)
			elif(self.value  == '>='):				
				v = '<'			
				ops = []
				for n in self.operands:
					ops.append(n.clone())
				node = Node(v, ops)
				#print('...'+self.value+'...'+v+ node.to_infix())
				return node
			else: 
				return self.clone()
	
	def removeEqual(self):
		if self.is_terminal() :
			if(self.is_number() or (self.isVariable() and not self.isLit())):
				n = self.clone()
				return n
			elif(self.isLit()):
				n = self.clone()
				n.value = self.value
				return n
				
			ops = []
			for n in self.operands:
				s = n.removeEqual()
				ops.append(s)
			return Node(self.value, ops)
		else:
			#print(','+self.value+',')
			if(self.value  == '='):
				v = '&'			
				ops = []
				for n in self.operands:
					s = n.clone()
					ops.append(s)
				node1 = Node('>=', ops).removeEqual()
				node2 = Node('<=', ops).removeEqual()
				return Node(v,[node1, node2])
			elif(self.value  == '!='):
				v = '|'			
				ops = []
				for n in self.operands:
					s = n.clone()
					ops.append(s)
				node1 = Node('>', ops).removeEqual()
				node2 = Node('<', ops).removeEqual()
				return Node(v,[node1, node2])
			else: 
				v = self.value		
				ops = []
				for n in self.operands:
					ops.append(n.removeEqual())
				node = Node(v, ops)
				#print('...'+self.value+'...'+v+ node.to_infix())
				return node
	
	def negateMin(self):
		if self.is_terminal() :
			if(self.is_number() or (self.isVariable() and not self.isLit())):
				n = self.clone()
				return n
			elif(self.isLit()):
				n = self.clone()
				n.value = negateLit(self.value)
				return n
				
			ops = []
			for n in self.operands:
				s = n.negateMin()
				ops.append(s)
			return Node(self.value, ops)
		else:
			#print(','+self.value+',')
			if(self.value.startswith('X')):
				ops = []
				for n in self.operands:
					s = n.negateMin()
					ops.append(s)
				return Node(self.value, ops)
				
			elif(self.value.startswith('U')):
				v = '!'
				ops = []
				op = self.clone()
				ops.append(op)
				node = Node(v, ops)
				#print('...'+self.value+'...'+v+ node.to_infix())
				return node
			elif(self.value.startswith('R')):
				v = self.value
				v = v.replace('R','U')
				ops = []
				for n in self.operands:
					s = n.negateMin()
					ops.append(s)
				node = Node(v, ops)
				#print('...'+self.value+'...'+v+ node.to_infix())
				return node
			elif(self.value.startswith('F')):
				v = '!'
				ops = []
				op = self.clone()
				ops.append(op)
				node = Node(v, ops)
				#print('...'+self.value+'...'+v+ node.to_infix())
				return node
			elif(self.value.startswith('G')):
				v = self.value.replace('G','F')				
				ops = []
				for n in self.operands:
					s = n.negateMin()
					ops.append(s)
				return Node(v, ops)
			elif(self.value  == '&'):
				v = '|'			
				ops = []
				for n in self.operands:
					s = n.negateMin()
					ops.append(s)
				return Node(v, ops)
			elif(self.value  == '|'):
				v = '&'			
				ops = []
				for n in self.operands:
					s = n.negateMin()
					ops.append(s)
				return Node(v, ops)
			elif(self.value  == '->'):
				v = '&'			
				n1 = self.operands[0].clone()
				n2 = self.operands[1].negateMin()
				ops = [n1, n2]
				return Node(v, ops)
			elif(self.value  == '!'):
				n= self.operands[0].clone()
				return n
			elif(self.value  == '='):
				v = '!'			
				node = self.clone()
				return Node(v,[node])
			elif(self.value  == '!='):
				v = '='			
				ops = []
				for n in self.operands:
					ops.append(n.clone())
				return Node(v, ops)
			elif(self.value  == '<'):
				v = '>='			
				ops = []
				for n in self.operands:
					ops.append(n.clone())
				return Node(v, ops)
			elif(self.value  == '>'):
				v = '<='			
				ops = []
				for n in self.operands:
					ops.append(n.clone())
				return Node(v, ops)
			elif(self.value  == '<='):
				v = '>'			
				ops = []
				for n in self.operands:
					ops.append(n.clone())
				return Node(v, ops)
			elif(self.value  == '>='):				
				v = '<'			
				ops = []
				for n in self.operands:
					ops.append(n.clone())
				node = Node(v, ops)
				#print('...'+self.value+'...'+v+ node.to_infix())
				return node
			else: 
				return self.clone()
	
	def to_cnf(self):
		if self.is_terminal() :
			if(self.is_number() or (self.isVariable() and not self.isLit())):
				n = self.clone()
				return n
			elif(self.isLit()):
				n = self.clone()
				return n		
			else: 
				n = self.clone()
				return n
				#ops = []
				#for n in self.operands:
				#	s = n.clone()
				#	ops.append(s)
				#return Node(self.value, ops)		
		else:
			#print(','+self.value+',')
			if(self.value  == '|'):
				v = '!'			
				ops = []
				for n in self.operands:
					s = n.negate().to_cnf()
					ops.append(s)
				node = Node('&', ops)
				return Node(v,[node])
				
			elif(self.value  == '->'):
				v = '|'						
				n1 = self.operands[0].negate()
				n2 = self.operands[1].clone()
				ops = [n1, n2]
				node = Node(v, ops)
				return node.to_cnf()
			else: 
				return self.clone()
	
	def delta_perturb(self, delta):
		if self.is_terminal() :
			if(self.is_number() or (self.isVariable() and not self.isLit())):
				n = self.value
				v = '-'
				node1 = Node(n)
				node2 = Node(str(delta))
				n11=  Node(v, [node1, node2])
				#print(n11.to_infix())
				return n11
			elif(self.isLit()):
				n = self.clone()
				return n	
			else: 
				ops = []
				for n in self.operands:
					s = n.delta_perturb(delta)
					ops.append(n)
				return Node(self.value, ops)			
		else:
		#	print(self.to_infix())
			if(self.value  == '='):
				v = '&'			
				ops = []
				for n in self.operands:
					s = n.clone()
					ops.append(s)
				node1 = Node('>=', ops).delta_perturb(delta)
				node2 = Node('<=', ops).delta_perturb(delta)
				return Node(v,[node1, node2])
			elif(self.value  == '!='):
				v = '|'			
				ops = []
				for n in self.operands:
					s = n.clone()
					ops.append(s)
				node1 = Node('>', ops).delta_perturb(delta)
				node2 = Node('<', ops).delta_perturb(delta)
				return Node(v,[node1, node2])					
			elif(self.value  == '>='):
				v = '>='						
				n1 = self.operands[0].clone()
				n2 = self.operands[1].delta_perturb(delta)
				ops = [n1, n2]
				node = Node(v, ops)
				return node
			elif(self.value  == '>'):
				v = '>'						
				n1 = self.operands[0].clone()
				n2 = self.operands[1].delta_perturb(delta)
				ops = [n1, n2]
				node = Node(v, ops)
				return node
			elif(self.value  == '<='):
				v = '<='						
				n1 = self.operands[0].delta_perturb(delta)
				n2 = self.operands[1].clone()
				ops = [n1, n2]
				node = Node(v, ops)
				return node
			elif(self.value  == '<'):
				v = '<'						
				n1 = self.operands[0].delta_perturb(delta)
				n2 = self.operands[1].clone()
				ops = [n1, n2]
				node = Node(v, ops)
				return node
			else: 
				ops = []
				for n in self.operands:
					s = n.delta_perturb(delta)
					ops.append(s)
				return Node(self.value, ops)
	
	def isOperator(self):
		var = self.value
		if var in '+-*/^':
			return True
		else:
			return False			
	def is_terminal(self):
		return len(self.operands) == 0
	
	def is_number(self):
		var = self.value
		m1 = False
		try:
			float(var)
			m1 = True
		except ValueError:
			m1 = False
		return m1
			
	def isVariable(self):
		var = self.value
		m1 = bool(re.match(r'[a-zA-Z_][a-zA-Z_\d]*', var))
		#return m1
		m2 = (var == 'G') or (var == 'F') or (var == 'U') or (var == 'X')
		return (m1 and not m2)
	
	def isLit(self):
		var = self.value
		return var == 'true' or var == 'false' or var == '(true)' or var == '(false)'

	def is_empty(self):
		return len(self.operands) == 0 and self.value == ""
		
	def clone(self):
		val = self.value
		operands = []
		for op in self.operands:
			op1 = op.clone()
			operands.append(op1)
		return Node(val, operands)
		
	def negateLit(lit):
		if var == 'true': 
			return 'false'
		if var == 'false':
			return 'true' 
		if var == '(true)': 
			return '(false)'
		if var == '(false)':
			return '(true)' 


