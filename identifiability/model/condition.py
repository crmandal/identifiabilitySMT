class Condition:
	def __init__(self, a, b = None, c= None):
		self.literal1 = a
		self.binop = b	
		self.literal2 = c

	def __str__(self):
		cond = "(" 
		cond += self.literal1.evaluate().to_infix()
		if not self.binop == None: 
			cond += " "+  self.binop 
			cond += self.literal2.evaluate().to_infix()
			#for l in self.literal2:
			#	cond += l
		cond += ")"
		return cond
	
	def to_infix(self, index=""):
		cond = "(" 
		cond += self.literal1.evaluate().to_infix()
		if not self.binop == None: 
			cond += " "+  self.binop 
			cond += self.literal2.evaluate().to_infix()
			#for l in self.literal2:
			#	cond += l
		cond += ")"
		return cond

	def to_prefix(self, index = ""):
		cond = "(" 
		if not self.binop == None: 
			cond +=   self.binop + " "
			#for l in :
			cond += self.literal1.evaluate().to_prefix(index)+ " "	
			#toString(infix2prefix(self.literal1))+ " "	
			#for l in infix2prefix(self.literal2):
			#	cond += l
			cond += self.literal2.evaluate().to_prefix(index)
			#toString(infix2prefix(self.literal2))	
		else:
			cond += self.literal1.evaluate().to_prefix(index)
		cond += ")"
		return cond
		
	def clone(self):
		l1 = self.literal1.clone()
		bop = self.binop
		l2 = self.literal2.clone()
		return Condition(l1, bop, l2)

	def replace(self, macros):
		l1 = self.literal1 #.evaluate().to_infix()
		bop = self.binop
		l2 = self.literal2 #.evaluate().to_infix()

		if not isinstance(l1, float) and not isinstance(l1, int):
			for key in reversed(macros.keys()):
				# if l1.find(str(key), 0) != -1 :	
				l1 = l1.replace(key, macros[key])
			for key in macros.keys():
				# if l1.find(str(key), 0) != -1 :	
				l1 = l1.replace(key, macros[key])
		if not isinstance(l2, float) and not isinstance(l2, int):
			for key in reversed(macros.keys()):
				# if l2.find(str(key), 0) != -1 :
				l2 = l2.replace(key, macros[key])
			for key in macros.keys():
				# if l2.find(str(key), 0) != -1 :
				l2 = l2.replace(key, macros[key])

				# if str(key) == 'off_t4':
				# 	print('in replace', key, macros[key], l1.to_infix(), l2.to_infix())
		return Condition(l1, bop, l2)

