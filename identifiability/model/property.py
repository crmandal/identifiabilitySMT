import sys, getopt

class Condition:
	def __init__(self, a, b, c):
		self.literal1 = a
		self.op = b	
		self.literal2 = c

class Properties:
	def __init__(self, a, b):
		self.min = a
		self.max = b
	def __str__(self):
		ps = "MIN : \n"
		for m1 in self.min:
			ps += str(m1)
		ps+= "\nMAX :\n"
		for m2 in self.max:
			ps += str(m2)	
		#print(ps)
		return ps

# class Goals:
# 	def __init__(self, a, b):
# 		self.op = a
# 		self.properties = b
# 	def __str__(self):
# 		goals = ""+str(self.op);
# 		for prop in self.properties:
# 			goals += str(prop) + " "		
# 		return goals

class Property:
	def __init__(self, a):
		self.goals = a	
	#	self.op = b
	def __str__(self):
		prop = ""
	#	prop = self.op +"(";
		for goal in self.goals:
			prop += str(goal)
			#print(prop)
	#	prop += ")"
		return prop

class Goal:
	def __init__(self, a, b):
		self.mode = a
		self.condition = b	
	def __str__(self):
		goal = "@"+ self.mode+"(and ";
		for cond in self.condition:
			goal += cond
		goal += ")"
		return goal
