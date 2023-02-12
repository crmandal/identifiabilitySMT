class Range:
	def __init__(self, a, b):
		self.left = a
		self.right = b

	def __str__(self):
		ran = "["
		left = self.left.to_infix()
		right = self.right.to_infix()
		ran += left
		ran += ","
		ran += right
		#for e in self.left:
		#	ran += str(e) + ' '
		
		#for e in self.right:
		#	ran += str(e)+ ' '
		ran += "]"
		return ran
	
	def getleft(self):
		left = self.left.evaluate()
		#toString(infix2prefix(self.left))
		return left
	
	def getright(self):
		right = self.right.evaluate()
		#toString(infix2prefix(self.right))
		return right
		
	def leftVal(self):
		return float((self.getleft()).value);
		
	def rightVal(self):
		return float((self.getright()).value);
		
	def mid(self):
		return (self.leftVal()+ self.rightVal())/2.0
		
	def getleftPre(self):
		left = self.left.evaluate().to_prefix()
		#toString(infix2prefix(self.left))
		return left
	
	def getrighttPre(self):
		right = self.right.evaluate().to_prefix()
		#toString(infix2prefix(self.right))
		return right
	
	def to_prefix(self):
		range1 = self.clone()
		left1 = range1.left
		right1 = range1.right
		#print('rleft', left1.to_infix())
		left = range1.left.evaluate().to_prefix()
		#print('right', right1.to_infix())
		right = range1.right.evaluate().to_prefix()
		#float(postfixEval(infix2postfix(right1))))
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
