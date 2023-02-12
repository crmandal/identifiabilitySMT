import sys
from pythonds.basic.stack import Stack
#from stack import *
import ast

import re

def main(argv):
	#expr = []
	#['(', '(', '(', '0.02', '*', '(', '1', ')', ')', '*', '(', '2', '*', '3.1412', '/', '(', '(', '2', '*', '3.1412', '/', '5', ')', '*', '24', ')', ')', '/', '2', ')', '*', 'cos(', '(', '2', '*', '3.1412', '/', '(', '(', '2', '*', '3.1412', '/', '5', ')', '*', '24', ')', ')', '*', 'tm', ')', '*', '(', '1', '-', '0.999', '*', 'exp(', '0', '-', '5', '*', 'exp(', '0', '-', '0.05', '*', 'tm', ')', ')', ')', ')', '+', '(', '(', '(', '0.02', '*', '(', '1', ')', ')', '+', '(', '0.02', '*', '(', '1', ')', ')', ')', '+', '(', '0.02', '*', '(', '1', ')', ')', '*', 'sin(', '(', '2', '*', '3.1412', '/', '(', '(', '2', '*', '3.1412', '/', '5', ')', '*', '24', ')', ')', '*', 'tm', ')', ')', '*', '(', '0', '-', '0.999', '*', '5', '*', '0.05', '*', 'exp(', '0', '-', '5', '*', 'exp(', '0', '-', '0.05', '*', 'tm', ')', ')', '*', 'exp(', '0', '-', '0.05', '*', 'tm', ')', ')']
	#exp = toString(expr)
	exp = '-0.0008636867115791952*(-81.96110535309376*(-1.4605211259734208*cos(-0.4167187504761705*1.00100  + 12.56637061435917)*exp(0.01641490513179676*1.00100 ) - 0.05753116628048063*exp(0.01641490513179676*1.00100 )*sin(-0.4167187504761705*1.00100  + 12.56637061435917) + 2.395982335265008)*( 0.8 * 1.0000000000 )  - 0.05035382217550032*(2380.9844232731057*( 0.8 * 1.0000000000 ) *4.00000 *1.00100 ^2*exp(0.01641490513179676*1.00100 ) - 44985.881405403336*( 0.8 * 1.0000000000 ) )*cos(-0.4167187504761705*1.00100  + 12.56637061435917)/(4.00000 *1.00100 ^2) + 0.05035382217550032*(2380.98442327311*( 0.8 * 1.0000000000 ) *4.00000 *1.00100 ^2*exp(0.01641490513179676*1.00100 ) + 3906.00075360178*( 0.8 * 1.0000000000 ) *1.00100 ^2 + (9.18311703497797e-12)*( 0.8 * 1.0000000000 ) *1.00100  - 44985.8814054033*( 0.8 * 1.0000000000 ) )/(4.00000 *1.00100 ^2))*cos(0.4167187504761705*tm) - 0.003454746846316781*(-10.24513816913672*(-2.9075289145283327*cos(-0.20835937523808526*1.00100  + 6.283185307179585)*exp(0.01641490513179676*1.00100 ) - 0.22906006147024952*exp(0.01641490513179676*1.00100 )*sin(-0.20835937523808526*1.00100  + 6.283185307179585) + 4.769796064291168)*( 0.8 * 1.0000000000 )  - 0.05035382217550032*(595.2461058182764*( 0.8 * 1.0000000000 ) *4.00000 *1.00100 ^2*exp(0.01641490513179676*1.00100 ) - 44985.881405403336*( 0.8 * 1.0000000000 ) )*cos(-0.20835937523808526*1.00100  + 6.283185307179585)/(4.00000 *1.00100 ^2) + 0.05035382217550032*(595.246105818276*( 0.8 * 1.0000000000 ) *4.00000 *1.00100 ^2*exp(0.01641490513179676*1.00100 ) + 976.500188400445*( 0.8 * 1.0000000000 ) *1.00100 ^2 + (2.29577925874449e-12)*( 0.8 * 1.0000000000 ) *1.00100  - 44985.8814054033*( 0.8 * 1.0000000000 ) )/(4.00000 *1.00100 ^2))*cos(0.20835937523808526*tm) + 0.0008636867115791952*(-81.96110535309376*(-0.05753116628048063*cos(-0.4167187504761705*1.00100  + 12.56637061435917)*exp(0.01641490513179676*1.00100 ) + 1.4605211259734208*exp(0.01641490513179676*1.00100 )*sin(-0.4167187504761705*1.00100  + 12.56637061435917) + 0.09437977697402647)*( 0.8 * 1.0000000000 )  + 0.05035382217550032*(2380.9844232731057*( 0.8 * 1.0000000000 ) *4.00000 *1.00100 ^2*exp(0.01641490513179676*1.00100 ) - 44985.881405403336*( 0.8 * 1.0000000000 ) )*sin(-0.4167187504761705*1.00100  + 12.56637061435917)/(4.00000 *1.00100 ^2) - 0.05035382217550032*(-(1.16634598111245e-12)*( 0.8 * 1.0000000000 ) *4.00000 *1.00100 ^2*exp(0.01641490513179676*1.00100 ) - (1.91338852814623e-12)*( 0.8 * 1.0000000000 ) *1.00100 ^2 + 18746.4602883289*( 0.8 * 1.0000000000 ) *1.00100  + (2.20367262679799e-11)*( 0.8 * 1.0000000000 ) )/(4.00000 *1.00100 ^2))*sin(0.4167187504761705*tm) + 0.003454746846316781*(-10.24513816913672*(-0.22906006147024952*cos(-0.20835937523808526*1.00100  + 6.283185307179585)*exp(0.01641490513179676*1.00100 ) + 2.9075289145283327*exp(0.01641490513179676*1.00100 )*sin(-0.20835937523808526*1.00100  + 6.283185307179585) + 0.37577262748024254)*( 0.8 * 1.0000000000 )  + 0.05035382217550032*(595.2461058182764*( 0.8 * 1.0000000000 ) *4.00000 *1.00100 ^2*exp(0.01641490513179676*1.00100 ) - 44985.881405403336*( 0.8 * 1.0000000000 ) )*sin(-0.20835937523808526*1.00100  + 6.283185307179585)/(4.00000 *1.00100 ^2) - 0.05035382217550032*(-(1.45793247639056e-13)*( 0.8 * 1.0000000000 ) *4.00000 *1.00100 ^2*exp(0.01641490513179676*1.00100 ) - (2.39173566018279e-13)*( 0.8 * 1.0000000000 ) *1.00100 ^2 + 9373.23014416444*( 0.8 * 1.0000000000 ) *1.00100  + (1.10183631339899e-11)*( 0.8 * 1.0000000000 ) )/(4.00000 *1.00100 ^2))*sin(0.20835937523808526*tm)'.replace('^', '**')
	#'((( 0.02 * ( 1 ) ) *( 2 * 3.1412 / ( ( 2 * 3.1412 / 5 ) * 24 ) ) /2)*cos(( 2 * 3.1412 / ( ( 2 * 3.1412 / 5 ) * 24 ) ) *tm)*(1-0.999 *exp(0-5 *exp(0-0.05 *tm))))+((( 0.02 * ( 1 ) ) +( 0.02 * ( 1 ) ) )+( 0.02 * ( 1 ) ) *sin(( 2 * 3.1412 / ( ( 2 * 3.1412 / 5 ) * 24 ) ) *tm))*(0-0.999 *5 *0.05 *exp(0-5 *exp(0-0.05 *tm))*exp(0-0.05 *tm))'
	#'cos ( 0.02 * tm + 0.5 * x - 0.33 * 0.99)'
	#exp1 = ast.literal_eval(exp)
	#~ expr = exp.split(' ')
	#~ print(exp, len(expr)) #, exp1)
	
	
	#~ exp1 = eval_expr(exp)
	#~ print(exp1)
	#~ node = ast.parse(exp1)
	#~ #print(str(node))
	
	#~ visitor = v()
	#~ visitor.visit(node)
	#print(visitor.tokens)
	tokens = infix2prefix(exp)
	expr1 = toString(tokens)
	print(expr1)
	#expr1 = infix2prefix(exp)
	

def eval_expr(expr):
	""" Eval and expression inside a #define using a suppart of python grammar """
	return _eval(ast.parse(expr, mode='eval').body)

def _eval(node):
		if isinstance(node, ast.Num):
			val = str(node.n)
			#print(val)
			return val
		elif isinstance(node, ast.BinOp):
			val = doMath(node.op, _eval(node.left), _eval(node.right))
			#print(val)
			return val
			#return OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
		elif isinstance(node, ast.UnaryOp):
			return doMath(node.op, '0.0', _eval(node.operand))
			#return OPERATORS[type(node.op)](_eval(node.operand))
		#~ elif isinstance(node, ast.BoolOp) and isinstance(node, ast.BitXor):			
			#~ values = [_eval(x) for x in node.values]			
			#~ val = doMath('^', values[0], values[1])
			#~ return val
#			return OPERATORS[type(node.op)](**values)
		elif isinstance(node, ast.Call):
			val = str(node.func.id) + '('
			for arg in node.args:
				val +=  _eval(arg)
			val +=')'
			#print(val)
			return val
		elif isinstance(node, str):
			val = node
			#print(val)
			return val
		elif isinstance(node, ast.Name):
			val = str(node.id)
			#print(val)
			return val
		#elif isinstance(node, ast.NameConstant):
		#	val = str(node.value)
			#print(val)
			return val
		else:
			#print(str(node))
			raise TypeError(node)
			
class v(ast.NodeVisitor):

	def __init__(self):
		self.tokens = []

	def f_continue(self, node):
		super(v, self).generic_visit(node)

	def visit_Add(self, node):
		#self.tokens.append("(")
		self.tokens.append('+')
		self.f_continue(node)	
		#self.tokens.append(")")
	
	def visit_Sub(self, node):
		#self.tokens.append("(")
		self.tokens.append('-')
		self.f_continue(node)		
		#self.tokens.append(")")

	def visit_And(self, node):
		#self.tokens.append("(")
		self.tokens.append('&&')
		self.f_continue(node)	
		#self.tokens.append(")")

	def visit_BinOp(self, node):
		#if isinstance(node.left, ast.Num) and isinstance(node.right, ast.Num):
		#	val = doMath(node.op, node.left.n, node.right.n)
		#	self.tokens.append(val)
		#else:
			self.tokens.append("(")
			self.visit(node.op)
			self.visit(node.left)
			self.visit(node.right)
			self.tokens.append(")")

	def visit_BoolOp(self, node):
		self.tokens.append("(")
		self.visit(node.op)
		for val in node.values:
			self.visit(val)
		self.tokens.append(")")

	def visit_Call(self, node):		
		self.tokens.append("(")
		self.visit(node.func)
		#self.tokens.append(node.func.id)
		self.tokens.append("(")
		for arg in node.args:
			self.visit(arg)
		self.tokens.append(")")		
		self.tokens.append(")")

	def visit_Div(self, node):
		#self.tokens.append("(")
		self.tokens.append('/')
		self.f_continue(node)
		#self.tokens.append(")")

	def visit_Expr(self, node):
		# print('visit_Expr')
		self.f_continue(node)

	def visit_Import(self, stmt_import):
		for alias in stmt_import.names:
			print('import name "%s"' % alias.name)
			print('import object %s' % alias)
		self.f_continue(stmt_import)

	def visit_Load(self, node):
		# print('visit_Load')
		self.f_continue(node)

	def visit_Module(self, node):
		# print('visit_Module')
		self.f_continue(node)

	def visit_Mult(self, node):
		#self.tokens.append("(")
		self.tokens.append('*')
		self.f_continue(node)
		#self.tokens.append(")")

	def visit_Name(self, node):
		self.tokens.append(node.id)
		self.f_continue(node)

	def visit_NameConstant(self, node):
		self.tokens.append(node.value)
		self.f_continue(node)

	def visit_Num(self, node):
		self.tokens.append(node.n)
		self.f_continue(node)

	def visit_Pow(self, node):
		#self.tokens.append("(")
		self.tokens.append('^')
		self.f_continue(node)
		#self.tokens.append(")")
	
def infix2prefix(expr):
	if not isinstance(expr, str):
		exp = toString(expr).replace('^', '**')
	else:
		exp = expr
	print(exp, len(expr))
	exp1 = eval_expr(exp)
	print(exp1)
	node = ast.parse(exp)
	#node = ast.parse(exp)
	#print(node)
	
	visitor = v()
	visitor.visit(node)
	return visitor.tokens
	
	#print(visitor.tokens)
	#expr1 = toString(visitor.tokens)
	#print(expr1)
	
def toString(path):
	pathstr = ''
	for p in path:
		pathstr += str(p)+ ' '
	return pathstr

def infix2postfix(infixexpr):
	prec = {}
	prec["^"] = 4
	prec["*"] = 3
	prec["/"] = 3
	prec["+"] = 2
	prec["-"] = 2
	prec["("] = 1
	opStack = Stack()
	postfixList = []
	tokenList = infixexpr #.split()

	for token in tokenList:
		if isVariable(token): # in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or token in "0123456789":
			postfixList.append(token)
		elif token == '(':
			opStack.push(token)
		elif token == ')':
			topToken = opStack.pop()
			while topToken != '(':
				postfixList.append(topToken)
				topToken = opStack.pop()
		else:
			while (not opStack.isEmpty()) and (prec[opStack.peek()] >= prec[token]):
				postfixList.append(opStack.pop())
			opStack.push(token)

	while not opStack.isEmpty():
		postfixList.append(opStack.pop())
	#return " ".join(postfixList)
	return postfixList


def infix2prefix1(infixexpr):
	prec = {}
	prec["^"] = 4
	prec["*"] = 3
	prec["/"] = 3
	prec["+"] = 2
	prec["-"] = 2
	prec["("] = 1
	opStack = Stack()
	postfixList = []
	rstr = infixexpr[::-1]
	tokenList = rstr.split(' ')
	#tokenList.reverse() #.split()

	for token in tokenList:
		if isVariable(token): # in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or token in "0123456789":
			postfixList.append(token)
		elif token == '(':
			opStack.push(token)
		elif token == ')':
			topToken = None
			if(not opStack.isEmpty()):
				topToken = opStack.pop()
			while topToken != '(':
				postfixList.append(topToken)
				topToken = None
				if(not opStack.isEmpty()):
					topToken = opStack.pop()
			# topToken = opStack.pop()
		else:
			while (not opStack.isEmpty()) and (prec[opStack.peek()] >= prec[token]):
				postfixList.append(opStack.pop())
			opStack.push(token)

	while not opStack.isEmpty():
		postfixList.append(opStack.pop())
		
   # prefixString = " ".join(postfixList)
	
	prefixString = postfixList
	prefixString.reverse()
	
	return prefixString

def isVariable(var):
	m1 = bool(re.match(r'[a-zA-Z_][a-zA-Z_\d]*', var))
#	print(var, m1, m2,m3)
	return (m1 or isNumber(var))

def isNumber(var):
	#m2 = bool(re.match(r'([\d]+[.][\d]*|[\d]*[.][\d]+)([eE][-+]?[\d]+)?', var))
	#m3 = bool(re.match(r'[\d]+', var))
	m1 = False
	try:
		float(var)
		m1 = True
	except ValueError:
		m1 = False
			
	#print(var, m2,m3)
	return (m1) # or m2 or m3)
	
def postfixEval(postfixExpr):
	operandStack = Stack()
	tokenList = postfixExpr #.split()

	for token in tokenList:
		if isNumber(token): #in "0123456789":
			operandStack.push(float(token))
		else:
			operand2 = operandStack.pop()
			operand1 = operandStack.pop()
			result = doMath(token,operand1,operand2)
			operandStack.push(result)
	return operandStack.pop()

def doMath(op, op1, op2):	
	opr = ''
	if isinstance(op, ast.Add):
		opr = '+'
	elif isinstance(op, ast.Sub):
		opr = '-'			
	elif isinstance(op, ast.Mult):
		opr = '*'			
	elif isinstance(op, ast.Div):
		opr = '/'
	elif isinstance(op, ast.Pow):
		opr = '^'
	else:
		opr = str(op)
	
	opr1 = op1
	opr2 = op2
#	print(op1, op, opr, op2)
	if(isNumber(op1)):
		opr1 = float(op1)
	if(isNumber(op2)):
		opr2 = float(op2)	
		
	if opr == '*':
		if isNumber(op1) and isNumber(op2):
			return str(opr1 * opr2)
		else:
			return op1 +" "+ opr+" "+ op2
	elif opr == '/':
		if isNumber(op1) and isNumber(op2):
			return str(opr1 / opr2)
		else:
			return op1 +" "+ opr+" "+ op2
	elif opr == '+':
		if isNumber(op1) and isNumber(op2):
			return str(opr1 + opr2)
		else:
			return op1 +" "+ opr+" "+ op2
	elif opr == '^':
		if isNumber(op1) and isNumber(op2):
			return str(pow(opr1,opr2))
		else:
			return op1 +" ** "+ op2
	else:
		if isNumber(op1) and isNumber(op2):
			return str(opr1 - opr2)
		else:
			return op1 +" "+ opr+" "+ op2

if __name__ == "__main__": 	
   main(sys.argv[1:])
