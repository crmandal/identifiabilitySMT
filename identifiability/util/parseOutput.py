#------------------------------------------------------------------
# dReal output parser
#------------------------------------------------------------------

import ply.lex as lex
import ply.yacc as yacc
import sys, getopt
from collections import OrderedDict
from util.smtOutput import *
from model.node import *
from model.range import *
	
################## lexer #########################
reserved = {
	'delta'		: 'DELTA',
	'sat'		: 'SAT',
	'with'		: 'WITH',
	'the'		: 'THE',
	'following' : 'FLW',
	'box'		: 'BOX',
	'Solution'	: 'SLN'
	#'CNF'		: 'CNF'
}

tokens = [
	'VAR', 'RAT', 'NUM', 'CLN', 'EQ',
	'ADD','SUB','MULT','DIV','POW', #'UMINUS',
	'CM','LB', 'RB', 'LP', 'RP', 'SC'
	] + list(reserved.values())

# Tokens

def t_VAR(t):
	r'[a-zA-Z_][a-zA-Z_\d]*'
	if t.value in reserved:
		t.type = reserved[t.value]
		#t.type = t.value
	return t
	
t_ADD  	= r'\+'
t_SUB	= r'\-'
t_MULT	= r'\*'
t_DIV	= r'/'
t_POW	= r'\^'
t_LB	= r'\['
t_RB	= r'\]'
t_LP	= r'\('
t_RP	= r'\)'
t_CM	= r'\,'
t_CLN	= r'\:'
t_EQ	= r'\='
t_SC	= r'\;'

def t_RAT(t):
	#r'[\d]*\.?[\d]+([eE][-+]?[\d]+)?'
	# r'([\d]+[.][\d]*|[\d]*[.][\d]+)([eE][-+]?[\d]+)?'
	r'([\d]+[.]?[\d]*|[\d]*[.][\d]+)([eE][-+]?[\d]+)?'
	# r'([\d]+[.][\d]*|[\d]*[.][\d]+)([eE][-+]?[\d]+)?'
	t.value = float(t.value)
	#print(t.value)
	return t
	
def t_NUM(t):
	r'[\d]+'
	t.value = int(t.value)
	#print(t.value)
	return t

# Ignored characters
t_ignore = " \t\n;"

#def t_newline(t):
	#r'\n+'
	#t.lexer.lineno += t.value.count("\n")

def t_error(t):
	print("Illegal character '%s'" % t.value[0])
	t.lexer.skip(1)

################## parser ##############
# Precedence rules for the arithmetic operators
precedence = (
	#('nonassoc', 'LT', 'GT', 'LE', 'GE'),
	('left', 'ADD','SUB'),
	('left','MULT', 'DIV'),
	('left','UMINUS'),
	('right','POW')
	) 

# dictionary of names (for storing variables)
names = { }

def p_satinstance(p):
	'instance : cmt CLN variables extraLines'
	lst = p[3]
	p[0] = SATInstance(lst)
	#print(p[1])
	
def p_cmt1(p):
	'cmt : DELTA SUB SAT WITH THE FLW BOX'
	p[0] = p[1]+ p[2]+ p[3]+ p[4]+ p[5]+ p[6]+ p[7]
	#print(p[0])

def p_cmt2(p):
	'cmt : SLN'
	p[0] = p[1]
	#print(p[0])

def p_extra1(p):
	'extraLines : extraLines extra'
	p[0] = p[1] + p[2]
	
def p_extra2(p):
	'extraLines : extra'
	p[0] = p[1] 
	
def p_extra3(p):
	'extra : VAR CLN VAR'
	p[0] = p[1] + p[2] + p[3]
	#print('extra: ', p[0])

def p_extra4(p):
	'extra : empty'		

def p_empty(p):
	'empty :'
	pass
	
def p_variables1(p):
	'variables :  variables variable'	
	lst = p[1]
	lst.append(p[2])
	p[0] = lst
	#print(p[1])

def p_variables2(p):
	'variables :  variable'	
	lst = []
	lst.append(p[1])
	p[0] = lst
	#print(p[1])

def p_variable1(p):
	'variable : VAR CLN LB expr CM expr RB EQ LB expr CM expr RB'  
	#print(p[1])
	name = p[1]
	initVal = Range(p[4], p[6])
	endVal = Range(p[10], p[12])
	p[0] = Variable(name, initVal, endVal)
	#print('p_variable1', str(p[0]))
	
def p_variable2(p):
	'variable : VAR CLN LB expr CM expr RB'  
	#print(p[1])
	name = p[1]
	initVal = Range(Node('0'), Node('0'))
	endVal = Range(p[4], p[6])
	p[0] = Variable(name, initVal, endVal)
	#print('p_variable2', str(p[0]))

def p_exp1(p):
	'''expr : expr ADD expr
	| expr SUB expr
	| expr MULT expr
	| expr DIV expr
	| expr POW expr'''
	lst = []
	lst.append(p[1])
	lst.append(p[3])
	p[0] = Node(p[2], lst)
	
def p_exp2(p):
	'expr : LP expr RP'
	#print('p_exp3 ', str(p[1]))
	lst = []
	#lst.append(p[2])
	p[0] = p[2] #Node(p[1], lst)

def p_exp3(p):
	'expr : SUB expr %prec UMINUS'
	lst = []
	lst.append(p[2])
	p[0] = Node(p[1], lst)
	
def p_exp4(p):
	'expr : range'
	p[0] = Node(p[1])

def p_range3(p):
	'''range : NUM
			| RAT
			| VAR'''
	p[0] = p[1]

#def p_error(p):
#   print("Syntax error at '%s'" % p.value)
###################################

# Build the lexer
lexer1 = lex.lex()

# Build the parser
#do_parser()
parser1 = yacc.yacc()
###################################

def p_error(p):
	print("Syntax error at '%s'" % repr(p)) #p.value)
	print("Syntax error at "+ p.value, p.type, p.lineno, p.lexpos)	
	#while True:      
	tok = parser1.token() # Get the next token
		# if not tok or tok.type == 'RC': 
		# 	break     
	print("next Token : ", tok.type, tok.value, tok.lineno, tok.lexpos)

def parseInstance(fileName):	
	try:
		with open(fileName, 'r') as f:
			s = f.read()
	except EOFError:
		print ("Could not open file %s." + inputfile)
	ha = getSATInstance(s)
	return ha
	
def getSATInstance(s):
	#lexer.input(s)
	#for tok in lexer:
	#	print(tok)
		
	# Build the lexer
	lexer = lex.lex()
	
	# Build the parser
	#do_parser()
	parser = yacc.yacc()
	satinstance = yacc.parse(s, tracking=True)
	# print(satinstance)
	return satinstance


def main(argv):
	inputfile = sys.argv[1]
	outputfile = sys.argv[2]
	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:
			print("parseModel.py -i <inputfile> -o <outputfile>")
			sys.exit(2)
			
	for opt, arg in opts:
		if opt == '-h':
			print("parseModel.py -i <inputfile> -o <outputfile>")
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile"):
			outputfile = arg
	
	print("Input file is :"+ inputfile)
	print("Output file is :"+ outputfile)
	
	si = parseInstance(inputfile)
	print(str(si))
	#ha.saveModel(outputfile)

if __name__ == "__main__": 	
   main(sys.argv[1:])
