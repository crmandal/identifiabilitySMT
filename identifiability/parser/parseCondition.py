#------------------------------------------------------------------
# dReach Model parser
#------------------------------------------------------------------

import ply.lex as lex
import ply.yacc as yacc
import sys, os, getopt
from collections import OrderedDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.haModel import *
from model.node import *

################## lexer #########################
reserved = {
# }
#    'mode'   : 'MODE',
#    'invt'   : 'INVT',
#    'flow'   : 'FLOW',
#    'jump'   : 'JUMP',   
#    'init'   : 'INIT',
#    'goal'   : 'GOAL',
   'true'	: 'T',
   'false'	: 'F',
   'and'	: 'AND',
   #'or'		: 'OR',
   'sin'	: 'SIN',
   'cos'	: 'COS',
   'tan'	: 'TAN',
   'exp'	: 'EXP',
   'log' : 'LOG',
	'sqrt'	: 'SQRT',
   'constraints'	: 'CONST'
}

tokens = [
	'VAR', 'RAT', 'NUM', 'DF',
	'ADD','SUB','MULT','DIV','POW',
	'LP','RP', 'CM','LB', 'RB', 'LC', 'RC',
	'EQ','GT', 'LT', 'GE', 'LE', 'TO', 'AT',
	'CLN', 'AP', 'DEFN' #, 'COMMENT'
	] + list(reserved.values())

# Tokens
t_DEFN	= r'\#define'


def t_DF(t):
	r'd/dt'
	return t

def t_VAR(t):
	r'[a-zA-Z_][a-zA-Z_\d]*'
	if t.value in reserved:
		t.type = reserved[t.value]
		#t.type = t.value
	return t

t_TO	= r'==>' 
t_ADD  	= r'\+'
t_SUB	= r'-'
t_MULT	= r'\*'
t_DIV	= r'/'
t_POW	= r'\^'
t_GT	= r'>='
t_LT	= r'<='
t_GE	= r'>'
t_LE	= r'<'
t_EQ	= r'='
t_LP	= r'\('
t_RP	= r'\)'
t_LC	= r'\{'
t_RC	= r'\}'
t_LB	= r'\['
t_RB	= r'\]'
t_CLN	= r':'
t_CM	= r','
t_AT	= r'@'
t_AP	= r'\''

def t_RAT(t):
	#r'[\d]*\.?[\d]+([eE][-+]?[\d]+)?'
	# r'([\d]+[.]?[\d]*|[\d]*[.][\d]+)([eE][-+]?[\d]+)?'
	r'([\d]+[.][\d]+|[\d]*[.][\d]+)([eE][-+]?[\d]+)?|([\d]+[\d]*[eE][-+]?[\d]+)' 
	#t.value = float(t.value)
	#print('rat', t.value)
	return t
	
def t_NUM(t):
	r'[\d]+'
	#t.value = float(t.value)
	#print('num', t.value)
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
	('nonassoc', 'LT', 'GT', 'LE', 'GE'),
	('left', 'ADD','SUB'),
	('left','MULT', 'DIV'),
	('left','UMINUS'),
	('right','POW')
	)

# dictionary of names (for storing variables)
names = { }

	
def p_condition1(p):
	'''condition : expr binop expr'''
	lit1 = p[1]
	bop = p[2]
	lit2 = p[3]
	#lst = [p[1], p[3]]
	#p[0] = Node(bop, lst)
	p[0] = Condition(lit1, bop, lit2)
	#print('condition', str(p[0]))

def p_condition3(p):
	'condition : LP condition RP'
	p[0] = p[2]	
	  
def p_condition2(p):
	'''condition : T
			   | F'''	
	node = Node('('+ p[1] + ')')
	p[0] = Condition(node)


def p_empty(p):
	'empty :'
	pass

def p_variables1(p):
	'variables : variables variable'
	p[0] = p[1]
	p[0].update(p[2])
	#print(p[1])
	
def p_variables2(p):
	'variables :  variable'	
	p[0] = OrderedDict()
	p[0].update(p[1])
	#print(p[1])
	
def p_variable1(p):
	'variable : LB expr CM expr RB VAR' 
	var = p[6]
	val = Range(p[2], p[4])
	p[0] = {var : val}

def p_range(p):
	'''range : RAT
			| NUM
			| VAR'''
	p[0] = p[1]
	#print(p[0])

def p_variable2(p):
	'variable : LB expr RB VAR'
	var = p[4]
	val = "["+ p[2]+"]"
	p[0] = {var : val}
	
	
def p_binop(p):
	'''binop : EQ
			  | LE
			  | GE
			  | LT
			  | GT'''
	p[0] = p[1]
			  
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
	
def p_exp5(p):
	'expr : LP expr RP'
	#print('p_exp3 ', str(p[1]))
	#lst = []
	#lst.append(p[2])
	p[0] = p[2] #Node(p[1], lst)

def p_exp2(p):
	'expr : SUB expr %prec UMINUS'
	lst = []
	lst.append(p[2])
	p[0] = Node(p[1], lst)
	
def p_exp3(p):
	'expr : range'
	p[0] = Node(p[1])

def p_exp4(p):
	'''expr : trig_func
		 | exp_func'''
	p[0] = p[1]
	
def p_trig_func(p):
	'trig_func : trig LP expr RP'
	lst = []
	lst.append(p[3])
	p[0] = Node(p[1], lst)

def p_exp_func(p):
	'''exp_func : EXP LP expr RP 
				| LOG LP expr RP
				| SQRT LP expr RP'''
	lst = []
	lst.append(p[3])
	p[0] = Node(p[1], lst)

def p_trig(p):
	'''trig : SIN
			| COS
			| TAN'''
	p[0] = p[1]

###################################

def p_error(p):
	
	# Build the lexer
	lexer1 = lex.lex()

	# Build the parser
	#do_parser()
	parser1 = yacc.yacc()
	###################################

	print("Syntax error at '%s'" % repr(p)) #p.value)
	print("Syntax error at "+ p.value, p.type, p.lineno, p.lexpos)	
	#while True:      
	tok = parser1.token() # Get the next token
		# if not tok or tok.type == 'RC': 
		# 	break     
	print("next Token : ", tok.type, tok.value, tok.lineno, tok.lexpos)

def getModel(fileName):	
	try:
		with open(fileName, 'r') as f:
			s = f.read()
	except EOFError:
		print ("Could not open file %s." + inputfile)
	ha = getCondition(s)
	return ha
	
def getCondition(s):
	#lexer.input(s)
	#for tok in lexer:
	#	print(tok)
		
	# Build the lexer
	lexer = lex.lex()
	
	# Build the parser
	#do_parser()
	parser = yacc.yacc()
	ha = yacc.parse(s, tracking=True)
	return ha


def main(argv):
	# inputfile = sys.argv[1]
	# outputfile = sys.argv[2]
	# try:
	# 	opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	# except getopt.GetoptError:
	# 		print("parseModel.py -i <inputfile> -o <outputfile>")
	# 		sys.exit(2)
			
	# for opt, arg in opts:
	# 	if opt == '-h':
	# 		print("parseModel.py -i <inputfile> -o <outputfile>")
	# 		sys.exit()
	# 	elif opt in ("-i", "--ifile"):
	# 		inputfile = arg
	# 	elif opt in ("-o", "--ofile"):
	# 		outputfile = arg
	
	# print("Input file is :"+ inputfile)
	# print("Output file is :"+ outputfile)
	
	cond = getCondition('((x/3.0) < y)')
	print(str(cond), type(cond))
	#print(ha)
	# ha.saveModel(outputfile)

if __name__ == "__main__": 	
   main(sys.argv[1:])
