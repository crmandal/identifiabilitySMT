#------------------------------------------------------------------
# STL property parser
#------------------------------------------------------------------
import ply.lex as lex
import ply.yacc as yacc
import sys, getopt

from model.node import *
#from condition import *
	
################## lexer #########################
reserved = {
	'true'	: 'T',
	'false'	: 'F',
	#'and'	: 'AND',
	#'min'	: 'MIN',
	#'max'	: 'MAX',
	'G'		: 'GLOBAL',
	'F'		: 'FUTURE',
	'U'		: 'UNTIL',
	'R'		: 'RELEASE',
	'X'		: 'NEXT',
	'sin'	: 'SIN',
	'cos'	: 'COS',
	'tan'	: 'TAN',
	'exp'	: 'EXP',
	'log'	: 'LOG',
	'sqrt'	: 'SQRT'
}

tokens = [
	'NUM', 'VAR', 'RAT', 
	'ADD','SUB','MULT','DIV','POW',
	'LP','RP', 'LB', 'RB', 'IMPLY',
	'EQ','GT', 'LT', 'GE', 'LE','AT', 'NEQ',
	'BOR', 'NOT', 'BAND',  'SC', 'CM' #'CLN',
	] + list(reserved.values())

# Tokens
def t_VAR(t):
	r'[a-zA-Z_][a-zA-Z0-9_]*'
	if t.value in reserved:
		t.type = reserved[t.value]
		#t.type = t.value
	return t
	
t_IMPLY	= r'->'
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
t_NEQ	= r'!='
t_LP	= r'\('
t_RP	= r'\)'
t_LB	= r'\['
t_RB	= r'\]'
t_AT	= r'@'
t_BOR 	= r'\|'
t_BAND	= r'\&'
t_NOT	= r'!'
#t_CLN	= r':'
t_SC	= r';'
t_CM	= r','


def t_RAT(t):
	#r'[\d]*\.?[\d]+([eE][-+]?[\d]+)?'
	# r'([\d]+[.]?[\d]*|[\d]*[.][\d]+)([eE][-+]?[\d]+)?'
	r'([\d]+[.][\d]+|[\d]*[.][\d]+)([eE][-+]?[\d]+)?|([\d]+[\d]*[eE][-+]?[\d]+)' 
	#t.value = float(t.value)
	#print(t.value)
	return t
	
def t_NUM(t):
	r'[\d]+'
	#t.value = float(t.value)
	#print(t.value)
	return t

# Ignored characters
t_ignore = " \t\n"

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
	('left','ADD','SUB'),
	('left','MULT','DIV'),
	('right','UMINUS'),
	)

# dictionary of names (for storing variables)
names = { }

def p_model(p):
	'model : properties'
	props = p[1]
	p[0] = props
	  
def p_empty(p):
	'empty :'
	pass
	
def p_list1(p):
	'properties : properties prop SC'
	p[0] = p[1]
	p[0].append(p[2])
	
def p_list2(p):
	'properties : empty'
	p[0] = []
	
def p_prop1(p):
	'prop : NOT prop'
	#op = p[1]	
	lst = []
	lst.append(p[2])
	p[0] = Node(p[1], lst)
	#print(str(ps))

def p_prop2(p):
	'prop : LP prop RP'
	p[0] = p[2]

def p_prop3(p):
	'''prop : prop BAND prop 
			| prop BOR prop
			| prop IMPLY prop'''
	lst = []
	lst.append(p[1])
	lst.append(p[3])
	p[0] = Node(p[2], lst)

def p_prop4(p):
	'''prop : NEXT LB NR RB prop'''
	lst = []
	lst.append(p[5])
	val = p[1]+ p[2]+p[3]+p[4]
	p[0] = Node(val, lst)

def p_prop5(p):
	'''prop : GLOBAL LB NR CM NR RB prop 
			| GLOBAL LP NR CM NR RP prop
			| GLOBAL LB NR CM NR RP prop
			| GLOBAL LP NR CM NR RB prop'''
	lst = []
	lst.append(p[7])
	val = p[1]+ p[2]+p[3]+p[4]+p[5]+p[6]
	p[0] = Node(val, lst)

def p_prop51(p):
	'''prop : GLOBAL prop'''
	lst = []
	lst.append(p[2])
	val = p[1] #+ '[0,infty)'
	p[0] = Node(val, lst)
	
def p_prop6(p):
	'''prop : FUTURE LB NR CM NR RB prop 
			| FUTURE LP NR CM NR RP prop 
			| FUTURE LB NR CM NR RP prop 
			| FUTURE LP NR CM NR RB prop'''
	lst = []
	lst.append(p[7])
	val = p[1]+ p[2]+p[3]+p[4]+p[5]+p[6]
	p[0] = Node(val, lst)

def p_prop61(p):
	'''prop : FUTURE prop'''
	lst = []
	lst.append(p[2])
	val = p[1] #+ '[0,infty)'
	p[0] = Node(val, lst)

def p_prop7(p):
	'''prop : prop UNTIL LB NR CM NR RB prop
			| prop UNTIL LP NR CM NR RP prop
			| prop UNTIL LB NR CM NR RP prop
			| prop UNTIL LP NR CM NR RB prop'''
	lst = []
	lst.append(p[1])
	lst.append(p[8])
	val = p[2]+p[3]+p[4]+p[5]+p[6]+p[7]
	p[0] = Node(val, lst)

def p_prop71(p):
	'''prop : prop UNTIL prop'''
	lst = []
	lst.append(p[1])
	lst.append(p[3])
	val = p[2] #+ '[0,infty)'
	p[0] = Node(val, lst)

def p_prop8(p):
	'''prop : prop RELEASE LB NR CM NR RB prop
			| prop RELEASE LP NR CM NR RP prop
			| prop RELEASE LB NR CM NR RP prop
			| prop RELEASE LP NR CM NR RB prop'''
	lst = []
	lst.append(p[1])
	lst.append(p[8])
	val = p[2]+p[3]+p[4]+p[5]+p[6]+p[7]
	p[0] = Node(val, lst)

def p_prop81(p):
	'''prop : prop RELEASE prop'''
	lst = []
	lst.append(p[1])
	lst.append(p[3])
	val = p[2] #+ '[0,infty)'
	p[0] = Node(val, lst)


def p_prop9(p):
	'prop : condition'
	p[0] = p[1]
#	print(str(p[0]))

def  p_condition(p):
	'condition : AT NUM'
	lit1 = Node('mode')
	bop = '='
	lit2 = Node(p[2])
	ls = [lit1, lit2]
	p[0] = Node(bop, ls) #Condition(lit1, bop, lit2)	

def p_condition1(p):
	'condition : expr op expr'
	lit1 = p[1]
	bop = p[2]
	lit2 = p[3]
	ls = [lit1, lit2]
	p[0] = Node(bop, ls)
	#p[0] = Condition(lit1, bop, lit2)

def p_condition3(p):
	'condition : LP condition RP'
	p[0] = p[2]	
	  
def p_condition2(p):
	'''condition : T
				| F'''	
	node = Node('('+ p[1] + ')')
	p[0] = node
	
def p_op(p):
	'''op : EQ
		  | LE
		  | GE
		  | LT
		  | GT
		  | NEQ'''
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
	# print('p_exp1', p[2], lst)
	p[0] = Node(p[2], lst)
	
def p_exp5(p):
	'expr : LP expr RP'
	# print('p_exp5 ', str(p[2]))
	p[0] = p[2] 
	
def p_exp2(p):
	'expr : SUB expr %prec UMINUS'
	lst = []
	lst.append(p[2])
	p[0] = Node(p[1], lst)
	
def p_exp3(p):
	'''expr : NR
		  | VAR'''
	p[0] = Node(p[1])
	
def p_exp31(p):
	'''NR : NUM
		  | RAT'''
	p[0] = p[1]
	
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

def p_error(p):
	#while True:		# Get the next token
	#	if not tok or tok.type == 'RC': 
	#		break

	##############################

	# Build the lexer
	lexer = lex.lex()

	# Build the parser
	#do_parser()
	parser = yacc.yacc()
	###################################

	print("Syntax error at '%s'" % repr(p)) #p.value)
	print("Syntax error at "+ p.value, p.type, p.lineno, p.lexpos)	
	
	tok = parser.token()		
	print("next Token : " +tok.value)

def getSTLfromfile(fileName):	
	try:
		with open(fileName, 'r') as f:
			s = f.read()
	except EOFError:
		print ("Could not open file %s." + inputfile)
	
	#print(s)
	props = getSTL(s)
	
	return props
	
def getSTL(s):
	#lexer.input(s)
	#for tok in lexer:
	#	print(tok)		

	# Build the lexer
	lexer = lex.lex()

	# Build the parser
	#do_parser()
	parser = yacc.yacc()

	properties = yacc.parse(s, tracking=True)
	return properties


def main(argv):
	inputfile = sys.argv[1]
	try:
		opts, args = getopt.getopt(argv,"hi:",["ifile="])
	except getopt.GetoptError:
			print("parseProperty.py -i <inputfile>")
			sys.exit(2)
			
	for opt, arg in opts:
		if opt == '-h':
			print("parseProperty.py -i <inputfile>")
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
	
	print("Input file is :"+ inputfile)
	
	#properties = getSTL(inputfile)
	#for prop in properties:
	#	print(prop.to_infix())
	#	neg = prop.negate()
	#	print(neg.to_infix())
		
	prop  = getSTL('((mode = 1) & (x = 9.386875) & (v = 0.24525));')
	print(prop[0].to_infix())
	

if __name__ == "__main__": 	
	main(sys.argv[1:])
