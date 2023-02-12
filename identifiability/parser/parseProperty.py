#------------------------------------------------------------------
# dReach property parser
#------------------------------------------------------------------

import ply.lex as lex
import ply.yacc as yacc
from model.property import *
	
################## lexer #########################
reserved = {
   'true'	: 'T',
   'false'	: 'F',
   'and'	: 'AND',
   'min'	: 'MIN',
   'max'	: 'MAX'
}

tokens = [
	'NUM', 'VAR',
	'ADD','SUB','MULT','DIV','POW',
	'LP','RP', 'LB', 'RB',
	'EQ','GT', 'LT', 'GE', 'LE','AT',
	'BOR', 'NOT', 'BAND', 'CLN', 'SC'
	] + list(reserved.values())

# Tokens
def t_VAR(t):
	r'[a-zA-Z_][a-zA-Z0-9_/]*'
	if t.value in reserved:
		t.type = reserved[t.value]
		#t.type = t.value
	return t
	
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
t_LB	= r'\['
t_RB	= r'\]'
t_AT	= r'@'
t_BOR 	= r'\|'
t_BAND	= r'\&'
t_NOT	= r'!'
t_CLN	= r':'
t_SC	= r';'


def t_NUM(t):
	r'-?[\d][\d.]*'
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
	'model : MIN CLN list MAX CLN list'
	prop_min = p[3]
	prop_max = p[6]
	ps = Properties(prop_min, prop_max)
	p[0] = ps
	  
def p_empty(p):
	'empty :'
	pass
	
def p_list1(p):
	'list : list property SC'
	p[0] = p[1]
	p[0].append(p[2])
	
def p_list2(p):
	'list : empty'
	p[0] = []
	
def p_property1(p):
	'property : prop'
	p[0] = Property(p[1])

def p_prop1(p):
	'''prop : BAND props 
			| BOR props'''	
	ps = []
	ps.append(p[1])
	for p2 in p[2]:
		ps.append(p2)
	p[0] = ps

def p_props1(p):
	'props : props prop'	
	p[0] = p[1]
	for p2 in p[2]:
		p[0].append(p2)
	#print(p[2])

def p_props2(p):
	'props : prop'	
	p[0] = []
	for p1 in p[1]:
		p[0].append(p1)
	#print(p[1])

def p_prop2(p):
	'prop : NOT prop'
	#op = p[1]
	ps = []
	ps.append(p[1])
	for p2 in p[2]:
		ps.append(p2)
	p[0] = ps
	#print(str(ps))

def p_prop3(p):
	'prop : LP prop RP'
	p[0] = p[2]

def p_prop4(p):
	'prop : goal'
	ps = []
	ps.append(p[1])
	p[0] = ps
#	print(str(p[0]))

def p_goal1(p):
	'goal : LP AT NUM formula RP'	
	mode = p[3]
	formula = p[4]
	pr = Goal(mode, formula)
	p[0] = pr
#	print(str(pr))	

def p_formula1(p):
	'formula : LP AND literals RP'
	p[0] = p[3]

def p_formula2(p):
	'formula :  condition'
	p[0] = p[1]	
	
def p_literals1(p) :
	'literals : literals condition'
	p[0] = p[1]
	p[0].append(p[2])
	
def p_literals2(p) :
	'literals : condition'
	p[0] = []
	p[0].append(p[1])


def p_condition1(p):
	'condition : expr op expr'
	p[0] = p[1] +" "+ p[2] +" "+ p[3]

def p_condition3(p):
	'condition : LP condition RP'
	p[0] = p[1] + p[2] + p[3]	
	  
def p_condition2(p):
	'''condition : T
			   | F'''	
	p[0] = p[1]
	
def p_op(p):
	'''op : EQ
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
	| expr POW expr
	| LP expr RP'''
	p[0] = p[1] +" "+ p[2] +" "+ p[3]
	
def p_exp2(p):
	'expr : SUB expr %prec UMINUS'
	p[0] = p[1] + p[2]
	
def p_exp3(p):
	'''expr : NUM
		  | VAR'''
	p[0] = p[1] 


def p_error(p):
	##############################

	# Build the lexer
	lexer = lex.lex()

	# Build the parser
	#do_parser()
	parser = yacc.yacc()
	###################################

	#while True:      # Get the next token
	#	if not tok or tok.type == 'RC': 
	#		break
	print("Syntax error at "+ p.value, p.type)	
	tok = parser.token()      
	print("next Token : " +tok.value)

def getProperty(fileName):	
	try:
		with open(fileName, 'r') as f:
			s = f.read()
	except EOFError:
		print ("Could not open file %s." + inputfile)
	
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
	
	properties = getProperty(inputfile)
	print(properties)

if __name__ == "__main__": 	
   main(sys.argv[1:])
