import getopt
from collections import OrderedDict
#from ha.util.exprEval import *

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from util.exprEval import *
from util.graph import *
from range import *
from condition import *
from node import *
from mode import *

class Parameter:
	def __init__(self, name, left, right):
		self.name = name
		self.range = Range(Node(left), Node(right))	
		
	def __str__(self):
		param = 'Name: '+ self.name+' Range: ['+self.range+']'
		return param

	def deleteGoal(self):
		self.goals = []

	def getRange(self):
		return self.range;
	
	def getName(self):
		return self.name

