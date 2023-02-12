from __future__ import print_function
import os
import subprocess
import re
import sys, getopt
import re
from util.stack import *

# Program to convert infix expression to prefix expression

# method to reverse a string and return the reversed string
def reverse(str):
	return("".join(reversed(str)))
	
	
# method that returns the priority of the operator
def priority(operator):
    if operator == '+' or operator == '-':
        return 1

    elif operator == '*' or operator == '/' or operator == '%':
        return 2

    elif operator == '^':
        return 3

    else:
        return 0

def infix2prefix(infixList):

	infixString = infixList.reverse()

	# taking a list variable to store operators
	stack = []
	# result string variable
	prefixString = ""
	i = 0

	# loop till i is in the range of the length of string
	while i in range(0, len(infixString)):

		# if an alphabet is found then copy it to the output string
		if isVariable(infixString[i]):
			prefixString += infixString[i]

		# as we have reversed the string closing bracket will be found first
		# if an closing bracket is found then put it in stack
		elif infixString[i] == ')' or infixString[i] == ']' or infixString[i] == '}':
			stack.append(infixString[i])


		# as we have reversed the string opening bracket will be found after the closing bracket
		# if an opening bracket is found then
		# keep removing the operators from the stack and add them to prefix string until you find the corresponding opening bracket
		elif infixString[i] == '(' or infixString[i] == '[' or infixString[i] == '{':

			if infixString[i] == '(':
				while stack[-1] != ')':
					prefixString += stack.pop()
				stack.pop()

			if infixString[i] == '[':
				while stack[-1] != ']':
					prefixString += stack.pop()
				stack.pop()

			if infixString[i] == '{':
				while stack[-1] != '}':
					prefixString += stack.pop()
				stack.pop()

		# if none of the above cases are satisfied then we surely have an operator
		else:

			# if the stack if empty then we simply put the operator in stack
			if len(stack) == 0:
				stack.append(infixString[i])

			# if not then we compare the priority of the stack top and current operator
			else:

				# if the priority of current operator is greater than or equal to the stack top then push it onto the stack
				if priority(infixString[i]) >= priority(stack[-1]):
					stack.extend(infixString[i])


				# if the priority of current operator is less than the stack top then
				# pop the stack top and add it to the prefix string
				elif priority(infixString[i]) < priority(stack[-1]):
					prefixString += stack.pop()
					position = len(stack) - 1

					# now if you have an operator that has less priority as of current operator then pop
					while position >= 0 and priority(stack[position]) > priority(infixString[i]):
						prefixString += stack.pop()
						position -= 1
						if position < 0:
							break

					stack.extend(infixString[i])

		# increment the value of i
		i += 1

	# at the end remove all the operators from the stack
	while len(stack) != 0:
		prefixString += stack.pop()
		
	# reverse the string before output	
	prefixString = reverse(prefixString)

	# return the result
	return prefixString

def isVariable(var):
	m1 = re.match("r'[a-zA-Z_][a-zA-Z_\d]*", var)
	m2 = re.match("r'([\d]+[.][\d]*|[\d]*[.][\d]+)([eE][-+]?[\d]+)?", var)
	m3 = re.match("r'[\d]+", var)
	return (m1 or m2 or m3)
	
def main(argv):
	# main function
	#infix = input("Enter the Infix Expression : ")
	prefix = in2prefix(infix)
	print("The converted Expression in Prefix is : " + prefix)

if __name__ == "__main__": 	
   main(sys.argv[1:])
