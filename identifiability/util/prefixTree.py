class Node:
   def __init__(self):
       self.children = {}  # mapping from character ==> Node 
       self.value = None

	def find(self, key):
	    for char in key:
	        if char in self.children:
	            node = self.children[char]
	        else:
	            return None
	    return node.value


	def insert(root, string, value):
	    node = root
	    i = 0
	    while i < len(string):
	        if string[i] in node.children:
	            node = node.children[string[i]]
	            i += 1
	        else:
	            break

	    # append new nodes for the remaining characters, if any
	    while i < len(string):
	        node.children[string[i]] = Node()
	        node = node.children[string[i]]
	        i += 1

	    # store value in the terminal node
	    node.value = value