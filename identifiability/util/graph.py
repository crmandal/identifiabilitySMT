# Python program to print all paths from a source to destination.
import os
import subprocess
import re
import sys, getopt
from collections import defaultdict
 
class Path:
	def __init__(self, list1=[]):
		self.list= list1

	def append(self, a):
		self.list.append(a)
		return self

	def concat(self, list2):
		for p in list2:
			self.list.append(p)
		return self

	def __str__(self):
		i = 0
		pathstr = '['
		for p in self.list:
			if(i == 0):
				pathstr += p
			else:
				pathstr += ',' + p
			i = i+1
		pathstr += ']'
		return pathstr

	def getPrefixes(self):
		# get prefixes for a list
		prefixes = []
		for i in range(len(self.list)+1):
			if(i == 0):
				continue;
			else:
				p = self.list[0:i]
				prefix = Path(p)
				prefixes.append(prefix)
		return prefixes


# This class represents a directed graph 
# using adjacency list representation

class Graph:
	
	def __init__(self,vertices):
		#No. of vertices
		self.V= vertices 
		
		# default dictionary to store graph
		self.graph = defaultdict(list) 
 
	# function to add an edge to graph
	def addEdge(self,u,v):
		if(v not in self.graph[u]):
			self.graph[u].append(v)     
		#print(u, self.graph[u])
		return self

	def printG(self):
		for i in range(self.V):
			print(i, self.graph[i])

	#A recursive function to print all paths from 'u' to 'd'.
	#visited[] keeps track of vertices in current path.
	#path[] stores actual vertices and path_index is current
	#index in path[]

	def printAllPathsUtil(self, u, d, visited, path):

		# Mark the current node as visited and store in path
		visited[u]= True
		path.append(u)

		# If current vertex is same as destination, then print
		# current path[]
		if u == d:
			print(path)
			yield(path)

		else:
			# If current vertex is not destination
			#Recur for all the vertices adjacent to this vertex
			for i in self.graph[u]:
				if visited[i]==False:
					self.printAllPathsUtil(i, d, visited, path)
					
		# Remove current vertex from path[] and mark it as unvisited
		path.pop()
		visited[u]= False
 
 
	# Prints all paths from 's' to 'd'
	def printAllPaths(self,s, d):

		# Mark all the vertices as not visited
		visited =[False]*(self.V)

		# Create an array to store paths
		path = []

		# Call the recursive helper function to print all paths
		self.printAllPathsUtil(s, d,visited, path)
		#print(paths)

	def getAllPaths(self,s, d):
		paths = self.bfs_paths(s, d)
		for path in paths:
			print(path)
		return paths

	def bfs_paths(start, goal):
		queue = [(start, [start])]
		while queue:
			(vertex, path) = queue.pop(0)
			for next in self.graph[vertex] - set(path):
				if next == goal:
					yield path + [next]
				else:
					queue.append((next, path + [next]))
	   # return queue

	def getPathsofLength(self, start, goal, length):
		paths = []
		path = []
		#path = Path()
		queue = [(start, [start], 0)]
		#print('paths of length ',length)
		#queue = [(start, Path([start]), 0)]
		i = 0
		k = 0
		l = int(length)
		if l == 0:
			if start == goal:
				path.append(start)
				paths.append(path) 
				yield path
			else:
				paths = []
		else:
			while(len(queue) > 0): # and k <= l):
				(u, path, k) = queue.pop(0)
			#   print(u, path, k, l)
				if(k < l):
					for p in self.graph[u]:
						#print(u, 'next', p)
						if p == goal:
							paths.append(path +[p])
							if(k <= l):
								yield path +[p]
								queue.append((p, path + [p], k+1))
							
						else:
							queue.append((p, path + [p], k+1))
				# else:
				# 	print(' k > length')
				# 	raise StopIteration
				
		
	def getKPaths(self, start, length):
		paths = []
		path = []
		#path = Path()
		queue = [(start, [start], 0)]
		#print('paths of length ',length)
		i = 0
		k = 0
		l = int(length)
		#if l == 0:
		path.append(start)
		paths.append(path) 
		yield path
		#else:
		
		while(len(queue) > 0): 
			(u, path, k) = queue.pop(0)
		#   print(u, path, k, l)
			if(k < l):
				for p in self.graph[u]:
					#print(u, 'next', p)
					paths.append(path +[p])
					if(k <= l):
						yield path +[p]
						queue.append((p, path + [p], k+1))
			#else:
			#   print('k > length', k, l)
				#raise StopIteration
	def getKPathsStop(self, start, length):
		paths = []
		path = []
		#path = Path()
		queue = [(start, [start], 0)]
		#print('paths of length ',length)
		i = 0
		k = 0
		l = int(length)
		#if l == 0:
		path.append(start)
		paths.append(path) 
		# yield path
		#else:
		
		while(len(queue) > 0): 
			(u, path, k) = queue.pop(0)
		#   print(u, path, k, l)
			if(k < l):
				for p in self.graph[u]:
					#print(u, 'next', p)
					if(k <= l):
						# yield path +[p]
						paths.append(path +[p])
						queue.append((p, path + [p], k+1))
			#else:
			#   print('k > length', k, l)
				#raise StopIteration
		return paths

def getPrefixes(path, reverse=False):
	# get prefixes for a list
	prefixes = []
	revPrefixes = []
	for i in range(len(path)+1):
		if(i == 0):
			continue;
		else:
			prefix = path[0:i]
			prefixes.append(prefix)

	revPrefixes = prefixes.reverse()
	if (reverse):
		return revPrefixes
	else:
		return prefixes

def string(path):
	i = 0
	pathstr = '['
	for p in path:
		if(i == 0):
			pathstr += str(p)
		else:
			pathstr += ',' + str(p)
		i = i+1
	pathstr += ']'
	return pathstr

def toString(path):
	pathstr = ''
	for p in path:
		pathstr += str(p)+ ' '
	return pathstr

def main(argv):
	# Create a graph given in the above diagram
	# g = Graph(4)
	# g.addEdge(0, 1)
	# g.addEdge(0, 2)
	# g.addEdge(0, 3)
	# g.addEdge(2, 0)
	# g.addEdge(2, 1)
	# g.addEdge(1, 3)

	g = Graph(6)
	g.addEdge(1, 2)
	g.addEdge(2, 3)
	g.addEdge(3, 2)
	g.addEdge(2, 4)
	g.addEdge(3, 4)
	g.addEdge(2, 5)
	g.addEdge(3, 5)
  
	s = 1 ; d = 3
	k = 10
	print ("Following are different paths (length <= %d) from %d to %d :" %(k, s, d))
	ps = g.getPathsofLength(s,d, k)
	# #This code is contributed by Neelam Yadav
	# for p in ps:
	#   print(p)
	#ps = 
	#This code is contributed by Neelam Yadav
	for p in g.getPathsofLength(s,d, k):
		print(p)

if __name__ == "__main__": 
   main(sys.argv[1:])
