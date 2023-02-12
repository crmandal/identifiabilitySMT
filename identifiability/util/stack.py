class Stack:
	def __init__(self):
		self.items = []

	def isEmpty(self):
		return self.items == []

	def push(self, item):
		self.items.append(item)
		#print('Stack pushed ', item)

	def pop(self):
		if not self.isEmpty():
			item = self.items.pop()
		#print('Stack poped ', item)
		else:
			item = None
		return item

	def peek(self):
		if not self.isEmpty():
			return self.items[len(self.items)-1]
		else:
			return None

	def size(self):
		return len(self.items)
