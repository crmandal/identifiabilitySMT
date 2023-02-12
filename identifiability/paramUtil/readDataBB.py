import csv
TIME=0
k = 1
radius =2 
rho = 3
g = 4
X=5
V=6
S=7
Mode = 8

FALLING=1
RISING=2

class Data:
	
	def __init__(self, fname):
		rows=[]
		with open(fname) as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for row in spamreader:
				rows.append(row)
				#print(row)
				#print(', '.join(row))
		self.values = rows
		self.row = len(rows)
		self.column = len(row)

	def getNumCols(self):
		return self.column
	
	def getNumRows(self):
		return self.row
	
	def getData(self):
		return self.values

	def getTypes(self):
		types = []
		types.append(FALLING)
		types.append(RISING)
		return types
		
def getNoiseMargins():
	noise = {}
	noiseX = 0.002
	noiseV = 0.005
	noise.update({X:noiseX})
	noise.update({V:noiseV})
	print('Noise: '+ str(noise))
	return noise
		
def convert2Prop(data, dtype):
	noise = getNoiseMargins()
	prop = '('
	print('convert2Prop', data[Mode], dtype)
	#if(int(data[Mode]) == 1 and dtype == FALLING): #FALLING		
		#prop += '(mode = 1) & '
	#elif (int(data[Mode]) == 2 and dtype == RISING): #RISING
		#prop += '(mode = 2) & '
	
	prop += ' (x < '+str(float(data[X])+noise[X])+') & (x > '+str(float(data[X])-noise[X])+') & (v < '+str(float(data[V])+noise[V])+') & (v > '+str(float(data[V])-noise[V])+')'
	#prop += ' (x = '+str(float(data[X]))+') & (v = '+str(float(data[V]))+')'
	prop += ');'
	propneg = '((x > '+str(float(data[X])+noise[X])+') | (x < '+str(float(data[X])-noise[X])+') | (v > '+str(float(data[V])+noise[V])+') | (v < '+str(float(data[V])-noise[V])+'));'
	return (prop, propneg)	
		
