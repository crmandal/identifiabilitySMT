import csv
'''
	Attribute Name						Possible Values
	--------------						---------------
1		age:							continuous.
2		sex:							M, F.
3		on thyroxine:					f, t.
4		query on thyroxine:				f, t.
5		on antithyroid medication:		f, t.
6		sick:							f, t.
7		pregnant:						f, t.
8		thyroid surgery:				f, t.
9		I131 treatment:					f, t.
10		query hypothyroid:				f, t.
11		query hyperthyroid:				f, t.
12		lithium:						f, t.
13		goitre:							f, t.
14		tumor:							f, t.
15		hypopituitary:					f, t.
16		psych:							f, t.
17		TSH measured:					f, t.
18		TSH:							continuous.
19		T3:								continuous.
20		TT4:							continuous.
21		T4U:							continuous.
22		FTI:							continuous.
23		class code						1,2,3
24		class name						EU, Hypo, Hyper
'''

Age=0
Sex=1
On_thyroxine=2
Query_on_thyroxine=3
On_antithyroid_medicatio=4
Sick=5
Thyroid_surgery=7
I131_treatment=8
Query_hypothyroid=9
Query_hyperthyroid=10
Lithium=11
Goitre=12
Tumor=13
Hypopituitary=14
Psych=15
TSH=17
T3=18
TT4=19
T4U=20
FTI=21
Class=22

NORMAL=1
HYPO=2
HYPER=3

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
		types.append(NORMAL)
		types.append(HYPO)
		types.append(HYPER)
		return types

def getNoiseMargins():
	noise = {}
	noise.update({TSH:0.1})
	noise.update({T3:0.2})
	noise.update({TT4:10})
	noise.update({T4U:0.1})
	noise.update({FTI:10})
	return noise

def convert2Prop(data, dtype):
	print('convert2Prop', data[Class], dtype)
	noise = getNoiseMargins()
	prop = '('
	#if(data[Class] == 1 and dtype == NORMAL): #normal
	#prop += '((mode = 2) | (mode = 3)) &'
	#elif (data[Class] == 2 and dtype == HYPO): #hypo
	#	prop += '(mode = 4)'
	#elif (data[Class] == 3 and dtype == HYPER): #hyper
	#	prop += '(mode = 5)'
	
	#prop += '& (TSH = '+data[TSH]+') & (T3 = '+data[T3]+') & (TT4 = '+data[TT4]+') & (T4U = '+data[T4U]+') & (FTI = '+data[FTI]+')'
	#prop += ')'
	
	prop += ' (tsh < '+str(float(data[TSH])+noise[TSH])+') & (tsh > '+str(float(data[TSH])-noise[TSH])+') & '+\
			'((t3 + bg_t3) < '+str(float(data[T3])+noise[T3])+') & ((t3 + bg_t3) > '+str(float(data[T3])-noise[T3])+') &' +\
			'((t4 + bg_t4) < '+str(float(data[TT4])+noise[TT4])+') & ((t4 + bg_t4) > '+str(float(data[TT4])-noise[TT4])+') &' +\
			'((k_t4_t3 + k_t4_rt3) < '+str(float(data[T4U])+noise[T4U])+') & ((k_t4_t3 + k_t4_rt3) > '+str(float(data[T4U])-noise[T4U])+') &' +\
			'(bg < '+str(float(data[FTI])/float(data[TT4]) +noise[FTI] )+' * (bg + bg_t4 + bg+t3)) & (bg > '+str(float(data[FTI]) / float(data[TT4]) -noise[FTI])+' * (bg + bg_t4 + bg+t3))'
	#prop += ' (x = '+str(float(data[X]))+') & (v = '+str(float(data[V]))+')'
	prop += ');'
	propneg = ' (tsh < '+str(float(data[TSH])+noise[TSH])+') & (tsh > '+str(float(data[TSH])-noise[TSH])+') & '+\
			'((t3 + bg_t3) < '+str(float(data[T3])+noise[T3])+') & ((t3 + bg_t3) > '+str(float(data[T3])-noise[T3])+') &' +\
			'((t4 + bg_t4) < '+str(float(data[TT4])+noise[TT4])+') & ((t4 + bg_t4) > '+str(float(data[TT4])-noise[TT4])+') &' +\
			'((k_t4_t3 + k_t4_rt3) < '+str(float(data[T4U])+noise[T4U])+') & ((k_t4_t3 + k_t4_rt3) > '+str(float(data[T4U])-noise[T4U])+') &' +\
			'(bg < '+str(float(data[FTI])/float(data[TT4]) +noise[FTI] )+' * (bg + bg_t4 + bg+t3)) & (bg > '+str(float(data[FTI]) / float(data[TT4]) -noise[FTI])+' * (bg + bg_t4 + bg+t3));'
	return (prop, propneg)	
		
