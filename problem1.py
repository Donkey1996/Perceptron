import pandas as pd
from visualize import visualize_scatter
import sys

class Perceptron:
	def __init__(self):
		self.x1 = []
		self.x2 = []
		self.y = []
		self.weights = [0, 0, 0]
		self.fx = []
	def read(self, data):

		self.x1 = list(data['x1'])
		self.x2 = list(data['x2'])
		self.y = list(data['y'])
		self.fx = [0]*len(self.x1)

	def f_x(self):
		w0, w1, w2 = self.weights[0], self.weights[1], self.weights[2]
		for i in range(len(self.x1)):
			if w0 + w1*self.x1[i] + w2*self.x2[i] > 0:
				self.fx[i] = 1
			else:
				self.fx[i] = -1
		return self.fx

	def is_convergent(self, fx):
		if fx == self.y:
			return True
		return False

	def fit(self, output):
		#implement PLA
		file = open(output, 'w')
		file.write(str(self.weights[2])+","+str(self.weights[0])+","+str(self.weights[1])+"\n")
		while not self.is_convergent(self.f_x()):
			#update weights using all examples until converged
			for i in range(len(self.x1)):
				if not self.y[i]*self.fx[i] > 0:
					self.weights[0] += self.y[i]
					self.weights[1] += self.x1[i]*self.y[i]
					self.weights[2] += self.x2[i]*self.y[i]

				#if self.is_convergent(self.f_x()):
				#	break
			file.write(str(self.weights[1])+","+str(self.weights[2])+","+str(self.weights[0])+"\n")
def main():
	input, output = sys.argv[1], sys.argv[2]
	data = pd.read_csv(input, names=['x1', 'x2', 'y'])
	p = Perceptron()
	p.read(data)
	p.fit(output)
	#visualize_scatter(data, feat1='x1', feat2='x2', labels='y', weights=[p.weights[1], p.weights[2], p.weights[0]])

if __name__ == '__main__':
	main()