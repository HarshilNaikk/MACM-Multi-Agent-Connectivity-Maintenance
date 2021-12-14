import sys
import os
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import matplotlib

class agent:
	def __init__(self, agentnum, xcoord, ycoord, obstacles):
		self.xcoord = xcoord			# X coordinates to simulate environment
		self.ycoord = ycoord			# Y coordinates to simulate environment
		self.agentnum = agentnum		# Current agent number
		self.obstacles = obstacles		# Obstacle data for handling (Simulating the environment)
		self.A = np.zeros((len(self.xcoord), len(self.ycoord)), dtype=float) 	# Graph Adjacency array
		self.N = []						# Neighbours list
		self.ND = []					# Neighbours Data
		self.sensrange = 4.0			# Sensing Range 
		self.agentrange = 4.0			# Min. Distance from Agents
		self.x = 5						# Arbitrary Estimate of eigenvector
		self.y1 = np.random.random()	#Arbitrary estimate for PI Consensus
		self.w1 = np.random.random()	#Arbitrary estimate for PI Consensus
		self.y2 = np.random.random()	#Arbitrary estimate for PI Consensus
		self.w2 = np.random.random()	#Arbitrary estimate for PI Consensus
		self.Kp = 500					# PI Consensus gain
		self.Ki = 200.0					# PI Consensus gain
		self.sigma = 10.0				# PI Consensus paramter

	def getWeights(self):									
		# -------------------------------------------------------------------------------------------------------------------
		# Update the adjacency matrix A
		# -------------------------------------------------------------------------------------------------------------------
		for i in range(len(self.xcoord)):
			for j in range(len(self.xcoord)):
				if math.sqrt((self.xcoord[i] - self.xcoord[j])**2 + (self.ycoord[i] - self.ycoord[j])**2) < 4.0:
					if j!=i:
						self.A[i,j] = math.exp(-1*((self.xcoord[i] - self.xcoord[j])**2 + (self.ycoord[i] - self.ycoord[j])**2)/(2*self.sigma**2))
				else:
					self.A[i,j] = 0.0

	def getNeighbours(self, neighboursdata):
		# -------------------------------------------------------------------------------------------------------------------				
		# Update Neighbours data based on sensor model
		# -------------------------------------------------------------------------------------------------------------------
		for j in range(len(self.xcoord)):
			if j!= self.agentnum:
				dist = math.sqrt((self.xcoord[self.agentnum] - self.xcoord[j])**2 + (self.ycoord[self.agentnum]-self.ycoord[j])**2)
				if dist < self.sensrange:
					self.N.append(j)
					self.ND.append(neighboursdata[j])	
	
	def PIaverageconsensus(self, agentnum, L, N, ND, xi, y1, y2, w1, w2):	
		# -------------------------------------------------------------------------------------------------------------------
		# PI Average consensus estimator function. Gives an estimate as the output.
		# -------------------------------------------------------------------------------------------------------------------
		xini = xi
		xfin = xi + 100
		delT = 0.1
		k1, k2, k3 = 0.01,0.01,0.01
		while (abs(xfin - xini)) > 0.1:
			xdot = -k1*y1 - k2*L*xfin - k3*(y2 - 1)*xfin 
			xfin += min(xdot[agentnum])*delT
			sum1, sum2, sum3, sum4 = 0.0,0.0,0.0,0.0
			for i in range(len(N)):
				sum1 += (y1 - ND[i][1])
				sum2 += (w1 - ND[i][2])
				sum3 += (y2 - ND[i][3])
				sum4 += (w2 - ND[i][4])
			w1dot = -self.Ki*sum1
			w2dot = -self.Ki*sum3
			w1 += w1dot*delT
			w2 += w2dot*delT
			y1dot = -self.Kp*sum1 + self.Ki*sum2
			y2dot = -self.Kp*sum3 + self.Ki*sum4
			y1 += y1dot*delT
			y2 += y2dot*delT
		return xfin

	def controlLaw(self, Ak, xfin, N, ND, x, y):
		# -------------------------------------------------------------------------------------------------------------------
		# Gives the control output u that maintains connectivity for the agent based on the adjacency matrix and the agent's estimate of the eigenvector
		# -------------------------------------------------------------------------------------------------------------------
		uk = np.array([0.0,0.0])
		for i in range(len(N)):
			xi, yi = N[i][0], N[i][1]
			uk += -Ak[i]*((ND[i][0] - xfin)**2)*(np.array([x, y]) - np.array([xi, yi]))/(self.sigma**2)
		return uk 


class main:
	def __init__(self):
		self.numagents = 6
		self.numobstacles = 10
		self.xcoord = [-2.0,-1.0,0.0,1.0,2.0,2.0]
		self.ycoord = [1.0, 3.0, 0.0, 3.0, 2.0, -1.0]
		self.obstacles = [[np.random.uniform(-7,7), np.random.uniform(-7,7)] for i in range(self.numobstacles)]
		self.neighboursdata = [[] for _ in range(self.numagents)]
		self.agents = []
		self.sigma = 0.00008

	def mainthread(self):
		# -------------------------------------------------------------------------------------------------------------------
		# Simulates the environment and handles the main thread processing.
		# -------------------------------------------------------------------------------------------------------------------
		avg1 = 0.0
		avg2 = 0.0
		xwei = []
		for i in range(self.numagents):				# Get the first initial estimates for variables going to be used ahead. 
			agentnum = i
			agentclass = agent(agentnum, self.xcoord, self.ycoord, self.obstacles)
			self.agents.append(agentclass)
			self.neighboursdata[i].extend([agentclass.x, agentclass.y1, agentclass.w1, agentclass.y2, agentclass.w2])
			agentclass.getWeights()
			avg1 += agentclass.x
			avg2 += (agentclass.x)**2
			xwei.append(agentclass.x)

		x = np.zeros(self.numagents)				# Estimated eigenvector
		k1, k2, k3 = 0.12,0.02,0.004
		xp = np.linspace(0,100, 500)
		yp = []
		plt.figure(figsize=(20,10))
		ax = plt.axes()
		plt.xlim((-20,20))
		plt.ylim((-10,10))
		for k in range(500):
			self.xcoord[0] += 0.5					# One agent is moving according to an external law, and other agents are maintaining connectivity accordingly. 
			self.ycoord[1] -= 0.5
			for i in range(self.numagents):
				agentclass = self.agents[i]
				agentclass.getWeights()
				agentclass.getNeighbours(self.neighboursdata)
				d = (agentclass.A).dot(np.transpose(np.ones(self.numagents)))
				D = np.diag(d)
				L = D - agentclass.A
				avg1 = 0.0
				avg2 = 0.0
				for p in range(self.numagents):
					avg1 += xwei[p]
					avg2 += xwei[p]**2
				xdot = -k1*(avg1/self.numagents)*np.ones(self.numagents) - k2*L*x - k3*(avg2/self.numagents - 1)*x
				x += xdot[i]*0.015
				xwei[i] = x[i]
				uk = np.array([0.0,0.0])
				for j in agentclass.N:
					uk += -agentclass.A[i][j]*((x[i] - x[j])**2)*(np.array([self.xcoord[i], self.ycoord[i]]) - np.array([self.xcoord[j], self.ycoord[j]]))/(self.sigma**2)
				if i==0:
					uk[0], uk[1] = 0.0,0.0
				if i != 0 and i != 1:							# Update the graph only for those points using the control input that are not being controlled. 
					self.xcoord[i] += uk[0]*0.1
					self.ycoord[i] += uk[1]*0.1
				plt.clf()
				for i in range(len(self.xcoord)):	# Plotting code
					for j in range(len(self.ycoord)):
						xd = [self.xcoord[int(i)], self.xcoord[int(j)]]
						yd = [self.ycoord[int(i)], self.ycoord[int(j)]]
						if math.sqrt((xd[0]-xd[1])**2 + (yd[0]-yd[1])**2) < 4.0:
							plt.plot(xd,yd, '--',color='g')
						plt.scatter(self.xcoord[int(i)], self.ycoord[int(i)], marker='o', color = 'r')
					plt.annotate("{}".format(i), (self.xcoord[i], self.ycoord[i]))
				plt.xlim([-5, 15])
				plt.ylim([-5, 5])
				plt.show(block=False)
				plt.pause(0.1)
			avg1 = 0.0
			avg2 = 0.0
			yp.append(max(x))
		# plt.plot(xp, yp)
		# plt.xlabel("time")
		# plt.ylabel("Fiedler Eigenvector")
		# plt.title("Convergence of Eigenvector estimate")
		plt.show()

if __name__=='__main__':
	mainthread = main()
	mainthread.mainthread()
