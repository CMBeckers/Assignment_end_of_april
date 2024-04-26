import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import random

class Node:

	def __init__(self, value, number, connections=None):

		self.index = number
		self.connections = connections
		self.value = value

class Network: 

	def __init__(self, nodes=None):

		if nodes is None:
		    self.nodes = []
		else:
		    self.nodes = nodes

	def get_mean_degree(self):
		#Your code  for task 3 goes here
		pass

	def get_mean_clustering(self):
		#Your code for task 3 goes here
		pass

	def get_mean_path_length(self):
		#Your code for task 3 goes here
		pass

	def make_random_network(self, N, connection_probability=0.5):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	def make_ring_network(self, N, neighbour_range=1):
		#Your code  for task 4 goes here
		pass

	def make_small_world_network(self, N, re_wire_prob=0.2):
		#Your code for task 4 goes here
		pass
	def plot(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def get_ops(population, i, j):
	x, y = population.shape
	neighbour_opinions = []
	neighbour_opinions.append(population[i-1, j])
	neighbour_opinions.append(population[(i+1)%x, j])
	neighbour_opinions.append(population[i, (j+1)%y])
	neighbour_opinions.append(population[i, j-1])
	return neighbour_opinions

'''
Function fo find the neighbours above, below, to the left and to the 
right of each cell. The code makes sure to wrap around the grid to 
make sure cells on the edge still have four neighbours 
'''

def calculate_agreement(population, row, col, external=0.0):
	current_value = population[row, col]
	total_agreement = 0
	opinion_list = get_ops(population, row, col)
	for opinion in opinion_list:
		total_agreement += current_value * opinion
	total_agreement += external * current_value
	return total_agreement

'''
Function to calculate the level of agreement between a cell and 
its neighbours
'''

def ising_step(population, external=0.0, alpha=10):
	n_rows, n_cols = population.shape
	row = np.random.randint(0, n_rows)
	col = np.random.randint(0, n_cols)
	agreement = calculate_agreement(population, row, col, external=0.0)

	if agreement < 0:
		population[row, col] *= -1
	elif alpha:
		p = math.e ** (-agreement / alpha)
		if random.random() < p:
			population[row, col] *= -1

'''
Performs a single update of the ising function by choosing a 
random cell and updating its value based on the calculation of 
the agreement of its neighbours
'''

def plot_ising(im, population):
	new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
	im.set_data(new_im)
	plt.pause(0.1)

'''
Displays the ising model
'''

def test_ising():
	'''
	This function will test the calculate_agreement function in the Ising model
	'''

	print("Testing ising model calculations")
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1)==4), "Test 1"

	population[1, 1] = 1.
	assert(calculate_agreement(population,1,1)==-4), "Test 2"

	population[0, 1] = 1.
	assert(calculate_agreement(population,1,1)==-2), "Test 3"

	population[1, 0] = 1.
	assert(calculate_agreement(population,1,1)==0), "Test 4"

	population[2, 1] = 1.
	assert(calculate_agreement(population,1,1)==2), "Test 5"

	population[1, 2] = 1.
	assert(calculate_agreement(population,1,1)==4), "Test 6"

	"Testing external pull"
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1,1)==3), "Test 7"
	assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
	assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
	assert(calculate_agreement(population,1,1, -10)==14), "Test 10"

	print("Tests passed")


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

def defuant_main():
	#Your code for task 2 goes here
	pass
def test_defuant():
	#Your code for task 2 goes here
	pass

'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
	#You should write some code for handling flags here
	parser = argparse.ArgumentParser()
	parser.add_argument("-ising_model", action="store_true")
	parser.add_argument("-external", type=float, default=0)
	parser.add_argument("-alpha", type=float, default=1)
	parser.add_argument("-testing_ising", action="store_true")

	args = parser.parse_args()
	external = args.external
	alpha = args.alpha

	if args.ising_model:
		population = np.random.choice([-1, 1], size=(100, 100))
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()
		im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

		# Iterating an update 100 times
		for frame in range(100):
			# Iterating single steps 1000 times to form an update
			for step in range(1000):
				ising_step(population, alpha, external)
			print('Step:', frame, end='\r')
			plot_ising(im, population)

	if args.test_ising():
		test_ising()

if __name__=="__main__":
	main()