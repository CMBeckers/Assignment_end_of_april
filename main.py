import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import random
import argparse
import math


class Node:

    def __init__(self, value, number, connections=None):
        self.index = number
        self.connections = connections
        self.value = value
        self.children = []
        self.parents = []

    def get_neighbours(self):
        return np.where(np.array(self.connections) == 1)[0]

    def get_connections(self, network):
        '''
        function that returns the nodes connected to self.node when the network is passed as an argument
        '''
        neighbours = []
        neighbour_indices = self.get_neighbours()
        for index in neighbour_indices:
            neighbour_node = network.nodes[index]
            neighbours.append(neighbour_node)
        return neighbours

    def parents(self, connections):
        '''
        Parent neighbour is a node which has a value larger than the current node
        returns parent nodes by comparing values of neighbouring nodes to current node
        '''
        parents = []
        for neighbour in connections:
            if neighbour.value > self.value:
                parents.append(neighbour)
        return parents

    def children(self, connections):
        '''
        Child neighbour is a node which has a value larger than the current node
        returns children nodes by comparing values of neighbouring nodes to current node
        '''
        children = []
        for neighbour in connections:
            if neighbour.value < self.value:
                children.append(neighbour)
        return children



class queue:
    def __init__(self):
        self.queue = []

    def push(self, item):
        self.queue.append(item)

    def pop(self):
        return self.queue.pop(0)

    def empty(self):
        return len(self.queue) == 0


class Network:

    def __init__(self, nodes=None):
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def mean(self, list):
        total = 0
        for item in list:
            total += item  # all items will be of type int
        return total / len(list)

    def get_mean_degree(self):
        '''
        This function returns the mean degree of a network
        The degree of a node is the number of edges entering that node
        Returns the mean degree of every node in the network
        '''
        degrees = []
        for node in self.nodes:
            degree = sum(node.connections)  # Assuming node.connections contains 0/1 values representing connections
            degrees.append(degree)
        if len(degrees) > 0:
            return sum(degrees) / len(degrees)
        else:
            return 0

    def get_mean_clustering(self):
        '''
        mean clustering describes if the neighbours of a node connect to each other and form a triangle
        return the mean clustering by dividing the cluster coefficient for each node by the length of the list "coefficient_per_node"
        '''
        coefficient_per_node = []
        for node in self.nodes:
            clustering = []
            visited = set()
            neighbour_indices = node.get_neighbours()
            number_of_neighbours = len(neighbour_indices)
            if number_of_neighbours < 2: # two neighbours required for potential clustering
                coefficient_per_node.append(0)
                continue
            else:
                for index_1 in neighbour_indices:
                    for index_2 in neighbour_indices:
                        if index_1 == index_2:
                            continue
                        else:
                            neighbour_1 = self.nodes[index_1]
                            neighbour_2 = self.nodes[index_2]
                            neighbour_1_indices = neighbour_1.get_neighbours()
                            neighbour_2_indices = neighbour_2.get_neighbours()
                        if (index_1, index_2) in visited or (index_2, index_1) in visited:
                            continue
                        else:
                            if index_1 in neighbour_2_indices or index_2 in neighbour_1_indices:
                                clustering.append(1)
                                visited.add((index_1, index_2))
                possible_connections = (number_of_neighbours ** 2 - number_of_neighbours) / 2
                coefficient_per_node.append(len(clustering) / possible_connections)
        return self.mean(coefficient_per_node)

    def Breadth_First_Search(self, start, target):
        '''
        Searches through a graph or tree structure
        returns the distance between the start and target node
        '''
        self.start_node = start
        self.goal = target
        self.search_queue = queue()
        self.search_queue.push(self.start_node)
        visited = []
        while not self.search_queue.empty():
            node_to_check = self.search_queue.pop()
            if node_to_check == self.goal:
                break
            for neighbour_index in node_to_check.get_neighbours():
                neighbour = self.nodes[neighbour_index]
                if neighbour_index not in visited:
                    self.search_queue.push(neighbour)
                    visited.append(neighbour_index)
                    neighbour.parent = node_to_check
        route = 0
        if node_to_check == self.goal:
            self.start_node.parent = None
            while node_to_check.parent:
                node_to_check = node_to_check.parent
                route += 1
        return route

    def get_mean_path_length(self):
        '''
        Finds the path lengths from one node to every other node in the graph
        The mean of these lengths is stores and calculated for every node in the graph
        The mean of these means is returned
        '''
        lengths = []
        means = []
        for value_1 in range(0, len(self.nodes)-1):
            lengths.append([])
            for value_2 in range(0, len(self.nodes)):
                if value_2 == value_1:
                    continue
                else:
                    route_length = self.Breadth_First_Search(self.nodes[value_1], self.nodes[value_2])
                    lengths[value_1].append(route_length)
        for length in lengths:
            print(length)
            if len(length) > 0:
                means.append((sum(length)) / (len(length)))
        return self.mean(means)

    def make_random_network(self, N, connection_probability):
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
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1
    def make_ring_network(self, N, neighbour_range=1):
        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for offset in range(-neighbour_range, neighbour_range + 1):
                if offset != 0:  # Skip connecting a node to itself
                    neighbor_index = (index + offset) % N
                    node.connections[neighbor_index] = 1
                    self.nodes[neighbor_index].connections[index] = 1

    def make_small_world_network(self, N, rewiring_prob=0.2):
        self.make_ring_network(N)  # Call make_ring_network without passing self

        for index in range(len(self.nodes)):
            node = self.nodes[index]
            connection_indexes = [indx for indx in range(N) if node.connections[indx] == 1]
            for connection_index in connection_indexes:
                if np.random.random() < rewiring_prob:
                    node.connections[connection_index] = 0
                    self.nodes[connection_index].connections[index] = 0

                    random_node = np.random.choice([indx for indx in range(N) if indx != index and indx not in connection_indexes])
                    self.nodes[random_node].connections[index] = 1
                    node.connections[random_node] = 1

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')
        plt.show()

'''
code for task 1
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
Code for task 2
'''
def spawn(num_people):
    return np.random.rand(num_people)

def update(opinion,beta,threshold,iterations):
    opinion_change = []
    for i in range(iterations):
        n = np.random.randint(len(opinion))
        if n == 0:
            neighbour = n + 1
        elif n == (len(opinion) - 1):
            neighbour = n - 1
        else:
            neighbour = (n+random.choice([-1,1]))
        difference = opinion[n] - opinion[neighbour]

        if abs(difference) < threshold:
            opinion[n] += (beta * (opinion[neighbour] - opinion[n]))
            opinion[neighbour] += (beta * (opinion[n] - opinion[neighbour])) #most important part so far
        opinion_change.append(opinion.copy()) #gives you a copy of the same list, not same as deep copy (compound list)
    return opinion_change



def plot_opinion(opinion_change, iterations, beta, threshold):
    fig = plt.figure()
    #first sublot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(opinion_change[-1], bins=10)
    ax1.set_xlabel('Opinion')
    ax1.set_ylabel('Number')
    ax1.set_xticks(np.arange(0, 1.1, 0.2))
    #second subplot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(range(iterations), opinion_change, 'ro')
    ax2.set_ylabel('Opinion')
    ax2.set_xlabel('Iteration')
    fig.suptitle(f'Coupling: {beta}, Threshold: {threshold}')
    plt.tight_layout()
    plt.show()

def defuant_main(beta, threshold):
    num_people = 2000
    iterations = 50000
    opinion_change = update(spawn(num_people), beta, threshold,iterations)

def test_defuant():
    defuant_main(0.5,0.1)
    defuant_main(0.5, 0.1)
    defuant_main(0.5, 0.1)
    defuant_main(0.5, 0.1) #check threshold values they asked to put in on assignment


def argparsing():
    parser = argparse.ArgumentParser()

    parser.add_argument("-network", type=int, nargs='?', const=10, default=False,
                        help='Number of nodes in a random network (default: 10)')
    parser.add_argument('-small_world', dest='small_world', type=int, nargs='?', const=10, default=False,
                        help='Number of nodes in a small-world network (default: 10)')
    parser.add_argument('-re-wire', dest='re_wire', metavar='p', type=float, nargs='?', const=0.2, default=False,
                        help='Rewiring probability for small-world network (default: 0.2)')
    parser.add_argument("-ring_network", type=int, nargs='?', const=10, default=False,
                        help='Number of nodes in a ring network (default: 10)')
    parser.add_argument("-defuant", default=False, nargs='?',
                        help='Run the defuant model with optional parameters: -beta <value>, -threshold <value>')
    parser.add_argument("-beta", type=float, nargs='?', default=False, const=10)
    parser.add_argument("-threshold", type=float, nargs='?', const=0.2, default=False)
    parser.add_argument("-test_defuant", action='store_true',
                        help='Run test functions for the defuant model')

    args = parser.parse_args()

    return args

def main():
    args = argparsing()

    if args.network:
        network_size = args.network
        prob = random.randint(1, 10) / 10
        network = Network()
        network.make_random_network(network_size, prob)
        print("Random Network:")
        print(network.get_mean_path_length())
        print(network.get_mean_clustering())
        print(network.get_mean_degree())

    if args.small_world:
        network_size = args.small_world
        re_wire_size = args.re_wire
        network = Network()
        network.make_small_world_network(network_size, re_wire_size)
        print("Small-World Network:")
        print(network.get_mean_path_length())
        print(network.get_mean_clustering())
        print(network.get_mean_degree())

    if args.ring_network:
        ring_network_size = args.ring_network
        network = Network()
        network.make_ring_network(ring_network_size)
        print("Ring Network:")
        print(network.get_mean_path_length())
        print(network.get_mean_clustering())
        print(network.get_mean_degree())

    if args.defuant:
        defuant_main(args.beta, args.threshold)



    if args.test_defuant:
        test_defuant()




def test_networks():
    # Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number - 1) % num_nodes] = 1
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert (network.get_mean_degree() == 2), network.get_mean_degree()
    assert (network.get_mean_clustering() == 0), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 2.777777777777778), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert (network.get_mean_degree() == 1), network.get_mean_degree()
    assert (network.get_mean_clustering() == 0), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 5), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert (network.get_mean_degree() == num_nodes - 1), network.get_mean_degree()
    assert (network.get_mean_clustering() == 1), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 1), network.get_mean_path_length()

    print("All tests passed")


if __name__ == "__main__":
    test_networks()
    test_defuant()
    test_ising()
    main()

