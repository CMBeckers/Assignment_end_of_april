import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import argparse
import math


class Node:
    """
    This class holds the information of each node: That would be the nodes' index, value, connections
    with other nodes, and also the neighbours, the children as well as the parents of the current node.
    """
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
    """
    This class holds the functions to create the different networks and also the function to the plot and visualise
    the networks created.
    """
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
                            continue # represents the same node
                        else: # checking neighbours of inital nodes neighbours
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
            while node_to_check.parent: # to back propagate until node has no parent which represents start node
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
    def make_ring_network(self, NR, neighbour_range=1):
        """
        This function creates a ring network of size NR nodes with each node connected to its immediate neighbours on
        each side. Like the random network function, it uses lists to represent the connections between each node and
        its immediate neighbours, where zeroes represent no connection and ones represent a connection. Here again, the
        position of each element in the lists represent the node index, therefore, the positioning stays consistent.

        Input: self (current instance of the class), number of nodes NR, neighbour_range (here indicating the immediate
        neighbours)
        Output: ring network composed of lists, describing the connections of each individual node to other nodes
        """

        self.nodes = []

        #for loop iterates over each node and makes a connections list for each node with no connections to other nodes
        for node_index in range(NR):
            value = np.random.random()
            connections = [0 for _ in range(NR)] #no connections are indicated as only zeroes in the list
            #new node object is created storing the value, index and connections of each node
            self.nodes.append(Node(value, node_index, connections))

        #first for loop iterates over each node in the network and accesses its index
        for (index, node) in enumerate(self.nodes):
            #nested for loop determines the neighbours of the current node but also iterates over itself
            for offset in range(-neighbour_range, neighbour_range + 1):
                if offset != 0 : #Skip connecting a node to itself
                    neighbor_index = (index + offset) % NR #calculates neigbouring nodes indices
                    node.connections[neighbor_index] = 1 #updates connection list by making neighbouring zeroes to ones
                    self.nodes[neighbor_index].connections[index] = 1 #updates connection lists of all appropriate nodes


    #creates a small world network of size N nodes, with a default re-wiring probability 0.2
    def make_small_world_network(self, N, rewiring_prob=0.2):
        """
        This function uses the ring network function to create a small world network of size N nodes with a default
        probability of 0.2 to re-wire the connections between the nodes.
        Input: number of nodes N, re-wiring probability between nodes (with a default value of 0.2 unless different
        value is parsed)
        Input: self (current instance of the class), number of nodes N, rewiring_pron (here indicating the probability
        that each individual connections will be re-wired (default probability set to 0.2))
        Output: small world network composed of lists, describing the connections of each individual node to other nodes
        """
        self.make_ring_network(N)  #create base structure for network with make_ring_network function of size N nodes

        #iterates over each node in network and stores node information in the variable node
        for index in range(len(self.nodes)):
            node = self.nodes[index]
            #iterates over connection list of current node and stores indices of nodes it is connected to in new list
            connection_indexes = [indx for indx in range(N) if node.connections[indx] == 1]
            for connection_index in connection_indexes:
                if np.random.random() < rewiring_prob:
                    #when if statement holds, the connection at hand is removed and the connections list is updated
                    node.connections[connection_index] = 0
                    self.nodes[connection_index].connections[index] = 0

                    #randomly selects new node to connect to besides itself and the nodes already connected to
                    random_node = np.random.choice([indx for indx in range(N) if indx != index and indx not in connection_indexes])
                    #connects current node to randomly selected node and updates connections list
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
        plt.pause(1)
        plt.close()

'''
code for task 1
'''
def get_ops(population, i, j):
    """
    Function to find the neighbours above, below, to the left and to the
    right of each cell. The code makes sure to wrap around the grid to
    make sure cells on the edge still have four neighbours
    :param population: Grid representing the population of people
    :param i: Represents the horizontal position of a cell
    :param j: Represents the vertical position of a cell
    :return: List of opinions of the surrounding neighbours of each cell
    """
    x, y = population.shape
    neighbour_opinions = []
    neighbour_opinions.append(population[i-1, j])
    neighbour_opinions.append(population[(i+1)%x, j])
    neighbour_opinions.append(population[i, (j+1)%y])
    neighbour_opinions.append(population[i, j-1])
    return neighbour_opinions


def calculate_agreement(population, row, col, external=0):
    """
    Function to calculate the level of agreement between a cell and
    its neighbours
    :param population: Grid representing the population of people
    :param row: Row of the grid
    :param col: Column of the grid
    :param external: External influence on a cells voting opinion
    :return: Total sum of the agreement of a cells neighbours
    """
    current_value = population[row, col]
    total_agreement = 0
    opinion_list = get_ops(population, row, col)
    for opinion in opinion_list:
        total_agreement += current_value * opinion
    total_agreement += external * current_value
    return total_agreement


def ising_step(population, alpha=1, external=0):
    """
    Performs a single update of the ising function by choosing a
    random cell and updating its value based on the calculation of
    the agreement of its neighbours
    :param population: Grid representing the population of people
    :param external: External influence on a cells voting opinion
    :param alpha: Tolerance of those who disagree with their neighbours within the society
    """
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


def plot_ising(im, population):
    '''
    Displays the ising model
    '''
    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)

def run_ising_simulation(population, num_steps=100, external=0, alpha=1):
    """
    Runs the Ising simulation for a specified number of steps and updates the plot
    :param population: Initial grid representing the population of people
    :param num_steps: Number of simulation steps to run
    :param external: External influence on a cell's voting opinion
    :param alpha: Tolerance of those who disagree with their neighbours within the society
    """
    fig, ax = plt.subplots()
    im = ax.imshow(population, cmap='gray', vmin=-1, vmax=1)

    for step in range(num_steps):
        population_copy = np.copy(population)  # Create a copy of the population grid
        ising_step(population_copy, external=external, alpha=alpha)  # Perform one step of Ising model
        plot_ising(im, population_copy)  # Update the plot with the new population grid copy
        plt.pause(0.1)  # Pause to display the updated plot

        # Update the original population grid after each step
        population[:] = population_copy
        plt.pause(0.1)

'''
Code for task 2
'''
def spawn(num_people):
    return np.random.rand(num_people)

def update(opinion,beta,threshold,iterations):
    '''
    updates opinion of random i, j in grid based on neighbours opinions
    :return: opinion changes of all elements of the grid
    '''
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
    '''
    plots opinions over time
    '''

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
    num_people = 100
    iterations = 10000
    opinion_change = update(spawn(num_people), beta, threshold,iterations)
    plot_opinion(opinion_change,iterations,beta,threshold)

'''
code for task 5
'''
def spawning(num_people):
    return np.random.rand(num_people)

def mean(list):
    total = 0
    for item in list:
        total += item  # all items will be of type int
    return total / len(list)

def updating(network, beta, threshold, iterations):
    '''
    Function that updated the opinions of nodes based on the opinions of its neighbours
    :return: opinion change of nodes over time
    '''
    opinion_change = []

    for i in range(iterations):
        opinion_snapshot = []

        for node in network.nodes:
            try: # some nodes may not have neighbours
                neighbour_index = random.choice(node.get_neighbours())
                neighbour = network.nodes[neighbour_index]
                difference = node.value - neighbour.value

                if abs(difference) < threshold:
                    node.value += beta * (neighbour.value - node.value)
                    neighbour.value += beta * (node.value - neighbour.value)

                opinion_snapshot.append(node.value)

            except IndexError: # append original opinion
                opinion_snapshot.append(node.value)

        opinion_change.append(opinion_snapshot)

        if i % 10 == 0:
            network.plot()  # Display network plot every 10 iterations
    plot_opinions_over_time(opinion_change)
    return opinion_change


def plot_opinions(opinion_change, iterations, beta, threshold):
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
    plt.pause(1)

def plot_opinions_over_time(opinions_over_time):
    '''
    plots the means of the opinions at each time step over time
    returns a graph of the change in opinions over time
    '''
    means = [mean(opinion) for opinion in opinions_over_time]
    time = []
    for value in range(0, len(opinions_over_time)):
        time.append(value)
    plt.plot(time, means)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Opinion')
    plt.title('Mean Opinion over Time')
    plt.show()
def defuant_main_task_5(network_size, beta, threshold):
    iterations = 100
    rewiring_prob = random.randint(1, 10)/10
    network = Network()
    network.make_small_world_network(network_size, rewiring_prob)
    opinion_change = updating(network, beta, threshold, iterations)
    plot_opinions_over_time(opinion_change)


def test_defuant_task_5():
    network_size = 100
    defuant_main_task_5(network_size, 0.5,0.5)
    defuant_main_task_5(network_size, 0.1, 0.5)
    defuant_main_task_5(network_size,0.5, 0.1)
    defuant_main_task_5(network_size, 0.1, 0.2)

def test_defuant():
    print("Testing defuant model")
    assert update([0.45, 0.55], 0.2, 0.2, 1) == [[0.466, 0.53]] or [[0.47000000000000003, 0.534]], "defuant 1"
    assert update([0.05, 0.5], 0.5, 0.5, 1)==[[0.16250000000000003, 0.275]] or [[0.275, 0.3875]], "defuant 2"
    assert update([0.6, 0.2], 0.5, 0.5, 1)==[[0.4, 0.30000000000000004]] or [[0.5, 0.4]], "defuant 3"
    print("Tests passed")



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
def argparsing():
    parser = argparse.ArgumentParser()

    parser.add_argument("-network", type=int, nargs='?', const=10, default=False,
                        help='Number of nodes in a random network (default: 10)')
    parser.add_argument("-test_network", action="store_true", default=False)
    parser.add_argument('-small_world', dest='small_world', type=int, nargs='?', const=10, default=False,
                        help='Number of nodes in a small-world network (default: 10)')
    parser.add_argument('-re-wire', dest='re_wire', metavar='p', type=float, default=0.2,
                        help='Rewiring probability for small-world network (default: 0.2)')
    parser.add_argument("-ring_network", type=int, nargs='?', const=10, default=False,
                        help='Number of nodes in a ring network (default: 10)')
    parser.add_argument("-defuant", default=False ,action="store_true",
                        help='Run the defuant model with optional parameters: -beta <value>, -threshold <value>')
    parser.add_argument("-beta", type=float, nargs='?', default=0.2)
    parser.add_argument("-threshold", type=float, nargs='?', default=0.2)
    parser.add_argument("-test_defuant", action='store_true',default=False,
                        help='Run test functions for the defuant model')
    parser.add_argument("-test_ising", action='store_true')
    parser.add_argument("-ising_model", default=False, action = "store_true")
    parser.add_argument("-external", type=float, nargs='?', default=False, const=0)
    parser.add_argument("-alpha", type=float, nargs='?', default=1)
    parser.add_argument("-use_network", default=False, nargs='?', const=100, type=int)

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
        network.plot()

    if args.test_network:
        test_networks()

    if args.small_world:
        network_size = args.small_world
        re_wire_size = args.re_wire
        network = Network()
        network.make_small_world_network(network_size, re_wire_size)
        print("Small-World Network:")
        print(network.get_mean_path_length())
        print(network.get_mean_clustering())
        print(network.get_mean_degree())
        network.plot()

    if args.ring_network:
        ring_network_size = args.ring_network
        network = Network()
        network.make_ring_network(ring_network_size)
        print("Ring Network:")
        print(network.get_mean_path_length())
        print(network.get_mean_clustering())
        print(network.get_mean_degree())
        network.plot()

    if args.defuant:
        if args.use_network:
            defuant_main_task_5(args.use_network, args.beta, args.threshold)
        else:
            defuant_main(args.beta, args.threshold)

    if args.ising_model:
        num_steps = 100
        population = np.random.choice([-1, 1], size=(100, 100))
        run_ising_simulation(population, num_steps, args.external, args.alpha)

    if args.test_defuant:
        test_defuant()

    if args.test_ising:
        test_ising()



if __name__ == "__main__":
    main()

