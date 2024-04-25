import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import random
import argparse


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

    def make_small_world_network(self, N, re_wire_prob=0.2):
        self.make_ring_network(self, N, neighbour=1)

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


def argparsing():
    show_network = False
    small_world = False
    re-wire = False
    ring_network = False
    parser = argparse.ArgumentParser()
    parser.add_argument("-network", action="store_true", default=False)
    parser.add_argument("-integer", type=int, default=10)
    parser.add_argument('-small_world', dest='smallworld', metavar='N', type=int, default=10,
                        help='number N of nodes in small-world network')
    parser.add_argument('-re-wire', dest='rewire', metavar='p', type=float, default=0.2,
                        help='value p of rewiring probability')
    parser.add_argument("-ring_network", type=int, default=10)
    args = parser.parse_args()
    if args.network:
        show_network = True
    if args.integer:
        network_size = args.integer
    if args.small_world:
        small_world = True
    if args.re-wire:
        re-wire = True
    if args.ring_network:
        ring_network = True
    return show_network, network_size, small_world, re-wire


def main():
    show_network, network_size, small_world, re-wire = argparsing()
    if show_network:
        prob = random.randint(1, 10) / 10
        network = Network()
        network.make_random_network(network_size, prob)
        print(network.get_mean_path_length(network))
        print(network.get_mean_clustering(network))
        print(network.get_mean_degree(network.nodes))
    if small_world:
        if re-wire: # using specific rewiring value
            network = Network()
            network.make_small_world_network()
        else:
            continue
    if small_world: # using default rewire value
        network = Network()
        network.make_small_world_network()




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
    main()

