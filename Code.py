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
            if neighbour.value < self.value:
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
        return self.queue.pop()
    def empty(self):
        return len(self.queue)==0

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

    def get_mean_degree(self, nodes):
        '''
        This function returns the mean degree of a network
        The degree of a node is the number of edges entering that node
        Returns the mean degree of every node in the network
        '''
        degrees = []
        for node in nodes:
            degree = 0
            connections = np.where(np.array(node.connections) == 1)[0]
            degree = sum(1 for connection in connections if connection != 0)
            degrees.append(degree)
        return Network.mean(self, degrees)

    def get_mean_clustering(self, Network):
        coefficient_per_node = []
        '''
        This function calculated the mean clustering coefficient of all the nodes in the network
        Clustering means how many neighbours of a node are connected to each other and divides this by the number of possible connections 
        Returns the mean clustering coefficient 
        '''
        for node in Network.nodes:
            clustering = []
            visited = []
            neighbour_index = node.get_neighbours()
            number_of_neighbours = len(neighbour_index)
            if number_of_neighbours == 0 or number_of_neighbours == 1: # clustering requires two neighbours.
                continue
            else:
                for index_1 in neighbour_index:
                    for index_2 in neighbour_index:
                        neighbour_1 = Network.nodes[index_1]
                        neighbour_2 = Network.nodes[index_2]
                        neighbour_1_index = neighbour_1.get_neighbours()
                        neighbour_2_index = neighbour_2.get_neighbours()
                        if index_1 in neighbour_2_index or index_2 in neighbour_1_index:
                            if [node.value, index_1, index_2] in visited:
                                continue
                            else:
                                clustering.append(1)
                                visited.append([node.value, index_1, index_2])
                possible_connections = (number_of_neighbours**2 - number_of_neighbours)/2
                coefficient_per_node.append((len(clustering))/possible_connections)
        return sum(coefficient_per_node)/len(coefficient_per_node)


    def Breadth_First_Search(self, start, target, network):
        self.start_node = start
        self.target_node = target
        self.search_queue = queue()
        self.search_queue.push(self.start_node)
        visited = []

        while not self.search_queue.empty():
            node_checking = self.search_queue.pop()
            if node_checking == self.target_node:
                break
            for neighbour_index in node_checking.get_neighbours():
                neighbour = network.nodes[neighbour_index]
                if neighbour_index not in visited:
                    self.search_queue.push(neighbour)
                    visited.append(neighbour_index)
                    neighbour.parents = node_checking

        node_checking = self.target_node
        self.start_node.parents = None
        route = []
        while node_checking.parents:
            route.append(node_checking)
            node_checking = node_checking.parents
        route.append(node_checking)
        return len(route)

    def get_mean_path_length(self, Network):
        lengths = []
        for value_1 in range(0, len(Network.nodes)-1):
            for value_2 in range(value_1+1, len(Network.nodes)):
                route_length = self.Breadth_First_Search(Network.nodes[value_1], Network.nodes[value_2], Network)
                lengths.append(route_length)
        return sum(lengths)/len(lengths)

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
            for neighbour_index in range(index+1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    # def make_ring_network(self, N, neighbour_range=1):
    # Your code  for task 4 goes here

    # def make_small_world_network(self, N, re_wire_prob=0.2):
    # Your code for task 4 goes here

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-network", action = "store_true", default=False)
    parser.add_argument("-integer", type=int, default=10)
    args = parser.parse_args()
    if args.network:
        show_network = True
    if args.integer:
        network_size = args.integer
    return show_network, network_size

def main():
    show_network, network_size = argparsing()
    if show_network:
        prob = random.randint(1, 10) / 10
        network = Network()
        network.make_random_network(network_size, prob)
        print(network.get_mean_path_length(network))
        print(network.get_mean_clustering(network))
        print(network.get_mean_degree(network.nodes))




A = Network()
A.make_random_network(10, 0.8)
A.plot()


if __name__ == "__main__":
    main()














