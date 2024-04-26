import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

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
        self.nodes = []
        for node_index in range(NR):
            value = np.random.random()
            connections = [0 for _ in range(NR)]
            self.nodes.append(Node(value, node_index, connections))

        for (index, node) in enumerate(self.nodes):
            for offset in range(-neighbour_range, neighbour_range + 1):
                if offset != 0 : #Skip connecting a node to itself
                    neighbor_index = (index + offset) % NR
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
    assert (network.get_clustering() == 0), network.get_clustering()
    assert (network.get_path_length() == 2.777777777777778), network.get_path_length()

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
    assert (network.get_clustering() == 0), network.get_clustering()
    assert (network.get_path_length() == 5), network.get_path_length()

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
    assert (network.get_clustering() == 1), network.get_clustering()
    assert (network.get_path_length() == 1), network.get_path_length()

    print("All tests passed")


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-ring_network', dest='ring', metavar='NR', type=int, default = 0, help='number N of nodes in ring network')
    parser.add_argument('-small_world', dest='smallworld', metavar='N', type=int, default=10, help='number N of nodes in small-world network')
    parser.add_argument('-re-wire', dest='rewire', metavar='p', type=float, default=0.2, help='value p of rewiring probability')

    args = parser.parse_args()

    NR = args.ring
    N = args.smallworld
    rewiring_prob = args.rewire

    network = Network()  # Initialize an empty network

    if NR > 0:
        # Create a ring network with NR nodes
        network.make_ring_network(NR)
        print(f"Creating ring network with {NR} nodes")
    elif N > 0:
        # Create a small-world network with N nodes and rewiring probability rewiring_prob
        network.make_small_world_network(N, rewiring_prob)
        print(f"Creating small-world network with {N} nodes and rewiring probability {rewiring_prob}")

    # Plot the generated network
    network.plot()
    plt.show()  # Display the plot
    # Parameters
    N = args.smallworld  # Number of nodes
    rewiring_prob = args.rewire  # Probability of rewiring each edge
    NR = args.ring

# You should write some code for handling flags here

if __name__ == "__main__":
    main()
