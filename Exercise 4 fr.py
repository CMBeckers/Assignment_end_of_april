def main():
    #initialise parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    #create flag for ring network with default value of NR = 0, so no ring network is created unless called
    parser.add_argument('-ring_network', dest='ring', metavar='NR', type=int, default = 0, help='number N of nodes in ring network')
    #create flag for small world network with default value of N = 10
    parser.add_argument('-small_world', dest='smallworld', metavar='N', type=int, default=10, help='number N of nodes in small-world network')
    #create flag for the re-wiring probability of the small world network with default value of p = 0.2
    parser.add_argument('-re_wire', dest='rewire', metavar='p', type=float, default=0.2, help='value p of rewiring probability')

    args = parser.parse_args()

    #assign parsed arguments to parameters in the functions
    NR = args.ring
    N = args.smallworld
    rewiring_prob = args.rewire

    #initialise an empty network
    network = Network()

    #initialise required functions depending on which arguments are parsed
    if NR > 0:
        #create a ring network with NR nodes
        network.make_ring_network(NR)
        print(f"Creating ring network with {NR} nodes")
    elif N > 0:
        #create a small-world network with N nodes and rewiring probability rewiring_prob
        network.make_small_world_network(N, rewiring_prob)
        print(f"Creating small-world network with {N} nodes and rewiring probability {rewiring_prob}")

    #plot network
    network.plot()
    # display the plot
    plt.show()

#ensure main function is executed
if __name__ == "__main__":
    main()
