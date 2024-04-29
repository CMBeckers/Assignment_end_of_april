import argparse
import numpy as np
import matplotlib.pyplot as plt
import random


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
    num_people = 100
    iterations = 10000
    opinion_change = update(spawn(num_people), beta, threshold,iterations)
    plot_opinion(opinion_change,iterations,beta,threshold)

def test_defuant():
    defuant_main(0.5,0.5)
    defuant_main(0.1, 0.5)
    defuant_main(0.5, 0.1)
    defuant_main(0.1, 0.2) #check threshold values they asked to put in on assignment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-beta", type=float, default=0.2,
                        help="Set Beta: . Default: 0.2")
    parser.add_argument("-threshold", type=float, default=0.2,
                        help="Set Threshold. Default: 0.2")
    parser.add_argument("-defuant",
                        help="Run Defuant Model.", action="store_true")
    parser.add_argument("-test_defuant",
                        help="Run Test Model.", action="store_true")
    args = parser.parse_args()

    if args.defuant:
        defuant_main(args.beta, args.threshold)
    if args.test_defuant:
        test_defuant()


if __name__ == "__main__":
    main()
