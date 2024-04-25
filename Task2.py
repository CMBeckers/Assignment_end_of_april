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

def defuant_main(beta, threshold):
    num_people = 2000
    iterations = 50000
    opinion_change = update(spawn(num_people), beta, threshold,iterations)

def test_defuant():
    defuant_main(0.5,0.1)
    defuant_main(0.5, 0.1)
    defuant_main(0.5, 0.1)
    defuant_main(0.5, 0.1) #check threshold values they asked to put in on assignment

def main():
    #need to put something to do with arg

if __name__ == "__main__":
    main()