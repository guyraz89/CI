import numpy as np

class Population:

    population_list = []
    board_size = 0
    pop_size = 0
    max_evals = 0


    ''' Initlize all paremeters for the algorithm '''
    def __init__(self, populationSize, boardSize, evals):
        self.population_list = []
        self.board_size = boardSize
        self.pop_size = populationSize
        self.max_evals = evals


    ''' fitness function get a specific board with queens already spotted in their places
        and calculate it's value by counting how many queens are on the same line.
        the fitness return value can accumulate at most as the number of queens on the given board.
    '''
    def fitness(self, Phenotype):


    ''' decode function transform Genome description of queens permutation into matrix of {0, 1}
        1 means there is a queen in the [i,j] address otherwise 0.
    '''
    def decode(self, g):


    ''' select the best 2 individual in the population (by fitness) to be the parents of the next generation
        then select the next best 2 again and again
    '''
    def select(self, pop):

    ''' display the chess board at the end of the algorithm '''
    def display(self):




    ''' The GA algorithm get for NQueens problem '''
    def GA(self, n, max_evals, decodefct, selectfct, fitnessfct, seed=None) :
        eval_cntr = 0
        history = []
        #
        # GA params
        mu = 1000
        # Probability of Crossover
        pc = 0.37
        # Probability of Mutation
        pm = 4 / n
        #    kXO = 1 # 1-point Xover
        local_state = np.random.RandomState(seed)
        Genome = local_state.randint(2, size=(n, mu))
        Phenotype = []
        for k in range(mu):
            Phenotype.append(decodefct(Genome[:, [k]]))
            fitness = fitnessfct(Phenotype)
        eval_cntr += mu
        fcurr_best = fmax = np.max(fitness)
        xmax = Genome[:, [np.argmax(fitness)]]
        history.append(fmax)

        while (eval_cntr < max_evals):




        return xmax, fmax, history




if __name__ == 'main':

    population8 = Population(20,8,10**5)
    population16 = Population(20,16,10**5)
    xMax, fMax, hist = population8.GA()
