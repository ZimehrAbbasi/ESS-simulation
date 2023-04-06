# Class Structures

# Player: Class of a player
#      name: Name of the player
#      options: dictionary of options of player against different players (probability distribution)
#      position: location of the player on the board
#      fitness: fitness of the player chosen from a normal distribution meaned according to the type of the player
#      neighbour_radius: radius of the neighbourhood that the player can interact with

# Game: Class of a game
#      players: list of players
#      payoffs: dictionary of payoffs of player against different players
#      n: number of players
#      board: board of the game
#      costs: dictionary of costs of player for existing in the game
#      fitness_distributions: list of fitness distributions of players in the game

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Game:

    class Player:

        def __init__(self, type, options):
            self.type = type # Type of the player
            self.options = options # dictionary of options of player against different players (probability distribution)
            self.position = (0, 0) # position of the player in the game
            self.fitness = 0 # fitness of the player chosen from a normal distribution meaned according to the type of the player
            self.neighbour_radius = 1 # radius of the neighbourhood of the player

        # get the options of the player against the opponent
        def get_options(self, opponent):
            return self.options[opponent]

    def __init__(self, players, options, payoffs, fitness_mean, fitness_std, neighbourhoods, costs, board_size, fitness=None):
        self.players = [self.Player(player, option) for player, option in zip(players, options)] # list of players
        self.payoffs = payoffs # dictionary of payoffs of player against different players
        self.n = board_size # size of the board
        positions = set() # set of positions of players
        self.board = [[0 for i in range(self.n)] for j in range(self.n)] # board of the game
        self.costs = costs # dictionary of costs of player for existing in the game

        # set fitness of players
        for player in self.players:
            # if fitness is not given, then choose a random fitness from a normal distribution
            if not fitness:
                player.fitness = np.random.normal(fitness_mean[player.type], fitness_std[player.type])
            else:
                player.fitness = fitness[player.type]

        self.fitness_distributions = [[(player.fitness, player.type) for player in self.players]]

        # set positions of players (TODO: change to a better designation of positions, by region perhaps?)
        for player in self.players:
            while True:
                player.position = (np.random.randint(0, self.n), np.random.randint(0, self.n))
                if player.position not in positions:
                    positions.add(player.position)
                    self.board[player.position[0]][player.position[1]] = player
                    break

        # set neighbourhood raadius of players
        for player in self.players:
            player.neighbour_radius = np.random.normal(neighbourhoods[player.type], 1)

    # get the payoff matrix of the player against the opponent
    def get_payoff_matrix(self, player, opponent):
        return self.payoffs[player][opponent]
    
    # move the player to a random position within the neighbourhood
    def move_player_within_neighbourhood(self, player):
        # get the neighbourhood of the player
        neighbourhood = []
        for i in range(player.position[0] - player.neighbour_radius, player.position[0] + player.neighbour_radius + 1):
            for j in range(player.position[1] - player.neighbour_radius, player.position[1] + player.neighbour_radius + 1):
                if i >= 0 and i < self.n and j >= 0 and j < self.n:
                    neighbourhood.append((i, j))

        # choose a random position from the neighbourhood
        position = np.random.choice(neighbourhood)

        # move the player to the position
        self.board[player.position[0]][player.position[1]] = 0
        self.board[position[0]][position[1]] = player
        player.position = position
    
    # simulate the game for iter iterations
    def simulate(self, iter, games=1):

        for it in range(iter):
            # choose a player
            player = np.random.choice(self.players)

            # check if there is anybody in the neighborhood
            neighbours = []
            for i in range(player.position[0] - player.neighbour_radius, player.position[0] + player.neighbour_radius + 1):
                for j in range(player.position[1] - player.neighbour_radius, player.position[1] + player.neighbour_radius + 1):
                    if i >= 0 and i < self.n and j >= 0 and j < self.n and self.board[i][j] != 0:
                        neighbours.append(self.board[i][j])

            # choose an opponent from the neighbourhood
            if len(neighbours) > 0:
                opponent = np.random.choice(neighbours)
            else:
                self.move_player_within_neighbourhood(player)
                continue

            # get the payoff matrix of the player against the opponent
            payoff_matrix = self.get_payoff_matrix(player.type, opponent.type)
            
            # get the option of the player against the opponent and vice versa
            options = player.get_options(opponent.type)
            opponent_options = opponent.get_options(player.type)

            # TODO: reinforced learning for both players?
            # TODO: play the game multiple times
            # play the game
            for g in range(games):
                # choose the option of the player and the opponent
                player_option = np.random.choice(options.keys(), p=options.values())
                opponent_option = np.random.choice(opponent_options.keys(), p=opponent_options.values())

                # get the payoffs of the player and the opponent
                player_payoff = payoff_matrix[player_option][opponent_option][0]
                opponent_payoff = payoff_matrix[opponent_option][player_option][1]

                # update the fitness of the player and the opponent
                player.fitness += player_payoff
                opponent.fitness += opponent_payoff

            # swap the positions of the player and the opponent
            self.board[player.position[0]][player.position[1]] = opponent
            self.board[opponent.position[0]][opponent.position[1]] = player
            player.position, opponent.position = opponent.position, player.position

            #TODO: how to handle if fitness is negative

            # subtract the cost of the player per n iterations
            if (it + 1) % self.n == 0:
                for player in self.players:
                    player.fitness -= self.costs[player.type]

            # add the fitness distribution of the players to the fitness distribution
            if (it + 1) % 100 == 0:
                self.fitness_distributions.append([(player.fitness, player.type) for player in self.players])
        
    # display the fitness distributions of the players
    def display_fitness_distributions(self):
        sns.set() 
        for i in range(len(self.fitness_distributions[0])):
            plt.plot([d[i][0] for d in self.fitness_distributions], label=self.fitness_distributions[0][i][1])
        plt.legend()
        plt.show()

