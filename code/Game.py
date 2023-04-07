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
#      n: board size
#      num_players: number of players
#      positions: set of positions of players
#      board: board of the game
#      costs: dictionary of costs of player for existing in the game
#      immovable: list of player types that cannot be moved
#      evolution_function: function that determines the evolution of the game
#      fitness_distributions: list of fitness distributions of players in the game

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import *

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

    def __init__(self, players, options, payoffs, fitness_mean, fitness_std, neighbourhoods, costs, board_size, immovable, proportions, population, evolution_function, fitness=None):

        self.players = [] # population of players
        # create the players for the population
        for player, option in zip(players, options):
            for i in range(floor(proportions[player.type] * population)):
                self.players.append(self.Player(player, option))

        self.payoffs = payoffs # dictionary of payoffs of player against different players
        self.n = board_size # size of the board
        self.num_players = len(self.players) # number of players
        self.positions = set() # set of positions of players
        self.board = [[0 for i in range(self.n)] for j in range(self.n)] # board of the game
        self.costs = costs # dictionary of costs of player for existing in the game
        self.immovable = set(immovable)
        self.evolution_function = evolution_function

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
                if player.position not in self.positions:
                    self.positions.add(player.position)
                    self.board[player.position[0]][player.position[1]] = player
                    break

        # set neighbourhood raadius of players
        for player in self.players:
            player.neighbour_radius = np.random.normal(neighbourhoods[player.type], 1)

        # set graph matrix of the game
        self.graph = [[0 for i in range(self.num_players)] for j in range(self.num_players)]
        for i in range(self.num_players):
            for j in range(i + 1, self.num_players):
                if self.distance(self.players[i], self.players[j]) <= self.players[i].neighbour_radius:
                    self.graph[i][j] = 1
                if self.distance(self.players[i], self.players[j]) <= self.players[j].neighbour_radius:
                    self.graph[j][i] = 1


    def distance(self, player1, player2):
        return sqrt((player1.position[0] - player2.position[0]) ** 2 + (player1.position[1] - player2.position[1]) ** 2)

    # get the payoff matrix of the player against the opponent
    def get_payoff_matrix(self, player, opponent):
        return self.payoffs[player][opponent]
    
    # move the player to a random position within the neighbourhood
    def move_player_within_neighbourhood(self, player):

        # choose a random position from the neighbourhood
        while True:
            (posX, posY) = np.random.choice(range(player.position[0] - player.neighbour_radius, player.position[0] + player.neighbour_radius + 1)), np.random.choice(range(player.position[1] - player.neighbour_radius, player.position[1] + player.neighbour_radius + 1))
            if posY >= 0 and posY < self.n and posX >= 0 and posX < self.n:
                if (posX, posY) not in self.positions:
                    self.positions.remove(player.position)
                    self.board[player.position[0]][player.position[1]] = 0
                    self.board[posX][posY] = player
                    player.position = (posX, posY)
                    self.positions.add(player.position)
                    break
    
    # simulate the game for iter iterations
    def simulate(self, iter, games=1):

        for it in range(iter):
            # play the game for each player
            for i, player in enumerate(self.players):
                for j, opponent in enumerate(self.players):

                    # if the player is the same as the opponent or the player and the opponent are not neighbours, then continue
                    if i >= j or self.graph[i][j] == 0:
                        continue

                    # get the payoff matrix of the player against the opponent
                    payoff_matrix = self.get_payoff_matrix(player.type, opponent.type)
                    
                    # get the option of the player against the opponent and vice versa
                    options = player.get_options(opponent.type)
                    opponent_options = opponent.get_options(player.type)

                    # TODO: reinforced learning for both players (updating probability distribution of options)
                    # TODO: play the game multiple times
                    # play the game
                    for g in range(games):
                        # choose the option of the player and the opponent
                        player_option = np.random.choice(options.keys(), p=options.values())
                        opponent_option = np.random.choice(opponent_options.keys(), p=opponent_options.values())

                        # get the payoffs of the player and the opponent
                        player_payoff = payoff_matrix[player_option][opponent_option][0]
                        opponent_payoff = payoff_matrix[opponent_option][player_option][1]
                        
                        # TODO: handle case for multiple games in the simulation
                        # update the fitness of the player and the opponent
                        player.fitness += player_payoff
                        opponent.fitness += opponent_payoff

                    # swap the positions of the player and the opponent if they are not immovable else move the player to a random position within the neighbourhood
                    if player.type in self.immovable or opponent.type in self.immovable:
                        self.move_player_within_neighbourhood(player)
                    else:
                        self.board[player.position[0]][player.position[1]] = opponent
                        self.board[opponent.position[0]][opponent.position[1]] = player
                        player.position, opponent.position = opponent.position, player.position

                    #TODO: how to handle if fitness is negative

                    # subtract the cost of the player per n iterations
                    if (it + 1) % self.num_players == 0:
                        for player in self.players:
                            player.fitness -= self.costs[player.type]

                    # add the fitness distribution of the players to the fitness distribution
                    if (it + 1) % 100 == 0:
                        self.fitness_distributions.append([(player.fitness, player.type) for player in self.players])

            # evolve the players
            self.evolve()

    def evolve(self, *args):
        self.players = self.evolution_function(args)
        
    # display the fitness distributions of the players
    def display_fitness_distributions(self):
        sns.set() 
        for i in range(len(self.fitness_distributions[0])):
            plt.plot([d[i][0] for d in self.fitness_distributions], label=self.fitness_distributions[0][i][1])
        plt.legend()
        plt.show()

# TODO: determine types of players
# TODO: determine fitness distributions of players
# TODO: determine neighbourhoods of players
# TODO: determine costs of players
# TODO: determine payoffs of players
# TODO: determine evolution function
# TODO: determine number of players
# TODO: determine mutation function
# TODO: determine immovable players
# TODO: determine board size
# TODO: determine to be played game between players 