
# import stat libraries
import numpy as np
import matplotlib.pyplot as plt
import random
from Graph import Graph

class Player:

    def __init__(self, name):
        self.name = name
        self.d_rate = 0
        self.size = 1
        self.strategy = [1, 0]
        self.payoff = 0
        self.new_strategy = None
        self.new_payoff = 0
        self.average_payoff = 0
        self.asset_value = 0
        self.liability_value = 0


class Game:

    def __init__(self, adjacency_matrix, size = 100):
        self.adjacency_matrix = adjacency_matrix
        self.players = [Player(i) for i in range(len(adjacency_matrix))]
        # name to player dictionary
        self.player_dict = {i: self.players[i] for i in range(len(adjacency_matrix))}
        self.d_rate_mean = 0.2
        self.d_rate_std = 0.1
        self.size_mean = 10
        self.size_std = 5
        self.strategy_mean = 0.5
        self.strategy_std = 0.3
        self.libailities_matrix = np.zeros((size, size))
        self.asset_matrix = np.zeros((size, size))
        self.graph = None

    # calculate payoffs for two players
    def payoff(self, player1, player2):

        # player 1
        x_A = player1.strategy[0]
        E_A = player1.size
        I_A = player1.d_rate

        # player 2
        x_B = player2.strategy[0]
        E_B = player2.size
        I_B = player2.d_rate

        p1 = x_A * x_B * (1 - abs(x_A - x_B)) * (1 - abs(I_A - I_B))
        p2 = x_A * (1 - x_B) * I_A
        p3 = (1 - x_A) * x_B * I_B
        p4 = (1 - x_A) * (1 - x_B) * 0.5
        return p1 + p2 + p3 + p4
    
    def game_payoff(self, player1, player2):
        return self.payoff(player1, player2) - self.payoff(player2, player1)

    # algorithm to return a list of all disjoint pairs of numbers upto n
    def disjointPairs(self, players):
        n = len(players)
        left = players[0:n//2]
        right = players[n//2:]
        pairs = []
        for i in range(n//2):
            pairs.append([left[i], right[i]])
        return pairs
    
    # initialize the game
    def initialize(self):
        for i, player in enumerate(self.players):
            self.players[i].d_rate = np.random.normal(self.d_rate_mean, self.d_rate_std)
            if self.players[i].d_rate < 0:
                self.players[i].d_rate = 0
            self.players[i].size = np.random.normal(self.size_mean, self.size_std)
            # uniformly random value between 0 and 1
            val = np.random.uniform(low=0.0, high=1.0, size=1)
            self.players[i].strategy = [val, 1 - val]
            self.adjacency_matrix[i][i] = 0

        self.graph = Graph(self.adjacency_matrix)
        # for each bank in the adjacency matrix, iterate through its neighbours and sum the total size
        for i in range(len(self.adjacency_matrix)):
            # invest in neighbouring banks
            neighbours = self.graph.getNeighbours(i)
            investment = self.players[i].d_rate * self.players[i].size
            for neighbour in neighbours:
                pass

    def simulate(self, epochs=1000):
        players = self.players

        # TODO: incorporate banks going bankrupt
        # TODO: incorporate banks merging?
        # TODO: incorporate assets per bank
        # TODO: incorporate random shocks to asset values

        for k in range(epochs):
            random.shuffle(players)

            # play mutliple rounds
            for j in range(10):
                pairs = self.disjointPairs(players)
                for pair in pairs:
                    payoff_player1 = self.game_payoff(pair[0], pair[1])
                    payoff_player2 = self.game_payoff(pair[1], pair[0])
                    pair[0].payoff += payoff_player1
                    pair[1].payoff += payoff_player2

                    # update average payoffs per player
                    pair[0].average_payoff += payoff_player1
                    pair[1].average_payoff += payoff_player2

            # update strategies
            for player in players:
                # go through adjacent players
                maximum_payoff = player.payoff
                new_strategy = player.strategy
                for i, adj in enumerate(self.adjacency_matrix[player.name]):
                    if adj == 1:
                        maximum_payoff = max(maximum_payoff, players[i].payoff)
                        if maximum_payoff == players[i].payoff:
                            new_strategy = players[i].strategy

                player.new_payoff = maximum_payoff
                player.new_strategy = new_strategy

            # update strategies and incorporate payoffs into size
            for player in players:
                player.strategy = player.new_strategy
                player.payoff = player.new_payoff
                player.size += player.payoff
                player.new_strategy = None
                player.new_payoff = 0

            if k % 50 == 0:
                print("Epoch: ", k)

        # update average payoffs
        for player in players:
            player.average_payoff /= (epochs * 10)

    def displayStrategy(self):
        x = []
        y = []
        for player in self.players:
            x.append(player.strategy[0])
            y.append(player.strategy[1])

        plt.scatter(x, y)
        plt.show()

    def displaySize(self):
        x = []
        y = []
        for player in self.players:
            x.append(player.size)
            y.append(player.average_payoff)

        plt.scatter(x, y)
        plt.show()

    def run(self):
        self.initialize()
        self.displayStrategy()
        self.simulate()
        self.displayStrategy()
        self.displaySize()

if __name__ == "__main__":

    # TODO: create liabilities matrix based on proportion of bank size
    
    size = 300
    # create random graph
    adjacency_matrix = np.random.randint(2, size=(size, size))
    
    game = Game(adjacency_matrix, size)
    game.run()


    

        
