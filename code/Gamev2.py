
# import stat libraries
import numpy as np
import matplotlib.pyplot as plt
import random

class Player:

    def __init__(self, name):
        self.name = name
        self.d_rate = 0
        self.size = 1
        self.strategy = [1, 0]


class Game:

    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix
        self.players = [Player(i) for i in range(len(adjacency_matrix))]
        self.d_rate_mean = 0.5
        self.d_rate_std = 0.1
        self.size_mean = 10
        self.size_std = 50
        self.strategy_mean = 0.5
        self.strategy_std = 0.3

    # calculate payoffs for two players
    def payoff(self, player1, player2):
        return (player1.d_rate * player2.size * player2.strategy[0] - player2.d_rate * player1.size * player1.strategy[0]) / (player1.d_rate * player2.size * player2.strategy[0] + player2.d_rate * player1.size * player1.strategy[0])

    # algorithm to return a list of all disjoint pairs of numbers upto n
    def disjointPairs(self, players):
        n = len(players)
        left = players[0:n//2]
        right = players[n//2:]
        random.shuffle(left)
        random.shuffle(right)
        pairs = []
        for i in range(n//2):
            pairs.append([left[i], right[i]])

        return pairs
    
    # initialize the game
    def initialize(self):
        for i, player in enumerate(self.players):
            self.players[i].d_rate = np.random.normal(self.d_rate_mean, self.d_rate_std)
            self.players[i].size = np.random.normal(self.size_mean, self.size_std)
            # uniformly random value between 0 and 1
            val = np.random.uniform(low=0.0, high=1.0, size=1)
            self.players[i].strategy = [val, 1 - val]

    def simulate(self):
        players = self.players
        for i in range(100):
            random.shuffle(players)
            pairs = self.disjointPairs(players)
            for pair in pairs:
                payoff_player1 = self.payoff(pair[0], pair[1])
                payoff_player2 = self.payoff(pair[1], pair[0])

                if payoff_player1 > payoff_player2:
                    pair[1].strategy = pair[0].strategy
                else:
                    pair[0].strategy = pair[1].strategy

    def displayStrategy(self):
        x = []
        y = []
        for player in self.players:
            x.append(player.strategy[0])
            y.append(player.strategy[1])

        plt.scatter(x, y)
        plt.show()

    def run(self):
        self.initialize()
        self.displayStrategy()
        self.simulate()
        self.displayStrategy()

if __name__ == "__main__":

    # create random graph
    adjacency_matrix = np.random.randint(2, size=(100, 100))
    game = Game(adjacency_matrix)
    game.run()


    

        
