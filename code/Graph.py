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

class Graph:

    def __init__(self, adjacency_matrix):
        # convert adjacency matrix to list of players
        self.players = [Player(i) for i in range(len(adjacency_matrix))]
        self.n = len(self.players)
        # convert adjacency matrix to dictionary of players
        self.player_dict = {i: self.players[i] for i in range(len(adjacency_matrix))}
        # convert adjacency matrix to adjacency dictionary
        self.adjacency_dict = {self.player_dict[i]: [] for i in range(len(adjacency_matrix))}
        for i in range(len(adjacency_matrix)):
            for j in range(len(adjacency_matrix)):
                if adjacency_matrix[i][j] == 1:
                    self.adjacency_dict[self.player_dict[i]].append(self.player_dict[j])

    def getNeighbors(self, player):
        return self.adjacency_dict[self.player_dict[player]]
    
    def getPlayers(self):
        return self.players
