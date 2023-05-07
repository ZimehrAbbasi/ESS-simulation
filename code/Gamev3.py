import numpy as np

class Bank:
    def __init__(self, name, strategy, asset_value, liabilities):
        self.name = name
        self.strategy = strategy
        self.asset_value = asset_value
        self.liabilities = liabilities
        self.new_asset_value = asset_value
        self.default = False

class Asset:

    def __init__(self, k, r, l, n, volatility):
        self.k = k
        self.r = r
        self.l = l
        self.n = n
        self.volatility = volatility

    def logistic(self, x):
        return self.r * (self.l/(1 + np.exp(-self.k*x)) - 0.5) + self.n

class Game:

    def __init__(self, n, m, num_rounds, assets, strategies, asset_mean, asset_std, liabilities_mean, liabilities_std):
        self.banks = []
        self.num_rounds = num_rounds
        self.num_banks = n
        self.num_assets = m
        self.strategies = strategies
        self.assets = assets
        self.num_strategies = len(strategies)
        self.asset_mean = asset_mean
        self.asset_std = asset_std
        self.liabilities_mean = liabilities_mean
        self.liabilities_std = liabilities_std

    def run(self):

        # Initialize the game
        self.initialize()
        strat = None

        # Run the game
        for i in range(self.num_rounds):
            strat = self.run_round()
            if strat:
                break
        
        print(strat)

        # Print the results
        self.print_results()

    def initialize(self):

        i = 0
        while i < self.num_banks:
            assets = np.random.lognormal(self.asset_mean, self.asset_std)
            liability = np.random.lognormal(self.liabilities_mean, self.liabilities_std)

            if assets > liability:
                strategy = np.random.randint(0, self.num_strategies)
                self.banks.append(Bank(i, self.strategies[strategy], assets, liability))
                i += 1

    def f(self, bank1, bank2, asset, bank1_d, bank2_d):
        return np.random.normal(asset.logistic(bank1.asset_value * bank1_d + bank2.asset_value * bank2_d), asset.volatility, 1)

    def bankOff2x2(self, bank1, bank2):

        # get index of maximum strategy
        max_asset_index = np.argmax([asset.n for asset in self.assets])
        
        dd = 0
        for asset in self.assets:
            dd += bank1.asset_value/self.num_assets * self.f(bank1, bank2, asset, 1/self.num_assets, 1/self.num_assets)

        dnd = 0
        for i, asset in enumerate(self.assets):
            dnd += bank1.asset_value/self.num_assets * self.f(bank1, bank2, asset, 1/self.num_assets, i == max_asset_index)

        ndd = 0
        for i, asset in enumerate(self.assets):
            ndd += bank1.asset_value * (i == max_asset_index) * self.f(bank1, bank2, asset, i == max_asset_index, 1/self.num_assets)

        ndnd = 0
        for i, asset in enumerate(self.assets):
            ndnd += bank1.asset_value * (i == max_asset_index) * self.f(bank1, bank2, asset, i == max_asset_index, i == max_asset_index)

        # create 2 * 2 matrix of payoffs
        payoff_matrix = np.array([[dd, dnd], [ndd, ndnd]])

        # create probability distribution from strategy
        prob_dist_1 = np.array(bank1.strategy)
        prob_dist_2 = np.array(bank2.strategy)

        # draw from probability distribution
        bank1_i = np.random.choice([0, 1], 1, p=prob_dist_1)
        bank2_i = np.random.choice([0, 1], 1, p=prob_dist_2)

        return payoff_matrix[bank1_i, bank2_i]


    def disjointPairs(self, banks):
        n = len(banks)
        left = banks[0:n//2]
        right = banks[n//2:]
        pairs = []
        for i in range(n//2):
            pairs.append([left[i], right[i]])
        return pairs
    
    def run_round(self):

        np.random.shuffle(self.banks)
        
        pairs = self.disjointPairs(self.banks)
        for pair in pairs:
            payoff_player1 = self.bankOff2x2(pair[0], pair[1])
            payoff_player2 = self.bankOff2x2(pair[1], pair[0])

            pair[0].new_asset_value += payoff_player1
            pair[1].new_asset_value += payoff_player2

        for bank in self.banks:
            if bank.new_asset_value <= bank.liabilities:
                bank.default = True
        
        strategies = []
        for bank in self.banks:
            if not bank.default:
                strategies.append(bank.strategy)

        if len(strategies) == 1:
            return strategies[0]

        for bank in self.banks:
            if bank.default:
                i = np.random.randint(0, len(strategies))
                bank.strategy = strategies[i]

            bank.new_asset_value = bank.asset_value

    def print_results(self):
        
        for bank in self.banks:
            print("Bank: " + str(bank.name))
            print("Strategy: " + str(bank.strategy))
            print("Asset Value: " + str(bank.asset_value))
            print("Liabilities: " + str(bank.liabilities))
            print("Default: " + str(bank.default))
            print()


if __name__ == "__main__":

    # Initialize assets
    assets = [Asset(0.1, 2, 1, 5, 4), Asset(0.1, 2, 1, 10, 12)]

    # Initialize strategies
    strategies = [[0.5, 0.5], [1, 0], [0, 1]]

    # Initialize game
    game = Game(100, 2, 100, assets, strategies, 10, 1, 10, 1)

    # Run game
    game.run()



