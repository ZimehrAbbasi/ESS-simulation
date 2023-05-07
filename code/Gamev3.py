import numpy as np
import matplotlib.pyplot as plt

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
        self.n_new = n
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
        self.last = 0

    def run(self):

        # Initialize the game
        self.initialize(self.num_banks, self.strategies, self.num_strategies)
        strat = None

        # Run the game
        for i in range(self.num_rounds):
            strat = self.run_round()
            if strat:
                break
            
            if np.random.uniform(0, 1) < 0.01:
                self.randomDevalue()
            if np.random.uniform(0, 1) < 0.01:
                self.devalueHighest()

        # Print the results
        self.print_results()

    def revalue(self):
        for asset in self.assets:
            asset.n = asset.n_new

    def devalueHighest(self):
        assets = [asset.n for asset in self.assets]
        max_index = np.argmax(assets)
        self.assets[max_index].n = np.random.normal(-50, self.assets[max_index].volatility, 1)[0]

    def randomDevalue(self):
        assets = [asset.volatility for asset in self.assets]
        stockBaseReturnRates = np.array(assets).flatten()
        total = np.sum(stockBaseReturnRates)
        stockBaseReturnRates = stockBaseReturnRates / total
        stockToDevalue = np.random.choice([i for i in range(self.num_assets)], 1, p=stockBaseReturnRates)
        asset = self.assets[stockToDevalue[0]]
        asset.n = np.random.normal(-50, asset.volatility, 1)[0]

    def initialize(self, n, strategies, num_strategies):

        i = self.last
        while i < n + self.last:
            assets = np.random.lognormal(self.asset_mean, self.asset_std)
            liability = np.random.lognormal(self.liabilities_mean, self.liabilities_std)

            if assets > liability:
                strategy = np.random.randint(0, num_strategies)
                self.banks.append(Bank(i, strategies[strategy], assets, liability))
                i += 1

        self.last = i

    def f(self, bank1, bank2, asset, bank1_d, bank2_d):
        return np.random.normal(asset.logistic(bank1.asset_value * bank1_d + bank2.asset_value * bank2_d), asset.volatility, 1)

    def bankOff2x2(self, bank1, bank2):

        # get index of maximum strategy
        max_asset_index = np.argmax([asset.n for asset in self.assets])
        dd = 0
        for asset in self.assets:
            return_rate = self.f(bank1, bank2, asset, 1/self.num_assets, 1/self.num_assets) / 100
            dd += bank1.asset_value/self.num_assets * return_rate

        dnd = 0
        for i, asset in enumerate(self.assets):
            return_rate = self.f(bank1, bank2, asset, 1/self.num_assets, i == max_asset_index) / 100
            dnd += bank1.asset_value/self.num_assets * return_rate

        ndd = 0
        for i, asset in enumerate(self.assets):
            return_rate = self.f(bank1, bank2, asset, i == max_asset_index, 1/self.num_assets) / 100
            ndd += bank1.asset_value * (i == max_asset_index) * return_rate

        ndnd = 0
        for i, asset in enumerate(self.assets):
            return_rate = self.f(bank1, bank2, asset, i == max_asset_index, i == max_asset_index) / 100
            ndnd += bank1.asset_value * (i == max_asset_index) * return_rate

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

        popped = 0
        for i, bank in enumerate(self.banks):
            if bank.new_asset_value <= bank.liabilities:
                self.banks.pop(i)
                popped += 1
            else:
                bank.default = False
        
        strategies = []
        for bank in self.banks:
            strategies.append(bank.strategy)

        for bank in self.banks:
            i = np.random.randint(0, len(strategies))
            bank.strategy = strategies[i]
            
            bank.new_asset_value = bank.asset_value

        self.initialize(popped, strategies, self.num_strategies)

    def print_results(self):
        strat = []
        for bank in self.banks:
            strat.append(int(bank.strategy[0] == 1))
  
        fig, ax = plt.subplots(1, 1)
        ax.hist(strat, bins=2)
        
        ax.set_title("Frequency of strategies")
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Frequency')

        # remove x-ticks
        ax.set_xticks([])
        
        rects = ax.patches
        labels = ["Highest return rate", "Diversify"]
        
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height+0.01, label,
                    ha='center', va='bottom')
        
        plt.show()


if __name__ == "__main__":

    # Initialize assets
    assets = [Asset(0.1, 2, 1, 4, 0.4), Asset(0.1, 2, 1, 0, 4)]

    # Initialize strategies
    strategies = [[1, 0], [0, 1]]

    # Initialize game
    game = Game(100, 2, 500, assets, strategies, 10, 1, 10, 1)

    # Run game
    game.run()



