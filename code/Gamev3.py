import numpy as np
import matplotlib.pyplot as plt
import time

# Class definitions

# Bank class
class Bank:
    def __init__(self, name, strategy, asset_value, liabilities):
        self.name = name # name of the bank
        self.strategy = strategy # strategy of the bank
        self.asset_value = asset_value # value of the assets
        self.liabilities = liabilities # value of the liabilities
        self.new_asset_value = asset_value # value of the assets after the round
        self.default = False # whether the bank has defaulted
        self.last_strategy = None # last strategy of the bank

class Asset:

    def __init__(self, k, r, l, n, volatility):
        self.k = k # growth rate
        self.r = r
        self.l = l # carrying capacity
        self.n = n # return rate
        self.n_new = n # return rate after the round
        self.volatility = volatility

    def logistic(self, x):
        return self.r * (self.l/(1 + np.exp(-self.k*x)) - 0.5) + self.n # logistic function

class Game:

    def __init__(self, n, m, num_rounds, assets, asset_mean, asset_std, liabilities_mean, liabilities_std):
        self.banks = []  # list of banks
        self.num_rounds = num_rounds # number of rounds
        self.num_banks = n # number of banks
        self.num_assets = m # number of assets
        self.assets = assets # list of assets
        # sort assets by return rate
        self.assets.sort(key=lambda x: x.n) # sort assets by return rate
        self.asset_mean = asset_mean # mean of the asset value
        self.asset_std = asset_std # standard deviation of the asset value
        self.liabilities_mean = liabilities_mean # mean of the liabilities
        self.liabilities_std = liabilities_std # standard deviation of the liabilities
        self.last = 0

        # strategies

        # 1. assets with highest expected return
        strat1 = np.array([1 if i == (self.num_assets - 1) else 0 for i in range(self.num_assets)])

        # 2. assets with lowest volatility
        min_asset_index = np.argmin([asset.volatility for asset in self.assets])
        strat2 = np.array([1 if i == min_asset_index else 0 for i in range(self.num_assets)])

        # 3. assets with median expected return
        median_asset_index = self.assets[self.num_assets//2]
        strat3 = np.array([1 if i == median_asset_index else 0 for i in range(self.num_assets)])

        # 4. all assets equally
        strat4 = np.array([1/self.num_assets for i in range(self.num_assets)])

        # 5. assets are invested in based on the proportion of their returns
        strat5 = np.array([asset.n for asset in self.assets])
        strat5 = strat5 / np.sum(strat5)

        # 6. assets are invested in inversel proportional of their volatility
        strat6 = np.array([1/asset.volatility for asset in self.assets])
        strat6 = strat6 / np.sum(strat6)

        # 7. assets in the top quartile are invested in equally
        strat7 = np.array([1 if i >= self.num_assets * 3/4 else 0 for i in range(self.num_assets)])
        strat7 = strat7 / np.sum(strat7)

        # indices of assets in the lowest quartile in volatility
        # create a list of tuples, with indices and volatility of a given asset
        asset_volatility = [(i, asset.volatility) for i, asset in enumerate(self.assets)]
        # sort the list by volatility
        asset_volatility.sort(key=lambda x: x[1])
        # get the indices of the assets in the lowest quartile
        lowest_quartile = [asset_volatility[i][0] for i in range(self.num_assets//4)]

        # 8. create a strategy that invests equally in the lowest quartile of volatility
        strat8 = np.array([1 if i in lowest_quartile else 0 for i in range(self.num_assets)])
        strat8 = strat8 / np.sum(strat8)

        # 9. random strategy
        strat9 = np.random.uniform(0, 1, self.num_assets)

        # create a list of strategies
        self.strategies = [strat1, strat2, strat3, strat4, strat5, strat6, strat7, strat8, strat9]
        for i, strat in enumerate(self.strategies):
            if np.isnan(strat).any() or np.sum(strat) != 1:
                self.strategies[i] = np.random.uniform(0, 1, self.num_assets)
                self.strategies[i] = self.strategies[i] / np.sum(self.strategies[i])
        self.num_strategies = len(self.strategies)

    def run(self, epoch=25):
        strategy_distribution = np.zeros(self.num_strategies)
        for i in range(epoch):
            print("Epoch", i)
            strategy_distribution += self.epoch()/epoch

        print("Game simulation finished.")
        self.print_results(strategy_distribution)

    def epoch(self):

        # Initialize the game
        print("Initializing the game for this epoch...")
        self.initialize(self.num_banks, None, self.num_strategies)
        print("Game initialization successful.")

        # Initialize strategy distribution
        strat = np.zeros(self.num_strategies)

        # Run the game
        print("Running the Evolutionary Game...")
        start_time = time.time()
        average_time = 0
        for i in range(self.num_rounds):
            self.run_round()
            
            # randomly devalue assets
            if np.random.uniform(0, 1) < 1/self.num_rounds:
                self.randomDevalue()
            if np.random.uniform(0, 1) < 1/self.num_rounds:
                self.devalueHighest()

            if i % 100 == 0:
                print("Round number", i)
                print("\tTime elapsed:", round(time.time() - start_time, 4), "seconds")
                average_time = (time.time() - start_time) / (i + 1)
                remaining = average_time * (self.num_rounds - i)
                print("\tEstimated time remaining:", round(remaining, 4), "seconds")

            for bank in self.banks:
                # convert all strategies to np arrays of type float
                bank.last_strategy = np.array(bank.last_strategy, dtype=float)
                for i, strategy in enumerate(self.strategies):
                    # check if 2 np arrays are equal
                    if np.array_equal(bank.last_strategy, strategy):
                        strat[i] += 1/self.num_rounds
                        break

        print("Evolutionary Game finished for this epoch.")

        return strat

    # revalue an asset
    def revalue(self):
        for asset in self.assets:
            asset.n = asset.n_new

    # Devalue the highest asset
    def devalueHighest(self):
        assets = [asset.n for asset in self.assets]
        max_index = np.argmax(assets)
        self.assets[max_index].n = np.random.normal(-50, self.assets[max_index].volatility, 1)[0]

    # randomly devalue an asset
    def randomDevalue(self):
        assets = [asset.volatility for asset in self.assets] # volatility of each asset
        stockBaseReturnRates = np.array(assets).flatten() # convert to np array
        total = np.sum(stockBaseReturnRates) # sum of all volatilities
        stockBaseReturnRates = stockBaseReturnRates / total # normalize
        stockToDevalue = np.random.choice([i for i in range(self.num_assets)], 1, p=stockBaseReturnRates) # randomly choose an asset
        asset = self.assets[stockToDevalue[0]] # get the asset
        asset.n = np.random.normal(-50, asset.volatility, 1)[0] # devalue the asset

    # initialize the game
    def initialize(self, n, strategies, num_strategies):

        i = self.last
        while i < n + self.last:

            # generate random assets and liabilities
            assets = np.random.lognormal(self.asset_mean, self.asset_std)
            liability = np.random.lognormal(self.liabilities_mean, self.liabilities_std)

            # if assets are greater than liabilities, create a bank
            if assets > liability:
                # create a probability distribution for each bank over the strategies
                if not strategies:
                    strat_dist = np.random.uniform(0, 1, num_strategies)
                    strat_dist = strat_dist / np.sum(strat_dist)
                else:
                    # generate random number in range [0, num_strategies]
                    num = np.random.uniform(0, num_strategies)
                    strat_dist = strategies[int(num)]
                self.banks.append(Bank(i, strat_dist, assets, liability))
                i += 1

        self.last = i

    def choose_strategy(self, i, strategy):
        
        # joker (random)
        if i == 8:
            j = np.random.randint(0, self.num_strategies)
            while j != 8:
                j = np.random.randint(0, self.num_strategies)
            return self.strategies[j]
        
        return strategy
        
    # logisitic function for the bank payoff
    def f(self, bank1, bank2, asset, bank1_d, bank2_d):
        return np.random.normal(asset.logistic(bank1.asset_value * bank1_d + bank2.asset_value * bank2_d), asset.volatility, 1)

    """ Deprecated """
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
    
    # play a round of the game
    def playOff(self, bank1, bank2):

        # create a matrix for the payoffs
        payoffs = np.zeros((self.num_strategies, self.num_strategies))

        # iterate through the strategies
        for i, strat1 in enumerate(self.strategies):
            for j, strat2 in enumerate(self.strategies):
                strat1 = self.choose_strategy(i, strat1) # choose strategy for bank 1
                strat2 = self.choose_strategy(j, strat2) # choose strategy for bank 2
                for k, asset in enumerate(self.assets):
                    return_rate = self.f(bank1, bank2, asset, strat1[k], strat2[k]) / 100 # get return rate
                    payoffs[i, j] += bank1.asset_value * strat1[k] * return_rate # add to payoff

        # create probability distribution from strategy
        prob_dist_1 = np.array(bank1.strategy)
        prob_dist_2 = np.array(bank2.strategy)

        startegy_choices = [i for i in range(self.num_strategies)]

        # draw from probability distribution
        bank1_i = np.random.choice(startegy_choices, 1, p=prob_dist_1)
        bank2_i = np.random.choice(startegy_choices, 1, p=prob_dist_2)

        bank1.last_strategy = self.strategies[int(bank1_i)]

        return payoffs[bank1_i, bank2_i]

    # create disjoint pairs of banks
    def disjointPairs(self, banks):
        n = len(banks)
        left = banks[0:n//2]
        right = banks[n//2:]
        pairs = []
        for i in range(n//2):
            pairs.append([left[i], right[i]])
        return pairs
    
    # run a round of the game
    def run_round(self):
        
        # shuffle the banks
        np.random.shuffle(self.banks)
        
        # create disjoint pairs
        pairs = self.disjointPairs(self.banks)

        # play a round of the game
        for pair in pairs:
            payoff_player1 = self.playOff(pair[0], pair[1])
            payoff_player2 = self.playOff(pair[1], pair[0])

            pair[0].new_asset_value += payoff_player1
            pair[1].new_asset_value += payoff_player2

        # check for defaults
        popped = 0
        for i, bank in enumerate(self.banks):
            if bank.new_asset_value <= bank.liabilities:
                self.banks.pop(i)
                popped += 1
            else:
                bank.default = False
        
        # check dominant strategies
        strategies = []
        for bank in self.banks:
            strategies.append(bank.strategy)

        # update the strategies
        for bank in self.banks:
            i = np.random.randint(0, len(strategies))
            bank.strategy = strategies[i]
            bank.new_asset_value = bank.asset_value

        # initialize new banks
        self.initialize(popped, strategies, len(strategies))

    def print_results(self, strat):
  
        plt.bar(np.array([i for i in range(self.num_strategies)]), np.array(strat), align='center', color='blue')
        plt.ylabel('Strategies')
        plt.xlabel('Abundance of Strategies')
        plt.title('Strategy Abundance')
        # x labels
        plt.xticks(np.array([i for i in range(self.num_strategies)]), ('1', '2', '3', '4', '5', '6', '7', '8', '9'))
        plt.show()


if __name__ == "__main__":

    # Set the random seed for reproducibility (optional)
    np.random.seed(42)

    # Initialize assets

    # Manually created external assets with mean and standard deviation
    asset_array = np.array([[0.02, 0.01], [0.04, 0.04], [0.06, 0.08], [0.08, 0.12], [0.1, 0.16]])

    # Create asset objects
    assets = []
    num_assets = 5
    for i in range(num_assets):
        assets.append(Asset(0.1, 2, 1, asset_array[i][0] * 100, asset_array[i][1] * 100))

    # Initialize game
    game = Game(50, num_assets, 50, assets, 7.00306, 0.69115, 4.60517, 0.92103)

    # Run game
    game.run(3)
