import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import copy

# Class definitions

# Bank class
class Bank:
    def __init__(self, name, strategy, asset_value, liabilities, internal_assets, internal_assets_value, internal_liabilities, internal_liabilities_value):
        self.name = name # name of the bank
        self.strategy = strategy # strategy of the bank
        self.asset_value = asset_value # value of the assets
        self.liabilities = liabilities # value of the liabilities
        self.new_asset_value = asset_value # value of the assets after the round
        self.internal_assets = internal_assets # value of the internal assets
        self.internal_assets_value = internal_assets_value # value of the internal assets after the round
        self.internal_liabilities = internal_liabilities # value of the internal liabilities
        self.internal_liabilities_value = internal_liabilities_value # value of the internal liabilities after the round
        self.default = False # whether the bank has defaulted
        self.last_strategy = None # last strategy of the bank

# Asset class
class Asset:

    def __init__(self, k, r, l, n, volatility):
        self.k = k # growth rate
        self.r = r
        self.l = l # carrying capacity
        self.n = n # return rate
        self.n_new = n # return rate after the round
        self.volatility = volatility
        self.m = 0 # amount of assset held by all bank

    def logistic(self, x):
        return self.r * (self.l/(1 + np.exp(-self.k*(x + self.m))) - 0.5) + self.n # logistic function

# Game class
class Game:

    def __init__(self, n, m, num_rounds, asset_array, asset_mean, asset_std, liabilities_mean, liabilities_std, p, showGraph):
        self.banks = []  # list of banks
        self.defaulted_banks = [] # list of defaulted banks
        self.num_rounds = num_rounds # number of rounds
        self.num_banks = n # number of banks
        self.num_assets = m # number of assets
        
        self.assets = []
        for i in range(m):
            self.assets.append(Asset(0.1, 2, 1, copy.deepcopy(asset_array[i][0]) * 100, copy.deepcopy(asset_array[i][1]) * 100))

        self.asset_array = asset_array

        # sort assets by return rate
        self.assets.sort(key=lambda x: x.n) # sort assets by return rate

        self.asset_mean = asset_mean # mean of the asset value
        self.asset_std = asset_std # standard deviation of the asset value
        self.liabilities_mean = liabilities_mean # mean of the liabilities
        self.liabilities_std = liabilities_std # standard deviation of the liabilities
        self.last = 0
        self.p = p
        self.showGraph = showGraph

        self.A = None
        self.L = None

        # Strategies

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
        # strat9 = np.random.uniform(0, 1, self.num_assets)

        # 10. median volatility
        median_volatility = np.median([asset.volatility for asset in self.assets])
        strat10 = np.array([1 if asset.volatility == median_volatility else 0 for asset in self.assets])
        # normalize the strategy
        strat10 = strat10 / np.sum(strat10)

        # create a list of strategies
        self.strategies = [strat1, strat2, strat3, strat4, strat5, strat6, strat7, strat8, strat10]
        for i, strat in enumerate(self.strategies):
            if np.isnan(strat).any() or np.sum(strat) != 1:
                self.strategies[i] = np.random.uniform(0, 1, self.num_assets)
                self.strategies[i] = self.strategies[i] / np.sum(self.strategies[i])
        self.num_strategies = len(self.strategies)

        self.strategy_space = None

    # initialize the game
    def initialize(self, n, strategies, num_strategies):

        self.banks = []
        self.defaulted_banks = []
        self.A = self.create_matrix(n)
        self.L = self.A.T
        B = self.A - self.L
        C = np.matmul(B, np.ones(len(self.A)))

        inc = 0

        for i in range(n):
            # generate random assets and liabilities
            total_assets = np.log(np.random.lognormal(self.asset_mean, self.asset_std))
            total_liability = -0.361664 + 1.008992 * total_assets

            assets = total_assets + np.sum(self.A[i])
            liability = total_liability + np.sum(self.L[i])

            # if assets are greater than liabilities, create a bank
            while assets - liability + C[i] <= self.asset_std and assets < 0 and assets > 30:
                total_assets = np.log(np.random.lognormal(self.asset_mean, self.asset_std + inc))
                total_liability = -0.361664 + 1.008992 * total_assets

                assets = total_assets + np.sum(self.A[i])
                liability = total_liability + np.sum(self.L[i])
                inc += 0.01

            inc = 0
                
            # if assets - liability > 0:
            # create a probability distribution for each bank over the strategies
            if not strategies:
                # create vector with 0s in all columns and 1 in 1 column
                strat_dist = np.zeros(num_strategies)
                strat_dist[np.random.randint(0, num_strategies)] = 1
            else:
                # generate random number in range [0, num_strategies]
                num = np.random.uniform(0, len(strategies))
                strat_dist = strategies[int(num)]
            # self.banks.append(Bank(i, strat_dist, assets, liability, self.A[i], np.sum(self.A[i]), self.L[i], np.sum(self.L[i])))
            self.banks.append(Bank(i, strat_dist, assets, liability, 0, 0, 0, 0))


    def run(self, epoch=25):

        # Initialize the game
        strategy_distribution = np.zeros(self.num_strategies)

        # Run the game
        if self.showGraph:
            plt.ion()
            fig, ax = plt.subplots()

        ax = 0

        for i in range(epoch):
            print("Generation", i)
            strat = self.epoch(ax)/epoch
            strategy_distribution += strat

        if self.showGraph:
            plt.ioff()
            ax.clear()

        # Normalize the strategy distribution
        strategy_distribution = strategy_distribution / epoch

        # Print the results
        print("Game simulation finished.")
        return strategy_distribution

    def epoch(self, ax):
        
        # Set the colors for the graph nodes
        colors = {True: 'red', False: 'green'}

        # Initialize the game
        print("Initializing the game for this epoch...")
        self.initialize(self.num_banks, self.strategy_space, self.num_strategies)
        print("Game initialization successful.")
        
        # Initialize strategy distribution
        strat = np.zeros(self.num_strategies)
        bank_copy = self.banks[:]

        G = nx.Graph()

        for bank in bank_copy:
            G.add_node(bank.name, attr=False)

        pos = nx.random_layout(G) 

        # Run the game
        print("Running the Evolutionary Game...")
        for i in range(self.num_rounds):
            if len(self.banks) == 0:
                return strat
            self.run_round()
            
            # randomly devalue assets
            # if np.random.uniform(0, 1) < 1/(self.num_rounds):
            #     print("#############################################")
            #     print()
            #     print("SHOCK: ASSET VALUE FALLEN BY 50%")
            #     print()
            #     print("#############################################")
            #     self.randomDevalue()
            if np.random.uniform(0, 1) < 1/(self.num_rounds):
                print("#############################################")
                print()
                print("SHOCK: HIGHEST ASSET VALUE FALLEN BY 50%")
                print()
                print("#############################################")
                self.devalueHighest()

            if np.random.uniform(0, 1) < 1/(self.num_rounds):
                print("#############################################")
                print()
                print("SHOCK: RANDOM BANK DEFAULTS")
                print()
                print("#############################################")
                self.defaultRandomBank()

            if i % 100 == 0:
                print("Round number", i)

                if self.showGraph:

                    ax.clear()

                    for bank in self.defaulted_banks:
                        print(bank.name, "has defaulted playing strategy ", bank.strategy)
                        G.nodes[bank.name]['attr'] = True

                    node_attributes = nx.get_node_attributes(G, 'attr')
                    node_colors = [colors[attr] for attr in node_attributes.values()]
                    nx.draw_networkx(G, pos, node_color=node_colors, with_labels=True, ax=ax)

                    # Show the graph
                    plt.show()
                    plt.pause(0.1)

            for bank in self.banks:
                # convert all strategies to np arrays of type float
                strat += np.array(bank.strategy, dtype=float)
                        
        # Evolution 
        # check dominant strategies
        strategies = []
        for bank in self.banks:
            strategies.append(bank.strategy)

        # update the strategies
        for bank in self.banks:
            i = np.random.randint(0, len(strategies))
            bank.strategy = strategies[i]
            bank.new_asset_value = bank.asset_value

        self.strategy_space = strategies

        # reset assets
        self.assets = []
        for i in range(self.num_assets):
            self.assets.append(Asset(0.1, 2, 1, copy.deepcopy(self.asset_array[i][0]) * 100, copy.deepcopy(self.asset_array[i][1]) * 100))

        print("Evolutionary Game finished for this epoch.")
        # self.print_results(strat)

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
        stockBaseReturnRates = stockBaseReturnRates / np.sum(stockBaseReturnRates) # normalize
        stockToDevalue = np.random.choice([i for i in range(self.num_assets)], 1, p=stockBaseReturnRates) # randomly choose an asset
        asset = self.assets[stockToDevalue[0]] # get the asset
        asset.n = np.random.normal(-50, asset.volatility, 1)[0] # devalue the asset

    def choose_strategy(self, i, strategy):
        
        # joker (random)
        if i == 8:
            j = np.random.randint(0, self.num_strategies)
            while j != 8:
                j = np.random.randint(0, self.num_strategies)
            return self.strategies[j]
        
        return strategy
    
    # logisitic function for the bank payoff
    def f(self, bank1, asset, bank1_d):
        return np.random.normal(asset.logistic(bank1.asset_value * bank1_d), asset.volatility, 1)

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
    def playOff(self, bank):

        # create a matrix for the payoffs
        payoff = 0

        i = np.argmax(bank.strategy)
        strat = self.strategies[i]
        for k, asset in enumerate(self.assets):
            return_rate = self.f(bank, asset, strat[k]) / 100 # get return rate
            payoff += bank.asset_value * strat[k] * return_rate # add to payoff
            asset.m += bank.asset_value * strat[k] # add to maket value

        return payoff
    
    def checkDefaulted(self, i, haircut=0.4):
        bank = self.banks[i]
        defaulted = False
        if bank.new_asset_value + bank.internal_assets_value <= bank.liabilities + bank.internal_liabilities_value:
        # if bank.new_asset_value <= bank.liabilities:
            # calculate difference in external assets and liabilities
            diff = bank.new_asset_value - bank.liabilities
            total_assets = diff + bank.internal_assets_value
            shortfall = bank.internal_liabilities_value - total_assets
            
            # calculate the haircut
            new_value = total_assets * (1 - haircut)
            # get column i of A
            col = self.A[:, i]
            for j in range(len(self.banks)):
                self.banks[j].internal_assets_value -= col[j]

            sum_col = np.sum(col)
            col = col / sum_col * new_value
            for j in range(len(self.banks)):
                self.banks[j].new_asset_value += col[j]

            # remove the ith column and row from A and L
            self.A = np.delete(self.A, i, 0)
            self.A = np.delete(self.A, i, 1)
            self.L = np.delete(self.L, i, 0)
            self.L = np.delete(self.L, i, 1)

            # remove the ith bank
            self.banks.pop(i)
            bank.default = True
            defaulted = True
            self.defaulted_banks.append(bank)

        return defaulted

    def checkDefault(self):

        defaulted = False
        haircut = 0.4
        i = len(self.banks) - 1
        while i > -1:
            defaulted = self.checkDefaulted(i, haircut)
            i -= 1

        return defaulted
    
    def defaultRandomBank(self):
        # generate uniformly random number
        i = np.random.randint(0, len(self.banks))
        # reduce external assets of bank by a certain percentage
        bank = self.banks[i]
        # p = np.random.uniform(0, 1) 
        # while p * bank.new_asset_value > bank.liabilities:
        #     p = np.random.uniform(0, 1)

        # bank.new_asset_value = p * bank.new_asset_value
        bank.new_asset_value = 0
        
        while self.checkDefault():
            pass
        
    
    # run a round of the game
    def run_round(self):

        # play a round of the game
        for bank in self.banks:
            # get the payoffs
            payoff = self.playOff(bank)
            # update the asset values
            bank.new_asset_value += payoff

        # check for defaults
        while self.checkDefault():
            pass

    def print_results(self, strat):
  
        plt.bar(np.array([i for i in range(self.num_strategies)]), np.array(strat) / np.sum(strat), align='center', color='blue')
        plt.xlabel('Investment Strategies')
        plt.ylabel('Abundance of Strategies')
        plt.title('Strategy Abundance')
        # x labels
        plt.xticks(np.array([i for i in range(self.num_strategies)]), ('0', '1', '2', '3', '4', '5', '6', '7', '8'))
        plt.show()

    def create_matrix(self, n):
        
        # create initial matrix
        A = np.zeros((n, n))
        cut = 10
        # create random matrix
        for i in range(n):
            row_sum = np.log(np.random.lognormal(self.asset_mean/cut, self.asset_std/np.sqrt(cut)))
            distribution = np.random.uniform(0, 1, n - 1) * np.random.choice([0, 1], size=n-1, p=[self.p, 1-self.p])
            distribution /= np.sum(distribution)
            distribution *= row_sum
            row_vec = np.insert(distribution, i, 0)
            A[i] = row_vec

        return A
    
    def runMultiple(self, generations, epochs):

        # run the game for a certain number of epochs
        strat = np.zeros(self.num_strategies)
        for i in range(epochs):
            print("Epoch: ", i)
            strat += self.run(generations)

        # print the results
        self.print_results(strat)

if __name__ == "__main__":

    # Set the random seed for reproducibility (optional)
    np.random.seed()

    # Initialize assets

    # Manually created external assets with mean and standard deviation
    asset_array = [[0.02, 0.01], [0.04, 0.04], [0.06, 0.08], [0.08, 0.12], [0.1, 0.16]]

    assets_means = 19.66890448
    assets_std = 2.31054634
    liabilities_means = 19.45820445
    liabilities_std = 2.27696596

    num_banks = 250
    num_assets = 5
    num_rounds = 1000
    p = 0.1
    showGraph = False

    # Initialize game
    game = Game(num_banks, num_assets, num_rounds, asset_array, assets_means, assets_std, liabilities_means, liabilities_std, p, showGraph)

    EPOCHS = 10
    GENERATIONS = 50
    # Run game
    game.runMultiple(GENERATIONS, EPOCHS)


