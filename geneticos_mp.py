# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 00:54:01 2021

@author: gorka
"""

import random
import csv
import numpy as np
import pandas as pd
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

class Symbol_long(object):

    def __init__(self, params) -> None:
        
        
        self.drop = params[0]/100   # Porcentaje del precio que tiene que caer para comprar
        self.profit = params[1]/100   # Porcentaje del precio que tiene que subir para vender
        self.status = True
        self.side = 1

        self.buy_amount = 1
        self.buy_amount_list = np.array([])
        self.buy_price_list = np.array([])
        self.profit_list = np.array([])
        self.acc = 0

        self.buy_point = 0.00001
        self.sell_point = 100000
        self.average_price = []
        self.buy_level = 0
        
        self.fund_list = []
        self.fund_time = []
        self.equity_list = []
        self.equity_time = []
        
        self.equity_abs = 0
        self.max_acc = 0
        self.rent_abs = 0
        self.num_ops = 0


    def buy_order(self, price):
        
        self.buy_amount_list = np.append(self.buy_amount_list, [self.buy_amount])
        self.buy_price_list = np.append(self.buy_price_list, [price])
        self.acc = np.sum([self.buy_amount_list])
        self.average_price = round(np.dot(self.buy_price_list, self.buy_amount_list)/self.acc)
        self.fund_list = np.append(self.fund_list, [-self.buy_amount])
        #print('#'*30)
        #print('#BUY_ORDER', self.coin)
        #print('Amount: ', self.buy_amount)
        
        self.buy_amount = self.buy_amount * 2
        self.buy_level = self.buy_level + 1
        self.trading_points(price)
        self.max_acc = max(self.max_acc,float(self.acc))

        return

    def sell_order(self, price):
        op_profit = np.around(self.acc * (price/self.average_price - 1),2)
        self.profit_list = np.append(self.profit_list, [op_profit])
        op_fund = self.acc 
        self.fund_list = np.append(self.fund_list, [op_fund])
        self.equity_list = np.append(self.equity_list, [op_profit])
        
        #print('#'*30)
        #print('#SELL_ORDER', time)
        #print('Buy Level: ', self.buy_level)
        #print('Amount: ', self.acc)
        #print('Profit: ', op_profit)
        
        self.buy_amount_list = []
        self.buy_price_list = []
        self.buy_amount = 1
        self.buy_level = 1
        self.trading_points(price)
        self.acc = 0
        return

    def trading_points(self, price):

        if len(self.buy_amount_list) == 0:
            self.buy_point = price * (1 - self.drop/100)
            self.average_price = price
            self.sell_point = 100000
        
        else:
            self.buy_point = price * (1 - self.drop/100)
            self.average_price = round(np.dot(self.buy_price_list, self.buy_amount_list)/self.acc)
            self.sell_point = round(self.average_price * (1 + self.profit/100))
        
        return 

    def backtest(self):
        
        #btc_data = np.genfromtxt('prices3.csv', dtype=None, delimiter=',', encoding=None)
        csvarchivo = open('prices.csv')
        
        btc_data = csv.reader(csvarchivo)


        for row in btc_data:
            
            price = float(row[1])
            time = row[0]
            #print(time, price)
            
            if price < self.buy_point:
                self.buy_order(price)
                self.fund_time.append(time)
                
            elif price > self.sell_point:
                self.sell_order(price)
                self.fund_time.append(time)
                self.equity_time.append(time)
                
            else:
                if len(self.buy_amount_list) == 0:
                    self.buy_point = max(price*(1-self.drop/100), self.buy_point)
        
        equity_abs = round(sum(self.profit_list),3)
        num_ops = len(self.equity_list)
        order_num = len(self.fund_list)
        rent_order = round(equity_abs/order_num*100,3)
        rent_anual = round(((1 + equity_abs/self.max_acc)**(1/3.6) - 1)*100,2)
        
        rent_anual_df.append(rent_anual)
        rent_order_df.append(rent_order)
        max_acc_df.append(self.max_acc)
        num_ops_df.append(num_ops)
        drop_df.append(self.drop)
        profit_df.append(self.profit)
        side_df.append(self.side)

        resultado = [rent_anual, rent_order, self.max_acc, num_ops, self.drop, self.profit, self.side]
        #print(resultado)
        
        return rent_anual
    
class Symbol_short(object):

    def __init__(self, params) -> None:
        
        
        self.drop = -params[0]/100   # Porcentaje del precio que tiene que caer para comprar
        self.profit = params[1]/100   # Porcentaje del precio que tiene que subir para vender
        self.status = True
        self.coin = 'Symbol__' + str(self.drop) + '--' + str(self.profit)
        self.side = -1

        self.buy_amount = 1
        self.buy_amount_list = np.array([])
        self.buy_price_list = np.array([])
        self.profit_list = np.array([])
        self.acc = 0

        self.buy_point = 100000
        self.sell_point = 0.00001
        self.average_price = []
        self.buy_level = 0
        
        self.fund_list = []
        self.fund_time = []
        self.equity_list = []
        self.equity_time = []
        
        self.equity_abs = 0
        self.max_acc = 0
        self.rent_abs = 0
        self.num_ops = 0


    def buy_order(self, price):
        
        self.buy_amount_list = np.append(self.buy_amount_list, [self.buy_amount])
        self.buy_price_list = np.append(self.buy_price_list, [price])
        self.acc = np.sum([self.buy_amount_list])
        self.average_price = round(np.dot(self.buy_price_list, self.buy_amount_list)/self.acc)
        self.fund_list = np.append(self.fund_list, [self.buy_amount])
        """
        print('#'*30)
        print('#BUY_ORDER', price)
        print('Amount: ', self.buy_amount)
        """
        
        self.buy_amount = self.buy_amount * 2
        self.buy_level = self.buy_level + 1
        self.trading_points(price)
        
        self.max_acc = max(self.max_acc,float(self.acc))

        """
        print('Buy Level: ', self.buy_level)
        print('Accumulated: ', self.acc)
        print('Buy Point: ', round(self.buy_point), 'Average Price: ', round(self.average_price), 'Sell Point: ', round(self.sell_point))
        """
        return
    

    def sell_order(self, price):
        op_profit = np.around(self.acc * (self.average_price/price - 1),2)
        self.profit_list = np.append(self.profit_list, [op_profit])
        #op_fund = -self.acc + op_profit
        op_fund = -self.acc
        self.fund_list = np.append(self.fund_list, [op_fund])
        #master.fund_time = np.append(master.fund_time, [time])
        self.equity_list = np.append(self.equity_list, [op_profit])
        #master.equity_time.append(time)
        
        """
        print('#'*30)
        print('#SELL_ORDER', price)
        print('Buy Level: ', self.buy_level)
        print('Amount: ', self.acc)
        print('Profit: ', op_profit)
        """
        
        #transaction = {'symbol' : str(self.coin), 'amount' : self.acc, 'buy level' : self.buy_level, 'profit' : self.op_profit}
        #master.transaction_list.append(transaction)
        #master.profit.append(float(self.op_profit))
        
        #entrada=pd.DataFrame([(datetime.today(), 'Sell', self.acc, self.op_profit)], columns = ['Time' , 'Side', 'Amount', 'Profit'])
        #df=df.append(entrada,ignore_index=True)
        
        self.buy_amount_list = []
        self.buy_price_list = []
        self.buy_amount = 1
        self.buy_level = 1
        self.trading_points(price)
        self.acc = 0
        return

    def trading_points(self, price):

        if len(self.buy_amount_list) == 0:
            self.buy_point = price * (1 + self.drop/100)
            self.average_price = price
            self.sell_point = 0.00001
        
        else:
            self.buy_point = price * (1 + self.drop/100)
            self.average_price = round(np.dot(self.buy_price_list, self.buy_amount_list)/self.acc)
            self.sell_point = round(self.average_price * (1 - self.profit/100))
        
        return 
    
    def backtest(self):
        
        csvarchivo = open('prices.csv')
        btc_data = csv.reader(csvarchivo)
        
        for row in btc_data:
            
            price = float(row[1])
            time = row[0]
            #print(time, price)
            
            if price > self.buy_point:
                self.buy_order(price)
                self.fund_time.append(time)
                
            elif price < self.sell_point:
                self.sell_order(price)
                self.fund_time.append(time)
                self.equity_time.append(time)
                
            else:
                if len(self.buy_amount_list) == 0:
                    self.buy_point = min(price*(1+self.drop/100), self.buy_point)
       
        equity_abs = round(sum(self.equity_list),3)
        #rent_abs = round(equity_abs/self.max_acc*100,2)
        num_ops = len(self.equity_list)
        #equity_plt = np.cumsum(self.profit_list)
        order_num = len(self.fund_list)
        rent_order = round(equity_abs/order_num*100,3)
        rent_anual = round(((1 + equity_abs/self.max_acc)**(1/3.6) - 1)*100,2)
                
        rent_anual_df.append(rent_anual)
        rent_order_df.append(rent_order)
        max_acc_df.append(self.max_acc)
        num_ops_df.append(num_ops)
        drop_df.append(-self.drop)
        profit_df.append(self.profit)
        side_df.append(self.side)
        
        resultado = [rent_anual, rent_order, self.max_acc, num_ops, -self.drop, self.profit, self.side]
        #print(resultado)
        return rent_anual


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_drop", random.randrange, -800, 800, 5)
toolbox.register("attr_profit", random.randrange, 1, 401, 5)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_drop, toolbox.attr_profit), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


rent_anual_df = []
rent_order_df = []
max_acc_df = []
num_ops_df = []
drop_df = []
profit_df = []
side_df = []

num_individuos = 3000
num_generaciones = 5

def evalOneMax(individual):
    
    #print('individual: ', individual)
    
    if individual[0] > 0:
        symbol = Symbol_long(individual)
    else:
        symbol = Symbol_short(individual)
        
    rent = symbol.backtest()
    
    return rent,


def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = np.array((5,6,7,8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2
    
    
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)
    
    pop = toolbox.population(n=num_individuos)
    
    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # np.array_equal or np.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=np.array_equal)
    
    
    # Rentabilidad
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
        
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=num_generaciones, stats=stats,
                        halloffame=hof)

    return pop, stats, hof

if __name__ == "__main__":
    main()


filename = 'geneticos_indiv.xlsx'
df = pd.DataFrame({'Rentabilidad' : rent_anual_df, 'Rentabilidad Orden' : rent_order_df, 'Acc Max' : max_acc_df, 'N Ops' : num_ops_df, 'Drop' : drop_df, 'Profit' : profit_df, 'Side' : side_df})
df.to_excel(filename)

