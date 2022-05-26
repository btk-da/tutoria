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
                total_funds_time.append(time)

                
            elif price > self.sell_point:
                self.sell_order(price)
                self.fund_time.append(time)
                self.equity_time.append(time)
                total_equity_time.append(time)
                total_funds_time.append(time)
                
            else:
                if len(self.buy_amount_list) == 0:
                    self.buy_point = max(price*(1-self.drop/100), self.buy_point)
        
        equity_abs = round(sum(self.profit_list),3)
        num_ops = len(self.equity_list)
        order_num = len(self.fund_list)
        rent_order = round(equity_abs/order_num*100,3)
        rent_anual = round(((1 + equity_abs/self.max_acc)**(1/3.6) - 1)*100,2)
        
        total_equity.append(self.equity_list)
        total_funds.append(self.fund_list)
        
        rent_anual_df.append(rent_anual)
        rent_order_df.append(rent_order)
        max_acc_df.append(self.max_acc)
        num_ops_df.append(num_ops)
        drop_df.append(self.drop)
        profit_df.append(self.profit)
        side_df.append(self.side)

        resultado = [rent_anual, rent_order, self.max_acc, num_ops, self.drop, self.profit, self.side]
        #print(resultado)
        
        return rent_anual, self.max_acc
    
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
                total_funds_time.append(time)

                
            elif price < self.sell_point:
                self.sell_order(price)
                self.fund_time.append(time)
                self.equity_time.append(time)
                total_equity_time.append(time)
                total_funds_time.append(time)
                
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
                
        total_equity.append(self.equity_list)
        total_funds.append(self.fund_list)
        
        rent_anual_df.append(rent_anual)
        rent_order_df.append(rent_order)
        max_acc_df.append(self.max_acc)
        num_ops_df.append(num_ops)
        drop_df.append(-self.drop)
        profit_df.append(self.profit)
        side_df.append(self.side)
        
        resultado = [rent_anual, rent_order, self.max_acc, num_ops, -self.drop, self.profit, self.side]
        #print(resultado)
        return rent_anual, self.max_acc


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_drop", random.randrange, -800, 800, 8)
toolbox.register("attr_profit", random.randrange, 1, 401, 8)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_drop, toolbox.attr_profit), n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


total_equity = []
total_funds = []
total_equity_time = []
total_funds_time = []

rent_anual_df = []
rent_order_df = []
max_acc_df = []
num_ops_df = []
drop_df = []
profit_df = []
side_df = []

combi_rent_anual_df = []
combi_rent_order_df = []
combi_max_acc_df = []
combi_num_ops_df = []
combi_n_df = []
combi_indiv_df = []

num_individuos = 1000
num_generaciones = 3


def evalOneMax(individual):
    
    num = random.randint(1,10)    
    #print('individual: ', list(individual), 'n: ', n)
    
    l1 = [individual[0], individual[1]]
    l2 = [individual[2], individual[3]]
    l3 = [individual[4], individual[5]]
    l4 = [individual[6], individual[7]]
    l5 = [individual[8], individual[9]]
    l6 = [individual[10], individual[11]]
    l7 = [individual[12], individual[13]]
    l8 = [individual[14], individual[15]]
    l9 = [individual[16], individual[17]]
    l10 = [individual[18], individual[19]]
    
    combi_list_total = [l1,l2,l3,l4,l5,l6,l7,l8,l9,l10]
    combi_list = combi_list_total[:num]
    
    combi = []
    
    for item in combi_list:
        
        if item[0] > 0:
            symbol = Symbol_long(item)
        else:
            symbol = Symbol_short(item)
            
        res = symbol.backtest()
        combi = combi + [res]
        #print('combi: ', combi)

    back_test = pd.DataFrame(combi, columns = ['Rentabilidad', 'Acc Max'])

    pond_base = back_test['Acc Max'][0]
    pond_list = []

    for i in back_test['Acc Max']:
        pond_param = i/pond_base
        pond_list.append(pond_param)

    equity_col = []

    for n in list(range(len(total_equity))):   
        for i in total_equity[n]:
            new = i/pond_list[n]
            equity_col.append(new)

    equity_df = pd.DataFrame({'Fecha' : total_equity_time, 'Equity' : equity_col})
    equity_df = equity_df.sort_values(by='Fecha')
    equity_df = equity_df.reset_index()
    
    funds_col = []
    for n in list(range(len(total_funds))):   
        for i in total_funds[n]:
            new = i/pond_list[n]
            funds_col.append(new)  
    
    funds_df = pd.DataFrame({'Fecha' : total_funds_time, 'Funds' : funds_col})
    funds_df = funds_df.sort_values(by='Fecha')

    acc_abs = [abs(ele) for ele in np.cumsum(funds_df['Funds'])]
    max_acc = round(max(acc_abs),2)
    equity_abs = round(sum(equity_df['Equity']),3)
    #rent_abs = round(equity_abs/max_acc*100,2)
    num_ops = len(equity_df['Equity'])
    #equity_plt = np.cumsum(equity_df['Equity'])
    order_num = len(funds_df['Funds'])
    rent_order = round(equity_abs/order_num*100,3)
    rent_anual = round(((1 + equity_abs/max_acc)**(1/3.6) - 1)*100,2)
    
    #print('Resultados totales')
    print('Rent', rent_anual, 'rent order', rent_order, 'max acc', max_acc, 'num ops', num_ops, 'n: ', num)  
    
    combi_rent_anual_df.append(rent_anual)
    combi_rent_order_df.append(rent_order)
    combi_max_acc_df.append(max_acc)
    combi_num_ops_df.append(num_ops)
    combi_n_df.append(num)
    combi_indiv_df.append(combi_list)
    
    
    total_equity.clear()
    total_funds.clear()
    total_equity_time.clear()
    total_funds_time.clear()
    
    """
    writer = pd.ExcelWriter('backtest_acc1.xlsx')
    funds_df.to_excel(writer, sheet_name="funds", index=False)
    equity_df.to_excel(writer, sheet_name="equity", index=False)
    writer.save()
    """
    
    return rent_anual,


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


indiv_df = pd.DataFrame({'Rentabilidad' : rent_anual_df, 'Rentabilidad Orden' : rent_order_df, 'Acc Max' : max_acc_df, 'N Ops' : num_ops_df, 'Drop' : drop_df, 'Profit' : profit_df, 'Side' : side_df})
combi_df = pd.DataFrame({'Rentabilidad' : combi_rent_anual_df, 'Rentabilidad Orden' : combi_rent_order_df, 'Acc Max' : combi_max_acc_df, 'N Ops' : combi_num_ops_df, 'n' : combi_n_df, 'Individuals' : combi_indiv_df})

writer = pd.ExcelWriter('geneticos.xlsx')
indiv_df.to_excel(writer, sheet_name="indiv", index=False)
combi_df.to_excel(writer, sheet_name="combi", index=False)
writer.save()
"""
gen	nevals	avg    	std    	min 	max 
0  	1000  	6.69986	3.01989	0.34	18.3
"""