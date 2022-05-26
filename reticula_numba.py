# -*- coding: utf-8 -*-
"""
Created on Tue May 24 18:07:56 2022

@author: gorka
"""


import csv
import numpy as np
import pandas as pd
import time
import multiprocessing
import itertools
from numba import jit

#dataset = 'prices.csv'
df = pd.read_csv('prices.csv')
arr = df.to_numpy()

@jit(nopython=True)
def martingala_long(params, btc_data):
    
    drop = params[0]   # Porcentaje del precio que tiene que caer para comprar
    profit = params[1]   # Porcentaje del precio que tiene que subir para vender
    
    buy_amount = 1
    buy_amount_list = np.array([])
    buy_price_list = np.array([])
    profit_list = np.array([])
    acc = 0

    buy_point = 0.00001
    sell_point = 100000
    average_price = []
    buy_level = 0
    
    fund_list = []
    fund_time = []
    equity_list = []
    equity_time = []
    
    equity_abs = 0
    max_acc = 0
    num_ops = 0
    
    #btc_data = np.genfromtxt('prices.csv', dtype=None, delimiter=',', encoding=None)
    #csvarchivo = open(dataset)
    #btc_data = csv.reader(csvarchivo)

    for row in btc_data:
        
        price = float(row[1])
        time = row[0]
        #print(time, price)
        
        if price < buy_point:
            
            buy_amount_list = np.append(buy_amount_list, [buy_amount])
            buy_price_list = np.append(buy_price_list, [price])
            acc = np.sum([buy_amount_list])
            average_price = round(np.dot(buy_price_list, buy_amount_list)/acc)
            fund_list = np.append(fund_list, [-buy_amount])
            #print('#'*30)
            #print('#BUY_ORDER')
            #print('Amount: ', buy_amount)
            buy_amount = buy_amount * 2
            buy_level = buy_level + 1
            max_acc = max(max_acc,float(acc))            
            fund_time.append(time)
            
            buy_point = price * (1 - drop/100)
            average_price = round(np.dot(buy_price_list, buy_amount_list)/acc)
            sell_point = round(average_price * (1 + profit/100))
            
            
        elif price > sell_point:
            
            op_profit = np.around(acc * (price/average_price - 1),2)
            profit_list = np.append(profit_list, [op_profit])
            op_fund = acc 
            fund_list = np.append(fund_list, [op_fund])
            equity_list = np.append(equity_list, [op_profit])
            
            #print('#'*30)
            #print('#SELL_ORDER', time)
            #print('Buy Level: ', buy_level)
            #print('Amount: ', acc)
            #print('Profit: ', op_profit)
            
            buy_amount_list = []
            buy_price_list = []
            buy_amount = 1
            buy_level = 1
            acc = 0
            fund_time.append(time)
            equity_time.append(time)
            
            buy_point = price * (1 + drop/100)
            average_price = price
            sell_point = 0.00001
            
            
        else:
            if len(buy_amount_list) == 0:
                buy_point = max(price*(1-drop/100), buy_point)
    
    equity_abs = round(sum(profit_list),3)
    num_ops = len(equity_list)
    order_num = len(fund_list)
    rent_order = round(equity_abs/order_num*100,5)
    rent_anual = ((1 + equity_abs/max_acc)**(1/3.6) - 1)*100

    resultado = [rent_anual, rent_order, max_acc, num_ops, drop, profit, 1]
    print(resultado)

    return resultado


drop = np.around(np.linspace(1, 2, num=3, endpoint=True),2)  
profit = np.around(np.linspace(1, 2, num=3, endpoint=True),2)
product = list(itertools.product(drop, profit))    # Todas las combinaciones posibles


start = time.time()
for i in product:
    martingala_long(i, arr)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))


# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
for i in product:
    martingala_long(i, arr)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))



"""
def __init__(self, params) -> None:
    
    
    self.drop = -params[0]   # Porcentaje del precio que tiene que caer para comprar
    self.profit = params[1]   # Porcentaje del precio que tiene que subir para vender

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
    
    print('#'*30)
    print('#BUY_ORDER', price)
    print('Amount: ', self.buy_amount)
    
    
    self.buy_amount = self.buy_amount * 2
    self.buy_level = self.buy_level + 1
    self.trading_points(price)
    
    self.max_acc = max(self.max_acc,float(self.acc))


    print('Buy Level: ', self.buy_level)
    print('Accumulated: ', self.acc)
    print('Buy Point: ', round(self.buy_point), 'Average Price: ', round(self.average_price), 'Sell Point: ', round(self.sell_point))
    
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
    
    
    print('#'*30)
    print('#SELL_ORDER', price)
    print('Buy Level: ', self.buy_level)
    print('Amount: ', self.acc)
    print('Profit: ', op_profit)
    
    
    
    
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
    
    #start = time.time()
    csvarchivo = open(dataset)
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
    rent_abs = round(equity_abs/self.max_acc*100,2)
    num_ops = len(self.equity_list)
    equity_plt = np.cumsum(self.profit_list)
    order_num = len(self.fund_list)
    rent_order = round(equity_abs/order_num*100,5)
    rent_anual = round(((1 + equity_abs/self.max_acc)**(1/3.6) - 1)*100,2)
    
    
    resultado = [rent_anual, rent_order, self.max_acc, num_ops, self.drop, self.profit, -1]
    return resultado

# Initialize

#@jit(nopython=True)


def iniciar(params):
    
    if params[0] > 0:
        symbol = Symbol_long(params)
    else:
        symbol = Symbol_short(params)
        
    res = symbol.backtest()
    return res
"""


"""
def reticula():
    
    #drop = np.around(np.linspace(-8, 8, num=200, endpoint=True),2)  
    #profit = np.around(np.linspace(0.01, 4.01, num=50, endpoint=True),2)
    drop = np.around(np.linspace(-2, 2, num=2, endpoint=True),2)  
    profit = np.around(np.linspace(1, 2, num=2, endpoint=True),2)
    product = list(itertools.product(drop, profit))    # Todas las combinaciones posibles
    
    
    #param_list = [[1.9,1.2], [0.4,2.1], [1,2], [2,3], [0.3,1], [1.4,0.7], [0.7,0.7], [1.44,1]]
    
    if __name__ == '__main__':
    
        start = time.time()
        pool = multiprocessing.Pool(processes=8)
        resultados = pool.map(martingala_long, product)
        #print(resultados)
        elapsed_time = time.time() - start
        print('total time:', elapsed_time)
        
        back_test = pd.DataFrame(resultados)
        back_test.columns = ['Rentabilidad', 'Rentabilidad Orden', 'Acc Max', 'N Ops', 'Drop', 'Profit', 'Side'] 
    
        back_test.to_excel('combi_numba' + str(dataset) + '.xlsx')
    
    return
"""
 
#csvarchivo = open(dataset)
#btc_data = list(csv.reader(csvarchivo))
#btc_data = csv.reader(csvarchivo)
#data = genfromtxt('sample.csv', delimiter=',', skip_header = 1)
#btc_data = np.recfromcsv(dataset)
#btc_data = np.genfromtxt('prices3.csv', dtype=None, delimiter=',', encoding=None)


