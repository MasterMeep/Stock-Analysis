import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.stats import lognorm
from Random_walk import Random_walk

plt.style.use('dark_background')

class Stock:
    def __init__(self, stock):
        self.stock = stock
        self.ticker = yf.Ticker(stock)
        
        self.period = None
        self.close_data = None
        self.last_price = None
        self.close_data_percent_change = None
        self.percent_change_mean = None
        self.percent_change_std = None
        self.pdf_percent_change = None
        self.walk_max_values = None
        self.breakeven = None
        self.num_walks = None
        
    def set_close_data(self, period):
        close = self.ticker.history(period=period, interval="1d")
        
        self.period = period
        self.close_data = close['Close']
        self.last_price = self.close_data[-1]
    
    def set_close_percent_change(self):
        close = self.close_data
        
        self.close_data_percent_change = close.pct_change().dropna()
        
        self.percent_change_mean = self.close_data_percent_change.mean()
        self.percent_change_std = self.close_data_percent_change.std()

    
    def set_pdf_percent_change(self):
        shape, loc, scale = lognorm.fit(self.close_data_percent_change)
        x = np.linspace(self.close_data_percent_change.min(), self.close_data_percent_change.max(), 100)
        
        self.pdf_percent_change = lognorm.pdf(x, shape, loc, scale)

    def set_walk_data(self, num_days, call_premium, num_walks, load_from_save=False):
        save_name = f"{self.stock}_Period_{self.period}_Num_Days_{num_days}_Call_Premium_{call_premium}_Num_Walks_{num_walks}"
        
        if load_from_save == False:
            random_walk = Random_walk(num_days, self.percent_change_mean, self.percent_change_std, self.last_price, num_walks)
            random_walk.walk(save_name=save_name)
            
        self.num_walks = num_walks
        self.walk_max_values = np.load(f"{save_name}.npy")
        self.breakeven = self.last_price + call_premium

    def plot_close(self, ax=None):
        
        if ax is None:
            plt.subplots(figsize=(16, 3))
            
            plt.plot(self.close_data)
            plt.ylabel("Close Price ($)")
            plt.title(f"{self.ticker.info['symbol']} Close Price Over {self.period} with End Price: {self.last_price:.2f}")

        else:
            ax.plot(self.close_data)
            ax.set_ylabel("Close Price ($)")
            ax.set_title(f"{self.ticker.info['symbol']} Close Price Over {self.period} with End Price: {self.last_price:.2f}")
            
    def plot_percent_change(self, ax=None):
        
        if ax is None:
            plt.subplots(figsize=(16, 3))
            
            plt.bar(self.close_data_percent_change.index, self.close_data_percent_change, color=['#9AC37A' if x >= 0 else '#FF9393' for x in self.close_data_percent_change])
            plt.ylabel("Percent Change")
            plt.xlabel("Date")
            plt.title(f"{self.ticker.info['symbol']} Percent Change Over {self.period}")
            plt.axhline(y=0, color='black', linestyle='dotted')
            
        else:
            ax.bar(self.close_data_percent_change.index, self.close_data_percent_change, color=['#9AC37A' if x >= 0 else '#FF9393' for x in self.close_data_percent_change])
            ax.set_ylabel("Percent Change")
            ax.set_xlabel("Date")
            ax.set_title(f"{self.ticker.info['symbol']} Percent Change Over {self.period}")
            ax.axhline(y=0, color='black', linestyle='dotted')
            
    def plot_percent_change_pdf(self, ax=None):
            x = np.linspace(self.close_data_percent_change.min(), self.close_data_percent_change.max(), 100)
            
            if ax is None:
                plt.subplots(figsize=(16, 3))
                
                plt.hist(self.close_data_percent_change, bins=20, color='skyblue', edgecolor='black', density=True, alpha=0.75, label='Percent Change')
                plt.plot(x, self.pdf_percent_change, color='red', label='Log-Normal Distribution')
                plt.xlabel("Percent Change")
                plt.ylabel("")
                plt.title(f"Log-normal distribution for percent change | mean {np.mean(self.close_data_percent_change):.2f} | std {np.std(self.close_data_percent_change):.2f}")
                
            else:
                ax.hist(self.close_data_percent_change, bins=20, color='skyblue', edgecolor='black', density=True, alpha=0.75)
                ax.plot(x, self.pdf_percent_change, color='red', label=f'Log-Normal Distribution | mean {np.mean(self.close_data_percent_change):.2f} | std {np.std(self.close_data_percent_change):.2f}')
                ax.set_xlabel("Percent Change")
                ax.set_ylabel("")
                ax.set_title(f"Log-normal distribution for percent change | mean {np.mean(self.close_data_percent_change):.2f} | std {np.std(self.close_data_percent_change):.2f}")
                
    def plot_walk(self, call_price_min=None, call_price_max=None, ax=None):
        plt.hist(self.walk_max_values, bins=40, color='skyblue', edgecolor='black', density=True, alpha=0.75)
        
        shape, loc, scale = lognorm.fit(self.walk_max_values)
        x = np.linspace(np.min(self.walk_max_values)-10, np.max(self.walk_max_values), 100)
        pdf = lognorm.pdf(x, shape, loc, scale)
        
        
        probability_of_profit = 1 - lognorm.cdf(self.breakeven, shape, loc, scale)
        probability_price_min = 1- lognorm.cdf(call_price_min, shape, loc, scale)
        
        most_probable = lognorm.mean(shape, loc, scale)
        probability_of_most_probable = lognorm.cdf(most_probable, shape, loc, scale)
        
        if ax==None:
            if call_price_min != None and call_price_max != None:
                probability_in_spread = lognorm.cdf(call_price_max, shape, loc, scale) - lognorm.cdf(call_price_min, shape, loc, scale)
                
                plt.axvline(x=call_price_min, color='green', linestyle='dashed', label=f'Call Min {call_price_min} $')
                plt.axvline(x=call_price_max, color='green', linestyle='dashed', label=f'Call Max {call_price_max} $')

                plt.fill_betweenx([plt.ylim()[0], plt.ylim()[1]], call_price_min, call_price_max, color='green', alpha=0.2)

            plt.axvline(x=most_probable, color='pink', linestyle='dashed', label=f'Most Probable MAX Future Price {most_probable:.2f} $ {probability_of_most_probable:.2f}%')
            plt.plot(x, pdf, color='red', label=f'Log-Normal Distribution | mean {np.mean(self.walk_max_values):.2f} | std {np.std(self.walk_max_values):.2f}')

            plt.axvline(x=self.breakeven, color='yellow', linestyle='dashed', label=f'Breakeven {self.breakeven:.2f} $')
            plt.axvline(x=self.last_price, color='white', linestyle='dashed', label=f'Last Close Price {self.last_price:.2f} $')

            plt.xlim(np.mean(self.walk_max_values) - 3 * np.std(self.walk_max_values), np.mean(self.walk_max_values) + 3 * np.std(self.walk_max_values))

            plt.xlabel("Maximum Value ($)")
            plt.ylabel("")  # Remove the y-axis label
            plt.title(f"Distribution of Maximum Values from {self.num_walks} Random Walks | Probability of Profit: {(probability_of_profit*100):.2f}% | Probability min: {probability_price_min*100:.2f}% | Probability in spread: {probability_in_spread*100:.2f}%")
            plt.legend()
            plt.gca().set_yticks([])
            
        else:
            if call_price_min != None and call_price_max != None:
                probability_in_spread = lognorm.cdf(call_price_max, shape, loc, scale) - lognorm.cdf(call_price_min, shape, loc, scale)
                
                ax.axvline(x=call_price_min, color='green', linestyle='dashed', label=f'Call Min {call_price_min} $')
                ax.axvline(x=call_price_max, color='green', linestyle='dashed', label=f'Call Max {call_price_max} $')

                ax.fill_betweenx([ax.get_ylim()[0], ax.get_ylim()[1]], call_price_min, call_price_max, color='green', alpha=0.2)

            ax.axvline(x=most_probable, color='pink', linestyle='dashed', label=f'Most Probable MAX Future Price {most_probable:.2f} $ {probability_of_most_probable:.2f}%')
            ax.plot(x, pdf, color='red', label=f'Log-Normal Distribution | mean {np.mean(self.walk_max_values):.2f} | std {np.std(self.walk_max_values):.2f}')

            ax.axvline(x=self.breakeven, color='yellow', linestyle='dashed', label=f'Breakeven {self.breakeven:.2f} $')
            ax.axvline(x=self.last_price, color='white', linestyle='dashed', label=f'Last Close Price {self.last_price:.2f} $')

            ax.set_xlabel("Maximum Value ($)")
            ax.set_ylabel("")
            
            ax.set_title(f"Distribution of Maximum Values | Probability of Profit: {(probability_of_profit*100):.2f}% | Most probable MAX Future Price: {most_probable:.2f} $")
            
            ax.set_xlim(np.mean(self.walk_max_values) - 3 * np.std(self.walk_max_values), np.mean(self.walk_max_values) + 3 * np.std(self.walk_max_values))