import numpy as np

class Random_walk():
    def __init__(self, num_days, percent_change_pdf_mean, percent_change_pdf_std, initial_price, num_walks):
        self.num_days = num_days
        self.num_walks = num_walks
        self.percent_change_pdf_mean = percent_change_pdf_mean
        self.percent_change_pdf_std = percent_change_pdf_std
        self.initial_price = initial_price
        self.num_walks = num_walks
                
        self.max_values = None
        
    def walk(self, save_name):
        max_values = []
        
        for _ in range(self.num_walks):
            percent_samples = np.random.normal(self.percent_change_pdf_mean, self.percent_change_pdf_std, self.num_days)
            walk = np.cumprod(1 + percent_samples) * self.initial_price
            max_value = np.max(walk)
            max_values.append(max_value)
            
        self.max_values = max_values
        
        np.save(f"{save_name}.npy", max_values)