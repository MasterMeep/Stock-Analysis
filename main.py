from Stock import Stock
import matplotlib.pyplot as plt

time_period = "6mo"

stock = Stock("MSFT")
stock.set_close_data(time_period)

num_days = 21
call_price = 15
num_walks = 10000

stock.set_close_percent_change()
stock.set_pdf_percent_change()
stock.set_walk_data(num_days, call_price, num_walks, load_from_save=False)
stock.set_random_walk_pdf()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

stock.plot_walk(ax=ax1)
stock.plot_one_minus_cdf(stock_value = 900, ax=ax2)

plt.savefig(f"{stock.stock}_dataFrom_{time_period}_daysForcast_{num_days}_walk_cdf.png")

plt.clf()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
fig.delaxes(ax4)

stock.plot_close(ax=ax1)
stock.plot_percent_change_pdf(ax=ax3)
stock.plot_percent_change(ax=ax2)

plt.savefig(f"{stock.stock}_dataFrom_{time_period}_close_percent_change_and_pdf.png")