from Stock import Stock
import matplotlib.pyplot as plt

stock = Stock("TSLA")
stock.set_close_data("1y")

num_days = 21
call_price = 12
num_walks = 10000
price_min = 200
price_max = 230

stock.set_close_percent_change()
stock.set_pdf_percent_change()
stock.set_walk_data(num_days, call_price, num_walks, load_from_save=True)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
fig.delaxes(ax4)

stock.plot_close(ax=ax1)
stock.plot_percent_change_pdf(ax=ax3)
stock.plot_percent_change(ax=ax2)

plt.savefig(f"{stock.stock}.png")

plt.clf()
stock.plot_walk(price_min, price_max)

plt.savefig(f"{stock.stock}_walk.png")