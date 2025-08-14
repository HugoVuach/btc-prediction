from ws_handler import start_websocket
from plot_setup import setup_plot

if __name__ == "__main__":
    start_websocket()
    plt, ani = setup_plot() 
    plt.show()
