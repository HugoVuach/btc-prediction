def setup_plot():
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from globals import INDICATOR_CLASS, INDICATOR_YLIM
    from anim_functions import animate_2_plots, animate_3_plots, animate_ichimoku
    from indicators.ichimoku_indicator.ichimoku import Ichimoku

    if INDICATOR_YLIM is not None:
            # === Matplotlib : 3 sous-graphiques (chandeliers + volumes + indicateurs) ===

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        ani = animation.FuncAnimation(fig, lambda i: animate_3_plots(i, ax1, ax2, ax3), interval=1000)


    elif INDICATOR_CLASS is Ichimoku:
            # === Matplotlib : 2 sous-graphiques (chandeliers avec ichimoku + volumes) ===

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [4, 1.2]})
        ani = animation.FuncAnimation(fig, lambda i: animate_ichimoku(i, ax1, ax2), interval=1000)
    else:
            # === Matplotlib : 2 sous-graphiques (chandeliers avec indicateur + volumes ) ===

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [4, 1.2]})
        ani = animation.FuncAnimation(fig, lambda i: animate_2_plots(i, ax1, ax2), interval=1000)

    return plt, ani
