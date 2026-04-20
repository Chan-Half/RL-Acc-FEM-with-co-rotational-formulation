"""
###########################################################################################
#  @filename:  fem_show_exp.py
#  @brief:     fem show result of exp
#  @author:    Hao Chen
#  @version:   1.0
#  @date:      2026.04.03
#  @Email:     chen.hao2020@ia.ac.cn
###########################################################################################
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def OnTimer(ax):
    global x, v, a, dE, t, co_mem, l_x, l_v, l_a, l_dE
    if co_mem['x'] is not None:
        x = x[1:] + [co_mem['x'][0]]
        v = v[1:] + [co_mem['v'][0]]
        a = a[1:] + [co_mem['a'][0]]
        dE = dE[1:] + [co_mem['dE'][0]]
        t = t[1:] + [co_mem['t'][0]]
    l_x.set_ydata(x)
    l_v.set_ydata(v)
    l_a.set_ydata(a)
    l_dE.set_ydata(dE)

    l_x.set_xdata(t)
    l_v.set_xdata(t)
    l_a.set_xdata(t)
    l_dE.set_xdata(t)
    if t[-1] > 10:
        ax.set_xlim([t[-1]-10, t[-1]])
    try:
        ax.draw_artist(l_dE)
        ax.draw_artist(l_x)
        ax.draw_artist(l_v)
        ax.draw_artist(l_a)

        # break
    except:
        pass

    ax.figure.canvas.draw()


def monitor_state(shared_state):
    global x, v, a, dE, t, co_mem, l_x, l_v, l_a, l_dE
    co_mem = shared_state
    POINTS = 1000
    fig, ax = plt.subplots(facecolor='#1D1E23', edgecolor='#1D1E23')
    ax.set_ylim([-90, 90])
    ax.set_xlim([0, 10])
    ax.set_autoscale_on(False)
    ax.set_xticks([])
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(True)
    # 执行用户进程的时间百分比
    dE = [None] * POINTS
    x = [None] * POINTS
    v = [None] * POINTS
    a = [None] * POINTS
    t = [None] * POINTS
    l_x, = ax.plot(range(POINTS), x, lw=2, label='x ')
    l_v, = ax.plot(range(POINTS), v, label='v')
    l_a, = ax.plot(range(POINTS), a, label='a')
    l_dE, = ax.plot(range(POINTS), dE, label='dE')
    ax.legend(loc='upper center', ncol=5, prop=font_manager.FontProperties(size=10))
    # bg = fig.canvas.copy_from_bbox(ax.bbox)
    ax.set_facecolor('#373E4B')

    timer = fig.canvas.new_timer(interval=50)
    timer.add_callback(OnTimer, ax)
    timer.start()
    OnTimer(ax)
    plt.show()

