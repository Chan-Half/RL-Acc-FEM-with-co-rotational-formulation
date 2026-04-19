"""
###########################################################################################
#  @copyright: 中国科学院自动化研究所智能微创医疗技术实验室
#  @filename:  fem_monitor.py
#  @brief:     fem monitor function
#  @author:    Hao Chen
#  @version:   1.0
#  @date:      2023.04.03
#  @Email:     chen.hao2020@ia.ac.cn
###########################################################################################
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import psutil as p



def cpu_usage():
    t = p.cpu_times()
    return [t.user, t.system, t.idle]


before = cpu_usage()


def get_cpu_usage():
    global before
    now = cpu_usage()
    delta = [now[i] - before[i] for i in range(len(now))]
    total = sum(delta)
    before = now
    return [(100.0 * dt) / (total + 0.1) for dt in delta]


def OnTimer(ax):
    global cpu, sys1, idle, bg, co_mem, l_cpu, l_sys, l_idle
    tmp = get_cpu_usage()
    cpu = cpu[1:] + [tmp[0]]
    sys1 = sys1[1:] + [co_mem['fps'][0]]
    idle = idle[1:] + [p.virtual_memory().percent]
    l_cpu.set_ydata(cpu)
    l_sys.set_ydata(sys1)
    l_idle.set_ydata(idle)

    try:
        ax.draw_artist(l_cpu)
        ax.draw_artist(l_sys)
        ax.draw_artist(l_idle)
        # break
    except:
        pass

    ax.figure.canvas.draw()


def start_monitor(shared_fps, conn2):
    global cpu, sys1, idle, bg, co_mem, l_cpu, l_sys, l_idle
    co_mem = shared_fps
    POINTS = 300
    fig, ax = plt.subplots(facecolor='#1D1E23', edgecolor='#1D1E23')
    ax.set_ylim([0, 100])
    ax.set_xlim([0, POINTS])
    ax.set_autoscale_on(False)
    ax.set_xticks([])
    ax.set_yticks(range(0, 101, 10))
    ax.grid(True)
    # 执行用户进程的时间百分比
    cpu = [None] * POINTS
    # 执行内核进程和中断的时间百分比
    sys1 = [None] * POINTS
    # CPU处于空闲状态的时间百分比
    idle = [None] * POINTS
    l_cpu, = ax.plot(range(POINTS), cpu, color='#00D5E9', lw=2, label='cpu %')
    l_sys, = ax.plot(range(POINTS), sys1, label='fps ')
    l_idle, = ax.plot(range(POINTS), idle, label='Memory %')
    ax.legend(loc='upper center', ncol=4, prop=font_manager.FontProperties(size=10))
    # bg = fig.canvas.copy_from_bbox(ax.bbox)
    ax.set_facecolor('#373E4B')

    timer = fig.canvas.new_timer(interval=50)
    timer.add_callback(OnTimer, ax)
    timer.start()
    OnTimer(ax)
    plt.show()
