from multiprocessing import Process, Pipe
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import psutil as p
import multiprocessing as mp
from Forcesensor import Sensor, GetForce, GetTorque, ForceSensor

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
    global shared_force
    global cpu, sys1, idle, bg, l_force1, l_force2, l_force3, l_torque1, l_torque2, l_torque3
    force = shared_force['force']
    torque = shared_force['torque']
    tmp = get_cpu_usage()
    # cpu = cpu[1:] + [tmp[0]]
    # sys1 = sys1[1:] + [co_mem['fps'][0]]
    # idle = idle[1:] + [p.virtual_memory().percent]
    l_force1.set_ydata(force[0])
    l_force2.set_ydata(force[1])
    l_force3.set_ydata(force[2])
    l_torque1.set_ydata(torque[0])
    l_torque2.set_ydata(torque[1])
    l_torque3.set_ydata(torque[2])

    try:
        ax.draw_artist(l_force1)
        ax.draw_artist(l_force2)
        ax.draw_artist(l_force3)
        ax.draw_artist(l_torque1)
        ax.draw_artist(l_torque2)
        ax.draw_artist(l_torque3)
        # break

    except:
        pass

    ax.figure.canvas.draw()


def start_monitor(shared_force0):
    global shared_force
    global bg, co_mem, l_force1, l_force2, l_force3, l_torque1, l_torque2, l_torque3
    shared_force = shared_force0
    POINTS = 3000
    fig, ax = plt.subplots(facecolor='#1D1E23', edgecolor='#1D1E23')
    ax.set_ylim([-2, 2])
    ax.set_xlim([0, POINTS])
    ax.set_autoscale_on(False)
    ax.set_xticks([])
    ax.set_yticks(range(-2, 2, 10))
    ax.grid(True)
    # 执行用户进程的时间百分比

    force0 = [None] * POINTS
    torque0 = [None] * POINTS
    force1 = [None] * POINTS
    torque1 = [None] * POINTS
    force2 = [None] * POINTS
    torque2 = [None] * POINTS
    # 执行内核进程和中断的时间百分比

    sys1 = [None] * POINTS
    # CPU处于空闲状态的时间百分比
    idle = [None] * POINTS

    # l_cpu, = ax.plot(range(POINTS), cpu, color='#00D5E9', lw=2, label='cpu %')
    # l_sys, = ax.plot(range(POINTS), sys1, label='fps ')
    # l_idle, = ax.plot(range(POINTS), idle, label='Memory %')
    l_force1, = ax.plot(range(POINTS), force0, label='Fx %')
    l_force2, = ax.plot(range(POINTS), force1, label='Fy ')
    l_force3, = ax.plot(range(POINTS), force2, label='Fz %')
    l_torque1, = ax.plot(range(POINTS), torque0, label='Tx %')
    l_torque2, = ax.plot(range(POINTS), torque1, label='Ty ')
    l_torque3, = ax.plot(range(POINTS), torque2, label='Tz %')
    ax.legend(loc='upper center', ncol=4, prop=font_manager.FontProperties(size=10))
    # bg = fig.canvas.copy_from_bbox(ax.bbox)

    ax.set_facecolor('#373E4B')
    timer = fig.canvas.new_timer(interval=50)
    timer.add_callback(OnTimer, ax)
    timer.start()
    OnTimer(ax)
    plt.show()


if __name__ == '__main__':
    manager = mp.Manager()
    shared_force = manager.dict({})

    shared_force['force'] = manager.list([0, 0, 0])
    shared_force['torque'] = manager.list([0, 0, 0])
    # model = FEM()

    # A = aa()

    process = [Process(target=start_monitor, args=(shared_force,)),
               Process(target=ForceSensor, args=(shared_force,)),
               # Process(target=A.pp3, args=(ss, )),
               ]
    [p.start() for p in process]
    [p.join() for p in process]
