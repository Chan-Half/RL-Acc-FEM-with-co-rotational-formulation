import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], # Times New Roman
    "font.size": 12,           
    "axes.labelsize": 14,     
    "xtick.labelsize": 12,    
    "ytick.labelsize": 12,  
    "figure.dpi": 300    
})


run_time = np.linspace(0, 5, 500)


beq_0_m = np.zeros_like(run_time)


for i, t in enumerate(run_time):
    if 0 <= t < 2.5:
        beq_0_m[i] = 0.04 * t
    elif 2.5 <= t <= 5:  
        beq_0_m[i] = 0.1 - 0.04 * (t - 2.5)


beq_0_mm = beq_0_m * 1000


fig, ax = plt.subplots(figsize=(5, 4))  


ax.plot(run_time, beq_0_mm, color='black', linewidth=2.0, linestyle='-')


ax.set_xlabel('Time (s)')
ax.set_ylabel('Displacement (mm)')


ax.set_xlim(0, 5)
ax.set_ylim(0, 120) 


ax.grid(True, linestyle='--', alpha=0.6)


plt.tight_layout()



plt.savefig('displacement_curve.png', dpi=300, bbox_inches='tight')


plt.show()
