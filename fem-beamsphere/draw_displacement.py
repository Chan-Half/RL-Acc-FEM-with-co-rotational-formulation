import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 论文格式设置 =================
# 设置字体格式，论文通常使用衬线字体 (如 Times New Roman)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], # 优先使用Times New Roman
    "font.size": 12,           # 全局字体大小
    "axes.labelsize": 14,      # 坐标轴标签字体大小
    "xtick.labelsize": 12,     # X轴刻度字体大小
    "ytick.labelsize": 12,     # Y轴刻度字体大小
    "figure.dpi": 300          # 设置高分辨率输出，满足期刊要求
})

# ================= 2. 生成数据 =================
# X轴数据: run_time，从0到5s，生成500个点以保证曲线平滑
run_time = np.linspace(0, 5, 500)

# Y轴数据: beq_0，初始化为一个与 run_time 长度相同、值全为0的数组
beq_0_m = np.zeros_like(run_time)

# 根据您提供的逻辑生成数据 (单位: m)
for i, t in enumerate(run_time):
    if 0 <= t < 2.5:
        beq_0_m[i] = 0.04 * t
    elif 2.5 <= t <= 5:  # 包含5秒这个点
        beq_0_m[i] = 0.1 - 0.04 * (t - 2.5)

# 单位转换：将米(m)转化为毫米(mm)
beq_0_mm = beq_0_m * 1000

# ================= 3. 绘制图表 =================
fig, ax = plt.subplots(figsize=(5, 4))  # 设置图片尺寸宽6英寸，高4英寸

# 绘制折线图，颜色设置为黑色(black)，线宽2.0更适合论文打印
ax.plot(run_time, beq_0_mm, color='black', linewidth=2.0, linestyle='-')

# 设置坐标轴标签
ax.set_xlabel('Time (s)')
ax.set_ylabel('Displacement (mm)')

# 设置坐标轴的范围
ax.set_xlim(0, 5)
ax.set_ylim(0, 120)  # 峰值在100，所以y轴最高设置到120留白好看些

# 添加网格线，使其半透明，增加可读性
ax.grid(True, linestyle='--', alpha=0.6)

# 调整布局防止标签被裁剪
plt.tight_layout()

# ================= 4. 保存与显示 =================
# 将图片保存为高分辨率矢量图或PNG格式（适合论文插入）
plt.savefig('displacement_curve.png', dpi=300, bbox_inches='tight')

# 显示图片
plt.show()