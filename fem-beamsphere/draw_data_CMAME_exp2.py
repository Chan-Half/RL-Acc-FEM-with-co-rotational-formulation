import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 您想要绘制的 node_number 列表，可根据您实际保存的数据修改
node_numbers = [30, 60, 50]

# 基础路径
base_dir = "data/CMAME2_fix"

plt.figure(figsize=(10, 6), dpi=120)

# ================= 1. 读取并绘制各个 node_number 的曲线 =================
for n in node_numbers:
    file_path = os.path.join(base_dir, f"N{n}-stable.pickle")

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)  # 数据格式 [[run_time, min_length], ...]

        data_array = np.array(data)
        run_time = data_array[:, 0]  # X轴: 时间
        min_length = data_array[:, 1]  # Y轴: 最短距离

        plt.plot(run_time, min_length, linewidth=1.5, label=f'Distance of Beam to $P_0$')
    else:
        print(f"提示：未找到文件 {file_path}")

# ================= 2. 读取并绘制 baseline 曲线 =================
baseline_path = os.path.join(base_dir, "baseline.pickle")

if os.path.exists(baseline_path):
    with open(baseline_path, "rb") as f:
        baseline_data = pickle.load(f)  # 数据格式 [[run_time, scaled_factor*40.0], ...]

    baseline_array = np.array(baseline_data)
    base_run_time = baseline_array[:, 0]  # X轴: baseline对应的时间
    base_value = baseline_array[:, 1]  # Y轴: baseline对应的数值

    # 将 baseline 绘制为黑色实线
    plt.plot(base_run_time, base_value, color='black', linestyle='-', linewidth=2.5, label='Radius of Sphere')
else:
    print(f"提示：未找到基准线文件 {baseline_path}")

# ================= 3. 图表格式设置 =================
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Minimum Distance (mm)", fontsize=16)  # 您的保存代码里乘了1000，单位应为mm
# plt.title("Minimum Length vs Run Time", fontsize=20)

# 显示图例 (自动汇集 node 和 baseline 的 label)
plt.legend(loc='best', fontsize=14)

# 开启网格线，使图表更易读
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

plt.savefig(os.path.join(base_dir, "CMAME_exp2_results_fix.png"), dpi=300)  # 保存图像到文件

plt.show()

