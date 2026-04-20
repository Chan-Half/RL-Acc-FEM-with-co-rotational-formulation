import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# The list of node_numbers you want to draw can be modified according to the actual data you have saved.
node_numbers = [30, 60, 50]

# base file direction
base_dir = "data/CMAME2_fix"

plt.figure(figsize=(10, 6), dpi=120)


for n in node_numbers:
    file_path = os.path.join(base_dir, f"N{n}-stable.pickle")

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)  # Data format [[run_time, min_length], ...]

        data_array = np.array(data)
        run_time = data_array[:, 0]  # X-axis: Time
        min_length = data_array[:, 1]  # Y-axis: Shortest distance

        plt.plot(run_time, min_length, linewidth=1.5, label=f'Distance of Beam to $P_0$')
    else:
        print(f"Note: File not found {file_path}")

baseline_path = os.path.join(base_dir, "baseline.pickle")

if os.path.exists(baseline_path):
    with open(baseline_path, "rb") as f:
        baseline_data = pickle.load(f)  # Data format [[run_time, scaled_factor*40.0], ...]

    baseline_array = np.array(baseline_data)
    base_run_time = baseline_array[:, 0]  # X-axis: baseline Time
    base_value = baseline_array[:, 1]  # Y-axis: Shortest baseline distance

    # Draw the baseline as a solid black line.
    plt.plot(base_run_time, base_value, color='black', linestyle='-', linewidth=2.5, label='Radius of Sphere')
else:
    print(f"Note: baseline File not found {baseline_path}")

plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Minimum Distance (mm)", fontsize=16)  
# plt.title("Minimum Length vs Run Time", fontsize=20)


plt.legend(loc='best', fontsize=14)

plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

plt.savefig(os.path.join(base_dir, "CMAME_exp2_results_fix.png"), dpi=650)  # save the fig

plt.show()

