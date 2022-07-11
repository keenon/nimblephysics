import matplotlib.pyplot as plt
import numpy as np
import csv
from typing import List


def plot(file: str, joints: List[str], nrows=1, ncols=1, errorPlot=False, limitTimesteps=-1):
    with open(file, newline='') as f:
        reader = csv.DictReader(f)

        rows = [row for row in reader]
        if limitTimesteps != -1:
            rows = rows[:limitTimesteps]

        time = np.asarray([float(row['time']) for row in rows])

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

        # plot
        fig: plt.Figure
        axs: List[plt.Axes]
        if ncols == 1 and nrows == 1 and (len(joints) > 1 or errorPlot):
            nrows = len(joints) + (1 if errorPlot else 0)
        fig, axs = plt.subplots(nrows, ncols)
        fig.set_size_inches(8, 9)
        axs = np.ndarray.flatten(axs)
        errAx: plt.Axes = axs[len(axs) - 1]

        globalMin = 0
        globalMax = 0
        for i in range(len(joints)):
            joint: str = joints[i]
            gold = np.asarray([float(row[joint+'_gold'])
                              for row in rows]) * 180.0 / 3.14159
            rec = np.asarray([float(row[joint+'_rec'])
                             for row in rows]) * 180.0 / 3.14159
            globalMin = min(globalMin, min(gold + rec))
            globalMax = max(globalMax, max(gold + rec))

        for i in range(len(joints)):
            joint: str = joints[i]
            gold = np.asarray([float(row[joint+'_gold'])
                              for row in rows]) * 180.0 / 3.14159
            rec = np.asarray([float(row[joint+'_rec'])
                             for row in rows]) * 180.0 / 3.14159
            error = gold - rec

            if errorPlot:
                errAx.plot(time, error, label='"'+joint +
                           '" Error', color=colors[i])

            ax: plt.Axes = axs[i]
            ax.plot(time, rec, label='Recovered "'+joint+'"',
                    linewidth=2.0, color='black')  # 0D6EFD
            ax.plot(time, gold, label='Original "'+joint+'"',
                    linewidth=2.0, color=colors[i])  # '#FF8A00'
            ax.axhline(0, linestyle='--', color='black', linewidth=1.0,
                       alpha=0.5)
            ax.fill_between(time, gold, rec, alpha=0.1,
                            facecolor='black')  # '#E12026'
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position (deg)')
            ax.set_title('"'+joint +
                         '" Joint (avg err = '+str(round(np.mean(np.abs(error)), 2))+' deg)')
            ax.legend(loc=4, prop={'size': 6})

            ax.set(xlim=(min(time), max(time)))
            # ax.set(xlim=(0, max(time)),
            #        ylim=(min(gold + rec + [0]), max(gold + rec + [0])))
            # ylim=(globalMin, globalMax))
            # ylim=(min(gold + rec + [0]), max(gold + rec)))

        if errorPlot:
            errAx.axhline(0, linestyle='--', color='black', linewidth=1.0,
                          alpha=0.5)
            errAx.set(ylim=(-7, 7))
            errAx.set(xlim=(min(time), max(time)))
            errAx.set_xlabel('Time (s)')
            errAx.set_ylabel('Error (deg)')
            errAx.set_title('Joint Reconstruction Error')
            errAx.legend(prop={'size': 6})

        fig.tight_layout()
        plt.show()


# To produce Figure 3, uncomment the below code:

plot("./data/run0500cms.csv",
     ['knee_angle_r', 'ankle_angle_r', 'knee_angle_l', 'ankle_angle_l', 'pelvis_list'], 3, 2, True, 173)

# To produce Figure 4, uncomment the below code:

# plot("./data/DJ2.csv",
#      ['pelvis_tilt', 'pelvis_list', 'hip_flexion_l', 'knee_angle_l', 'lumbar_extension'], 3, 2, True)
