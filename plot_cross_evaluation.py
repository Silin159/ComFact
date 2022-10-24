import matplotlib.pyplot as plt
import seaborn as sns

# model: roberta-large, window: nlg, task: fact_full
cross_accuracy = [[0.883, 0.826, 0.852, 0.828],
                  [0.841, 0.835, 0.836, 0.836],
                  [0.861, 0.835, 0.874, 0.835],
                  [0.843, 0.825, 0.808, 0.850]]

cross_precision = [[0.735, 0.773, 0.762, 0.751],
                   [0.595, 0.702, 0.613, 0.672],
                   [0.716, 0.764, 0.709, 0.760],
                   [0.596, 0.669, 0.553, 0.635]]

cross_recall = [[0.698, 0.566, 0.524, 0.238],
                [0.775, 0.748, 0.787, 0.392],
                [0.568, 0.623, 0.774, 0.285],
                [0.793, 0.789, 0.887, 0.628]]

cross_f1 = [[0.716, 0.654, 0.621, 0.361],
            [0.673, 0.724, 0.689, 0.495],
            [0.634, 0.686, 0.740, 0.414],
            [0.680, 0.724, 0.682, 0.631]]

font_size = 35
heatmap_scale = 2.5

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 8, forward=True)
# fig.set_dpi(100)

# sns.set_theme()
sns.set(font_scale=heatmap_scale)
sns.heatmap(cross_accuracy, ax=ax1, cbar=True, annot=True, vmin=0.000, vmax=1.000, fmt='.3', linewidths=0.01, linecolor="black")
sns.heatmap(cross_f1, ax=ax2, cbar=True, annot=True, vmin=0.000, vmax=1.000, fmt='.3', linewidths=0.01, linecolor="black")

x_axis_labels = ["Per", "Mut", "Roc", "Mov"]
y_axis_labels = ["Per", "Mut", "Roc", "Mov"]

ax1.set_title("Acc.", fontsize=font_size+10, weight="bold")
ax1.set_xlabel("Testing Set", fontsize=font_size)
ax1.set_ylabel("Training Set", fontsize=font_size)
ax1.set_xticklabels(x_axis_labels, fontsize=font_size, rotation=0, style="italic")
ax1.set_yticklabels(y_axis_labels, fontsize=font_size, rotation=0, style="italic")

ax2.set_title("F1", fontsize=font_size+10, weight="bold")
ax2.set_xlabel("Testing Set", fontsize=font_size)
ax2.set_ylabel("Training Set", fontsize=font_size)
ax2.set_xticklabels(x_axis_labels, fontsize=font_size, rotation=0, style="italic")
ax2.set_yticklabels(y_axis_labels, fontsize=font_size, rotation=0, style="italic")

plt.tight_layout()

# plt.show()
plt.savefig("cross.pdf", bbox_inches="tight", pad_inches=0.05)
