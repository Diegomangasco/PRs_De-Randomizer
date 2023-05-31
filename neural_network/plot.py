import matplotlib.pyplot as plt


def graph_plot(figure_path, threshold):
    iterations = list()
    res = list()
    diff_same = list()
    diff_different = list()
    with open(f"{figure_path}/stats.txt", "r") as fp:
        line = fp.readline()
        while line:
            line = line[:-1]
            el = line.split(" ")
            iterations.append(int(el[0]))
            res.append(float(el[1]))
            diff_same.append(float(el[2]))
            diff_different.append(float(el[3]))
            line = fp.readline()

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(12.5, 19.625)
    fig.suptitle(f"10 Devices (Iterations = {max(iterations)})")
    ax[0].set_title(f"Detection accuracy")
    ax[0].plot(iterations, res, color="blue", marker=".", label="Accuracy")
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles, labels, loc="lower right")
    ax[0].set_ylabel("% of correctly recognized\nprobe requests")
    ax[0].set_xlabel("Iteration number")
    ax[0].set_ylim((0, 100))
    ax[1].set_title(f"Distance of probe requests")
    ax[1].plot(iterations, diff_same, color="blue", marker=".", label="Same device")
    ax[1].plot(iterations, diff_different, color="green", marker=".", label="Different devices")
    ax[1].plot(iterations, [threshold for _ in range(len(iterations))], color="red", linestyle='dashed', label=f"Threshold: {threshold}")
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles, labels, loc="lower right")
    ax[1].set_ylabel("Euclidean distance\nprobe requests")
    ax[1].set_xlabel("Iteration number")
    ax[1].set_ylim((0, max([max(diff_same), max(diff_different)])+3))

    plt.savefig(f"{figure_path}/_devices", dpi=220)


# graph_plot("./graphs", 3.0)
