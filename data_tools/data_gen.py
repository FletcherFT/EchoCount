from pathlib import Path
import json
import numpy as np
from GA import Model
import matplotlib.pyplot as plt


def mean_target_fitness(targets, bounds, beq):
    bounds = np.arange(*bounds)

    def get_fitness(chromosomes):
        m, n = chromosomes.shape
        f = np.zeros((m, bounds.size))
        g = np.zeros((m, bounds.size))
        for row in range(m):
            for b in bounds:
                idx = chromosomes[row, :] == b
                g[row, b] = idx.sum()
                if g[row, b] > 0:
                    f[row, b] = targets[idx].mean()
                else:
                    f[row, b] = 0

        f = np.abs(np.diff(f, axis=1)).sum(axis=1)
        g = np.abs(g - beq).sum(axis=1)
        return f, g

    return get_fitness


if __name__ == "__main__":
    trn_split = 0.7
    val_split = 0.15
    tst_split = 0.15
    # Get the data directory
    data_dir = Path("../raw_data/Data/Batch_0001").resolve()
    # Get the annotation file in the data directory
    data_path = list(data_dir.glob("Annotations.json"))[0]
    # Load in the json as a dictionary
    with open(data_path) as f:
        annotations = json.load(f)
    # Goals:
    # 1. Get the finished annotations from the data set
    # 2. Get the number of targets for each valid annotation
    # 3. Flag valid annotations for empty targets (true) and has targets (false)
    done = np.argwhere(
        [annotation["file_attributes"]["metadata"]["isfinished"] for annotation in annotations.values()]).squeeze()
    targets = np.array([len(annotation["regions"]) for annotation in annotations.values()])[done].squeeze()
    empty = np.array([i == 0 for i in targets]).squeeze()
    # To balance the data, undersample the larger of the two categories (background or target)
    n_bg = empty.sum()
    n_tg = np.invert(empty).sum()
    if n_bg > n_tg:
        n_bg = n_tg
    elif n_bg < n_tg:
        n_tg = n_bg
    # Calculate the split
    n_samples = n_bg + n_tg
    n_train = int(trn_split * n_samples)
    n_valid = int(val_split * n_samples)
    n_tests = int(tst_split * n_samples)
    # If the split results in a deficit or surplus allocation, then either add to the training or subtract from the val
    if n_train + n_valid + n_tests < n_samples:
        n_train += n_samples - n_train - n_valid - n_tests
    elif n_train + n_valid + n_tests > n_samples:
        n_valid -= n_train + n_valid + n_tests - n_samples
    # Now, allocate ~50% background images to each allocation
    empty_ind = np.argwhere(empty).squeeze()
    n_bg_train = int(n_train / 2)
    n_bg_valid = int(n_valid / 2)
    n_bg_tests = int(n_tests / 2)
    sampled_bg = np.random.choice(empty_ind, (n_bg_train + n_bg_valid + n_bg_tests,), replace=False)
    train = sampled_bg[:n_bg_train]
    valid = sampled_bg[n_bg_train:n_bg_train + n_bg_valid]
    tests = sampled_bg[n_bg_train + n_bg_valid:]
    # Now, allocate singletarget images to each allocation.
    # The train/validation/test split has to be balanced so that mean(targets) is close for all sets
    target_ind = np.argwhere(np.invert(empty)).squeeze()
    n_tg_train = n_train - n_bg_train
    n_tg_valid = n_valid - n_bg_valid
    n_tg_tests = n_tests - n_bg_tests
    #
    tg_targets = targets[target_ind]
    beq = np.array([n_tg_train, n_tg_valid, n_tg_tests])
    f = mean_target_fitness(tg_targets, (0, 3), beq)
    ga = Model.PartitionModel(200, n_tg, fitness=f, bounds=(0, 2), k=1, gamma=0.01, lmbda=0.5, gain=1.1)
    fig, ax = plt.subplots(2, 1)
    f = ga._f[0]
    g = ga._g[0]
    ln1 = ax[0].plot(f)[0]
    ln2 = ax[1].plot(g)[0]
    for i in range(1000):
        s, f, g = ga.update()
        x, y = ln1.get_data()
        x = np.hstack((x, x[-1] + 1))
        y = np.hstack((y, f))
        ln1.set_data((x, y))
        ax[0].set_xlim(xmin=x.min(), xmax=x.max())
        ax[0].set_ylim(ymin=y.min(), ymax=y.max())
        x, y = ln2.get_data()
        x = np.hstack((x, x[-1] + 1))
        y = np.hstack((y, g))
        ln2.set_data((x, y))
        ax[1].set_xlim(xmin=x.min(), xmax=x.max())
        ax[1].set_ylim(ymin=y.min(), ymax=y.max())
        print("{:05d}\t{:.3f}\t{:.3f}".format(i, f, g))
        if True:
            fig.canvas.draw()
            fig.canvas.flush_events()
    train = np.hstack((train, target_ind[s == 0]))
    valid = np.hstack((valid, target_ind[s == 1]))
    tests = np.hstack((tests, target_ind[s == 2]))
    keys = list(annotations.keys())
    train_keys = [keys[i] for i in train]
    valid_keys = [keys[i] for i in valid]
    tests_keys = [keys[i] for i in tests]
    train_annotations = {k: annotations[k] for k in train_keys if k in annotations}
    valid_annotations = {k: annotations[k] for k in valid_keys if k in annotations}
    tests_annotations = {k: annotations[k] for k in tests_keys if k in annotations}
    out_path = data_dir.joinpath("train.json")
    with open(out_path, 'w') as f:
        json.dump(train_annotations, f)
    out_path = data_dir.joinpath("valid.json")
    with open(out_path, 'w') as f:
        json.dump(valid_annotations, f)
    out_path = data_dir.joinpath("tests.json")
    with open(out_path, 'w') as f:
        json.dump(tests_annotations, f)
