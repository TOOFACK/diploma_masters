import matplotlib.pyplot as plt

def visualize_sample(xyz, labels, title="Sample"):
    xyz = xyz.cpu().numpy()
    labels = labels.cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.scatter(xyz[:,0], xyz[:,1], c=labels, s=1, cmap="tab20")
    plt.title(title)
    plt.axis("equal")
    plt.show()
