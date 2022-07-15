from matplotlib import pyplot as plt


def plot_subplots(shape, imgs, titles=None):
    """
    Plot a set of images in a grid of subplots.
    """
    rows, cols = shape

    plt.figure()

    if len(imgs) > rows * cols:
        raise ValueError(f"Too many images to plot. Max {rows * cols} images can be plotted.")

    for i in range(len(imgs)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imgs[i])

        if titles is not None:
            plt.title(titles[i])
        
        plt.axis('off')

    plt.show()
