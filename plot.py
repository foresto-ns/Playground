import PIL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def make_grid(x_train, x_test, y_train, y_test, h=.02):
    x_min = min(x_train[:, 0].min(), x_test[:, 0].min()) - .5
    y_min = min(y_train.min(), y_test.min()) - .5

    x_max = max(x_train[:, 0].max(), x_test[:, 0].max()) + .5
    y_max = max(y_train.max(), y_test.max()) + .5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    return xx, yy


def predict_proba_on_mesh(clf, xx, yy):
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    return Z


def plot_predictions(xx,
                     yy,
                     Z,
                     x_train=None,
                     x_test=None,
                     y_train=None,
                     y_test=None,
                     figsize=(10, 10),
                     title="predictions",
                     cm=plt.cm.RdBu,
                     cm_bright=ListedColormap(['#FF0000', '#0000FF'])):
    fig = plt.figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    if x_train is not None:
        ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, alpha=0.2)
    if x_test is not None:
        ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k')

    plt.xlim((xx.min(), xx.max()))
    plt.ylim((yy.min(), yy.max()))
    plt.title(title)
    fig.tight_layout()
    canvas.draw()  # draw the canvas, cache the renderer
    return PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
