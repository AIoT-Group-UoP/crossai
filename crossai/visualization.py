import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

LABELS_COLORS = sns.color_palette("tab10", 26)


def plot_ts(data,
            title: str = None,
            labels: list = None,
            path_to_save=None,
            show: bool = True,
            return_artifact: bool = False,
            **kwargs):
    """Plots only the fundamental axes, if they exist in the Dataframe.

    Args:
        data (np.array or pd.Series): An array or Series with the defined
            signal or prediction data. Each row is expected to contain a
            different modality.
        title (str): The title of the plot
        labels:
        path_to_save (str, optional): When defined, the plot is saved
            on the given path. Default None.
        show (bool): (Optional) If True, the plot will be shown.
        return_artifact (bool): (Optional) If True, the plot will be returned
            as an artifact using return
        **kwargs:
            figsize (tuple): Default value (16,8)
            num (int): Figure number. Default value None
            segments (array of arrays):
                If segments are passed as arguments, they are colored
                in the plots as vertical lines. segments are shaped like this;
                [[start_indice, end_indice,"label or compatible color"],[...]]
                Coloring can be done either using the label color or a color
                compatible with matplotlib color format.
            labels (list): List of strings with the names of the segments.
            xlabel (str): Label of the x axis. Default "Time"
            ylabel (str): Label of the y axis. Default ""
    Returns:
        plot artifact if return_artifact is True
    """
    figsize = kwargs.get("figsize", (16, 8))

    fig, axs = plt.subplots(1, 1, figsize=figsize, sharex=True,
                            num=kwargs.get("num", None),
                            constrained_layout=True)

    # for each row of df plot
    for i in range(len(data)):
        axs.plot(data[i], label=labels[i])
    axs.grid()
    axs.legend()
    segments = kwargs.get("segments", None)
    if segments is not None and labels is not None:
        for ax in fig.axes:
            for seg in segments:
                if isinstance(seg[2], str):
                    try:
                        label_color = LABELS_COLORS[labels.index(seg[2])]
                    except ValueError:
                        label_color = seg[2]
                else:
                    label_color = LABELS_COLORS[seg[2]]
                ax.axvspan(seg[0], seg[1], facecolor=label_color,
                           alpha=0.3, label=seg[2])

    xlabel = kwargs.get("xlabel", "Time")
    ylabel = kwargs.get("ylabel", "")

    # set global x and y labels
    fig.text(0.5, 0.04, xlabel, ha='center')
    fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')

    if title is not None:
        fig.suptitle(title)
    if path_to_save is not None:
        plt.savefig(path_to_save)
        plt.close()
    if show:
        plt.show()
    if return_artifact:
        return fig


def plot_predictions(predictions,
                     title: str = None,
                     labels: list = None,
                     color_threshold: float = 0.5,
                     path_to_save=None,
                     show: bool = True,
                     return_artifact: bool = False,
                     **kwargs):
    """Plots the prediction results of a model in regard to time.

    Args:
        predictions (list): List of predictions
        title (str): (Optional) Title of the plot
        labels (list): List of labels. If none the plot will not have a legend
        color_threshold (float): (Optional) Prediction threshold for the
                                instance to be colored with the class color
                                in a vertical line. Default 0.5
        path_to_save (str, optional): When defined, the plot is saved
                                    on the given path. Default None.
        show (bool): (Optional) If True, the plot will be shown.
        return_artifact (bool): (Optional) If True, the plot will be returned
                                as an artifact using return
        **kwargs:
            figsize (tuple): Default value (16,8)
            num (int): Figure number. Default value None
            segments (array of arrays):
                If segments are passed as arguments, they are colored
                in the plots as vertical lines. segments are shaped like this;
                [[start_indice, end_indice,"label or compatible color"],[...]]
                Coloring can be done either using the label color or a color
                compatible with matplotlib color format.
                Note that if segments are passed, the normal coloring of the
                predictions will be overriden.
            s (int): (Optional) Size of the markers
    Returns:
        plot artifact if return_artifact is True

    """
    predictions = np.array(predictions)
    predictions = np.transpose(predictions)

    figsize = kwargs.get("figsize", (16, 8))

    fig, axs = plt.subplots(1, 1, figsize=figsize, sharex=True,
                            num=kwargs.get("num", None),
                            constrained_layout=True)
    s = kwargs.get("s", 5)
    for i in range(len(predictions)):
        index = 0
        axs.scatter(np.arange(len(predictions[i])), predictions[i],
                    label=labels[i], marker="_", s=s)
        for prediction in predictions[i]:
            if prediction > color_threshold:
                axs.axvspan(index, index + 1, facecolor=LABELS_COLORS[i],
                            alpha=0.3)
            index += 1

    axs.grid()
    axs.legend()
    segments = kwargs.get("segments", None)

    if segments is not None and labels is not None:
        # remove any axvline that might have been added
        for line in axs.lines:
            line.remove()

        for ax in fig.axes:
            for seg in segments:
                if isinstance(seg[2], str):
                    label_color = LABELS_COLORS[labels.index(seg[2])]
                else:
                    label_color = LABELS_COLORS[seg[2]]
                ax.axvspan(seg[0], seg[1], facecolor=label_color,
                           alpha=0.3, label=seg[2])

    if title is not None:
        fig.suptitle(title)
    if path_to_save is not None:
        plt.savefig(path_to_save)
    if show:
        plt.show()
    if return_artifact:
        return fig
