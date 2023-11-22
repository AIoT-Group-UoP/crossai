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

    if kwargs.get("plot_features", None) is None:
        fig, axs = plt.subplots(1, 1, figsize=figsize, sharex=True,
                                num=kwargs.get("num", None),
                                constrained_layout=True)
        # for each row of df plot
        for i in range(len(data)):
            axs.plot(data[i], label=labels[i])
        axs.grid()
        axs.legend()
    else:
        features_list = kwargs.get("plot_features")
        # create subplots
        fig, axs = plt.subplots(len(features_list), 1, figsize=figsize,
                                sharex=True, num=kwargs.get("num", None),
                                constrained_layout=True, squeeze=False)
        for i in range(len(features_list)):
            # each feature is a row that has a column for each modality
            for j in range(len(features_list[i])):
                data_to_plot = data[features_list[i][j]]
                axs[i][0].plot(data_to_plot, label=labels[features_list[i][j]])
            axs[i][0].grid()
            axs[i][0].legend()

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
    if show:
        plt.show()
    plt.close()
    if return_artifact:
        return fig


def plot_predictions(predictions,
                     title: str = None,
                     labels: list = None,
                     color_threshold: float = 0.5,
                     path_to_save=None,
                     show: bool = True,
                     return_artifact: bool = False,
                     window_length: int = 1,
                     window_overlap: int = 0,
                     **kwargs):
    """Plots the prediction results of a model in regard to time and by taking
    into account the windowing process. The predictions are plotted as a
    horizontal line for each class, with the color of the line indicating the
    dominant class and the length of the line indicating the length of the
    window that the prediction corresponds to.

    Args:
        predictions (numpy.ndarray): The predictions of the model.
        title (str, optional): The title of the plot. Defaults to None.
        labels (list, optional): The labels of the classes. Defaults to None.
        color_threshold (float, optional): The threshold for the color of the
            lines. Defaults to 0.5.
        path_to_save ([type], optional): The path to save the plot. Defaults to
            None.
        show (bool, optional): Whether to show the plot. Defaults to True.
        return_artifact (bool, optional): Whether to return the plot as an
            artifact. Defaults to False.
        window_length (int, optional): The length of the window. Defaults to 1.
        window_overlap (int, optional): The overlap of the windows. Defaults to
            0.

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

    Returns:
        Plot artifact if return_artifact is True.
    """
    plot_dict = {}

    for i in range(len(predictions[0])):  # create dict keys
        plot_dict[i] = []
    figsize = kwargs.get("figsize", (16, 8))
    fig, axs = plt.subplots(1, 1, figsize=figsize, sharex=True,
                            num=kwargs.get("num", None),
                            constrained_layout=True)

    for i in range(len(predictions)):
        seg_start = i * (window_length - window_overlap)
        seg_end = seg_start + window_length
        for j in range(len(predictions[i])):
            axs.plot([seg_start, seg_end],
                     [predictions[i][j], predictions[i][j]],
                     color=LABELS_COLORS[j])
            plot_dict[j].append([seg_start, seg_end])

    # find consecutive segments with the same label
    transposed_preds = np.transpose(predictions)
    for i in range(len(transposed_preds)):
        start_color = None
        end_color = None
        for j in range(len(transposed_preds[i])):
            if transposed_preds[i][j] > color_threshold:
                if start_color is None:
                    start_color = plot_dict[i][j][0]
                end_color = plot_dict[i][j][1]
            else:
                if start_color is not None:
                    axs.axvspan(start_color, end_color,
                                color=LABELS_COLORS[i], alpha=0.3)
                    start_color = None
                    end_color = None

    axs.grid()
    axs.legend(labels)
    axs.legend(labels, bbox_to_anchor=(1.02, 0.5), loc='center right')
    segments = kwargs.get("segments", None)

    if segments is not None and labels is not None:
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

    axs.set_ylim([-0.02, 1.02])

    if title is not None:
        fig.suptitle(title)
    if path_to_save is not None:
        plt.savefig(path_to_save)
    if show:
        plt.show()
    plt.close()
    if return_artifact:
        return fig
