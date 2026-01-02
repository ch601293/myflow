from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF


def render_figure(fig: plt.Figure) -> Image.Image:
    """Render a matplotlib figure into a Pillow image."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)


def plot_figures(tss, forecasts, context_length, prediction_length, trainer, set="val"):
    fig = plt.Figure(figsize=(6, 2 * len(tss)), dpi=300)
    for i in range(len(tss)):
        dims = len(tss[i].shape)
        if dims == 2:
            gt = tss[i].to_numpy()[:, 0]
        ax = fig.add_subplot(len(tss), 1, i + 1)
        ax.plot(gt[-context_length - prediction_length :], label="Context", zorder=1)
        ax.plot(
            np.arange(prediction_length) + context_length,
            forecasts[i].quantile(0.5) if dims == 1 else forecasts[i].quantile(0.5),
            label="Forecast",
            zorder=1,
        )
        ax.fill_between(
            np.arange(prediction_length) + context_length,
            forecasts[i].quantile(0.05) if dims == 1 else forecasts[i].quantile(0.05),
            forecasts[i].quantile(0.95) if dims == 1 else forecasts[i].quantile(0.95),
            alpha=0.5 - 0.95 / 3,
            facecolor="C3",
            label="95% CI",
        )
        ax.legend(loc="upper left", fontsize="xx-small")

    # Convert PIL Image to tensor for TensorBoard
    pil_image = render_figure(fig)
    # Convert PIL to tensor (C, H, W) format
    image_tensor = TF.to_tensor(pil_image)

    # Log to TensorBoard
    for logger in trainer.loggers:
        if hasattr(logger.experiment, 'add_image'):
            # TensorBoard logger
            logger.experiment.add_image(
                f"{set}/sample",
                image_tensor,
                global_step=trainer.global_step
            )
        else:
            # Fallback for other loggers
            metrics = {f"{set}/sample": pil_image}
            logger.log_metrics(metrics, step=trainer.global_step)


def save_figures(tss, forecasts, context_length, prediction_length, logdir):
    fig = plt.Figure(figsize=(6, 2 * len(tss)), dpi=300)
    for i in range(len(tss)):
        ax = fig.add_subplot(len(tss), 1, i + 1)
        ax.plot(
            tss[i].to_numpy()[-context_length - prediction_length :, 0],
            label="Context",
            zorder=1,
        )
        ax.plot(
            np.arange(prediction_length) + context_length,
            forecasts[i].quantile(0.5),
            label="Forecast",
            zorder=1,
        )
        ax.fill_between(
            np.arange(prediction_length) + context_length,
            forecasts[i].quantile(0.05),
            forecasts[i].quantile(0.95),
            # Clamp alpha betwen ~16% and 50%.
            alpha=0.5 - 0.95 / 3,
            facecolor="C3",
            label="95% CI",
        )
        ax.legend(loc="upper left", fontsize="xx-small")
    fig.savefig(f"{logdir}/example_forecast.png")
