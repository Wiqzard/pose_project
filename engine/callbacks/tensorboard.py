from utils import LOGGER, colorstr
from torch.utils.tensorboard import SummaryWriter

writer = None  # TensorBoard SummaryWriter instance


def _log_scalars(scalars, step=0):
    """Logs scalar values to TensorBoard."""
    if writer:
        for k, v in scalars.items():
            writer.add_scalar(k, v, step)


def on_pretrain_routine_start(trainer):
    """Initialize TensorBoard logging with SummaryWriter."""
    if SummaryWriter:
        try:
            global writer
            writer = SummaryWriter(str(trainer.save_dir))
            prefix = colorstr("tensorboard: ")
            LOGGER.info(
                f"{prefix}Start with 'tensorboard --logdir {trainer.save_dir}', view at http://localhost:6006/"
            )
        except Exception as e:
            LOGGER.warning(
                f"WARNING ⚠️ TensorBoard not initialized correctly, not logging this run. {e}"
            )


def on_batch_end(trainer):
    """Logs scalar statistics at the end of a training batch."""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix='train'), trainer.epoch + 1)


def on_fit_epoch_end(trainer):
    """Logs epoch metrics at end of training epoch."""
    _log_scalars(trainer.metrics, trainer.epoch + 1)

callbacks = {
    "on_pretrain_routine_start": on_pretrain_routine_start,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_batch_end": on_batch_end,
}
