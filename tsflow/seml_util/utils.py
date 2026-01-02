import os
from pathlib import Path

from pytorch_lightning.loggers import TensorBoardLogger
from seml.database import get_collection

from tsflow.seml_util.logger import SemlLogger


def create_logdir(logdir, ex):
    _id = ex.current_run.config["overwrite"]
    db_collection = get_collection(ex.current_run.config["db_collection"])
    name = db_collection.find_one({"_id": _id})["seml"]["name"]
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = ex.current_run._id  # os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job_id is None:
        array_job_id = "local"
        array_task_id = ex.current_run._id
    logdir = os.path.join(logdir, f"{name}_{array_job_id}_{array_task_id}")
    logdir = os.path.abspath(logdir)
    Path(logdir).mkdir(parents=True, exist_ok=True)
    return logdir, array_job_id, array_task_id


def get_loggers(ex, run_name, logdir, hparams, media_logger):
    seml_logger = SemlLogger(ex, logdir, media_logger)

    # Create TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=logdir,
        name=run_name,
        default_hp_metric=False,
    )
    tb_logger.log_hyperparams(hparams)

    return [tb_logger, seml_logger]
