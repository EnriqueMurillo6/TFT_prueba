import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import copy
from pathlib import Path
import warnings
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import TorchNormalizer, EncoderNormalizer
from pytorch_forecasting.metrics import RMSE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)


path_train = "/../df_train.parquet"
path_test = "/../df_test.parquet"

df_train = pd.read_parquet(path_train)
df_test = pd.read_parquet(path_test)

df_train["carb"] = df_train["carb"].fillna(0)
df_train["bolus"] = df_train["bolus"].fillna(0)
df_train["basal_rate"] = df_train["basal_rate"].ffill().bfill()

df_prueba = df_train[:30000]
df_prueba["time_idx"] = df_prueba.groupby("group_id").cumcount()
df_prueba["group_id"] = df_prueba["group_id"].astype(str)

max_encoder_length = 6
max_prediction_length = 1

training_cutoff = df_prueba["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df_prueba[df_prueba.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Value",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["group_id"],
    time_varying_known_reals=["time_idx", "bolus", "basal_rate", "carb"],
    time_varying_unknown_reals=["Value"],
    target_normalizer=EncoderNormalizer(),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)
from torch.utils.data import DataLoader

batch_size = 64

train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)

validation = TimeSeriesDataSet.from_dataset(training, df_prueba, predict=False, stop_randomization=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

from pytorch_lightning import seed_everything

seed_everything(42)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=RMSE(),
    log_interval=10,
    log_val_interval=1,
)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="auto",
    gradient_clip_val=0.1,
    limit_train_batches=30,
    callbacks=[lr_logger, early_stop_callback],
)

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

predictions = best_tft.predict(
    val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
)

print(RMSE()(predictions.output, predictions.y))
raw_predictions = best_tft.predict(
    val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu")
)

for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(
        raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True
    )
