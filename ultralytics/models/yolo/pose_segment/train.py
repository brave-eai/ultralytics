# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseSegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK


class PoseSegmentationTrainer(yolo.detect.DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose_segment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
            self,
            cfg: str | Path | dict[str, Any] | None = None,
            weights: str | Path | None = None,
            verbose: bool = True,
    ) -> PoseSegmentationModel:
        model = PoseSegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]
        kpt_names = self.data.get("kpt_names")
        if not kpt_names:
            names = list(map(str, range(self.model.kpt_shape[0])))
            kpt_names = {i: names for i in range(self.model.nc)}
        self.model.kpt_names = kpt_names

    def get_validator(self):
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "seg_loss", "cls_loss", "dfl_loss"
        return yolo.pose_segment.PoseSegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose/")
        return data
