# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class PoseSegmentationPredictor(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose_segment"

    def construct_result(self, pred, img, orig_img, img_path):
        result = super().construct_result(pred, img, orig_img, img_path)
        # Extract keypoints from prediction and reshape according to model's keypoint shape
        pred_kpts = pred[:, 6:].view(pred.shape[0], *self.model.kpt_shape)
        # Scale keypoints coordinates to match the original image dimensions
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        result.update(keypoints=pred_kpts)
        return result
