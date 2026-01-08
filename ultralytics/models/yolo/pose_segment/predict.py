# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import math

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class PoseSegmentationPredictor(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose_segment"

    def postprocess(self, preds, img, orig_imgs):
        protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        return super().postprocess(preds[0], img, orig_imgs, protos=protos)

    def construct_results(self, preds, img, orig_imgs, protos):
        return [
            self.construct_result(pred, img, orig_img, img_path, proto)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        kpt_shape = self.model.kpt_shape
        kptn = math.prod(kpt_shape)

        if pred.shape[0] == 0:  # save empty boxes
            masks, kpts = None, None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6 + kptn:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto, pred[:, 6 + kptn:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            kpts = pred[:, 6:6 + kptn].view(pred.shape[0], *kpt_shape)
            kpts = ops.scale_coords(img.shape[2:], kpts, orig_img.shape)
            keep = masks.amax((-2, -1)) > 0  # only keep predictions with masks
            if not all(keep):  # most predictions have masks
                pred, masks, kpts = pred[keep], masks[keep], kpts[keep]  # indexing is slow

        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks, keypoints=kpts)
