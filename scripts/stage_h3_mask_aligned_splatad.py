"""SplatAD adapter that crops a downscaled training mask with its RGB target."""

from __future__ import annotations

from typing import Any

from nerfstudio.models.splatad import SplatADModel


class MaskAlignedSplatADModel(SplatADModel):
    """Keep the optional mask aligned with SplatAD's existing GT crop."""

    _mask_target_shape: tuple[int, int] | None = None

    def _downscale_if_required(self, image: Any) -> Any:
        result = super()._downscale_if_required(image)
        if (
            self._mask_target_shape is not None
            and result.shape[-1] == 1
        ):
            height, width = self._mask_target_shape
            result = result[:height, :width, :]
        return result

    def get_loss_dict(
        self, outputs: Any, batch: Any, metrics_dict: Any = None
    ) -> Any:
        if "mask" not in batch or "rgb" not in outputs:
            return super().get_loss_dict(outputs, batch, metrics_dict)
        self._mask_target_shape = tuple(outputs["rgb"].shape[:2])
        try:
            return super().get_loss_dict(outputs, batch, metrics_dict)
        finally:
            self._mask_target_shape = None
