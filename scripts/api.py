from scripts.pil_nude_detector import pil_nude_detector, nudenet_labels_index, mask_shapes_func_dict
from scripts.censor_image_filters import apply_filter, filter_dict
from modules.api.api import decode_base64_to_image, encode_pil_to_base64
import modules.script_callbacks as script_callbacks
from fastapi import FastAPI, Body
from PIL import ImageFilter
from modules import shared
from math import sqrt
import gradio as gr
import numpy as np


def nudenet_censor_api(_: gr.Blocks, app: FastAPI):
    @app.post("/nudenet/censor")
    async def censor(
            input_image: str = Body(None, title="base64 input image"),
            input_mask: str = Body(None, title="base64 mask (optional)"),
            enable_nudenet: bool = Body(True, title="Enable NudeNet mask detection"),
            output_mask: bool = Body(None, title="return mask"),
            filter_type: str = Body(None, title=f"Name of censor filter: {list(filter_dict)}"),
            blur_radius: float = Body(None, title="Gaussian blur radius: float [0, inf]"),
            blur_strength_curve: float = Body(None, title="Blur strength curve: float [0, 6]"),
            pixelation_factor: float = Body(None, title="Pixelation factor float [0, inf]"),
            fill_color: str = Body(None, title="Fill color hex color codes [#000000, #FFFFFF]"),
            mask_blend_radius: float = Body(None, title="Mask blend radius, [0, inf]"),
            mask_shape: str = Body(None, title=f"Name Mask shape: {list(mask_shapes_func_dict)}"),
            nms_threshold: float = Body(None, title="NMS threshold: float [0, 1]"),
            rectangle_round_radius: float = Body(None, title="Rectangle round radius: float [0, inf]"),
            thresholds: list = Body(None, title=f"list of float for thresholds of: {list(nudenet_labels_index)}"),
            expand_horizontal: list = Body(None, title=f"List of float for expand horizontal: {list(nudenet_labels_index)}"),
            expand_vertical: list = Body(None, title=f"List of float for expand vertical: {list(nudenet_labels_index)}"),
    ):
        input_image = decode_base64_to_image(input_image)
        if not input_image:
            return {'image': None, 'mask': None}
        censor_mask = None
        censored_image_base64 = None
        censor_mask_base64 = None

        if input_mask:
            censor_mask = decode_base64_to_image(input_mask).convert('L').resize(input_image.size)

        if enable_nudenet:
            nms_threshold = nms_threshold if nms_threshold else shared.opts.nudenet_nsfw_censor_nms_threshold
            mask_shape = mask_shape if mask_shape else shared.opts.nudenet_nsfw_censor_mask_shape
            rectangle_round_radius = rectangle_round_radius if rectangle_round_radius else shared.opts.nudenet_nsfw_censor_rectangle_round_radius
            if pil_nude_detector.thresholds is None:
                pil_nude_detector.refresh_label_configs()

            thresholds = np.asarray(thresholds) if thresholds else pil_nude_detector.thresholds
            expand_horizontal = np.asarray(expand_horizontal) if expand_horizontal else pil_nude_detector.expand_horizontal
            expand_vertical = np.asarray(expand_vertical) if expand_vertical else pil_nude_detector.expand_vertical

            nudenet_mask = pil_nude_detector.get_censor_mask(input_image, nms_threshold, mask_shape, rectangle_round_radius, thresholds, expand_horizontal, expand_vertical).convert('L')
            if nudenet_mask and censor_mask:
                censor_mask.paste(nudenet_mask, nudenet_mask)
            else:
                censor_mask = nudenet_mask

        if censor_mask:
            filter_type = filter_type if filter_type else shared.opts.nudenet_nsfw_censor_extras_filter_type
            scale_factor = sqrt((input_image.size[0] ** 2 + input_image.size[1] ** 2) / 524288)
            mask_blend_radius = mask_blend_radius if mask_blend_radius else (shared.opts.nudenet_nsfw_censor_mask_blend_radius_variable_blur if filter_type == 'Variable blur' else shared.opts.nudenet_nsfw_censor_mask_blend_radius)
            censor_mask = censor_mask.filter(ImageFilter.GaussianBlur(mask_blend_radius * scale_factor))
            filter_settings = {
                'blur_radius': blur_radius if blur_radius else shared.opts.nudenet_nsfw_censor_blur_radius * scale_factor,
                'blur_strength_curve': blur_strength_curve if blur_strength_curve else shared.opts.nudenet_nsfw_censor_blur_strength_curve,
                'color': fill_color if fill_color else shared.opts.nudenet_nsfw_censor_fill_color,
                'pixelation_factor': pixelation_factor if pixelation_factor else shared.opts.nudenet_nsfw_censor_pixelation_factor,
            }

            if filter_type:
                censored_image_base64 = encode_pil_to_base64(apply_filter(input_image, censor_mask, filter_type, **filter_settings))

            if output_mask:
                censor_mask_base64 = encode_pil_to_base64(censor_mask)

        return {
            'image': censored_image_base64,
            'mask': censor_mask_base64,
        }


script_callbacks.on_app_started(nudenet_censor_api)
