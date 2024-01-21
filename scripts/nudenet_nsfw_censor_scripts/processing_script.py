from scripts.nudenet_nsfw_censor_scripts.censor_image_filters import apply_filter
from scripts.nudenet_nsfw_censor_scripts.pil_nude_detector import pil_nude_detector
from modules import scripts, shared, images, processing
from PIL import ImageFilter
from math import sqrt


def assign_current_image_wrapper(original_function):
    def wrapper_function(*args, **kwargs):
        try:
            image = args[0]
            censor_mask = pil_nude_detector.get_censor_mask(image, shared.opts.nudenet_nsfw_censor_nms_threshold, shared.opts.nudenet_nsfw_censor_mask_shape, shared.opts.nudenet_nsfw_censor_rectangle_round_radius, pil_nude_detector.thresholds, pil_nude_detector.expand_horizontal, pil_nude_detector.expand_vertical)
            if censor_mask:
                scale_factor = sqrt((image.size[0] ** 2 + image.size[1] ** 2) / 524288)
                mask_blur_radis = (shared.opts.nudenet_nsfw_censor_mask_blend_radius_variable_blur if shared.opts.nudenet_nsfw_censor_gen_filter_type == 'Variable blur' else shared.opts.nudenet_nsfw_censor_mask_blend_radius) * scale_factor
                censor_mask = censor_mask.convert('L').filter(ImageFilter.GaussianBlur(mask_blur_radis))
                blur_radius = shared.opts.nudenet_nsfw_censor_blur_radius * scale_factor
                filter_settings = {
                    'blur_radius': blur_radius,
                    'blur_strength_curve': shared.opts.nudenet_nsfw_censor_blur_strength_curve,
                    'pixelation_factor': shared.opts.nudenet_nsfw_censor_pixelation_factor,
                    'color': shared.opts.nudenet_nsfw_censor_fill_color,
                }
                censored_image = apply_filter(image, censor_mask, shared.opts.nudenet_nsfw_censor_live_preview_filter_type, **filter_settings)
                new_args = censored_image, *args[1:]
                return original_function(*new_args, **kwargs)
        except Exception as e:
            print(e)
        return original_function(*args, **kwargs)

    return wrapper_function


def close_wrapper(fun):
    def wrapper(*args, **kwargs):
        try:
            original_assign_current_image = getattr(shared.state, 'original_assign_current_image', None)
            if original_assign_current_image:
                shared.state.assign_current_image = original_assign_current_image
                shared.state.original_assign_current_image = None
        except Exception as e:
            print(e)
        return fun(*args, **kwargs)
    return wrapper


class ScriptNudenetCensor(scripts.Script):
    def __init__(self):
        pass

    def title(self):
        return 'NudeNet NSFW Censor'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_image_after_composite(self, p, pp, *args):
        """
        Detect image with nudenet and apply censor filter selected areas of the pp.image
        pp.image will be replaced with censor image if filter is applied
        Args:
            p:
            pp:
            *args:

        Returns:
        """
        if not shared.opts.nudenet_nsfw_censor_enable or shared.opts.nudenet_nsfw_censor_gen_filter_type == 'Disable':
            return

        if pil_nude_detector.thresholds is None:
            pil_nude_detector.refresh_label_configs()

        censor_mask = pil_nude_detector.get_censor_mask(pp.image, shared.opts.nudenet_nsfw_censor_nms_threshold, shared.opts.nudenet_nsfw_censor_mask_shape, shared.opts.nudenet_nsfw_censor_rectangle_round_radius, pil_nude_detector.thresholds, pil_nude_detector.expand_horizontal, pil_nude_detector.expand_vertical)
        if censor_mask:
            info_text = None
            if shared.opts.nudenet_nsfw_censor_save_before_censor:
                info_text = processing.create_infotext(p, p.prompts, p.seeds, p.subseeds, index=p.batch_index, all_negative_prompts=p.negative_prompts)
                images.save_image(pp.image, p.outpath_samples, '', p.seeds[p.batch_index], p.prompts[p.batch_index], shared.opts.samples_format, info=info_text, p=p, suffix='-before-censor')

            scale_factor = sqrt((pp.image.size[0] ** 2 + pp.image.size[1] ** 2) / 524288)

            mask_blur_radis = (shared.opts.nudenet_nsfw_censor_mask_blend_radius_variable_blur if shared.opts.nudenet_nsfw_censor_gen_filter_type == 'Variable blur' else shared.opts.nudenet_nsfw_censor_mask_blend_radius) * scale_factor
            censor_mask = censor_mask.convert('L').filter(ImageFilter.GaussianBlur(mask_blur_radis))
            if shared.opts.nudenet_nsfw_censor_save_mask:
                if not info_text:
                    info_text = processing.create_infotext(p, p.prompts, p.seeds, p.subseeds, index=p.batch_index, all_negative_prompts=p.negative_prompts)
                images.save_image(censor_mask, p.outpath_samples, '', p.seeds[p.batch_index], p.prompts[p.batch_index], shared.opts.samples_format, info=info_text, p=p, suffix='-censor-mask')

            blur_radius = shared.opts.nudenet_nsfw_censor_blur_radius * scale_factor
            filter_settings = {
                'blur_radius': blur_radius,
                'blur_strength_curve': shared.opts.nudenet_nsfw_censor_blur_strength_curve,
                'pixelation_factor': shared.opts.nudenet_nsfw_censor_pixelation_factor,
                'color': shared.opts.nudenet_nsfw_censor_fill_color,
            }

            if shared.opts.nudenet_nsfw_censor_gen_filter_type:
                pp.image = apply_filter(pp.image, censor_mask, shared.opts.nudenet_nsfw_censor_gen_filter_type, **filter_settings)

    if not hasattr(scripts.Script, 'postprocess_image_after_composite'):
        postprocess_image = postprocess_image_after_composite

    def setup(self, p, *args):
        """
        Live preview censor, inject code by wrapping shared.state.assign_current_image()
        unwrap with p.close
        Args:
            p:
            *args:

        Returns:
        """
        if getattr(shared.state, 'original_assign_current_image', None):
            # Extra Protection, don't think this is necessary but just to be safe.
            p.close = close_wrapper(p.close)
        elif shared.opts.nudenet_nsfw_censor_enable and shared.opts.nudenet_nsfw_censor_live_preview_filter_type != 'Disable':
            if pil_nude_detector.thresholds is None:
                pil_nude_detector.refresh_label_configs()
            shared.state.original_assign_current_image = shared.state.assign_current_image
            shared.state.assign_current_image = assign_current_image_wrapper(shared.state.assign_current_image)
            p.close = close_wrapper(p.close)
