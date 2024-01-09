from scripts.nudenet_nsfw_censor_scripts.pil_nude_detector import pil_nude_detector, nudenet_labels_index, mask_shapes_func_dict, available_onnx_providers, default_onnx_provider, nudenet_labels_dict
from scripts.nudenet_nsfw_censor_scripts.censor_image_filters import filter_dict
from modules import shared, ui_components
import gradio as gr


censor_region_settings = {
    # todo better description
    'nudenet_nsfw_censor_enable': shared.OptionInfo(True, 'Enable Region censor for image generation').info('default True'),
    'nudenet_nsfw_censor_save_before_censor': shared.OptionInfo(True, 'Save a copy of the image before applying censor').info('default True'),
    'nudenet_nsfw_censor_save_mask': shared.OptionInfo(False, 'Save censor mask').info('default False'),
    "nudenet_nsfw_censor_gen_filter_type": shared.OptionInfo('Gaussian Blur', 'Generation censor filter', gr.Radio, {'choices': list(filter_dict)}).info('default Gaussian Blur'),
    "nudenet_nsfw_censor_live_preview_filter_type": shared.OptionInfo('Gaussian Blur', 'Live preview censor filter', gr.Radio, {'choices': list(filter_dict)}).info('default Gaussian Blur'),
    "nudenet_nsfw_censor_extras_filter_type": shared.OptionInfo('Variable blur', 'Extras censor filter', gr.Radio, {'choices': list(filter_dict)}).info('default Variable blur'),
    'nudenet_nsfw_censor_mask_shape': shared.OptionInfo('Ellipse', 'Mask shape', gr.Radio, {'choices': list(mask_shapes_func_dict)}).info('default Ellipse'),
    'nudenet_nsfw_censor_blur_radius': shared.OptionInfo(10, f'Blur strength', gr.Slider, {'minimum': 0, 'maximum': 100}).info('default 10'),
    'nudenet_nsfw_censor_rectangle_round_radius': shared.OptionInfo(0.5, 'Rectangle round radius', gr.Number).info('default 0.5'),
    'nudenet_nsfw_censor_mask_blend_radius': shared.OptionInfo(0, f'Mask blend strength', gr.Slider, {'minimum': 0, 'maximum': 100}).info('default 0'),
    'nudenet_nsfw_censor_mask_blend_radius_variable_blur': shared.OptionInfo(10, f'Mask blend strength - Variable blur', gr.Slider, {'minimum': 0, 'maximum': 100}).info('default 10'),
    'nudenet_nsfw_censor_blur_strength_curve': shared.OptionInfo(3, 'Blur strength curve - Variable blur', gr.Slider, {'minimum': 0, 'maximum': 6}).info('default 3'),
    'nudenet_nsfw_censor_pixelation_factor': shared.OptionInfo(5, 'Pixelation factor', gr.Slider, {'minimum': 1, 'maximum': 10}).info('default 5'),
    'nudenet_nsfw_censor_fill_color': shared.OptionInfo('#000000', 'Fill color', gr.ColorPicker, {}).info('default #000000'),
    'nudenet_nsfw_censor_nms_threshold': shared.OptionInfo(0.5, 'Non-Maximum Suppression threshold', gr.Slider, {'minimum': 0, 'maximum': 1}).info('default 0.5'),
    'nudenet_nsfw_censor_verbose_detection': shared.OptionInfo(False, 'Print detection info in terminal').info('default False'),
    "nudenet_nsfw_censor_onnx_provider": shared.OptionInfo(default_onnx_provider, 'ONNX provider', gr.Radio, {'choices': available_onnx_providers}, onchange=pil_nude_detector.change_onnx_provider).info(f'CPU is recommended, default {default_onnx_provider}'),
    'nudenet_nsfw_censor_selected_labels': shared.OptionInfo(
        ['Anus exposed', 'Female breast exposed', 'Female genitalia exposed', 'Male genitalia exposed'],
        'Categories to be censored', ui_components.DropdownMulti, {'choices': list(nudenet_labels_index)},
        onchange=pil_nude_detector.refresh_label_configs
    ).info("default 'Anus exposed', 'Female breast exposed', 'Female genitalia exposed', 'Male genitalia exposed'"),
}


for key, value in nudenet_labels_index.items():
    censor_region_settings[f'nudenet_nsfw_censor_label_threshold_{value[1]}'] = shared.OptionInfo(value[2][0], f'{key} - threshold', gr.Slider, {'minimum': 0, 'maximum': 1}, onchange=pil_nude_detector.refresh_label_configs).info(f'default {value[2][0]}')
    censor_region_settings[f'nudenet_nsfw_censor_label_horizontal_{value[1]}'] = shared.OptionInfo(value[2][1], f'{key} - horizontal multiplier', gr.Number, onchange=pil_nude_detector.refresh_label_configs).info(f'default {value[2][1]}')
    censor_region_settings[f'nudenet_nsfw_censor_label_vertical_{value[1]}'] = shared.OptionInfo(value[2][2], f'{key} - vertical multiplier', gr.Number, onchange=pil_nude_detector.refresh_label_configs).info(f'default {value[2][2]}')

shared.options_templates.update(shared.options_section(('nudenet_nsfw_censor', 'NudeNet NSFW Censor'), censor_region_settings))

extension_version = {
    'nudenet_nsfw_censor_version': shared.OptionInfo('2.0', 'extension_version', component_args={'interactive': False})
}

if not hasattr(shared.opts, 'nudenet_nsfw_censor_version') and hasattr(shared.opts, 'nudenet_nsfw_censor_enable'):
    # update to version 2.0, force reset horizontal and vertical multipliers to new default value of 1.0, reason the old default value are obsolete
    [setattr(shared.opts, f'nudenet_nsfw_censor_label_horizontal_{label}', 1.0) for label in nudenet_labels_dict]
    [setattr(shared.opts, f'nudenet_nsfw_censor_label_vertical_{label}', 1.0) for label in nudenet_labels_dict]

shared.options_templates.update(shared.options_section(('nudenet_nsfw_censor', 'NudeNet NSFW Censor'), extension_version))
