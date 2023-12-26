from scripts.nudenet_nsfw_censor_scripts.pil_nude_detector import pil_nude_detector, mask_shapes_func_dict
from scripts.nudenet_nsfw_censor_scripts.censor_image_filters import apply_filter, filter_dict
from modules import shared, images, scripts_postprocessing
from PIL import Image, ImageFilter
from math import sqrt
import gradio as gr


if hasattr(scripts_postprocessing.ScriptPostprocessing, 'process_firstpass'):  # webui >= 1.7
    from modules.ui_components import InputAccordion
else:
    InputAccordion = None

filter_opt_ui_show_dict = {
    # [blur_radius, blur_strength_curve, pixelation_factor, fill_color, mask_blend_radius, mask_blend_radius_variable_blur]
    'Variable blur': [True, True, False, False, False, True],
    'Gaussian Blur': [True, False, False, False, True, False],
    'Pixelate': [False, False, True, False, True, False],
    'Fill color': [False, False, False, True, True, False],
    'Detect only': [False, False, False, False, True, False],
}
mask_shape_opt_ui_show_dict = {
    # [(mask_blend_radius, mask_blend_radius_variable_blur), rectangle_round_radius, nms_threshold]
    'Ellipse': [True, False, True],
    'Rectangle': [True, False, True],
    'Rounded rectangle': [True, True, True],
    'Entire image': [False, False, False]
}


class ScriptPostprocessingNudenetCensor(scripts_postprocessing.ScriptPostprocessing):
    name = 'NudeNet NSFW censor'
    order = 100000

    def ui(self):
        with (
            InputAccordion(False, label="NSFW Censor", elem_id='nudenet_nsfw_censor_extras') if InputAccordion
            else gr.Accordion('NSFW Censor', open=False, elem_id='nudenet_nsfw_censor_extras')
            as enable
        ):
            with gr.Row():
                if not InputAccordion:
                    enable = gr.Checkbox(False, label='Enable', elem_id='nudenet_nsfw_censor_extras-visible-checkbox')
                enable_nudenet = gr.Checkbox(True, label='NudeNet Auto-detect')
                save_mask = gr.Checkbox(False, label='Save mask')
                override_settings = gr.Checkbox(False, label='Override filter configs')
            with gr.Row():
                filter_type = gr.Dropdown(value='Variable blur', label='Censor filter', choices=list(filter_dict), visible=False)
                mask_shape = gr.Dropdown(value='Ellipse', choices=list(mask_shapes_func_dict), label='Mask shape', visible=False)
            with gr.Row():
                blur_radius = gr.Slider(0, 100, 10, label='Blur radius', visible=False)  # Variable blur Gaussian Blur
                blur_strength_curve = gr.Slider(0, 6, 3, label='Blur strength curve', visible=False)  # Variable blur
                pixelation_factor = gr.Slider(1, 10, 5, label='Pixelation factor', visible=False)  # Pixelate
                fill_color = gr.ColorPicker(value='#000000', label='fill color', visible=False)  # Fill color
                mask_blend_radius = gr.Slider(0, 100, 0, label='Mask blend radius', visible=False)  # except Variable blur
                mask_blend_radius_variable_blur = gr.Slider(0, 100, 10, label='Variable blur mask blend radius', visible=False)  # Variable blur
                nms_threshold = gr.Slider(0, 1, 1, label='NMS threshold', visible=False)  # NMS threshold
                rectangle_round_radius = gr.Number(value=0.5, label='Rectangle round radius', visible=False)  # Rounded rectangle
            with gr.Row():
                create_canvas = gr.Button('Create canvas')
                mask_source = gr.CheckboxGroup(['Draw mask', 'Upload mask'], value=['Draw mask'], label="Canvas mask source")
                mask_brush_color = gr.ColorPicker('#000000', label='Brush color', info='visual only, use when brush color is hard to see')
            with gr.Row():
                input_mask = gr.Image(
                    label="Censor mask",
                    show_label=False,
                    elem_id="nsfw_censor_mask",
                    source="upload",
                    interactive=True,
                    type="pil",
                    tool="sketch",
                    image_mode="RGBA",
                    brush_color='#000000'
                )

                def update_mask_brush_color(color):
                    return gr.Image.update(brush_color=color)

                mask_brush_color.change(
                    fn=update_mask_brush_color,
                    inputs=[mask_brush_color],
                    outputs=[input_mask]
                )

                def get_current_image(image):
                    # ToDo if possible make this a client side operation
                    if image:
                        return gr.Image.update(image)

                dummy_component = gr.Label(visible=False)
                create_canvas.click(
                    fn=get_current_image,
                    _js='getCurrentExtraSourceImg',
                    inputs=[dummy_component],
                    outputs=[input_mask],
                    postprocess=False,
                )

            def update_opt_ui(_filter_type, _mask_shape, _override_settings, _enable_nudenet):
                filter_opt_enable_list = filter_opt_ui_show_dict[_filter_type]
                mask_shape_opt_show_list = mask_shape_opt_ui_show_dict[_mask_shape]
                # blur_radius, blur_strength_curve, pixelation_factor, fill_color, mask_blend_radius, mask_blend_radius_variable_blur, rectangle_round_radius, nms_threshold
                return (
                    gr.Dropdown.update(visible=_override_settings),  # filter_type
                    gr.Dropdown.update(visible=_override_settings),  # mask_shape
                    gr.Slider.update(visible=_override_settings and filter_opt_enable_list[0]),  # blur_radius
                    gr.Slider.update(visible=_override_settings and filter_opt_enable_list[1]),  # blur_strength_curve
                    gr.Slider.update(visible=_override_settings and filter_opt_enable_list[2]),  # pixelation_factor
                    gr.ColorPicker.update(visible=_override_settings and filter_opt_enable_list[3]),  # fill_color
                    gr.Slider.update(visible=_override_settings and filter_opt_enable_list[4] and mask_shape_opt_show_list[0]),  # mask_blend_radius
                    gr.Slider.update(visible=_override_settings and filter_opt_enable_list[5] and mask_shape_opt_show_list[0]),  # mask_blend_radius_variable_blur
                    gr.Number().update(visible=_override_settings and mask_shape_opt_show_list[1]),  # rectangle_round_radius
                    gr.Slider.update(visible=_override_settings and mask_shape_opt_show_list[2] and _enable_nudenet),  # nms_threshold
                )

            for element in [override_settings, filter_type, mask_shape, enable_nudenet]:
                element.change(update_opt_ui, inputs=[filter_type, mask_shape, override_settings, enable_nudenet], outputs=[filter_type, mask_shape, blur_radius, blur_strength_curve, pixelation_factor, fill_color, mask_blend_radius, mask_blend_radius_variable_blur, rectangle_round_radius, nms_threshold])

        controls = {
            'enable': enable,
            'enable_nudenet': enable_nudenet,
            'override_settings': override_settings,
            'save_mask': save_mask,
            'filter_type': filter_type,
            'blur_radius': blur_radius,
            'pixelation_factor': pixelation_factor,
            'fill_color': fill_color,
            'mask_shape': mask_shape,
            'blur_strength_curve': blur_strength_curve,
            'mask_blend_radius': mask_blend_radius,
            'mask_blend_radius_variable_blur': mask_blend_radius_variable_blur,
            'rectangle_round_radius': rectangle_round_radius,
            'nms_threshold': nms_threshold,
            'input_mask': input_mask,
            'mask_source': mask_source,
        }
        return controls

    def process(self, pp: scripts_postprocessing.PostprocessedImage, **args):
        if not args['enable']:
            return
        censor_mask = None

        if args['input_mask']:
            if 'Upload mask' in args['mask_source']:
                censor_mask = args['input_mask']['image'].convert('L').resize(pp.image.size)
            if 'Draw mask' in args['mask_source']:
                censor_mask = Image.new('L', pp.image.size, 0) if censor_mask is None else censor_mask
                draw_mask = args['input_mask']['mask'].convert('L').resize(pp.image.size)
                censor_mask.paste(draw_mask, draw_mask)

        if args['enable_nudenet']:
            if args['override_settings']:
                nms_threshold = args['nms_threshold']
                mask_shape = args['mask_shape']
                rectangle_round_radius = args['rectangle_round_radius']
            else:
                nms_threshold = shared.opts.nudenet_nsfw_censor_nms_threshold
                mask_shape = shared.opts.nudenet_nsfw_censor_mask_shape
                rectangle_round_radius = shared.opts.nudenet_nsfw_censor_rectangle_round_radius

            if pil_nude_detector.thresholds is None:
                pil_nude_detector.refresh_label_configs()
            nudenet_mask = pil_nude_detector.get_censor_mask(pp.image, nms_threshold, mask_shape, rectangle_round_radius, pil_nude_detector.thresholds, pil_nude_detector.expand_horizontal, pil_nude_detector.expand_vertical)
            if nudenet_mask is not None:
                nudenet_mask = nudenet_mask.convert('L')

            if nudenet_mask and censor_mask:
                censor_mask.paste(nudenet_mask, nudenet_mask)
            else:
                censor_mask = nudenet_mask

        if censor_mask:

            scale_factor = sqrt((pp.image.size[0] ** 2 + pp.image.size[1] ** 2) / 524288)
            save_mask = args['save_mask']
            if args['override_settings']:
                filter_type = args['filter_type']
                mask_blend_radius = args['mask_blend_radius_variable_blur'] if filter_type == 'Variable blur' else args['mask_blend_radius']
                filter_settings = {
                    'blur_radius': args['blur_radius'],
                    'blur_strength_curve': args['blur_strength_curve'],
                    'color': args['fill_color'],
                    'pixelation_factor': args['pixelation_factor'],
                }
            else:
                filter_type = shared.opts.nudenet_nsfw_censor_extras_filter_type
                mask_blend_radius = shared.opts.nudenet_nsfw_censor_mask_blend_radius_variable_blur if filter_type == 'Variable blur' else shared.opts.nudenet_nsfw_censor_mask_blend_radius
                filter_settings = {
                    'blur_radius': shared.opts.nudenet_nsfw_censor_blur_radius * scale_factor,
                    'blur_strength_curve': shared.opts.nudenet_nsfw_censor_blur_strength_curve,
                    'color': shared.opts.nudenet_nsfw_censor_fill_color,
                    'pixelation_factor': shared.opts.nudenet_nsfw_censor_pixelation_factor,
                }

            censor_mask = censor_mask.filter(ImageFilter.GaussianBlur(mask_blend_radius * scale_factor))

            if filter_type:
                pp.image = apply_filter(pp.image, censor_mask, filter_type, **filter_settings)

            if save_mask:
                # ToDo save mask with info text and to same dir
                images.save_image(censor_mask, shared.opts.outdir_samples or shared.opts.outdir_extras_samples, 'censor_mask', extension=shared.opts.samples_format)
