from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageFilter
import numpy as np


def combine_results(input_image, input_mask, processed):
    canvas = input_image.copy()
    canvas.paste(processed, input_mask)
    return canvas


def variable_blur(input_image: Image, control_mask: Image, blur_radius: float = 10, blur_strength_curve: float = 3, *args, **kwargs):
    """
    Apply gaussian blur of varying strength to the image base of te value of the mask
    Args:
        input_image: PIL image to be filtered
        control_mask: PIL in L mode (8-bit pixels, grayscale)
        blur_radius: A float max blur radius
        blur_strength_curve: A float [0, 6]
            0 no blur (just return a copy of the input image with no change)
            3 linear blur strength (Default)
            4 linear blur radius
            6 Applies uniform blur_radius to the entire image, same as ImageFilter.GaussianBlur(blur_radius)

    Returns: filtered PIL image
    """
    if blur_strength_curve <= 0:
        return input_image.copy()
    elif blur_strength_curve > 6:
        return input_image.filter(ImageFilter.GaussianBlur(blur_radius))

    control_mask_np = np.asarray(control_mask)
    blur_levels = np.unique(control_mask_np)
    blur_radius_array = np.power(blur_levels / 255, (12 - 2 * blur_strength_curve) / blur_strength_curve) * blur_radius  # [0, 6] 4 is norm 3 is linear

    with ThreadPoolExecutor(1024) as executor:

        def mask_array_to_img(i):
            return Image.frombuffer('L', input_image.size, np.where(control_mask_np == blur_levels[i], np.uint8(255), np.uint8(0)))

        futures_mask_array_to_img = [executor.submit(mask_array_to_img, i) for i in range(blur_levels.size)]

        def img_gaussian_blur(i):
            return input_image.filter(ImageFilter.GaussianBlur(blur_radius_array[i]))

        futures_img_gaussian_blur = [executor.submit(img_gaussian_blur, i) for i in range(blur_levels.size)]

        futures_combine, futures_combine_mask = [None] * blur_levels.size, [None] * blur_levels.size

        def combine_mask(index_1, index_2, pre_step_size):
            if pre_step_size:
                pre_step_index_1, pre_step_index_2 = index_2 - pre_step_size, index_2 + pre_step_size
                futures_combine_mask[pre_step_index_1].result()
                if pre_step_index_2 < blur_levels.size:
                    futures_combine_mask[index_2 + pre_step_size].result()
            futures_mask_array_to_img[index_1].result().paste(futures_mask_array_to_img[index_2].result(), futures_mask_array_to_img[index_2].result())

        def combine(index_1, index_2, pre_step_size):
            if pre_step_size:
                pre_step_index_1, pre_step_index_2 = index_2 - pre_step_size, index_2 + pre_step_size
                futures_combine[pre_step_index_1].result(), futures_combine_mask[pre_step_index_1].result()
                if pre_step_index_2 < blur_levels.size:
                    futures_combine[pre_step_index_2].result(), futures_combine_mask[pre_step_index_2].result()
            futures_img_gaussian_blur[index_1].result().paste(futures_img_gaussian_blur[index_2].result(), futures_mask_array_to_img[index_2].result())

        step_pre, step_combine, step_size, stride_size_limit = 0, 1, 2, blur_levels.size * 2
        while step_size < stride_size_limit:
            for index in range(0, blur_levels.size - step_combine, step_size):
                index_combine = index + step_combine
                futures_combine_mask[index_combine] = executor.submit(combine_mask, index, index_combine, step_pre)
                futures_combine[index_combine] = executor.submit(combine, index, index_combine, step_pre)
            step_pre, step_combine, step_size = step_combine, step_size, step_size * 2

    return futures_img_gaussian_blur[0].result()


def gaussian_blur(input_image, input_mask, blur_radius, *args, **kwargs):
    blured_image = input_image.filter(ImageFilter.GaussianBlur(blur_radius))
    return combine_results(input_image, input_mask, blured_image)


def pixelate(input_image, input_mask, pixelation_factor, *args, **kwargs):
    scale_size = tuple(round(d / pixelation_factor**2) for d in input_image.size)
    pixelated_image = input_image.resize(scale_size, Image.Resampling.BILINEAR).resize(input_image.size, Image.Resampling.NEAREST)
    return combine_results(input_image, input_mask, pixelated_image)


def fill_color(input_image, input_mask, color, *args, **kwargs):
    color_image = Image.new(input_image.mode, input_image.size, color)
    return combine_results(input_image, input_mask, color_image)


def do_nothing(input_image, *args, **kwargs):
    return input_image


filter_dict = {
    'Variable blur': variable_blur,
    'Gaussian Blur': gaussian_blur,
    'Pixelate': pixelate,
    'Fill color': fill_color,
    'No censor': do_nothing,
    'Disable': None
}


def apply_filter(input_image, input_mask, filter_type, *args, **kwargs):
    return filter_dict[filter_type](input_image, input_mask, *args, **kwargs)
