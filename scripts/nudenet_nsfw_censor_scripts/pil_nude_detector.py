from onnxruntime import InferenceSession, get_available_providers
from PIL import Image, ImageDraw
from cv2.dnn import NMSBoxes
from modules import shared
from pathlib import Path
from math import sqrt
import numpy as np

nudenet_labels_dict = {
    'Female_genitalia_covered': [0.25, 1.0, 1.0],
    'Face_female': [0.25, 1.0, 1.0],
    'Buttocks_exposed': [0.25, 1.0, 1.0],
    'Female_breast_exposed': [0.25, 1.0, 1.0],
    'Female_genitalia_exposed': [0.25, 1.0, 1.0],
    'Male_breast_exposed': [0.25, 1.0, 1.0],
    'Anus_exposed': [0.25, 1.0, 1.0],
    'Feet_exposed': [0.25, 1.0, 1.0],
    'Belly_covered': [0.25, 1.0, 1.0],
    'Feet_covered': [0.25, 1.0, 1.0],
    'Armpits_covered': [0.25, 1.0, 1.0],
    'Armpits_exposed': [0.25, 1.0, 1.0],
    'Face_male': [0.25, 1.0, 1.0],
    'Belly_exposed': [0.25, 1.0, 1.0],
    'Male_genitalia_exposed': [0.25, 1.0, 1.0],
    'Anus_covered': [0.25, 1.0, 1.0],
    'Female_breast_covered': [0.25, 1.0, 1.0],
    'Buttocks_covered': [0.25, 1.0, 1.0],
}

nudenet_labels_index = {key.replace('_', ' '): (index, key, value) for index, (key, value) in enumerate(nudenet_labels_dict.items())}
nudenet_labels_friendly_name = list(nudenet_labels_index)
nudenet_labels_index = {key: nudenet_labels_index[key] for key in sorted(nudenet_labels_index)}


def draw_ellipse(draw, left_expanded, top_expanded, right_expanded, down_expanded, *args, **kwargs):
    draw.ellipse((left_expanded, top_expanded, right_expanded, down_expanded), 1)


def draw_rectangle(draw, left_expanded, top_expanded, right_expanded, down_expanded, *args, **kwargs):
    draw.rectangle((left_expanded, top_expanded, right_expanded, down_expanded), 1)


def rounded_rectangle(draw, left_expanded, top_expanded, right_expanded, down_expanded, width_expanded, height_expanded, rectangle_round_radius, *args, **kwargs):
    if rectangle_round_radius > 1:
        # scale with mask size
        round_radius = rectangle_round_radius * sqrt((width_expanded ** 2 + height_expanded ** 2) / 524288)
    elif rectangle_round_radius < 0:
        # negative value don't scale
        round_radius = -rectangle_round_radius
    else:
        # if 1 >= rectangle_round_radius > 0, rectangle_round_radius is rounded fraction
        round_radius = (width_expanded if width_expanded < height_expanded else height_expanded) / 2 * rectangle_round_radius
    draw.rounded_rectangle((left_expanded, top_expanded, right_expanded, down_expanded), round(round_radius), 1)
    draw.rounded_rectangle((round(left_expanded), round(top_expanded), round(right_expanded), round(down_expanded)),
                           round(round_radius), 1)


mask_shapes_func_dict = {
    'Ellipse': draw_ellipse,
    'Rectangle': draw_rectangle,
    'Rounded rectangle': rounded_rectangle,
    'Entire image': None,
}

available_onnx_providers = get_available_providers()
if 'CPUExecutionProvider' in available_onnx_providers:
    default_onnx_provider = 'CPUExecutionProvider'
else:
    default_onnx_provider = available_onnx_providers[0]


class PilNudeDetector:
    def __init__(self):
        # NudeNet is sufficiently lightweight that running on CPU is preferable
        self.onnx_session = None
        self.input_name = None
        self.input_width = None
        self.input_height = None

        self.label_config = None
        self.thresholds = None
        self.expand_horizontal = None
        self.expand_vertical = None

    def init_onnx(self):
        self.onnx_session = InferenceSession(
            str(Path(__file__).parent.parent.parent.joinpath('nudenet', 'best.onnx')),
            providers=[shared.opts.nudenet_nsfw_censor_onnx_provider],
        )
        model_inputs = self.onnx_session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_name = model_inputs[0].name
        self.input_width = input_shape[2]  # 320
        self.input_height = input_shape[3]  # 320

    def change_onnx_provider(self):
        if self.onnx_session is None:
            self.init_onnx()
        self.onnx_session.set_providers([shared.opts.nudenet_nsfw_censor_onnx_provider])

    def refresh_label_configs(self):
        """
        Read configs from settings initialize or refresh self.thresholds, self.expand_horizontal, self.expand_vertical
        """
        enabled_labels = np.zeros(len(nudenet_labels_dict), dtype=np.bool_)
        for i in shared.opts.nudenet_nsfw_censor_selected_labels:
            enabled_labels[nudenet_labels_index[i][0]] = True
        self.thresholds = np.array([getattr(shared.opts, f'nudenet_nsfw_censor_label_threshold_{label}', 1,) for label in nudenet_labels_dict], dtype=np.float32)
        self.thresholds[~enabled_labels] = 1
        self.expand_horizontal = np.array([getattr(shared.opts, f'nudenet_nsfw_censor_label_horizontal_{label}', 1) for label in nudenet_labels_dict], dtype=np.float32)
        self.expand_vertical = np.array([getattr(shared.opts, f'nudenet_nsfw_censor_label_vertical_{label}', 1) for label in nudenet_labels_dict], dtype=np.float32)

    def pre_process_pil(self, pil_image):
        """Resize and pad image with white background if not to scale for detection
        Args:
            pil_image:

        Returns: numpy.float32 array
        """
        # resize
        width, height = pil_image.size
        new_size = (self.input_width, round(self.input_height * height / width)) if width > height else (round(self.input_width * width / height), self.input_height)
        resized_image = pil_image.resize(new_size)

        # pad image with white background
        pad_image = Image.new(resized_image.mode, (self.input_width,  self.input_height), (255, 255, 255))
        offset = ((self.input_width - new_size[0]) // 2, (self.input_height - new_size[1]) // 2)
        pad_image.paste(resized_image, offset)

        # convert to desired format
        return np.expand_dims(np.array(pad_image, dtype=np.float32).transpose(2, 0, 1) / 255.0, axis=0)

    def calculate_censor_mask(self, detection_results, img_size, thresholds, expand_horizontal, expand_vertical, nms_threshold, nudenet_nsfw_censor_mask_shape, rectangle_round_radius):
        """
        Generate binary mask from detection results of nudenet filtered and adjusted based on label_configs
        Args:
            detection_results: nudenet output
            img_size: (width, height) original image width
            thresholds:
            expand_horizontal:
            expand_vertical:
            nms_threshold: float [0, 1] Non-Maximum Suppression threshold for cv2.dnn.NMSBoxes
            nudenet_nsfw_censor_mask_shape:
            rectangle_round_radius:
        Returns: PIL binary mask
        """
        # if self.thresholds is None:
        #     self.refresh_label_configs()

        # [x_center, y_center, box_width, box_height, score_0, score_2, ..., score_16, score_17]
        outputs = np.transpose(np.squeeze(detection_results[0]))

        # get a bool array all boxes with its max score greater than the defined threshold of its category
        filter_results = np.max(outputs[:, 4:], axis=1) > thresholds[np.argmax(outputs[:, 4:], axis=1)]

        if np.any(filter_results):
            draw_func = mask_shapes_func_dict[nudenet_nsfw_censor_mask_shape]
            if draw_func is None:
                # just return a mask for the entire image
                return Image.new('1', img_size, 1)
            else:
                image_mask = Image.new('1', img_size, 0)
                draw = ImageDraw.Draw(image_mask)
                verbose = ''

                max_score_indices = np.argmax(outputs[:, 4:], axis=1)
                detection_results = outputs[filter_results]

                boxes = detection_results[:, :4]
                scores = detection_results[:, 4:][np.arange(detection_results.shape[0]), max_score_indices[filter_results]]
                class_index = max_score_indices[filter_results]

                # convert detected box coordinates (x_center, y_center, box_width, box_height) to (x_1, y_1, box_width, box_height)
                boxes[:, 0:2] -= boxes[:, 2:4] / 2

                # Non-Maximum Suppression
                if nms_threshold < 1:
                    nms = NMSBoxes(boxes, scores, 0, nms_threshold)
                    boxes = boxes[nms]
                    scores = scores[nms]
                    class_index = class_index[nms]

                # scale to original image width
                offset = abs(img_size[0] - img_size[1]) / 2
                if img_size[0] > img_size[1]:
                    factor = img_size[0] / self.input_width
                    boxes *= factor
                    boxes[:, 1] -= offset
                else:
                    factor = img_size[1] / self.input_height
                    boxes *= factor
                    boxes[:, 0] -= offset

                wh_e = boxes[:, 2:4] * np.vstack((expand_horizontal[class_index], expand_vertical[class_index])).T
                boxes[:, 0:2] -= (wh_e - boxes[:, 2:4])/2
                # x1y1x2y2
                boxes[:, 2:4] = boxes[:, 0:2] + wh_e
                boxes = boxes.round()

                for i in range(scores.shape[0]):
                    x1y1x2y2 = boxes[i]
                    wh = wh_e[i]
                    draw_func(draw, *x1y1x2y2, *wh, rectangle_round_radius)

                    if shared.opts.nudenet_nsfw_censor_verbose_detection:
                        verbose += (
                            f'\n{nudenet_labels_friendly_name[class_index[i]]}: score {scores[i]}, x1 {x1y1x2y2[0]} y1 {x1y1x2y2[1]}, w {wh[0].round()} h {wh[1].round()}, x2 {x1y1x2y2[2]} y2 {x1y1x2y2[3]}'
                        )

                if verbose:
                    print(verbose)

                return image_mask

    def get_censor_mask(self, pil_image, nms_threshold, nudenet_nsfw_censor_mask_shape, rectangle_round_radius, thresholds, expand_horizontal, expand_vertical):
        if self.onnx_session is None:
            self.init_onnx()
        detection_results = self.onnx_session.run(None, {self.input_name: self.pre_process_pil(pil_image)})
        return self.calculate_censor_mask(detection_results, pil_image.size, thresholds, expand_horizontal, expand_vertical, nms_threshold, nudenet_nsfw_censor_mask_shape, rectangle_round_radius)


pil_nude_detector = PilNudeDetector()
