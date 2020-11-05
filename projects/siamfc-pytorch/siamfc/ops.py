import numbers

import cv2
import numpy as np
import torch.nn as nn


def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img


def show_image(img,
               boxes=None,
               box_fmt='ltwh',
               colors=None,
               thickness=3,
               fig_n=1,
               delay=1,
               visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)

    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale

    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]

        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])

        if colors is None:
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
                      (255, 0, 255), (255, 255, 0), (0, 0, 128), (0, 128, 0),
                      (128, 0, 0), (0, 128, 128), (128, 0, 128), (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)

    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img


def crop_and_resize(img,
                    center,
                    size,
                    out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR,
                    faster=True):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((np.round(center - (size - 1) / 2),
                              np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    if faster:
        patch = get_cropped_input(img, corners, 1, out_size, interp,
                                  border_value)[0]
        return patch
    # pad image if necessary
    pads = np.concatenate((-corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad, border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size), interpolation=interp)

    return patch


def get_cropped_input(inputImage,
                      bbox,
                      padScale,
                      outputSize,
                      interpolation=cv2.INTER_LINEAR,
                      pad_color=0):
    bbox = np.array(bbox)
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    imShape = np.array(inputImage.shape)
    if len(imShape) < 3:
        inputImage = inputImage[:, :, np.newaxis]
    xC = float(bbox[0] + bbox[2]) / 2
    yC = float(bbox[1] + bbox[3]) / 2
    boxOn = np.zeros(4)
    boxOn[0] = float(xC - padScale * width / 2)
    boxOn[1] = float(yC - padScale * height / 2)
    boxOn[2] = float(xC + padScale * width / 2)
    boxOn[3] = float(yC + padScale * height / 2)
    outputBox = boxOn.copy()
    boxOn = np.round(boxOn).astype(int)
    boxOnWH = np.array([boxOn[2] - boxOn[0], boxOn[3] - boxOn[1]])
    imagePatch = inputImage[max(boxOn[1], 0):min(boxOn[3], imShape[0]),
                            max(boxOn[0], 0):min(boxOn[2], imShape[1]), :]
    boundedBox = np.clip(boxOn, 0, imShape[[1, 0, 1, 0]])
    boundedBoxWH = np.array(
        [boundedBox[2] - boundedBox[0], boundedBox[3] - boundedBox[1]])

    if imagePatch.shape[0] == 0 or imagePatch.shape[1] == 0:
        patch = np.zeros((int(outputSize), int(outputSize), 3),
                         dtype=imagePatch.dtype)
    else:
        patch = cv2.resize(
            imagePatch,
            (
                max(1, int(
                    np.round(outputSize * boundedBoxWH[0] / boxOnWH[0]))),
                max(1, int(
                    np.round(outputSize * boundedBoxWH[1] / boxOnWH[1]))),
            ),
            interpolation=interpolation,
        )
        if len(patch.shape) < 3:
            patch = patch[:, :, np.newaxis]
        patchShape = np.array(patch.shape)

        pad = np.zeros(4, dtype=int)
        pad[:2] = np.maximum(0, -boxOn[:2] * outputSize / boxOnWH)
        pad[2:] = outputSize - (pad[:2] + patchShape[[1, 0]])

        if np.any(pad != 0):
            if len(pad[pad < 0]) > 0:
                patch = np.zeros((int(outputSize), int(outputSize), 3))
            else:
                if isinstance(pad_color, numbers.Number):
                    patch = np.pad(
                        patch, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                        'constant',
                        constant_values=pad_color)
                else:
                    patch = cv2.copyMakeBorder(
                        patch,
                        pad[1],
                        pad[3],
                        pad[0],
                        pad[2],
                        cv2.BORDER_CONSTANT,
                        value=pad_color)

    return patch, outputBox
