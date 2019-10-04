import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json
from collections import OrderedDict
import torch
import json

#plot bbox for prediction result, bbox_data includes the dictionary of pred_bboxes with confidence score
def plot_bbox(image, bbox_data, result_path="somewhere"):
    # image = np.array(Image.open(img_path), dtype=np.uint8)
    try:
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for bbox in bbox_data:
            x1 = bbox['xmin']
            y1 = bbox['ymin']
            x2 = bbox['xmax']
            y2 = bbox['ymax']
            score = bbox['confidence_score']
            rect = patches.Rectangle((x1, y1), x2-x1+1, y2-y1+1, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1, score, color='red', verticalalignment='top')

        # plt.show()
        plt.axis('off')
        plt.savefig(result_path, dpi=200, bbox_inches='tight', pad_inches=0.0)
        plt.close()
    except:
        pass

def compute_IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def visualize_both_detection(img_path, gt_bboxs, pred_bboxs):
    img = Image.open(img_path)
    pixdata = img.load()
    (w, h) = img.size

    for bbox in gt_bboxs:
        [xmin, ymin, xmax, ymax] = bbox[1]
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        xmax -= 1
        ymax -= 1
        for i in range(xmin, xmax + 1):
            for j in range(3):
                pixdata[i, ymin+j] = RED
                pixdata[i, ymax-j] = RED

        for i in range(3):
            for j in range(ymin, ymax + 1):
                pixdata[xmin+i, j] = RED
                pixdata[xmax-i, j] = RED

    for bbox in pred_bboxs:
        [xmin, ymin, xmax, ymax] = bbox[1]
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        xmax -= 1
        ymax -= 1
        for i in range(xmin, xmax + 1):
            for j in range(3):
                pixdata[i, ymin+j] = BLUE
                pixdata[i, ymax-j] = BLUE

        for i in range(3):
            for j in range(ymin, ymax + 1):
                pixdata[xmin+i, j] = BLUE
                pixdata[xmax-i, j] = BLUE

    return img

def verify(x, w):
    return min(max(int(x), 0), w-1)

def get_gt_bbox(label_path):
    bboxes = list(open(label_path, 'r'))
    gt_bboxes = []
    for bbox in bboxes:
        bbox = bbox.strip().split(' ')
        bbox = [int(item) for item in bbox]

        gt_bboxes.append(bbox[:-1])

    return gt_bboxes

def visualize_bbox(image, bboxes):
    RED = (255, 0, 0)
    pixdata = image.load()
    (w, h) = image.size

    for bbox in bboxes:
        [xmin, ymin, xmax, ymax] = bbox

        for i in range(xmin, xmax + 1):
            for j in range(3):
                pixdata[i, ymin+j] = RED
                pixdata[i, ymax-j] = RED

        for i in range(3):
            for j in range(ymin, ymax + 1):
                pixdata[xmin+i, j] = RED
                pixdata[xmax-i, j] = RED

    return image


def visualize_bboxes(sk_image, bboxes, color):
    def relocate(x, t):
        x = max(0, x)
        x = min(t-1, x)
        return x
    color = np.array(color)
    (height, width, _) = sk_image.shape
    for bbox in bboxes:
        x1 = bbox['x']
        y1 = bbox['y']
        x2 = x1 + bbox['w'] - 1
        y2 = y1 + bbox['h'] - 1
        sk_image[y1:y2, relocate(x1-3, width):relocate(x1+3, width)] = color
        sk_image[y1:y2, relocate(x2-3, width):relocate(x2+3, width)] = color

        sk_image[relocate(y1-3, height):relocate(y1+3, height), x1:x2] = color
        sk_image[relocate(y2-3, height):relocate(y2+3, height), x1:x2] = color

    return sk_image


class KeepAspect(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']

        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        image_new = np.pad(image, pad, 'constant', constant_values=128)

        return {'image': image_new, 'padding': pad}


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1,  keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))

    return output

def get_bbox_data(pred_bboxes_data):
    """
    input: prediction bbox list
    output: pred_bboxes, json_bboxes
    """
    pred_bboxes = pred_bboxes_data.bbox
    pred_confidences = pred_bboxes_data.extra_fields['scores']
    num_pred_bboxes = pred_bboxes.size(0)

    pred_bboxes_list = [[] for _ in range(num_pred_bboxes)]
    for i in range(num_pred_bboxes):
        pred_bbox = []
        for j in range(4):
            pred_bbox.append(int(pred_bboxes[i][j]))
        pred_bboxes_list[i] = pred_bbox
    pred_bboxes = pred_bboxes_list
    

    json_bboxes = {"bboxes": []}
    for pred_id, [x1, y1, x2, y2] in enumerate(pred_bboxes):
        json_bbox = OrderedDict()
        json_bbox['x1'] = x1
        json_bbox['y1'] = y1
        json_bbox['x2'] = x2
        json_bbox['y2'] = y2
        json_bbox['confidence'] = pred_confidences[pred_id].item()

        json_bboxes["bboxes"].append(json_bbox)


    return pred_bboxes, json_bboxes