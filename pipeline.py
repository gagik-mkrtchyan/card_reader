from torchvision import io
import torchvision
import torch
import numpy as np
import cv2
import time

def read_decode_image(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_tensor = io.decode_jpeg(input=torch.tensor(list(image_bytes), dtype=torch.uint8), apply_exif_orientation=True, mode=torchvision.io.ImageReadMode.RGB)
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image_tensor

def bilinear_interpolate_vectorized(input, size):
    """
    Resize an image tensor using bilinear interpolation with vectorized operations.

    Args:
        input (torch.Tensor): The input image tensor of shape (C, H, W).
        size (tuple): The desired output size (new_H, new_W).

    Returns:
        torch.Tensor: The resized image tensor of shape (1, C, new_H, new_W).
    """
    c, h, w = input.shape
    new_h, new_w = size
    # Create an empty tensor for the resized image
    resized_image = torch.zeros((c, new_h, new_w), device=input.device)
    # Scale factors
    scale_y = torch.linspace(0, h - 1, new_h, device=input.device)
    scale_x = torch.linspace(0, w - 1, new_w, device=input.device)
    # Get the coordinates of the four neighbors
    x0 = torch.floor(scale_x).long()
    x1 = torch.clamp(x0 + 1, max=w - 1)
    y0 = torch.floor(scale_y).long()
    y1 = torch.clamp(y0 + 1, max=h - 1)
    # Calculate the distances to the neighbors
    dx = scale_x - x0.float()
    dy = scale_y - y0.float()
    for i in range(c):
        # Gather the values of the four neighbors
        q11 = input[i, y0][:, x0]
        q12 = input[i, y0][:, x1]
        q21 = input[i, y1][:, x0]
        q22 = input[i, y1][:, x1]
        # Interpolate in x direction
        top = q11 * (1 - dx) + q12 * dx
        bottom = q21 * (1 - dx) + q22 * dx

        # Interpolate in y direction
        resized_image[i] = top * (1 - dy).unsqueeze(1) + bottom * dy.unsqueeze(1)

    resized_image = resized_image.unsqueeze(0) # adding batch_size 1 [1, C, new_H, new_W]

    return resized_image

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else torch.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def iou(box_a, box_b):

    x1, y1, x2, y2 = box_a
    x3, y3, x4, y4 = box_b

    intersect_x1 = max(x1, x3)
    intersect_y1 = max(y1, y3)
    intersect_x2 = min(x2, x4)
    intersect_y2 = min(y2, y4)

    intersection_area = torch.clamp(intersect_x2 - intersect_x1, min=0) * torch.clamp(intersect_y2 - intersect_y1, min=0)
    box_a_area = (x2 - x1) * (y2 - y1)
    box_b_area = (x4 - x3) * (y4 - y3)

    iou_value = intersection_area / (box_a_area + box_b_area - intersection_area)
    return iou_value

def nms(boxes, scores, iou_thres):
    keep_indices = scores >= iou_thres
    filtered_boxes = boxes[keep_indices]
    filtered_scores = scores[keep_indices]
    sorted_indices = torch.argsort(filtered_scores, descending=True)
    filtered_boxes = filtered_boxes[sorted_indices]
    filtered_scores = filtered_scores[sorted_indices]
    F = []

    while len(filtered_boxes) != 0:
        current_box = filtered_boxes[0]
        F.append(current_box)
        ious = torch.tensor([iou(current_box, box) for box in filtered_boxes[1:]])
        keep_indices = ious < iou_thres
        filtered_boxes = filtered_boxes[1:][keep_indices]
        filtered_scores = filtered_scores[1:][keep_indices]

    F = torch.stack(F)
    rows_all_true = torch.all(torch.isin(boxes, F), dim=1)

    indices_rows_all_true = torch.nonzero(rows_all_true, as_tuple=True)
    indices_rows_all_true = torch.stack(indices_rows_all_true).squeeze(0)
    return indices_rows_all_true

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
        in_place=True,
        rotated=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    # nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    # print('prediction', prediction[:, 4:mi].shape)

    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)

    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = torch.split(tensor = x, split_size_or_sections = (4, nc, nm), dim = 1)

        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS

        c = x[:, 5:6] * max_wh  # classes
        scores = x[:, 4]  # scores

        boxes = x[:, :4] + c  # boxes (offset by class)
        i = nms(boxes, scores, iou_thres)  # NMS

        i = i[:max_det]  # limit detections
        output[xi] = x[i]

    return output

# Load the image using byte decoding

image_path = "exmple_img.jpeg" #TODO
image_tensor = read_decode_image(image_path)

credit_card_resized = bilinear_interpolate_vectorized(input=image_tensor, size=(640, 640))
credit_card_resized = credit_card_resized / 255

credit_card_resized = credit_card_resized
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
first_model = torch.jit.load('s3://fastbank-ml-models-archive/OCR/Weights/first_model.pt')

with torch.no_grad():
    first_model_results = first_model(credit_card_resized)

post_process = non_max_suppression(prediction=first_model_results, nc=3)
post_process = [row for row in post_process[0] if row[4] >= 0.7]

data = sorted(post_process, key=lambda x : x[-1], reverse=True)

try:
	expire_date_img, cardholder_img, cardnumber_img = data

	w1 = int(torch.ceil(expire_date_img[2]) - torch.ceil(expire_date_img[0]))
	w2 = int(torch.ceil(cardholder_img[2]) - torch.ceil(cardholder_img[0]))
	w3 = int(torch.ceil(cardnumber_img[2]) - torch.ceil(cardnumber_img[0]))

	cropped_img1 = credit_card_resized[:, :, int(torch.ceil(expire_date_img[1])):int(torch.ceil(expire_date_img[3])), int(torch.ceil(expire_date_img[0])):int(torch.ceil(expire_date_img[2]))]
	cropped_img2 = credit_card_resized[:, :, int(torch.ceil(cardholder_img[1])):int(torch.ceil(cardholder_img[3])), int(torch.ceil(cardholder_img[0])):int(torch.ceil(cardholder_img[2]))]
	cropped_img3 = credit_card_resized[:, :, int(torch.ceil(cardnumber_img[1])):int(torch.ceil(cardnumber_img[3])), int(torch.ceil(cardnumber_img[0])):int(torch.ceil(cardnumber_img[2]))]

	max_w = max(w1, w2, w3)


	img1_padding = torch.zeros((1, 3, cropped_img1.shape[2], max_w-w1))
	img2_padding = torch.zeros((1, 3, cropped_img2.shape[2], max_w-w2))
	img3_padding = torch.zeros((1, 3, cropped_img3.shape[2], max_w-w3))

	expire_date_img = torch.concatenate((cropped_img1, img1_padding), axis=3)
	cardholder_img = torch.concatenate((cropped_img2, img2_padding), axis=3)
	cardnumber_img = torch.concatenate((cropped_img3, img3_padding), axis=3)

	stacked_image = torch.concatenate((cardholder_img, expire_date_img, cardnumber_img), axis=2) # shape [3, h, w]

	stacked_image_resized = bilinear_interpolate_vectorized(stacked_image.squeeze(0), size=(320, 800))

	second_model = torch.jit.load('s3://fastbank-ml-models-archive/OCR/Weights/second_model.pt')
	stacked_image_resized = stacked_image_resized

	with torch.no_grad():
	    second_model_results = second_model(stacked_image_resized)

	post_process = non_max_suppression(prediction=second_model_results, nc=36)

	data = sorted(post_process[0], key=lambda x:x[5])
	card_holder = [row for row in data if row[5] > 9 and row[4] >= 0.65]

	card_holder_sorted = sorted(card_holder, key=lambda x:x[0])
	card_holder_sorted = [int(item[-1]) for item in card_holder_sorted]

	letter_dict = {
	    10: 'A',
	    11: 'B',
	    12: 'C',
	    13: 'D',
	    14: 'E',
	    15: 'F',
	    16: 'G',
	    17: 'H',
	    18: 'I',
	    19: 'J',
	    20: 'K',
	    21: 'L',
	    22: 'M',
	    23: 'N',
	    24: 'O',
	    25: 'P',
	    26: 'Q',
	    27: 'R',
	    28: 'S',
	    29: 'T',
	    30: 'U',
	    31: 'V',
	    32: 'W',
	    33: 'X',
	    34: 'Y',
	    35: 'Z'
	}

	for idx in card_holder_sorted:
	    print(letter_dict[idx], end=' ')
	print()

	date_card_number = [row for row in data if row[5] <= 9 and row[4] >= 0.5 and row[1] >= 20]

	date_card_number_sorted = sorted(date_card_number, key=lambda x:x[1])


	expire_date = [row for row in date_card_number_sorted if row[1] < date_card_number_sorted[0][3]-10]
	card_number = [row for row in date_card_number_sorted if row[1] > date_card_number_sorted[0][3]-10]

	card_number = sorted(card_number, key=lambda x:x[0])
	card_number = [int(item[-1]) for item in card_number if item[4] >= 0.7]
	print(*card_number, sep=' ')

	expire_date = sorted(expire_date, key=lambda x:x[0])
	expire_date = [int(item[-1]) for item in expire_date if item[4] >= 0.5]
	print(*expire_date, sep=' ')
except:
     print("The 3 fields were not found.")
