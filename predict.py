import sys
import numpy as np
import cv2
import torch

from models.model_with_tcn_big import Model
from utils.hwdb2_0_chars import char_set
from utils.pred_utils import get_pred_str

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


device = torch.device('cpu')
model = Model(num_classes=3000, line_height=32, is_transformer=True, is_TCN=True).to(device)
model.load_state_dict(torch.load('./output/model.pth', map_location=device))


def predict_from_image(image_path):
    image_np = cv2.imread(image_path)
    height, width = image_np.shape[0:2]
    image_np = cv2.resize(image_np, (int(1600 / height * width), 1600))
    image_tensor = torch.from_numpy(image_np).float().unsqueeze(0).transpose(1, 3)
    print(image_tensor.shape)
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        kernel, out_chars, sub_img_nums, line_top_lefts, line_contours = model(image_tensor, None, is_train=False)

        line_contours = line_contours[0]
        prediction_char = out_chars
        prediction_char = prediction_char.log_softmax(-1)
        pred_strs = get_pred_str(prediction_char, char_set)

        print(line_contours)
        print(pred_strs)


if __name__ == '__main__':
    image_path = sys.argv[1]
    predict_from_image(image_path)
