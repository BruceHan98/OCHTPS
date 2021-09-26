import sys
import cv2
import torch
from torchvision.transforms import ToTensor

from models.model_with_tcn_big import Model
from utils.hwdb2_0_chars import char_set
from utils.pred_utils import get_pred_str

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda')
model = Model(num_classes=3000, line_height=32, is_transformer=True, is_TCN=True).to(device)
model.load_state_dict(torch.load('./output/model.pth', map_location=device))


def predict_from_image(image_path):
    image = cv2.imread(image_path)
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_np.shape[0:2]
    short_edge = max(height, width)
    if short_edge > 1600:
        # 保证短边 >= input size
        scale = 1600 / short_edge
        image_np = cv2.resize(image_np, dsize=None, fx=scale, fy=scale)
    img_transform = ToTensor()

    image_tensor = img_transform(image_np).unsqueeze(0)
    # print(image_tensor.shape)
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        kernel, out_chars, sub_img_nums, line_top_lefts, line_contours = model(image_tensor, None, is_train=False)

        line_contours = line_contours[0]
        line_contours = list(map(lambda x: x * 4, line_contours))
        prediction_char = out_chars
        prediction_char = prediction_char.log_softmax(-1)
        pred_strs = get_pred_str(prediction_char, char_set)

        cv2.drawContours(image_np, line_contours, -1, (0, 0, 255), 2)
        cv2.imwrite('result.jpg', image_np)

        #         print(line_contours)
        print(pred_strs)


if __name__ == '__main__':
    image_path = sys.argv[1]
    predict_from_image(image_path)
