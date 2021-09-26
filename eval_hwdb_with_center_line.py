import os
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
from torchvision import transforms

from models.model_with_tcn_big import Model
from utils.hwdb2_0_chars import char_set
from utils.get_dgrl_data import get_pred_data
from utils.pred_utils import get_ar_cr, get_pred_str, polygon_IOU, normal_leven

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def predict(model, pred_iter, file_path, show=False):
    with torch.no_grad():
        img_np, img_tensor, boxes, page_label = next(pred_iter)
        label_np = np.ones_like(img_np, dtype=np.uint8) * 255
        boxes = boxes[0]
        imgs = img_tensor.to(device)
        print(imgs.shape)

        kernel, out_chars, sub_img_nums, line_top_lefts, line_contours = model(imgs, None, is_train=False)
        line_contours = line_contours[0]

        prediction_char = out_chars
        prediction_char = prediction_char.log_softmax(-1)
        pred_strs = get_pred_str(prediction_char, char_set)

        pred_str_group = ['' for _ in range(len(page_label))]
        not_in_char = ''
        TP = 0
        FP = 0
        FN = 0

        for pred_i in range(len(pred_strs)):
            pred_str_poly = line_contours[pred_i]
            pred_str_poly = np.squeeze(pred_str_poly, 1)
            find_flag = 0
            for label_i in range(len(boxes)):
                label_box = boxes[label_i] / 4
                pred_iou = polygon_IOU(pred_str_poly, label_box)
                if pred_iou > 0.9:
                    pred_str_group[label_i] += pred_strs[pred_i]
                    find_flag = 1
                    break
            if find_flag == 0:
                FP += 1
                not_in_char += pred_strs[pred_i]

        for i in range(len(pred_str_group)):
            if len(pred_str_group[i]) / len(page_label[i]):
                TP += 1
            else:
                FN += 1

        pred_strs_s = ''.join(pred_str_group) + not_in_char
        # CR, AR, All = get_ar_cr(pred_strs_s, ''.join(page_label))
        CR, AR, All = 0, 0, 0
        char_c = len(''.join(page_label))
        edit_d = normal_leven(pred_strs_s, ''.join(page_label))
        for sub_p, sub_l in zip(pred_str_group, page_label):
            sub_cr, sub_ar, sub_all = get_ar_cr(sub_p, sub_l)
            CR += sub_cr
            AR += sub_ar
            All += sub_all
        AR -= len(not_in_char)

        if show:
            line_contours = list(map(lambda x: x*4, line_contours))
            for box in line_contours:
                box = np.int_(box)
                cv2.polylines(img_np, [box], True, 128, 1)
            char_size = int(label_np.shape[1] / len(page_label) / 5)
            if isinstance(label_np, np.ndarray):
                label_np = Image.fromarray(cv2.cvtColor(label_np, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(label_np)

            fontText = ImageFont.truetype('simfang.ttf', char_size, encoding="utf-8")
            draw.text((0, 0), 'CR:{:.6f} AR:{:.6f}'.format(CR / All, AR / All), (0, 0, 0), font=fontText)
            for i in range(len(pred_str_group)):
                left = boxes[i][0][0]
                top = boxes[i][0][1]
                draw.text((left, top), 'label:' + page_label[i], (0, 0, 0), font=fontText)
                draw.text((left, top + char_size), 'preds:' + pred_str_group[i], (0, 0, 0), font=fontText)

            label_np = cv2.cvtColor(np.asarray(label_np), cv2.COLOR_RGB2BGR)

            show_np = np.hstack([img_np, label_np])
            show_np = cv2.resize(show_np, None, fx=0.7, fy=0.7)

            print("labels:", page_label)
            print("predicts:", pred_str_group)
            # cv2.drawContours(img_np, line_contours, -1, (0, 0, 255), 1)
            (path, filename) = os.path.split(file_path)
            save_name = filename.split('.')[0] + '_cl.jpg'
            save_dir = './output/result'
            save_path = os.path.join(save_dir, save_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(save_path, show_np)

    return CR, AR, All, edit_d, char_c, TP, FP, FN


if __name__ == '__main__':
    device = torch.device('cuda')
    img_transform = transforms.ToTensor()
    model = Model(num_classes=3000, line_height=32, is_transformer=True, is_TCN=True).to(device)
    model.load_state_dict(torch.load('./output/model.pth', map_location=device))
    model.eval()

    test_file_dir = '../dgrl_test'
    file_paths = []
    for file_path in os.listdir(test_file_dir):
        if file_path.endswith('dgrl'):
            file_paths.append(os.path.join(test_file_dir, file_path))

    CR_all, AR_all, All_all = 0, 0, 0
    EDIT_DISTANCE_ALL, CHAR_COUNT_ALL = 0, 0
    TP_all, FP_all, FN_all = 0, 0, 0
    pbar = tqdm(total=len(file_paths))
    pred_iter = iter(get_pred_data(file_paths, 1600))

    for i in range(len(file_paths)):
        cr, ar, all, edit_d, char_c, TP, FP, FN = predict(model, pred_iter, file_paths[i], False)
        CR_all += cr
        AR_all += ar
        All_all += all
        EDIT_DISTANCE_ALL += edit_d
        CHAR_COUNT_ALL += char_c
        TP_all += TP
        FP_all += FP
        FN_all += FN

        Precision = TP_all / (TP_all + FP_all)
        Recall = TP_all / (TP_all + FN_all)
        F1 = 2 / (1 / Precision + 1 / Recall)

        pbar.display('CR:{:.6f} AR:{:.6f} edit_d:{:.6f} Precision:{:.6f} Recall:{:.6f} F1:{:.6f}\n'.format(
            CR_all / All_all, AR_all / All_all, (CHAR_COUNT_ALL - EDIT_DISTANCE_ALL) / CHAR_COUNT_ALL,
            Precision, Recall, F1))
        pbar.update(1)
