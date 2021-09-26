import torch
import torch.optim as optim
import torch.multiprocessing
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast  # 自动混合精度，提高计算速度，降低显存使用
from torch.utils.data import DataLoader
from torchvision import transforms

from models.model_with_tcn_big import Model
from models.loss_kernels import DICE_loss
from models.loss_ctc import ctc_loss
from dataset.data_utils_kernel_box_from_dgrl import MyDataset, AlignCollate
from dataset.hwdb2_0_chars import char_dict, char_set
from utils.logger import logger
from utils.config import Config

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

scaler = GradScaler()
torch.multiprocessing.set_sharing_strategy('file_system')  # 设置共享CPU张量的策略
device = torch.device('cuda')
config = Config('config.yml')

# dataloader
train_dataset = MyDataset(config.train_data_dir, char_dict, data_shape=1600, n=2, m=0.6, transform=transforms.ToTensor(), max_text_length=80)
eval_dataset = MyDataset(config.eval_data_dir, char_dict, data_shape=1600, n=2, m=0.6, transform=transforms.ToTensor(), max_text_length=80, is_train=False)
train_dataloader = DataLoader(dataset=train_dataset, collate_fn=AlignCollate(), batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
eval_dataloader = DataLoader(dataset=eval_dataset, collate_fn=AlignCollate(), batch_size=config.eval_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
train_steps = len(train_dataloader)
eval_steps = len(eval_dataloader)
print("Training steps: %d, evaluation steps: %d" % (train_steps, eval_steps))

model = Model(num_classes=config.num_classes, line_height=config.line_height, is_transformer=True, is_TCN=True).to(device)

criterion_kernel = DICE_loss().to(device)
criterion_char = torch.nn.CTCLoss(blank=0, zero_infinity=True).to(device)

max_CR = 0


def train():
    for epoch in range(0, config.epochs):
        optimizer = optim.Adam(model.parameters(), lr=config.lr * 0.9 ** epoch, betas=(0.5, 0.999))
        train_iter = iter(train_dataloader)
        model.train()
        # model.PAN_layer.eval()
        a_CR_correct_chars, a_AR_correct_chars, a_all_chars = 0, 0, 0
        loss_all = 0
        loss_char_all = 0
        loss_kernel_all = 0
    
        for train_step in range(train_steps):
            imgs, kernel_labels, text_polys, label_tensors, text_lengths = next(train_iter)
            # torch.cuda.empty_cache()
            imgs = imgs.to(device)
            kernel_labels = kernel_labels.to(device)
            with autocast():  # 自动混合精度
                kernels_pred, out_chars, sub_img_nums = model(imgs, text_polys, is_train=True)
                loss_kernel = criterion_kernel(kernels_pred, kernel_labels)
    
                if (train_step + 1) % config.print_pred == 0:
                    is_print = True
                else:
                    is_print = False
                loss_char, CR_correct_chars, AR_correct_chars, all_chars = ctc_loss(criterion_char, out_chars, label_tensors, text_lengths, sub_img_nums, char_set, is_print)
                a_CR_correct_chars += CR_correct_chars
                a_AR_correct_chars += AR_correct_chars
                a_all_chars += all_chars
                loss_char_all += loss_char.item()
                loss_kernel_all += loss_kernel.item()
    
                if loss_kernel.item() > 0.13:
                    loss = 0.1 * loss_kernel + loss_char
                else:
                    loss = loss_char
    
                loss_all += loss.item()
            
            optimizer.zero_grad()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            scaler.scale(loss).backward()  # 梯度放大
            scaler.step(optimizer)
            scaler.update()
            
            AR = a_AR_correct_chars / (a_all_chars + 1)
            CR = a_CR_correct_chars / (a_all_chars + 1)
    
            if (train_step + 1) % config.display_interval == 0:
                torch.cuda.empty_cache()
                logger.info("train epoch: %d, iters: %4d/%4d, loss_char_all: %.6f, loss_char: %.4f,"
                            " loss_kernel_all: %.6f, loss_kernel: %.4f, AR: %.4f, CR: %.4f, AR_all: %.4f, CR_all: %.4f"
                            % (epoch+1, train_step+1, train_steps, loss_char_all/(train_step+1), loss_char.item(),
                               loss_kernel_all/(train_step+1), loss_kernel.item(), AR_correct_chars/all_chars, CR_correct_chars/all_chars, AR, CR, ))

        logger.info('train epoch:{} loss_kernel:{:.4f} loss_char:{:.4f} AR:{:.4f} CR:{:.4f}\n'.format(
            epoch, loss_kernel_all / (train_steps + 1), loss_char_all / (train_steps + 1), a_AR_correct_chars / a_all_chars, a_CR_correct_chars / a_all_chars))
        
        evaluate(epoch)


def evaluate(epoch, is_save=True):
    eval_iter = iter(eval_dataloader)
    model.eval()
    a_CR_correct_chars, a_AR_correct_chars, a_all_chars = 0, 0, 0
    loss_all = 0
    loss_kernel_all = 0
    loss_char_all = 0
    edit_distance = []

    with torch.no_grad():
        for eval_step in range(eval_steps):
            imgs, kernel_labels, text_polys, label_tensors, text_lengths = next(eval_iter)
            # torch.cuda.empty_cache()
            imgs = imgs.to(device)
            kernel_labels = kernel_labels.to(device)
            kernels_pred, out_chars, sub_img_nums = model(imgs, text_polys, is_train=False)

            loss_kernel = criterion_kernel(kernels_pred, kernel_labels)

            if (eval_step + 1) % config.display_interval == 0:
                torch.cuda.empty_cache()
                is_print = True
            else:
                is_print = False
            loss_char, CR_correct_chars, AR_correct_chars, all_chars = ctc_loss(criterion_char, out_chars, label_tensors, text_lengths, sub_img_nums, char_set, is_print)

            a_CR_correct_chars += CR_correct_chars
            a_AR_correct_chars += AR_correct_chars
            a_all_chars += all_chars
            loss_all += 0.0 * loss_kernel.item() + loss_char.item()
            loss_char_all += loss_char.item()
            loss_kernel_all += loss_kernel.item()

            AR = a_AR_correct_chars / (a_all_chars + 1)
            CR = a_CR_correct_chars / (a_all_chars + 1)

            if (eval_step + 1) % config.display_interval == 0:
                logger.info("eval epoch: %d, iters: %4d/%4d, loss_char_all: %.6f, loss_char: %.4f,"
                            " loss_kernel_all: %.6f, loss_kernel: %.4f, AR: %.4f, CR: %.4f, AR_all: %.4f, CR_all: %.4f"
                            % (epoch+1, eval_step+1, eval_steps, loss_char_all/(eval_step+1), loss_char.item(),
                               loss_kernel_all/(eval_step+1), loss_kernel.item(), AR_correct_chars/all_chars, CR_correct_chars/all_chars, AR, CR, ))

    AR = a_AR_correct_chars / (a_all_chars + 1)
    CR = a_CR_correct_chars / (a_all_chars + 1)

    global max_CR
    if is_save:
        if CR > max_CR:
            max_CR = CR
            torch.save(model.state_dict(), './output/model_epoch_{}_loss_char_all_{:.4f}_loss_kernel_all_{:.4f}_AR_{:.6f}_CR_{:.6f}.pth'.format(
                epoch, loss_char_all / (eval_steps + 1), loss_kernel_all / (eval_steps + 1), AR, CR))
    logger.info('eval epoch:{} loss_kernel:{:.4f} loss_char:{:.4f} AR:{:.4f} CR:{:.4f}\n'.format(
        epoch, loss_kernel_all / (eval_steps + 1), loss_char_all / (eval_steps + 1), a_AR_correct_chars / a_all_chars, a_CR_correct_chars / a_all_chars))


if __name__ == '__main__':
    # pre_dict = torch.load(
    #     './output/with_tcn_big_icdar/model_new1_epoch_13_loss_char_all_0.3923_loss_kernel_all_0.1185_AR_0.911840_CR_0.920156.pth')

    # # pre_dict.pop('DenseNet_layer.classifier.weight')
    # # pre_dict.pop('DenseNet_layer.classifier.bias')
    # model_dict = model.state_dict()
    # pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}

    # model_dict.update(pre_dict)
    # model.load_state_dict(model_dict)

    # model.load_state_dict(torch.load(
    #     r'./output/with_tcn_big_hwdb_all_t'
    #     r'/model_c_epoch_50_loss_char_all_0.0642_loss_kernel_all_0.1226_AR_0.987677_CR_0.990463.pth'))

    # eval(model, eval_data, criterion_kernel, criterion_char, 0,is_save=False)

    train()
