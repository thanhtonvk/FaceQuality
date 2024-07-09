import torch
import torchvision.transforms as T


class Config:
    # dataset
    img_list = 'annotations/quality_pseudo_labels.txt'
    finetuning_model = 'generate_pseudo_labels/extract_embedding/model/MobileFaceNet_MS1M.pth'
    # save settings
    checkpoints = "checkpoints/MS1M_Quality_Regression/S1"
    checkpoints_name = "MFN"
    # data preprocess
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        T.Resize((112,112))
        ])
    # training settings
    device = 'cuda:0' if torch.cuda.is_available() else 'mps'
    seed = 0
    multi_GPUs = [0]
    backbone = 'MFN'  # [MFN, R_50]
    pin_memory = True
    num_workers = 12
    batch_size = 128
    epoch = 20
    lr = 0.0001
    stepLR = [5, 10]
    weight_decay = 0.0005
    display = 100
    saveModel_epoch = 1
    loss = 'SmoothL1'   # ['L1', 'L2', 'SmoothL1']

config = Config()
