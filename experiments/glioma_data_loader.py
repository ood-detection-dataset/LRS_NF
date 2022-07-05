from matplotlib import pyplot
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
from collections import Counter
from collections import defaultdict
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.augmentations.transforms import HorizontalFlip, Normalize, RandomBrightness  # , GaussianBlur
from albumentations.pytorch import ToTensor
from albumentations import Compose
import os
import time
import cv2
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections import defaultdict
from collections import Counter
# from sklearn.metrics import roc_curve
# from sklearn.metrics import roc_auc_score

# device = torch.device('cuda') if torch.cuda.is_available(
# ) else torch.device('cpu')

# print('\n Using : ', device)

# device = torch.device(
#     'cuda') if torch.cuda.is_available() else torch.device('cpu')


class My_Transform(object):
    '''
    My transform: 
    '''

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label, p_id, path = sample['data'], sample['target'], sample['p_id'], sample['path']
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])

        augmented = aug(image=image)
        image_medium = augmented['image']
        # print('augment ',image_medium.shape)

        return {'data': image_medium, 'target': label, 'p_id': p_id, 'path': path}


class My_Normalize(object):
    '''
    My Normalize (TRail)
    '''

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label, p_id, path = sample['data'], sample['target'], sample['p_id'], sample['path']
        normal_aug = A.Normalize()
        augmented_img = normal_aug(image=image)
        image = augmented_img['image']
        image = image.transpose((2, 0, 1))
        # image = image/255.0
        # print('normal ',image.shape)
        return {'data': torch.from_numpy(image), 'target': torch.FloatTensor([label]), 'p_id': p_id, 'path': path}


class WSIDataset:
    """Generate dataset."""

    def __init__(self, root, transform=None, use_shuffle=True):

        stage_csvs = ['_Train_All.csv', '_Valid_All.csv',
                      '_Test_All.csv', '_Test_MUH.csv', 
                      '_Cov_Shift_TCGA.csv', '_Cov_Shift_GAN.csv', 
                      '_Cov_Shift_MUH.csv', '_Sem_Shift_BreakHis.csv']
        self.train_data = os.path.join(root, stage_csvs[0])
        self.valid_data = os.path.join(root, stage_csvs[1])
        self.test_data = os.path.join(root, stage_csvs[2])
        self.test_data_2 = os.path.join(root, stage_csvs[3])
        self.ood_TCGA_data = os.path.join(root, stage_csvs[4])
        self.ood_GAN_data = os.path.join(root, stage_csvs[5])
        self.ood_MUH_data = os.path.join(root, stage_csvs[6])
        self.ood_BeakHis_data = os.path.join(root, stage_csvs[7])
        
        self.transform_train = transforms.Compose(
            [My_Transform(), My_Normalize()])
        self.transform_valid = transforms.Compose([My_Normalize()])

    def Obtain_dataset(self, stage):
        if stage == 'Train_GAN_Normalized':
            self.dataset = StageDataset(
                [self.train_data], self.transform_train)
        elif stage == 'Valid_GAN_Normalized':
            self.dataset = StageDataset(
                [self.valid_data], self.transform_valid)
        elif stage == 'Test_GAN_Normalized':
            self.dataset = StageDataset([self.test_data], self.transform_valid)
        elif stage == 'Test_MUH_GAN_Normalized':
            self.dataset = StageDataset(
                [self.test_data_2], self.transform_valid)
        elif stage == 'TCGA_Unnormalized':
            self.dataset = StageDataset(
                [self.ood_TCGA_data], self.transform_valid)
        elif stage == 'GAN_Generated':
            self.dataset = StageDataset(
                [self.ood_GAN_data], self.transform_valid)
        elif stage == 'MUH_Unnormalized':
            self.dataset = StageDataset(
                [self.ood_MUH_data], self.transform_valid)
        elif stage == 'BreakHis':
            self.dataset = StageDataset(
                [self.ood_BeakHis_data], self.transform_valid)
        else:
            raise ValueError("'{}' is not a valid dataset name".format(stage))
        return self.dataset

    def Obtain_loader(self, stage, batch_size, n_jobs=4):
        self.Obtain_dataset(stage)
        self.loader = DataLoader(self.dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=n_jobs,
                                 drop_last=True)
        return self.loader


class StageDataset(Dataset):
    def __init__(self, ds, transform):
        self.map_dict = {'MU': 0, 'WT': 1}
        if len(ds) > 1:
            ds_list = []
            for i in ds:
                ds_list.append(pd.read_csv(i))
            self.ds = pd.concat(ds_list)
        else:
            self.ds = pd.read_csv(ds[0])
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        '''image	label	patent'''
        img_path = self.ds.iloc[idx, 0]
        label = self.map_dict[self.ds.iloc[idx, 1]]
        patent = self.ds.iloc[idx, 2]
        # print(img_path, patent, label)
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        # image = np.repeat(image[np.newaxis,:,:,:], 10, axis=0)
        sample = {'data': image, 'target': label,
                  'p_id': patent, 'path':img_path}

        if self.transform:
            sample = self.transform(sample)

        return sample


def test_model(t_model, testloader, device, weight_trained=None):
    since = time.time()
    t_model.to(device)
    if weight_trained:
        t_model.load_state_dict(torch.load(weight_trained))
    t_model.eval()
    p_classification = defaultdict(list)
    p_classification_prob = defaultdict(list)
    p_label = {}
    TP = TN = FP = FN = 0
    slide_label = []
    slide_prob = []
    for data in tqdm(testloader):
        inputs = data['data'].to(device=device, dtype=torch.float)
        labels = data['target'].to(device=device, dtype=torch.int64)
        p_id = data['p_id']
        output = t_model(inputs)
        for i in range(output.size()[0]):
            prediction = torch.argmax(output[i])
            curr_p_id = p_id[i]
            p_label[curr_p_id] = labels[i].cpu().item()
            p_classification[curr_p_id].append(prediction.cpu().item())
            p_classification_prob[curr_p_id].append(
                [output[i][0].cpu().item(), output[i][1].cpu().item()])
            if prediction.cpu().item() == labels[i].cpu().item() == 0:
                TN += 1
            elif prediction.cpu().item() == labels[i].cpu().item() == 1:
                TP += 1
            elif prediction.cpu().item() == 0 and labels[i].cpu().item() == 1:
                FN += 1
            else:
                FP += 1
            slide_label.append(labels[i].cpu().item())
            slide_prob.append(output[i][1].cpu().item())

    Specificity = TN / (TN + FP)  # Specificity = TN / (TN + FP)
    Sensitivity = TP / (FN + TP)  # Sensitivity = TP / (FN + TP)
    Precision = TP / (TP + FP)  # Precision = TP / (TP + FP)
    F1_Score = 2*(Precision * Sensitivity) / (Precision + Sensitivity)
    AUC = roc_auc_score(slide_label, slide_prob)
    return p_label, p_classification, p_classification_prob


def patent_class(p_label, p_classification, p_classification_prob):
    p_classification_check = defaultdict(list)

    TP = TN = FP = FN = 0
    pat_label = []
    pat_pred = []
    pat_prob = []
    for k in p_classification:
        # classification of slide for one patient
        p_list = p_classification[k]
        x = Counter(p_list)
        s = sum(p_list)
        # majority vote for patient level classification
        p_pred = x.most_common(1)[0][0]
        # keep the prob for the positive label
        p_prob = p_classification_prob[k]
        p_pos = list(map(lambda x: x[1], p_prob))
        p_pos = sum(p_pos)/len(p_pos)
        pat_prob.append(p_pos)
        p_gt = p_label[k]
        if p_pred == p_gt == 0:
            TN += 1
        elif p_pred == p_gt == 1:
            TP += 1
        elif p_pred == 0 and p_gt == 1:
            FN += 1
        else:
            FP += 1
        pat_label.append(p_gt)
        pat_pred.append(p_pred)
    Specificity = TN / (TN + FP)  # Specificity = TN / (TN + FP)
    Sensitivity = TP / (FN + TP)  # Sensitivity = TP / (FN + TP)
    Precision = TP / (TP + FP)  # Precision = TP / (TP + FP)
    F1_Score = 2*(Precision * Sensitivity) / (Precision + Sensitivity)

    AUC = roc_auc_score(pat_label, pat_prob)
    ns_fpr, ns_tpr, _ = roc_curve(pat_label, pat_prob)
    print(TP, TN, FN, FP)
    print('Statistic: ')
    print('Specificity: ', Specificity)
    print('Sensitivity: ', Sensitivity)
    print('Precision: ', Precision)
    print('Acc: ', (TP+TN)/(TP+TN+FN+FP))
    print('F1-Score: ', F1_Score)
    print('AUC ', AUC)


def change_csv_pth(csv_pth, new_pth, top_layers=4):
    '''
    csv_pth: path where csv file are store
    new_pth: new path where csv files are stored
    top_layers: number of parent folders
    '''
    df = pd.read_csv(csv_pth)
    df_pth = csv_pth.split('/')
    new_pth = df_pth[:-1]
    df_name = df_pth[-1].replace('.csv', '')
    # print(new_pth)
    # print(df_name)
    # print(df.head())
    for i in range(len(df)):
        img_pth = df.iloc[i, 0]
        img_pth = img_pth.split('/')
        prev = img_pth.index(df_name)
        img_pth = new_pth + img_pth[prev:]
        img_pth = '/'.join(img_pth)
        # print(img_pth)
        # print(os.path.exists(img_pth))
        df.iloc[i, 0] = img_pth

    df.to_csv(csv_pth, index=False)


# root = "/nobackup/datasets/gdrive/UoW_MQ_Glioma/example"

# for csv_file in ['/_Train_All.csv', '/_Valid_All.csv', '/_Test_All.csv', '/_Test_MUH.csv']:
#     old_csv_pth = root + csv_file
#     new_pth = root
#     change_csv_pth(old_csv_pth, new_pth)

# define root path, need to have '_Train_All.csv','_Valid_All.csv','_Test_All.csv','_Test_MUH.csv' under the root pathHi
# batch_size = 16

# wsi_dataset = WSIDataset(root)
# testloader = wsi_dataset.Obtain_loader('Test', batch_size)

# # Testing
# trained_model_pth = root + '/ResNet50_Classifier.pt'
# model = torch.load(trained_model_pth)
# p_label, p_classification, p_classification_prob = test_model(
#     model, testloader)
# patent_class(p_label, p_classification, p_classification_prob)
