import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from model import Net
from data import HandWritingDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 32
train_dataset = HandWritingDataset(split='train')
test_dataset = HandWritingDataset(split='test')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = Net().to(device)

l1 = nn.L1Loss().to(device)
ce = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# for epoch in range(10):
#     epoch_loss = 0
#     model.train()
#     for _, data_dict in enumerate(tqdm(train_loader)):
#         images = data_dict["img"].to(device)
#         bbox = data_dict["boxes"].to(device) / images.shape[2]
#         labels = data_dict["labels"].squeeze(-1).to(device)
#         # print(data_dict["img"].shape)
#         # print(data_dict["boxes"].shape)
#         # print(data_dict["labels"].shape)

#         optimizer.zero_grad()
#         pred_bbox, pred_label = model(images)
#         loss = l1(pred_bbox, bbox) + ce(pred_label, labels)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     epoch_loss = epoch_loss / len(train_loader)
#     print("Train Loss: {}".format(epoch_loss))

#     epoch_loss = 0
#     model.eval()
#     for _, data_dict in enumerate(tqdm(test_loader)):
#         images = data_dict["img"].to(device)
#         bbox = data_dict["boxes"].to(device) / images.shape[2]
#         labels = data_dict["labels"].squeeze(-1).to(device)

#         pred_bbox, pred_label = model(images)
#         loss = l1(pred_bbox, bbox) + ce(pred_label, labels)
#         epoch_loss += loss.item()
#     epoch_loss = epoch_loss / len(train_loader)
#     print("Test Loss: {}".format(epoch_loss))
# torch.save(model.state_dict(), 'model.t7')

total_count = 0
total_correct = 0
model.load_state_dict(torch.load('model.t7'))
for _, data_dict in enumerate(tqdm(test_loader)):
    images = data_dict["img"].to(device)
    bbox = data_dict["boxes"].to(device) / images.shape[2]
    labels = data_dict["labels"].squeeze(-1).to(device)

    pred_bbox, pred_label = model(images)
    # pred_label = pred_label.argmax(-1).detach().cpu().numpy()
    # labels = labels.detach().cpu().numpy()

    # correct = (labels == pred_label).sum()
    # total_count += images.shape[0]
    # total_correct += correct

    img = images[0].permute(1, 2, 0).detach().cpu().numpy()
    box = pred_bbox[0].detach().cpu().numpy() * images.shape[2]
    label = pred_label[0].argmax(-1).detach().cpu().numpy()

    fig,ax = plt.subplots(1)
    ax.imshow(img)
    rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(10, 10, "text on plot")
    plt.show()
    print('Correct Label: {}, Predicted Label: {}'.format(labels[0].detach().cpu().numpy(), label))
    exit()
# print('Accuracy: {}'.format(total_correct/total_count))
