import time

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 2))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    # print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
        # if (batch_idx + 1) % 50 == 0:
        #     print('Train Epoch: {}\tLoss: {:.6f}'.format(
        #         epoch, loss.item()))
    ave_loss = sum_loss / len(train_loader)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))
    return ave_loss

def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)/5
    # print(total_num, len(test_loader))
    with torch.no_grad():
        all_preds=[]
        all_targets=[]
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
            all_preds+=pred.tolist()
            all_targets+=target.tolist()
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset)/5, 100 * acc))
        cm=confusion_matrix(all_targets, all_preds)

        f1 = f1_score(all_targets, all_preds, average='weighted')

        fpr, tpr, thresholds = roc_curve(all_targets, all_preds)
        roc_auc = auc(fpr, tpr)

        return avgloss, cm, f1, fpr, tpr, roc_auc

modellr = 1e-4
BATCH_SIZE = 30
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ('non_plastic', 'plastic')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30)


])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset_train = datasets.ImageFolder('dataset/CSC249 Dataset/new_train', transform)

print(dataset_train.class_to_idx)
# dataset_test = datasets.ImageFolder('dataset/CSC249 Dataset/val', transform_test)
#
# print(dataset_test.class_to_idx)


indices = np.arange(len(dataset_train))
# Define the K-fold cross-validation splitter
kf = KFold(n_splits=5, shuffle=True)

train_loss_saver=[]
val_loss_saver=[]
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
    t1=time.time()
    fold_train_loss_saver=[]
    fold_val_loss_saver = []
    print(f'Fold {fold_idx + 1}')
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler=val_sampler)
    criterion = nn.CrossEntropyLoss()
    model = torchvision.models.resnet50(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = True
    # input_feature = model.fc.in_features
    # model.fc = nn.Linear(input_feature, 2)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=modellr, weight_decay=1e-4)
    # cm=[]
    # f1=0
    # fpr
    for epoch in range(1, EPOCHS + 1):
        adjust_learning_rate(optimizer, epoch)
        train_loss=train(model, DEVICE, train_loader, optimizer, epoch)
        fold_train_loss_saver.append(train_loss)
        val_loss,cm,f1, fpr, tpr, roc_auc=val(model, DEVICE, val_loader)
        fold_val_loss_saver.append(val_loss)
    train_loss_saver.append(fold_train_loss_saver)
    val_loss_saver.append(fold_val_loss_saver)
    # confusion mtrix
    sns.set()
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion matrix')
    plt.show()
    # f1
    print('F1 score: {:.2f}'.format(f1))
    # roc
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()
    t2=time.time()
    print(t2-t1)
    torch.save(model, 'model'+str(fold_idx)+'.pth')
    break



avg_loss_train=[]
avg_loss_val=[]
x = range(1,EPOCHS+1,1)
for i in range(0,EPOCHS):
    s1=0
    s2=0
    for j in range(0,1):
        s1=s1+train_loss_saver[j][i]
        s2=s2+val_loss_saver[j][i]
    avg_loss_train.append(s1/1)
    avg_loss_val.append(s2/1)
print(avg_loss_train)
print(avg_loss_val)

plt.plot(x,avg_loss_train)
plt.title('5-fold train loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.xticks(x)
plt.show()

plt.plot(x,avg_loss_val)
plt.title('5-fold val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.xticks(x)
plt.show()




