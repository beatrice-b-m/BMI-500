import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchvision.models.feature_extraction import create_feature_extractor
from torchinfo import summary
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class ResNet(nn.Module):
    def __init__(self, num_classes: int = 1):
        super(ResNet, self).__init__()
        # load the pretrained core
        self.base_model = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        self.base_model_out = create_feature_extractor(
            self.base_model, 
            {'layer4': 'feat1'}
        )
        
        # remove the last two layers
        self.base_model_out = nn.Sequential(*list(self.base_model.children())[:-2])
        
        # define classifier
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=3),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, xi):
        x = self.base_model_out(xi)
        xo = self.classifier(x)
        return xo


def load_dataset(data_path: str, transform_list, batch_size: int, shuffle: bool = True):
    """
    function to load dataset from root image folder
    """
    dataset = datasets.ImageFolder(
        data_path, 
        transform=transform_list
    )

    # send dataset to dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=4,
        prefetch_factor=2
    )
    return dataloader


def train_model(epochs, train_loader, val_loader,
                test_loader, model, criterion,
                optimizer, metric_collection):
    for t in range(epochs):
        print(f"epoch {t}\n-------------------------------")
        train_epoch(
            train_loader,
            val_loader,
            model,
            criterion,
            optimizer,
            metric_collection
        )
    test_epoch(
        test_loader,
        model,
        criterion,
        metric_collection
    )
    print("training complete !")

    
def train_epoch(train_loader, val_loader,
                model, criterion, optimizer,
                metric_collection):
    # run each epoch
    train_size = len(train_loader)
    model.train()
    with tqdm(train_loader, unit="batch", total=train_size) as t_epoch:
        for batch, (x, y) in enumerate(t_epoch):  # iterate over batches
            x, y = x.to(device), y.to(device)
            y = y.float()
            # compute prediction error
            pred = model(x)
            pred = pred.squeeze()
            loss = criterion(pred, y)
            # backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # update metric collection
            metric_collection.update(pred, y)

            if batch % (train_size//10) == 0:
                metrics = metric_collection.compute()
                t_epoch.set_postfix(
                    loss=loss.item(),
                    acc=metrics['acc'].item(),
                    auc=metrics['auc'].item(),
                    prec=metrics['prec'].item(),
                    rec=metrics['rec'].item(),
                    f1=metrics['f1'].item()
                )
        metric_collection.reset()
    # validation
    # validation doesnt requires gradient
    model.eval()
    with torch.no_grad():
        val_size = len(val_loader)
        val_loss = 0
        with tqdm(val_loader, unit="batch", total=val_size) as v_epoch:
            for x, y in v_epoch:
                x, y = x.to(device), y.to(device)
                y = y.float()
                pred = model(x)
                pred = pred.squeeze()
                val_loss += criterion(pred, y).item()
                metric_collection.update(pred, y)
        val_loss /= val_size
        metrics = metric_collection.compute()
        v_epoch.set_postfix(
            loss=loss.item(),
            acc=metrics['acc'].item(),
            auc=metrics['auc'].item(),
            prec=metrics['prec'].item(),
            rec=metrics['rec'].item(),
            f1=metrics['f1'].item()
        )
        metric_collection.reset()


def test_epoch(test_loader, model, criterion,
               metric_collection):
    model.eval()
    with torch.no_grad():
        test_size = len(test_loader)
        test_loss = 0
        with tqdm(test_loader, unit="batch", total=test_size) as te_epoch:
            for x, y in te_epoch:
                x, y = x.to(device), y.to(device)
                y = y.float()
                pred = model(x)
                pred = pred.squeeze()
                test_loss += criterion(pred, y).item()
                metric_collection.update(pred, y)
        test_loss /= test_size
        metrics = metric_collection.compute()
        te_epoch.set_postfix(
            loss=loss.item(),
            acc=metrics['acc'].item(),
            auc=metrics['auc'].item(),
            prec=metrics['prec'].item(),
            rec=metrics['rec'].item(),
            f1=metrics['f1'].item()
        )
        metric_collection.reset()


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"using {device} device...")

# image preprocessing transformations
transform_list = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

batch_size = 24
train_dir = '/data/beatrice/unet/encoder_images/train'
val_dir = '/data/beatrice/unet/encoder_images/val'
test_dir = '/data/beatrice/unet/encoder_images/test'
train_loader = load_dataset(train_dir, transform_list, batch_size)
val_loader = load_dataset(val_dir, transform_list, batch_size)
test_loader = load_dataset(test_dir, transform_list, batch_size)

# define model
model = ResNet().to(device)
print(summary(model, input_size=(8, 3, 1536, 1248)))

# define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# define metric collection
metric_collection = MetricCollection({
    'acc': BinaryAccuracy(),
    'auc': BinaryAUROC(),
    'prec': BinaryPrecision(),
    'rec': BinaryRecall(),
    'f1': BinaryF1Score()
})

metric_collection.to(device)

epochs = 30
train_model(epochs, train_loader, val_loader, test_loader, model, criterion, optimizer, metric_collection)
