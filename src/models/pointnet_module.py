#!/usr/bin/env python
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MeanMetric, MaxMetric
from lightning import LightningModule

from src.utils import pylogger
log = pylogger.get_pylogger(__name__)


class TNet(nn.Module):
    """
    T-Net for (input / feature) transformation in the PointNet paper.
    Structure: mlp --> maxpool --> mlp.
    TNet is data-dependent , so it's size is (batch_size, dim, dim).
    TNet provides a new viewpoint of a pcd of an object.
    """

    def __init__(self, k: int):
        """
        @param k: input dimension.
        """
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(
            nn.Conv1d(k, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        # max(x, 2) + view(-1, 1024)
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(256, k*k)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.max(x, 2, keepdim=True)[0]  # (batch_size, 1024, num_pts)
        x = x.view(-1, 1024)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)  # (B, k*k)
        x = x.view(-1, self.k, self.k)  # (B, k, k)

        identity = torch.eye(self.k).view(1, self.k, self.k).repeat(batch_size, 1, 1)
        if x.is_cuda:
            identity = identity.cuda()
        return x + identity


class FeatureNet(nn.Module):
    """
    Extract point embeddings and global features.
    Structure: input TNet --> optional feature TNet --> MLP --> max pool
    """
    def __init__(self, in_channels=3, classification=True, feature_transform=False):
        """
        @param in_channels: input data channels. Default is 3 for (x, y, z) coordinates.
        @param classification: True - for classification. Extract global feature only.
                               False - for segmentation. Cat([point_embedding, global_feature]
        @param feature_transform:
        """
        super().__init__()
        self.classification = classification
        self.feature_transform = feature_transform
        self.in_channels = in_channels
        if in_channels <= 3:
            self.tnet_in = TNet(in_channels)
        else:
            # (x, y, z) + feature (like nx, ny, nz or intensity, R, G, B, ...)
            self.tnet_in = TNet(3)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        if feature_transform:
            self.tnet64 = TNet(64)
        self.mlp = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, in_channels, num_pts)
        if self.in_channels > 3:
            x, feature = x.split(3, dim=1)
        trans_in = self.tnet_in(x)  # (batch_size, in_channels, in_channels)

        x = torch.transpose(x, 1, 2)  # (batch_size, num_pts, in_channels)
        x = torch.bmm(x, trans_in)  # (batch_size, num_pts, in_channels)
        x = torch.transpose(x, 1, 2)  # (batch_size, in_channels, num_pts)
        if self.in_channels > 3:
            x = torch.cat([x, feature], dim=1)
        x = self.conv1(x)  # (batch_size, 64, num_pts)
        point_feat = x

        if self.feature_transform:
            trans_feat = self.tnet64(x)
            x = torch.transpose(x, 1, 2)
            x = torch.bmm(x, trans_feat)  # (batch_size, num_pts, 64)
            x = torch.transpose(x, 1, 2)  # (batch_size, 64, num_pts)
        else:
            trans_feat = None

        x = self.mlp(x)  # (batch_size, 1024, num_pts)
        x = torch.max(x, 2, keepdim=True)[0]  # (batch_size, 1024, 1)
        global_feat = x.view(-1, 1024)  # (batch_size, 1024)

        if self.classification:
            return global_feat, trans_in, trans_feat
        # segmentation
        return torch.cat([point_feat, global_feat], dim=1), trans_in, trans_feat


def regularize_feat_transform(feat_trans, reg_scale=0.001):
    """
    Regularization loss over the feature transformation matrix
    @param feat_trans: (batch_size, 64, 64)
    @param reg_scale: regularization
    @return:
    """
    k = feat_trans.shape[-1]
    I = torch.eye(k)
    if feat_trans.is_cuda:
        I = I.cuda()
    tmp = (torch.bmm(feat_trans, torch.transpose(feat_trans, 1, 2)) - I)
    reg_loss = torch.mean(torch.norm(tmp, dim=(1, 2)))
    return reg_loss * reg_scale


class PointNetCls(nn.Module):
    """
    PointNet for classification
    """
    def __init__(self, num_classes, in_channels=3, feature_transform=False):
        super().__init__()
        self.num_classes = num_classes

        self.feat_net = FeatureNet(in_channels=in_channels, classification=True, feature_transform=feature_transform)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256, bias=False),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x, trans_in, trans_feat = self.feat_net(x)
        logits = self.mlp(x)  # (batch_size, num_classes)
        return logits, trans_in, trans_feat


class PointNetPartSeg(nn.Module):
    """
    PointNet for Part Segmentation
    """
    def __init__(self, num_classes, in_channels=3, feature_transform=False):
        super().__init__()
        self.num_classes = num_classes

        self.feat_net = FeatureNet(in_channels=in_channels, classification=True, feature_transform=feature_transform)

        self.mlp = nn.Sequential(
            nn.Linear(1024 + 64, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.out_conv = nn.Linear(128, )

    def forward(self, x):
        x, trans_in, trans_feat = self.feat_net(x)
        # x: (batch_size, 1024, num_pts); trans_feat: (batch_size, 64, num_pts)
        x = torch.cat([trans_feat, x], dim=1)
        x = self.mlp(x)
        logits = self.out_conv(x)  # (batch_size, num_classes)
        return logits, trans_feat


class PointNetClsModule(LightningModule):
    """
    PointNet Classifier - Lightning Module
    """
    def __init__(self,
                 net: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 compile: bool = False
                 ):
        """
        Init function of the LightningModule
        @param net: The model to train.
        @param optimizer: The optimizer to use for training.
        @param scheduler: The learning rate scheduler to use for training.
        @param compile: True - compile model for faster training with pytorch 2.0.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.criterion = nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=net.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=net.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=net.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        @param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step on a batch of data from the training set.

        @param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        @param batch_idx: The index of the current batch.
        @return: A tensor of losses between model predictions and targets.
        """
        points, labels = batch
        points = torch.transpose(points, 1, 2)  # (batch_size, 3, num_point)

        logits, trans_in, trans_feat = self.forward(points)  # preds are logits

        loss = self.criterion(logits, labels)
        if trans_feat is not None:
            loss += regularize_feat_transform(trans_feat, reg_scale=0.001)

        preds = torch.argmax(logits, dim=1)  # (batch_size, num_classes)
        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, labels)

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step on a batch of data from the validation set.
        @param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        @param batch_idx: The index of the current batch.
        """
        points, labels = batch
        points = torch.transpose(points, 1, 2)  # (batch_size, 3, num_point)

        logits, trans_in, trans_feat = self.forward(points)  # preds are logits
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)  # (batch_size, num_classes)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, labels)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step on a batch of data from the test set.

        @param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        @param batch_idx: The index of the current batch.
        """
        points, labels = batch
        points = torch.transpose(points, 1, 2)  # (batch_size, 3, num_point)

        logits, _, _ = self.forward(points)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)  # (batch_size, num_classes)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, labels)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        @return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class PointNetPartSegModule(LightningModule):
    """
    Lightning Module of PointNet for Part Segmentation on Point Clouds
    """
    def __init__(self, net: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler,
                 compile: bool = False):
        """
        Init function of the LightningModule
        @param net: The model to train. Either PointNet2MSGCls or PointNet2SSGCls.
        @param optimizer: The optimizer to use for training.
        @param scheduler: The learning rate scheduler to use for training.
        @param compile: True - compile model for faster training with pytorch 2.0.
        """
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt.
        # logger=True: send the hyperparameters to the logger.
        self.save_hyperparameters(logger=False)

        self.net = net

        self.criterion = nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for instance average IoU (intersection over union)
        self.train_iou = MeanMetric()
        self.val_iou = MeanMetric()
        self.test_iou = MeanMetric()
        self.test_ious = []

        # for tracking best so far validation accuracy
        self.val_iou_best = MaxMetric()

    def setup(self, stage: str) -> None:
        """
        Lightning hook that is called at the beginning of fit (train + validate),
        validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        @param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def on_train_start(self) -> None:
        """
        Lightning hook that is called when training begins
        """
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_iou_best.reset()
        self.val_iou.reset()

    def forward(self, x):
        return self.net(x)

    def model_step(self, batch):
        """
        Forward + Loss + Predict a batch of data
        """
        points, seg_labels, encoded_segments = batch
        points = points.permute(0, 2, 1)  # (batch_size, 3+num_features, num_point)
        logits, trans_feat = self.forward(points)  # logits: batch_size * num_points * num_classes

        # calculate instance IoU
        preds = torch.argmax(logits, dim=-1)
        instance_ious = []
        for i in range(points.shape[0]):
            # decode the segments. an encoded str of segments: '12,13,14,15'
            segments = [int(s) for s in encoded_segments[i].split(',')]
            for segment in segments:
                total_union = torch.sum((preds[i] == segment) | (seg_labels[i] == segment))
                if total_union == 0:
                    instance_ious.append(1.0)
                else:
                    intersection = torch.sum((preds[i] == segment) & (seg_labels[i] == segment))
                    instance_ious.append((intersection / total_union).item())

        logits = logits.reshape(-1, self.net.num_classes)
        labels = seg_labels.view(-1, 1)[:, 0]
        loss = self.criterion(logits, labels)

        # It's reported by others, the feature transformation seems not to improve the performance.
        if trans_feat is not None:
            loss += regularize_feat_transform(trans_feat, reg_scale=0.001)
        return loss, instance_ious

    def training_step(self, batch: tuple, batch_idx: int):
        """
        Perform a single training step on a batch of data from the training set.

        @param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        @param batch_idx: The index of the current batch.
        @return: A tensor of losses between model predictions and targets.
        """
        loss, instance_ious = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_iou(torch.mean(torch.tensor(instance_ious)))

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/iou", self.train_iou, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step on a batch of data from the validation set
        """
        loss, instance_ious = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_iou(torch.mean(torch.tensor(instance_ious)))

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step on a batch of data from the test set.
        """
        loss, instance_ious = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_iou(torch.mean(torch.tensor(instance_ious)))
        self.test_ious.extend(instance_ious)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou", self.test_iou, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_end(self) -> None:
        # calculate instance average iou
        ave_iou = torch.mean(torch.tensor(self.test_ious))
        log.info("Final instance average iou on the test dataset: {:.2f}".format(ave_iou))

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        @return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        return {'optimizer': optimizer}


if __name__ == '__main__':
    batch_size, num_classes = 4, 5
    sim_data = torch.rand(batch_size, 6, 2000)

    pointfeat = PointNetCls(num_classes, in_channels=6)
    out, _, _ = pointfeat(sim_data)
    print(f"Expect out shape: {batch_size} * {num_classes}")
    print('Point feat', out.shape)
    print("Total number of parameters:", sum(p.numel() for p in pointfeat.parameters() if p.requires_grad))
