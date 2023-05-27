import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import AdaCos, ArcFace, CosFace


class ShopeeMultiModel(nn.Module):
    def __init__(
        self,
        num_image_features,
        num_text_features,
        n_classes,
        device,
        use_fc=True,
        fc_dim=512,
        dropout=0.0,
        loss_module="softmax",
        s=30.0,
        margin=0.50,
        ls_eps=0.0,
        theta_zero=0.785,
    ):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(ShopeeMultiModel, self).__init__()

        self.num_image_features = num_image_features
        self.num_text_features = num_text_features

        self.use_fc = use_fc
        if self.use_fc:
            self.fc = nn.Linear(
                (self.num_image_features + self.num_text_features), fc_dim
            )
            self.bn = nn.BatchNorm1d(fc_dim)
            self.dropout = nn.Dropout(p=dropout)
            self._init_params()
            final_in_features = fc_dim
        else:
            final_in_features = self.num_image_features + self.num_text_features

        self.loss_module = loss_module
        if loss_module == "arcface":
            self.final = ArcFace(
                final_in_features,
                n_classes,
                s=s,
                m=margin,
                easy_margin=False,
                ls_eps=ls_eps,
                device=device,
            )
        elif loss_module == "cosface":
            self.final = CosFace(
                final_in_features, n_classes, s=s, m=margin, device=device
            )
        elif loss_module == "adacos":
            self.final = AdaCos(
                final_in_features, n_classes, m=margin, theta_zero=theta_zero
            )
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x_image, x_text, label):
        fused_features = self.fuse_features(x_image, x_text)
        if self.loss_module in ("arcface", "cosface", "adacos"):
            logits = self.final(fused_features, label)
        else:
            logits = self.final(fused_features)
        return logits

    def fuse_features(self, x_image, x_text):
        image_features = x_image.view(-1, self.num_image_features)
        text_features = x_text.view(-1, self.num_text_features)
        fused_features = torch.cat((image_features, text_features), 1)
        fused_features_normalized = F.normalize(fused_features, dim=1)

        if self.use_fc:
            x = self.fc(fused_features_normalized)
            x = self.bn(x)
            x = self.dropout(x)
        return x
