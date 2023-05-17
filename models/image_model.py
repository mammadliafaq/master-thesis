import timm
import torch.nn as nn

from .head import AdaCos, ArcFace, CosFace


class ImageModel(nn.Module):
    def __init__(
        self,
        n_classes,
        device,
        model_name="efficientnet_b0",
        use_fc=False,
        fc_dim=512,
        dropout=0.0,
        loss_module="softmax",
        s=30.0,
        margin=0.50,
        ls_eps=0.0,
        theta_zero=0.785,
        pretrained=True,
    ):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(ImageModel, self).__init__()
        print("Building Model Backbone for {} model".format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        if "resnet" in model_name:
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        self.backbone.global_pool = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == "arcface":
            self.final = ArcFace(
                final_in_features,
                n_classes,
                s=s,
                m=margin,
                easy_margin=False,
                ls_eps=ls_eps,
                device=device
            )
        elif loss_module == "cosface":
            self.final = CosFace(final_in_features, n_classes, s=s, m=margin, device=device)
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

    def forward(self, x, label):
        feature = self.extract_features(x)
        if self.loss_module in ("arcface", "cosface", "adacos"):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return logits

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.fc(x)
            x = self.bn(x)
            x = self.dropout(x)

        return x
