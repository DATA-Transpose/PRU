import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViT(nn.Module):
    def __init__(self, num_classes):
        super(ViT, self).__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Sequential(
            nn.Linear(768, num_classes)
        )
        self.classifier = self.backbone.heads
        # self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def get_representation(self, x):
        # Reshape and permute the input tensor
        x = self.backbone._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.backbone.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self.backbone._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.backbone.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.classifier(x)

        return x