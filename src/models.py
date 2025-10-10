import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 stride=1, use_se=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels) if use_se else None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.se is not None:
            out = self.se(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AdaptiveCNN(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.in_channels = config['in_channels']
        self.num_classes = config['num_classes']
        self.dataset = config['dataset']
        self.use_se = config.get('use_se', True)

        if self.dataset == 'mnist':
            self._build_mnist_architecture()
        elif self.dataset == 'cifar':
            self._build_cifar_architecture()
        else:
            self._build_custom_architecture(config)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(config.get('dropout', 0.5))

        logger.info(f"Initialized AdaptiveCNN for {self.dataset} dataset")

    def _build_mnist_architecture(self):
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ),
            ResidualBlock(32, 64, stride=2, use_se=self.use_se),
            ResidualBlock(64, 64, stride=1, use_se=self.use_se),
        ])
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)
        )

    def _build_cifar_architecture(self):
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
            ResidualBlock(64, 128, stride=2, use_se=self.use_se),
            ResidualBlock(128, 128, stride=1, use_se=self.use_se),
            ResidualBlock(128, 256, stride=2, use_se=self.use_se),
            ResidualBlock(256, 256, stride=1, use_se=self.use_se),
            ResidualBlock(256, 512, stride=2, use_se=self.use_se),
        ])
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)
        )

    def _build_custom_architecture(self, config):
        layers = []
        in_ch = self.in_channels

        for layer_config in config['architecture']:
            if layer_config['type'] == 'conv':
                layers.append(nn.Sequential(
                    nn.Conv2d(in_ch, layer_config['out_channels'],
                             kernel_size=layer_config['kernel_size'],
                             padding=layer_config.get('padding', 1)),
                    nn.BatchNorm2d(layer_config['out_channels']),
                    nn.ReLU(inplace=True)
                ))
                in_ch = layer_config['out_channels']
            elif layer_config['type'] == 'residual':
                layers.append(ResidualBlock(in_ch, layer_config['out_channels'],
                                          stride=layer_config.get('stride', 1),
                                          use_se=self.use_se))
                in_ch = layer_config['out_channels']

        self.features = nn.ModuleList(layers)

        classifier_layers = []
        in_features = in_ch
        for fc_size in config['classifier_layers']:
            classifier_layers.extend([
                nn.Linear(in_features, fc_size),
                nn.ReLU(inplace=True),
                nn.Dropout(config.get('dropout', 0.5))
            ])
            in_features = fc_size
        classifier_layers.append(nn.Linear(in_features, self.num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def get_feature_maps(self, x, layer_idx=0):
        with torch.no_grad():
            if layer_idx < len(self.features):
                for i, layer in enumerate(self.features):
                    x = layer(x)
                    if i == layer_idx:
                        return x
        return x

class ModelFactory:

    _models = {
        'adaptive_cnn': AdaptiveCNN,
    }

    @classmethod
    def create_model(cls, model_name, config):
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")

        model_class = cls._models[model_name]
        return model_class(config)

    @classmethod
    def register_model(cls, name, model_class):
        cls._models[name] = model_class
