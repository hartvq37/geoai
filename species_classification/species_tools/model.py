
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        width = planes * 2  # Wide ResNet doubles the number of channels
        
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class WideResNet50_2(nn.Module):
    def __init__(self, num_classes=1000):
        super(WideResNet50_2, self).__init__()
        
        self.inplanes = 64
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        # Final pooling and fc layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # 2048-dimensional features
        logits = self.fc(features)      # class logits
        
        if return_features:
            features = features.detach()
            return logits,features
        else:
            return logits

    def load_pretrained(self, path):
        """Load pretrained weights"""
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet18, self).__init__()
        
        # Load the pretrained ResNet18 model
        base_model = models.resnet18(pretrained=pretrained)
        
        # Get all layers except the final fully connected layer
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        
        # Replace the fully connected layer to match the desired number of classes
        self.fc = nn.Linear(512, num_classes)  # ResNet18 has 512 features after avgpool
        
        # Initialize the new fc layer
        if pretrained:
            for m in self.fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        # Extract features from all layers except the fully connected layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global Average Pooling
        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # 512-dimensional features for ResNet18
        
        # Compute class logits
        logits = self.fc(features)
        
        if return_features:
            return logits, features
        return logits

#class ResNet50(nn.Module):
#    def __init__(self, num_classes=1000, pretrained=False):
#        super(ResNet50, self).__init__()
#        # Load the ResNet50 model
#        self.resnet50 = models.resnet50(pretrained=pretrained)
#        
#        # Replace the fully connected layer to match the desired number of classes
#        in_features = self.resnet50.fc.in_features
#        self.resnet50.fc = nn.Linear(in_features, num_classes)
#    
#    def forward(self, x, return_features=False):
#        # Extract features from all layers except the fully connected layer
#        x = self.resnet50.conv1(x)
#        x = self.resnet50.bn1(x)
#        x = self.resnet50.relu(x)
#        x = self.resnet50.maxpool(x)
#        
#        x = self.resnet50.layer1(x)
#        x = self.resnet50.layer2(x)
#        x = self.resnet50.layer3(x)
#        x = self.resnet50.layer4(x)
#        
#        # Global Average Pooling
#        x = self.resnet50.avgpool(x)
#        x = torch.flatten(x, 1)  # 2048-dimensional features for ResNet50
#        
#        # Compute class logits
#        logits = self.resnet50.fc(x)
#        
#        if return_features:
#            features = x.detach()  # Detach features from the computation graph
#            return logits, features
#        else:
#            return logits

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet50, self).__init__()
        # Load the ResNet50 model
        base_model = models.resnet50(pretrained=pretrained)
        
        # Get all layers except the final fully connected layer
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        # Replace the fully connected layer to match the desired number of classes
        self.fc = nn.Linear(2048, num_classes)  # ResNet101 has 2048 features after avgpool
        # Initialize the new fc layer
        if pretrained:
            for m in self.fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global Average Pooling
        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # 2048-dimensional features for ResNet101
        
        # Compute class logits
        logits = self.fc(features)
        
        if return_features:
            return logits, features
        return logits

class ResNet101(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet101, self).__init__()
        
        # Load the pretrained ResNet101 model
        base_model = models.resnet101(pretrained=pretrained)
        
        # Get all layers except the final fully connected layer
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        
        # Replace the fully connected layer to match the desired number of classes
        self.fc = nn.Linear(2048, num_classes)  # ResNet101 has 2048 features after avgpool
        
        # Initialize the new fc layer
        if pretrained:
            for m in self.fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        # Extract features from all layers except the fully connected layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global Average Pooling
        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # 2048-dimensional features for ResNet101
        
        # Compute class logits
        logits = self.fc(features)
        
        if return_features:
            return logits, features
        return logits

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(MobileNetV2, self).__init__()
        
        # Load the pretrained MobileNetV2 model
        base_model = models.mobilenet_v2(pretrained=pretrained)
        
        # Extract the feature extractor part of MobileNetV2
        self.features = base_model.features  # All convolutional layers
        
        # Get the output size of the feature extractor
        feature_output_size = base_model.last_channel  # Typically 1280 for MobileNetV2
        
        # Replace the classifier to match the desired number of classes
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Linear(feature_output_size, num_classes)
        
        # Initialize the new fc layer
        if not pretrained:
            nn.init.kaiming_normal_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x, return_features=False):
        # Extract features using the MobileNetV2 feature extractor
        x = self.features(x)
        
        # Global Average Pooling
        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # Flatten the features
        
        # Compute class logits
        logits = self.fc(features)
        
        if return_features:
            return logits, features
        return logits

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(MobileNetV3Small, self).__init__()
        
        # Load the pretrained MobileNetV3 Small model
        if pretrained:
            # Use pretrained weights if specified
            base_model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        else:
            # Initialize with random weights
            base_model = models.mobilenet_v3_small(weights=None)
        
        # Extract the feature extractor part of MobileNetV3 Small
        self.features = base_model.features  # All convolutional layers
        
        # Get the output size of the feature extractor
        self.feature_output_size = base_model.classifier[0].in_features  # Typically 576 for MobileNetV3 Small
        
        # Replace the classifier to match the desired number of classes
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Linear(self.feature_output_size, num_classes)
        
        # Initialize the new fc layer
        if not pretrained:
            nn.init.kaiming_normal_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x, return_features=False):
        # Extract features using the MobileNetV3 Small feature extractor
        x = self.features(x)
        
        # Global Average Pooling
        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # Flatten the features
        
        # Compute class logits
        logits = self.fc(features)
        
        if return_features:
            return logits, features
        return logits

class FeatureExtractor(nn.Module):
    def __init__(self, model,feature_dim=2048):
        super().__init__()
        # Extract the feature layers (all layers except the last classification layer)
        self.features = nn.Sequential(*list(model.children())[:-1])
        # Save the classification layer (or the final fully connected layers)
        #self.classifier = list(model.children())[-1]
        self.classifier = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, 128)
            )           
    def forward(self, x):
        # Get the features from the feature layers
        features = self.features(x)
        # Flatten if necessary before passing through the classifier
        if isinstance(features, torch.Tensor):
            features = features.view(features.size(0), -1)
        # Get the logits from the classifier
        logits = self.classifier(features)
        # Return both features and logits
        return logits,features

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(2048, num_classes)
    def forward(self, features):
        return self.classifier(features)