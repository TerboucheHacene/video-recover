import torch
import torchvision


def get_resnet50_model():
    # extract the features from the images using ResNet50
    resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights)
    resnet50.eval()
    model = torch.nn.Sequential(*list(resnet50.children())[:-1])
    return model


def get_vgg16_features():
    # extract the features from the images using VGG16
    vgg16 = torchvision.models.vgg16(pretrained=True)
    vgg16.eval()
    model = torch.nn.Sequential(
        *list(vgg16.children())[:-1], torch.nn.AdaptiveAvgPool2d((1, 1))
    )
    return model


def get_transforms():
    # define the transforms
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return transforms
