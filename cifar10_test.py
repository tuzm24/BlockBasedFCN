import torch
import torchvision
import torchvision.transforms as transforms

from model.resnet import *

from tqdm import tqdm


if __name__=='__main__':
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    valdir = 'data/imagenet_1k/val'
    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transform,
        ]))
    # testset =
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=12)
    net = resnext101_32x8d(pretrained=True).cuda()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data[0].cuda(), data[1].cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))