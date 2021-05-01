import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class EarlyStopping(object):
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


# AlexNet with dropout
class Net(nn.Module):
    def __init__(self, num_classes=10, init_weights=False, p=0.5):
        # trained on 2 GPUs, change conv size if trained on CPU
        # trained on CIFAR10 so changed some sizes
        super(Net, self).__init__()
        self.p = p

        self.conv1 = nn.Conv2d(3, 48 * 2, kernel_size=7, stride=2, padding=2)  # input[3, 32, 32]  output[48, 15, 15]
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # output[48, 27, 27]
        self.conv2 = nn.Conv2d(48 * 2, 128 * 2, kernel_size=5, padding=2)  # output[128, 27, 27]
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # output[128, 13, 13]

        self.conv3 = nn.Conv2d(128 * 2, 192 * 2, kernel_size=3, padding=1)  # output[192, 13, 13]
        self.conv4 = nn.Conv2d(192 * 2, 192 * 2, kernel_size=3, padding=1)  # output[192, 13, 13]
        self.conv5 = nn.Conv2d(192 * 2, 128 * 2, kernel_size=3, padding=1)  # output[128, 13, 13]

        self.dr1 = nn.Dropout(p=self.p)
        self.fc1 = nn.Linear(128 * 2 * 3 * 3, 1024)
        self.dr2 = nn.Dropout(p=self.p)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(self.dr1(x)))
        x = F.relu(self.fc2(self.dr2(x)))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# to be changed
class BatchNorm_Net(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(BatchNorm_Net, self).__init__()
        print("BatchNorm AlexNet")
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]

            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]

            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.conv1 = nn.Conv2d(3, 48 * 2, kernel_size=7, stride=2, padding=2)  # input[3, 32, 32]  output[48, 15, 15]
        self.bn1 = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # output[48, 27, 27]
        self.conv2 = nn.Conv2d(48 * 2, 128 * 2, kernel_size=5, padding=2)  # output[128, 27, 27]
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # output[128, 13, 13]
        self.conv3 = nn.Conv2d(128 * 2, 192 * 2, kernel_size=3, padding=1)  # output[192, 13, 13]
        self.bn3 = nn.BatchNorm2d(192)
        self.conv4 = nn.Conv2d(192 * 2, 192 * 2, kernel_size=3, padding=1)  # output[192, 13, 13]
        self.bn4 = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192 * 2, 128 * 2, kernel_size=3, padding=1)  # output[128, 13, 13]
        self.bn5 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 2 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(self.dr1(x)))
        x = F.relu(self.fc2(self.dr2(x)))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def verbose_forward(self, x):
        # TODO: when it's used? hard to implement
        saved_vals = []
        x = F.relu(self.conv1(x))
        saved_vals.append(x)
        print("Conv 1: ", x.shape)
        x = self.pool(x)
        saved_vals.append(x)
        print("Post Pooling: ", x.shape)
        x = F.relu(self.conv2(x))
        saved_vals.append(x)
        print("Conv 2: ", x.shape)
        x = self.pool(x)
        saved_vals.append(x)
        print("Post Pooling 2")
        x = x.view(-1, 16 * 5 * 5)
        saved_vals.append(x)
        print("Flattening: ", x.shape)
        x = F.relu(self.bn1(self.fc1(x)))
        saved_vals.append(x)
        print("FC1: ", x.shape)
        x = F.relu(self.bn2(self.fc2(x)))
        saved_vals.append(x)
        print("FC2: ", x.shape)
        x = self.fc3(x)
        saved_vals.append(x)
        print("FC3: ", x.shape)
        saved_vals.append(x)
        return x, saved_vals


# data function
def get_cifar_data(augment_data=False, batch_size=128):
    if augment_data:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Normalize the test set same as training set without augmentation
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # flip, rotate, crop

        trainset = torchvision.datasets.CIFAR10(
            root="../data", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR10(
            root="../data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    else:
        # return normal dataloaders
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="../data", train=True, download=True, transform=transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR10(
            root="../data", train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    return trainloader, testloader, classes
