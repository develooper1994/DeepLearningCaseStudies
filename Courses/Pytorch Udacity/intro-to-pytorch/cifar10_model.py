# __author = Mustafa Selçuk Çağlar
# __version__ = 0.01


# torch imports
import torch
import torchvision
import torchvision.transforms as transforms
# to build a NN
import torch.nn as nn
import torch.nn.functional as F
# optimizers
import torch.optim as optim
# learning rate decay
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# other 3rd party improvements
import matplotlib.pyplot as plt
import numpy as np


# Python libraries
# ...

# Other hand written modules
# ...


# input: A ses + gürültü
# encode
# self.fc1 = nn.Linear(16 * 5 * 5, 120) # conv2 output channel is 16 and kernel=5x5 => 16*5*5 number of features is input of linear layer.
# self.fc2 = nn.Linear(120, 84)
# self.fc3 = nn.Linear(84, 10)
# decode
# self.fc3 = nn.Linear(10, 84)
# self.fc2 = nn.Linear(84, 120)
# self.fc1 = nn.Linear(120, 16 * 5 * 5)
# output: A ses


class Net(nn.Module):
    def __init__(self, image_shape=torch.Size([3, 32, 32]), number_of_classes=10):
        super(Net, self).__init__()
        channel = image_shape[0]  # torch.Size([3, 32, 32])[0]
        self.conv1 = nn.Conv2d(channel, 6, 5)  # input channel, output channel, kernel size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)  # conv1 output channel is 6 so that conv2 input channel is 6
        self.fc1 = nn.Linear(16 * 4 * 4,  # 5*5
                             120)  # conv2 output channel is 16 and kernel=5x5 => 16*5*5 number of features is input of linear layer.
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, number_of_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RunNet():
    def __init__(self, learning_rate_decay=True, epochs=10, learning_rate=0.01, print_every=2000, num_workers=2,
                 batch_size=64):
        """ 3. Define a Loss function and optimizer
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            Let's use a Classification Cross-Entropy loss and SGD with momentum. <br/>
            3-1. [Learning rate scheduler](https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/) """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(self.device))

        ### training parameters
        self.epochs = epochs  # 2
        self.steps = 0
        # torch.detach_(self.epochs) # test detach and make "algo" faster
        self.learning_rate = learning_rate  # 0.001
        self.running_loss = 0
        self.print_every = print_every
        self.learning_rate_decay = learning_rate_decay
        # inspection parameters
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.test_loss = 0

        ### transform and loaders
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_path = '~/pytorch/cifar10'
        self.trainset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=True,
                                                     download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                       shuffle=True)
        print("Train loader lenght: {}".format(len(self.trainloader)))

        self.testset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=False,
                                                    download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                      shuffle=False)
        print("Test loader lenght: {}".format(len(self.testloader)))

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        number_of_classes = len(self.classes)
        print("Number of classes: {}".format(number_of_classes))

        image_shape, _ = self.get_data_shape()
        print("Image shape: {}".format(image_shape))

        ### NN(Neural Network)
        self.net = Net(image_shape, number_of_classes).to(self.device)
        self.weights_init()

        ### optimizers
        self.criterion = nn.CrossEntropyLoss()  # there is no log_softmax ;)
        if self.learning_rate_decay == True:
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9, nesterov=True)
            ### learning rate decay
            # self.gamma = decaying factor
            # self.scheduler = StepLR(optimizer, step_size=1, gamma=0.1) #classic decay

            # Reduce learning rate whenever loss plateaus
            # Turkish: Öğrenme katsayısı(learning rate) devamlı aynı yada yakın bir sayı verirse değiştir.
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=0,
                                               verbose=True)  # factor=0.5
        else:
            self.optimizer = optim.Adam(self.net.parameters(),
                                        lr=self.learning_rate)  # adam automaticly estimates learning rate. it don't need any scheduler.

    ### Configure class parameters
    def reset_all(self):
        self.optimizer.zero_grad()
        self.steps = 0
        self.running_loss = 0
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.test_loss = 0

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def configure_(self, epochs, running_loss, print_every, train_accuracy, test_accuracy, test_loss, device,
                   learning_rate,
                   criterion, optimizer, scheduler, transform, num_workers, batch_size, trainset, testset, classes):

        self.architecture()

        # training parameters
        self.epochs = 10  # 2
        # torch.detach_(self.epochs) # test detach and make "algo" faster
        self.learning_rate = 0.01  # 0.001
        self.running_loss = 0
        self.print_every = 2000
        self.learning_rate_decay = learning_rate_decay
        # inspection parameters
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.test_loss = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(device))

        # optimizers
        self.criterion = nn.CrossEntropyLoss()  # there is no log_softmax ;)
        self.optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

        # self.gamma = decaying factor
        # self.scheduler = StepLR(optimizer, step_size=1, gamma=0.1) #classic decay

        # Reduce learning rate whenever loss plateaus
        # Turkish: Öğrenme katsayısı(learning rate) devamlı aynı yada yakın bir sayı verirse değiştir.
        self.scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)  # factor=0.5

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        self.num_workers = 2
        self.batch_size = 4

        self.dataset_path = '~/pytorch/cifar10'
        self.trainset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=True,
                                                     download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                       shuffle=True, num_workers=num_workers)
        print("Train loader lenght: {}".format(len(self.trainloader)))

        self.testset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=False,
                                                    download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                      shuffle=False, num_workers=num_workers)
        print("Test loader lenght: {}".format(len(self.testloader)))

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print("Test loader lenght: {}".format(len(self.classes)))

    def configure_(self, params: list):
        reconfigure_(self, epochs, running_loss, print_every, train_accuracy, test_accuracy, test_loss, device,
                     learning_rate,
                     criterion, optimizer, scheduler, transform, num_workers, batch_size, trainset, testset, classes)

    def reconfigure_(self, epochs, running_loss, print_every, train_accuracy, test_accuracy, test_loss, device,
                     learning_rate,
                     criterion, optimizer, scheduler, transform, num_workers, batch_size, trainset, testset, classes):
        self.epochs, self.running_loss, self.print_every, self.train_accuracy, self.test_accuracy, self.test_loss, self.device, self.learning_rate,
        self.criterion, self.optimizer, self.scheduler, self.transform, self.num_workers, self.batch_size, self.trainset, self.testset, self.classes = epochs, running_loss, print_every, train_accuracy, test_accuracy, test_loss, device, learning_rate,
        criterion, optimizer, scheduler, transform, num_workers, batch_size, trainset, testset, classes

    def reconfigure_(self, params: list):
        self.epochs, self.running_loss, self.print_every, self.train_accuracy, self.test_accuracy, self.test_loss, self.device, self.learning_rate,
        self.criterion, self.optimizer, self.scheduler, self.transform, self.num_workers, self.batch_size, self.trainset, self.testset, self.classes = params

    ### transforms
    def torch_image_to_numpy_image(self, torch_img):
        # torch -> C(channel), H(height), W(width)
        # numpy -> H(height), W(width), C(channel)
        # PIL -> H(height), W(width)
        numpy_img = torch_img.numpy()
        return np.transpose(numpy_img, (1, 2, 0))
        # torch_img = torchvision.transforms.ToPILImage()(torch_img)

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    ### inspect the data, parameters and NN(Neural Network)
    def imshow(self, img):
        img_shape = img.shape
        print("img_shape: {}".format(img.shape))
        img = img / 2 + 0.5  # unnormalize
        trans_img = self.torch_image_to_numpy_image(img)
        print("img_shape: {}".format(img.shape))
        plt.imshow(trans_img)  # numpy and torch dimension orders are different so that need to change dims.
        # torch dim order: CHW(channel, height, width)
        plt.show()
        # print("image size: {}".format( np.transpose(npimg, (1, 2, 0)).size ) )

    def inspect_data(self, loader: torch.utils.data.dataloader.DataLoader):
        # get some random training images
        loader_type = loader.dataset.train
        if loader_type:
            images, labels = self.get_one_iter(self.trainloader)
        else:
            images, labels = self.get_one_iter(self.testloader)
        # show images
        self.imshow(torchvision.utils.make_grid(images))
        # print labels
        print('classes: ', ''.join('%5s' % self.classes[labels[j]] for j in range(self.batch_size)))

    def inspect_one_data(self, images):
        image = images[0, ...]
        img = np.squeeze(image)
        img = self.torch_image_to_numpy_image(img)  # PIL or numpy image format
        img = self.rgb2gray(img)  # gray scale

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')
        width, height = img.shape
        thresh = img.max() / 2.5
        for x in range(width):
            for y in range(height):
                val = round(img[x][y], 2) if img[x][y] != 0 else 0
                ax.annotate(str(val), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white' if img[x][y] < thresh else 'black')
        plt.show()

    def weights_init(self):
        """
        - Tanh/Sigmoid vanishing gradients can be solved with "Xavier initialization" -> keras default
            -- Good range of constant variance
        - ReLU/Leaky ReLU exploding gradients can be solved with "Kaiming He initialization"
            -- Good range of constant variance
        note: Pytorch uses lecun initialization by default - "Yann Lecun"
        """
        # cifar10 dataset convert weight distribution to normal(gaussian like) distribution.
        # normal distributions are more
        for module in self.net.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(
                    module.weight)  # xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_
                nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight)  # xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_
                nn.init.constant_(module.bias, 0)

    def plot_conv2d_weights(self):
        for i, module in enumerate(self.net.modules()):
            if isinstance(module, nn.Conv2d):
                module_name = "Conv" + str(i)
                weights = module.weight
                weights = weights.reshape(-1).detach().cpu().numpy()
                print("{} bias: ".format(module_name), module.bias)  # Bias to zero
                plt.hist(weights)
                plt.title(module_name)
                plt.show()

    def print_all(self, epoch):
        print("Epoch: {}/{}.. ".format(epoch + 1, self.epochs),
              "steps: {}.. ".format(self.steps + 1),
              "learning rate: {} ".format(self.get_lr_()),
              "Train loss: {0:.3f}.. ".format(self.running_loss / self.print_every),
              "Train accuracy: {0:.3f}".format(self.train_accuracy / self.print_every),
              "Test loss: {0:.3f}.. ".format(self.test_loss / len(self.testloader)),
              "Test accuracy: {0:.3f}".format(self.test_accuracy / len(self.testloader))
              )

    def print_net(self):
        print(self.net)

    ### train and validate
    def validate_classes(self, n_classes=10):
        class_correct = list(0. for i in range(n_classes))
        class_total = list(0. for i in range(n_classes))
        with torch.no_grad():
            for data in net_runner.testloader:
                # data to device
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                # predictions
                outputs = self.net(images)
                # accuracy but not all of them
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()

                for i in range(self.batch_size):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(n_classes):
            print('Accuracy of %5s : %2d %%' % (
                net_runner.classes[i], 100 * class_correct[i] / class_total[i]))

    def validate_step_(self, test_data):
        test_images, test_labels = test_data
        test_images, test_labels = test_images.to(self.device), test_labels.to(self.device)
        ps = self.net(test_images)
        self.test_loss += self.criterion(ps, test_labels).item()  # there isn't log at output

        # ps = torch.exp(ps)
        # top_test_p, top_test_class = ps.topk(1, dim=1)
        # test_equals = top_test_class == test_labels.view(*top_test_class.shape)
        # self.test_accuracy += torch.mean(test_equals.type(torch.cuda.FloatTensor)).item()

        self.test_accuracy += self.accuracy_(ps, test_labels).item()
        # return ps, test_images, test_labels

    def validate_all(self, epoch, outputs, train_labels):
        with torch.no_grad():
            # train accuracy
            # top_train_p, top_train_class = outputs.topk(1, dim=1)
            # train_equals = top_train_class == train_labels.view(*top_train_class.shape)
            # self.train_accuracy += torch.mean(train_equals.type(torch.cuda.FloatTensor)).item()

            self.train_accuracy += self.accuracy_(outputs, train_labels).item()

            for _, test_data in enumerate(self.testloader, 0):
                self.validate_step_(test_data)  # ps, test_images, test_labels = self.validate_step_(test_data)

            self.print_all(epoch)
            # print('[%d, %5d] loss: %.3f' % (epoch + 1, steps + 1, self.running_loss / 2000))

    def train_step_(self, train_data):
        # get the inputs; data is a list of [inputs, labels]
        train_inputs, train_labels = train_data
        train_inputs, train_labels = train_inputs.to(self.device), train_labels.to(self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.net(train_inputs)
        train_loss = self.criterion(outputs, train_labels)

        train_loss.backward()
        self.optimizer.step()
        return train_labels, train_loss, outputs

    def train(self):
        self.reset_all()
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            self.running_loss = 0.0
            self.steps = 0
            for data in self.trainloader:
                train_labels, train_loss, outputs = self.train_step_(data)

                # print statistics
                self.running_loss += train_loss.item()
                if self.steps % self.print_every == self.print_every - 1:  # 1999    # print every 2000 mini-batches
                    self.net.eval()
                    self.test_loss = 0
                    self.test_accuracy = 0

                    # validate all of them
                    self.validate_all(epoch, outputs, train_labels)
                    self.running_loss = 0.0
                    self.net.train()
                self.steps += 1

            if self.learning_rate_decay == True:
                # Correct way to decay Learning Rate
                # self.scheduler.step() # StepLR    
                self.scheduler.step(self.test_accuracy)  # ReduceLROnPlateau needs metrics parameter
        print('Finished Training')

    ### save and load model
    def model_save(self, PATH):
        return torch.save(self.net.state_dict(), PATH)

    def model_load_(self, PATH):
        "loading network itself"
        return self.net.load_state_dict(torch.load(PATH))

    def model_load(self, PATH):
        "loading network to another network"
        return torch.load(PATH)

    ### get possibilities
    def evaluate(self, data):
        return self.net(data)

    ### accuracy
    def accuracy_(self, outputs, labels):
        top_p, top_class = outputs.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        if self.device == "cpu":
            accuracy = torch.mean(equals.type(torch.FloatTensor))
        else:
            accuracy = torch.mean(equals.type(torch.cuda.FloatTensor))
        return accuracy

    def accuracy2_(self, outputs, labels):
        correct = 0
        total = 0
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().item()
        accuracy = correct / total
        return accuracy

    ### setters
    def set_classes(self, classes):
        if not isinstance(classes, tuple):
            Warning(
                "classes should looks like: ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')")
        self.classes = classes

    def set_loader(self, trainloader, testloader):
        self.trainloader = trainloader
        self.testloader = testloader

    ### getters
    # data getters
    def get_one_iter(self, loader):
        dataiter = iter(loader)
        images, labels = dataiter.next()
        return images, labels

    def get_first_data(self, loader):
        images, labels = self.get_one_iter(loader)
        return images[0], labels[0]

    def get_first_train_data(self):
        return self.get_first_data(self.trainloader)

    def get_first_test_data(self):
        return self.get_first_data(self.testloader)

    def get_data_shape(self):
        # returns torch shape
        image, label = self.get_first_train_data()
        return image.shape, label.shape

    # class parameter getters
    def get_lr_(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def get_info(self):
        return {
            "lr": self.get_lr_(optimizer),
            "loss": {"train": self.running_loss / self.print_every, "test": self.test_loss / len(self.testloader)},
            "accuracy": {"train": self.train_accuracy / self.print_every,
                         "test": self.test_accuracy / len(self.testloader)}
        }


if __name__ == "__main__":
    net_runner = RunNet(learning_rate_decay=True)
    # net_runner.plot_conv2d_weights()
    # print(net_runner.net)

    ### try to train net
    # net_runner.train()

    ### try to make image to numeric
    # dataiter = iter(net_runner.testloader)
    # images, labels = dataiter.next()
    # net_runner.inspect_one_data(images)

    net_runner.validate_classes(n_classes=10)
