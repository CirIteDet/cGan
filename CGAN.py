import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(10, embedding_size)
        self.model = nn.Sequential(
            nn.Linear(z_size + embedding_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embed = self.embedding(labels)
        output = self.model(torch.cat([z, label_embed], -1))
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(10, embedding_size)
        self.model = nn.Sequential(
            nn.Linear(28 * 28 + embedding_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, im, labels):
        label_embed = self.embedding(labels)
        inp = im.reshape(batch_size, -1)
        iii = torch.cat([inp, label_embed], -1)
        output = self.model(iii)
        return output


z_size = 100
embedding_size = 32
batch_size = 64
dataset_train = torchvision.datasets.MNIST("./data", train=True,
                                     transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                               torchvision.transforms.Normalize(0.5,
                                                                                                                0.5)
                                                                               ]), download=True)
dataset_test = torchvision.datasets.MNIST("./data", train=False,
                                     transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                               torchvision.transforms.Normalize(0.5,
                                                                                                                0.5)
                                                                               ]), download=True)
dataloader_train = DataLoader(dataset_train, batch_size, shuffle=True, drop_last=True)
dataloader_test = DataLoader(dataset_test, batch_size, shuffle=True, drop_last=True)
Generator_net = Generator()
Generator_net = Generator_net.cuda()
Discriminator_net = Discriminator()
Discriminator_net = Discriminator_net.cuda()
d_optim = torch.optim.Adam(Discriminator_net.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
g_optim = torch.optim.Adam(Generator_net.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
loss_fn = nn.BCELoss()
loss_fn = loss_fn.cuda()

num_epoch = 100
ones = torch.ones(batch_size, 1)
ones = ones.to("cuda")
zeros = torch.zeros(batch_size, 1)
zeros = zeros.to("cuda")
one = torch.ones(16, 1)
one = one.to("cuda")
for epoch in range(num_epoch):
    print("第{}轮".format(epoch))
    for i, minibatch in enumerate(dataloader_train):
        img, label = minibatch
        img = img.cuda()
        label = label.cuda()
        z = torch.randn(batch_size, z_size)
        z = z.cuda()
        d_optim.zero_grad()
        label_shuffle = torch.zeros_like(label)
        for j in range(len(label) - 1):
            label_shuffle[j + 1] = label[j]
        label_shuffle[0] = label[len(label) - 1]
        loss_realMiss = loss_fn(Discriminator_net(img, label_shuffle), zeros)
        loss_real = loss_fn(Discriminator_net(img, label), ones)
        loss_fake = loss_fn(Discriminator_net(Generator_net(z, label).detach(), label), zeros)
        d_loss = loss_fake + loss_real + loss_realMiss
        d_loss.backward()
        d_optim.step()

        img_fake = Generator_net(z, label)
        fake_img = Discriminator_net(img_fake, label)
        mean = fake_img.sum() / fake_img.shape[0]
        print(mean)
        # if mean <= 0.1:
        #     for j, batch in enumerate(dataloader_test):
        #         imgs, labels = batch
        #         imgs = imgs.cuda()
        #         labels = labels.cuda()
        #         los1 = loss_fn(Discriminator_net(Generator_net(torch.randn(batch_size, z_size).cuda(), labels).detach(), labels), ones)
        #         los2 = loss_fn(
        #             Discriminator_net(imgs, labels),
        #             zeros)
        #         l = los2 + los1
        #         d_optim.zero_grad()
        #         l.backward()
        #         d_optim.step()
        #
        #         fake_img = Discriminator_net(img_fake, label)
        #         mean = fake_img.sum() / fake_img.shape[0]
        #         print("<0.4 mean{}".format(mean))
        #         if mean > 0.1:
        #             break
        # if mean >= 0.3:
        #     for j, batch in enumerate(dataloader_test):
        #         imgs, labels = batch
        #         imgs = imgs.cuda()
        #         labels = labels.cuda()
        #         lo1 = loss_fn(Discriminator_net(imgs, labels), ones)
        #         lo2 = loss_fn(Discriminator_net(Generator_net(torch.randn(batch_size, z_size).cuda(), labels).detach(), labels), zeros)
        #         lo = lo2 + lo1
        #         d_optim.zero_grad()
        #         lo.backward()
        #         d_optim.step()
        #
        #         fake_img = Discriminator_net(img_fake, label)
        #         mean = fake_img.sum() / fake_img.shape[0]
        #         print(">0.6 mean{}".format(mean))
        #         if mean < 0.3:
        #             break
        g_loss = loss_fn(fake_img, ones)
        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

    if epoch % 1 == 0:
        img_fake = img_fake.reshape(batch_size, 1, 28, 28).cpu().detach()
        plt.figure()

        for num in range(batch_size):
            plt.subplot(8, 8, num+1)
            if num < 8:
                plt.title("{} plot".format(epoch))
            plt.imshow(img_fake[num].permute(1, 2, 0))
        plt.show()
        # print("want output {}".format(epoch % 8))
        # img = Generator_net(torch.randn(z_size).cuda(), torch.tensor(epoch % 8, device="cuda"))
        # plt.figure()
        # plt.imshow(img.reshape(1, 28, 28).cpu().detach().permute(1, 2, 0))
        # plt.show()
