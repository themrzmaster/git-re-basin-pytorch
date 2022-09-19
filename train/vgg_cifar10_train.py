from models.vgg import VGG
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms, models
from utils.training import train, test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
    args = parser.parse_args()

    # Get data
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)

    model = VGG("VGG16").to(device)
    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, True)
        test(model, device, test_loader, True)
        scheduler.step()

    torch.save(model.state_dict(), f"cifar10_{str(args.seed)}_vgg.pt")


if __name__ == "__main__":
  main()