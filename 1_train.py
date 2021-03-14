import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from ssd import SSD300
from loss import MultiBoxLoss
from datasets.datasets import PascalVOCDataset
from utils import adjust_learning_rate, save_checkpoint, AverageMeter, clip_gradient, load_maps

cudnn.benchmark = True


def main(
    data_folder, 
    lp, # learning_parameters
    device,
):
    """
    Training.
    """
    label_map, rev_label_map, label_color_map = load_maps(os.path.join(data_folder, 'label_map.json'))
    checkpoint_path = os.path.join(data_folder, "checkpoint_ssd300.pkl")  # path to model checkpoint, None if none
    n_classes = len(label_map)  # number of different types of objects


    # Initialize model or load checkpoint
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lp['lr']}, {'params': not_biases}],
                                    lr=lp['lr'], momentum=lp['momentum'], weight_decay=lp['weight_decay'])

    else:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, device=device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=lp['batch_size'], shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=lp['workers'],
                                               pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = lp['iterations'] // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in lp['decay_lr_at']]

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, lp['decay_lr_to'])

        # One epoch's training
        train(
            lp,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=device
        )

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, checkpoint_path)


def train(lp, train_loader, model, criterion, optimizer, epoch, device):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if lp['grad_clip'] is not None:
            clip_gradient(optimizer, lp['grad_clip'])

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % lp['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data parameters
    data_folder = 'trial_dataset_dumps'  # folder with data files

    learning_parameters = {
        'batch_size': 8,  # batch size
        'iterations': 120000,  # number of iterations to train
        'workers': 4,  # number of workers for loading data in the DataLoader
        'print_freq': 200,  # print training status every __ batches
        'lr': 1e-3,  # learning rate
        'decay_lr_at': [80000, 100000],  # decay learning rate after these many iterations
        'decay_lr_to': 0.1,  # decay learning rate to this fraction of the existing learning rate
        'momentum': 0.9,  # momentum
        'weight_decay': 5e-4,  # weight decay
        'grad_clip': None,  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
    }

    main(
        data_folder, 
        learning_parameters,
        device,
    )
