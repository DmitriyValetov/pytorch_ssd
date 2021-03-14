from utils import *
from datasets.datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()



def evaluate(test_loader, model,label_map,rev_label_map, device):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)


        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, label_map, rev_label_map, device=device)
        # Print AP for each class
        pp.pprint(APs)
        print('\nMean Average Precision (mAP): %.3f' % mAP)

        # Calculate mAP
        APs, mAP = calculate_mmAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, label_map, rev_label_map, device=device)
        # Print AP for each class
        pp.pprint(APs)
        print('\nMean Average Precision (mAP): %.3f' % mAP)





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

    label_map, rev_label_map, label_color_map = load_maps(os.path.join(data_folder, 'label_map.json'))
    
    test_dataset = PascalVOCDataset(data_folder,
                                    split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=learning_parameters['batch_size'], shuffle=False,
                                            collate_fn=test_dataset.collate_fn, num_workers=learning_parameters['workers'], pin_memory=True)


    checkpoint_path = os.path.join(data_folder, "checkpoint_ssd300.pkl")
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']
    model.quantize = False
    model.to(device)
    model.eval()


    evaluate(
        test_loader, 
        model,
        label_map,
        rev_label_map,
        device
    )

