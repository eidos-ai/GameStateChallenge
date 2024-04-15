import sys
sys.path.append("./SNGS_dataset")
sys.path.append("./models")
sys.path.append("./losses")
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
import logging

from gsr.utils.neptune_logger import NeptuneLogger

from SNGS_dataset.dataset import SoccerNetGSDataset
from SNGS_dataset.sampler import SoccerNetGSBatchSampler
from losses.triplet_loss import OnlineTripletLoss, TripletSelector, RandomNegativeTripletSelector, HardestNegativeTripletSelector, SemihardNegativeTripletSelector
from losses.triplet_loss import distance as triplet_loss_distance

from models.resnet import ResnetMLP
from models.resnet import TRANSFORM as RESNET_TRANSFORM

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, device, log_interval, metrics=[],
        start_epoch=0, neptune_logger: NeptuneLogger=None, inference_transform=None):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    best_accuracy = 0
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, _ = train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics, neptune_logger)
        neptune_logger.log("train_loss", train_loss)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        # for metric in metrics:
        #     message += '\t{}: {}'.format(metric.name(), metric.value())

        # valid_loss, _ = valid_epoch(val_loader, model, loss_fn, device, metrics, neptune_logger)
        # valid_loss /= len(val_loader)
        # neptune_logger.log("valid_loss", valid_loss)
        # message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
        #                                                                          valid_loss)
        # for metric in metrics:
        #     message += '\t{}: {}'.format(metric.name(), metric.value())
        
        # if (epoch % 5 == 0) or (epoch == n_epochs - 1):
        #     # accuracy = accuracy_eval.eval_identifier(model, inference_transform, euclidean_distance)
        #     accuracy = accuracy_eval.eval_identifier(model, inference_transform, triplet_loss_distance)
        #     neptune_logger.log("Accuracy", accuracy)
        #     message += '\t{}: {}'.format("Accuracy", round(accuracy, 3))
        
        # if accuracy > best_accuracy:
        #     best_model = copy.deepcopy(model)
        #     best_accuracy = accuracy

        print(message)
    return model
        


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics, neptune_logger):
    # for metric in metrics:
    #     metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        logger.debug(f"batch image shape {images.shape}")
        labels = labels["role_id"] if len(labels) > 0 else None
        
        logger.debug(f"Moving to device")
        
        images = images.to(device)
        labels = labels.to(device)

        logger.debug(f"Going into model")
    
        optimizer.zero_grad()
        
        _ , embeddings = model(images)

        loss_outputs = loss_fn(embeddings, labels)
        
        logger.debug("loss")
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        # for metric in metrics:
        #     metric(outputs_images, target, loss_outputs)
        ap_distance = loss_outputs[2]
        an_distance = loss_outputs[3]
        neptune_logger.log("ap_distance_train", ap_distance)
        neptune_logger.log("an_distance_train", an_distance)
        
        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())
            message += '\t Ap Distance: {:.6f}, An Distance {:.6f}'.format(ap_distance, an_distance)
            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics

def valid_epoch(val_loader, model, loss_fn, device, metrics, neptune_logger):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (images, labels) in enumerate(val_loader):
            logger.debug(f"batch image shape {images.shape}")
            labels = labels["role_id"] if len(labels) > 0 else None
            labels = labels.to(device)
            images = images.to(device)

            _ , embeddings = model(images)

            loss_outputs = loss_fn(embeddings, labels)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()
            
            ap_distance = loss_outputs[2]
            an_distance = loss_outputs[3]
            neptune_logger.log("ap_distance_valid", ap_distance)
            neptune_logger.log("an_distance_valid", an_distance)


            # for metric in metrics:
            #     metric(outputs, target, loss_outputs)
        val_loss /= (batch_idx + 1)
    return val_loss, metrics

# Custom function to parse a comma-separated list of integers
def parse_int_list(arg):
    try:
        # Split the input string by commas and convert each part to an integer
        int_list = [int(item) for item in arg.split(',')]
        return int_list
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid integer list format. Use comma-separated integers.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', '--train', required=True, help='Path to directory with SoccerNetGSR train split')
    parser.add_argument('-te', '--test', required=False, help='Path to directory with SoccerNetGSR test split')
    parser.add_argument('-v', '--valid', required=True, help='Path to directory with SoccerNetGSR valid split')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=5, help='number of epochs to train model')
    parser.add_argument('-lr', '--learning_rate', type=float, required=False, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--n_videos', type=int, required=False, default=1, help='number of different videos per batch')
    parser.add_argument('--n_samples', type=int, required=False, default=99, help='number of samples per video per batch')
    parser.add_argument('--selector', type=str, required=False, default="HardestNegative", help='triplet selector to use')
    
    parser.add_argument("--backbone", required=False, type=str, default="resnet18", help="Resnet arch to use as backbone")
    parser.add_argument("--out_features", required=False, type=int, default=64, help="MLP output features")
    parser.add_argument("--hidden", required=False, type=parse_int_list, default=[256, 128], help="MLP hidden layers as comma-separated list of integers")

    parser.add_argument("--neptune", required=False, action="store_true", default=False, help="Log experiment results to neptune")
    parser.add_argument("--project", required=False, type=str, default="eidos/GameStateChallenge", help="The name of the project")
    parser.add_argument("--out_dir", required=False, type=str, default="experiments/identifier", help="Path to the directory where a directory will be created to store output from this experiment")
    parser.add_argument("--name", required=False, type=str, help="The name of the experiment")
    
    args = vars(parser.parse_args())
    
    # Get cpu, gpu or mps device for training.
    device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
    print(f"Using {device} device")
    
    train_dataset_path = args["train"]
    test_dataset_path = args["test"]
    valid_dataset_path = args["valid"]
    epochs = args["epochs"]
    n_videos_batch = args["n_videos"]
    n_samples_per_video_batch = args["n_samples"]
    out_features = args["out_features"]
    hidden_layers = args["hidden"]
    selector_str = args["selector"]
    lr = args["learning_rate"]
    backbone = args["backbone"]
    
    # Add training parameters to neptune
    neptune_logger = NeptuneLogger(enable_run=args["neptune"], project=args["project"], run_name_or_id=args["name"])
    neptune_logger.param("epochs", epochs)
    neptune_logger.param("n_videos_batch", n_videos_batch)
    neptune_logger.param("n_samples_per_class", n_samples_per_video_batch)
    neptune_logger.param("features", out_features)
    neptune_logger.param("backbone", backbone)
    neptune_logger.param("hidden layers", str(hidden_layers))
    neptune_logger.param("selector", selector_str)
    neptune_logger.param("lr", lr)
    neptune_logger.tags(["TrainIdentifier"])
    
    # Set output path for model
    experiment_id = neptune_logger.get_id()
    experiment_output_dir = Path(args["out_dir"], experiment_id)
    model_output_path = experiment_output_dir.joinpath("trained_model.pt")
    model_output_path.parent.mkdir(parents=True, exist_ok=False)
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if device == "cuda" else {}
    
    # Load adequate model and transform for input
    model = ResnetMLP("resnet18", mlp_hidden_layers=hidden_layers, mlp_output_size=out_features).to(device)
    transform = RESNET_TRANSFORM
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    neptune_logger.param("Total parameters", total_params)
    neptune_logger.param("Trainable parameters", trainable_params)

    logger.info("Creating train dataset and dataloader")
    train_dataset = SoccerNetGSDataset(train_dataset_path, transform=transform)
    train_batch_sampler = SoccerNetGSBatchSampler(train_dataset.labels, n_videos=n_videos_batch, n_samples=n_samples_per_video_batch)
    online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    
    logger.info("Creating validation dataset and dataloader")
    valid_dataset = SoccerNetGSDataset(valid_dataset_path, transform=transform)
    valid_batch_sampler = SoccerNetGSBatchSampler(valid_dataset.labels, n_videos=n_videos_batch, n_samples=n_samples_per_video_batch)
    online_valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_sampler=valid_batch_sampler, **kwargs)
    
    margin = 1.
    selector_class: TripletSelector = None
    if selector_str == "RandomNegative":
        selector_class = RandomNegativeTripletSelector
    elif selector_str == "SemihardNegative":
        selector_class = SemihardNegativeTripletSelector
    elif selector_str == "HardestNegative":
        selector_class = HardestNegativeTripletSelector
    else:
        raise Exception("Selector not recognized")
    
    loss_fn = OnlineTripletLoss(margin, selector_class(margin))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = epochs
    log_interval = 100
    
    trained_model = fit(online_train_loader, online_valid_loader, model, 
        loss_fn, optimizer, scheduler, n_epochs, device, 
        log_interval, neptune_logger=neptune_logger)
    

    torch.save(trained_model.state_dict(), model_output_path)
    
    # Upload model, disabled to save space
    # neptune_logger.upload_file(Path(output_model_path).name, output_model_path)    