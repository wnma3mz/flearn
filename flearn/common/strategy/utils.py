# coding: utf-8
import numpy as np
import torch


def convert_to_np(weights):
    for k, v in weights.items():
        if isinstance(v, torch.Tensor):
            weights[k] = v.cpu().numpy()
        elif isinstance(v, np.ndarray):
            pass
        elif isinstance(v, list):
            weights[k] = np.array(v)
        else:
            raise SystemError("NOT SUPPORT THE DATATYPE", type(v))

    return weights


def convert_to_tensor(weights):
    for k, v in weights.items():
        if isinstance(v, torch.Tensor):
            pass
        elif isinstance(v, np.ndarray):
            weights[k] = torch.from_numpy(v)
        elif isinstance(v, list):
            weights[k] = torch.from_numpy(np.array(v))
        else:
            raise SystemError("NOT SUPPORT THE DATATYPE", type(v))

    return weights


def cdw_feature_distance(old_model, new_model, device, train_loader):
    """cosine distance weight (cdw): calculate feature distance of
    the features of a batch of data by cosine distance.
    old_classifier,
    """
    old_model = old_model.to(device)
    # old_classifier = old_classifier.to(device)

    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(device)

        with torch.no_grad():
            # old_out = old_classifier(old_model(inputs))
            old_out = old_model(inputs)
            new_out = new_model(inputs)

        distance = 1 - torch.cosine_similarity(old_out, new_out)
        return torch.mean(distance).cpu().numpy()
