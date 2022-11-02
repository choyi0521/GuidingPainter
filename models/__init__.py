from models.ask_model import AskModel
import os
import torch


def get_model(args, phase, path=None):
    if phase == 'train':
        model = AskModel(args)
    elif phase == 'test':
        device = torch.device('cuda:{0}'.format(args.test_gpu) if torch.cuda.is_available() else 'cpu')
        model = AskModel.load_from_checkpoint(
            os.path.join(args.train_dir, 'checkpoints', args.checkpoint) if path is None else path,
            map_location = device,
            args = args
        )
        model = model.to(device)
        model.eval()
        model.freeze()
    else:
        raise NotImplementedError()
    return model
    