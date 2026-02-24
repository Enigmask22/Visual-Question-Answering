import argparse
import collections
import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader

import model.loss as module_loss
import model as module_arch
from parse_config import ConfigParser
from trainer import VQATrainer
from utils import prepare_device

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    data_dir = config['data_loader']['args']['data_dir']
    img_size = config['data_loader']['args']['img_size']
    batch_size = config['data_loader']['args']['batch_size']
    num_workers = config['data_loader']['args']['num_workers']
    
    train_json = os.path.join(data_dir, 'train.json')
    val_json = os.path.join(data_dir, 'val.json')
    
    from data_loader.data_loaders import build_answer_vocab, get_transforms
    from data_loader.dataset import VQADataset
    
    vocab, idx_to_ans = build_answer_vocab(train_json, val_json)
    num_classes = len(vocab)
    logger.info(f'Number of answer classes: {num_classes}')
    
    train_transform, val_transform = get_transforms(img_size)
    
    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    train_questions = [item['question'] for item in train_data]
    train_dataset = VQADataset(train_json, data_dir, train_transform, vocab)
    val_dataset = VQADataset(val_json, data_dir, val_transform, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model_type = config['arch']['type']
    model_args = config['arch']['args']
    if model_type == 'MLPBaseline':
        model = module_arch.MLPBaseline(num_classes, train_questions=train_questions, **model_args)
    elif model_type == 'CNNLSTMBaseline':
        model = module_arch.CNNLSTMBaseline(num_classes, train_questions=train_questions, **model_args)
    elif model_type == 'ViTBERTBaseline':
        model = module_arch.ViTBERTBaseline(num_classes, **model_args)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    logger.info(model)

    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = getattr(module_loss, config['loss'])
    metrics = []

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = VQATrainer(model, criterion, metrics, optimizer, config=config,
                         device=device, data_loader=train_loader,
                         valid_data_loader=valid_loader, idx_to_ans=idx_to_ans,
                         lr_scheduler=lr_scheduler)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='VQA Training')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path')
    args.add_argument('-r', '--resume', default=None, type=str, help='checkpoint path')
    args.add_argument('-d', '--device', default=None, type=str, help='GPU indices')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
