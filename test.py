import argparse
import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

import model.loss as module_loss
import model as module_arch
from parse_config import ConfigParser
from model import compute_vqa_accuracy, compute_anls


def main(config):
    logger = config.get_logger('test')

    data_dir = config['data_loader']['args']['data_dir']
    img_size = config['data_loader']['args']['img_size']
    batch_size = config['data_loader']['args']['batch_size']
    num_workers = config['data_loader']['args']['num_workers']
    
    train_json = os.path.join(data_dir, 'train.json')
    val_json = os.path.join(data_dir, 'val.json')
    test_json = os.path.join(data_dir, 'test.json')
    
    from data_loader.data_loaders import build_answer_vocab, get_transforms
    from data_loader.dataset import VQADataset
    
    vocab, idx_to_ans = build_answer_vocab(train_json, val_json)
    num_classes = len(vocab)
    logger.info(f'Number of answer classes: {num_classes}')
    
    _, test_transform = get_transforms(img_size)
    
    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    train_questions = [item['question'] for item in train_data]
    
    test_dataset = VQADataset(test_json, data_dir, test_transform, vocab)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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

    loss_fn = getattr(module_loss, config['loss'])

    logger.info(f'Loading checkpoint: {config.resume}')
    checkpoint = torch.load(config.resume, weights_only=False)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_valid_vocab = 0
    sum_vqa_acc = 0
    sum_vqa_top5 = 0
    sum_anls = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device)
            questions = batch['question']
            answers = batch['answer'].to(device)
            raw_answers = batch['raw_answer']
            gt_answers_list = batch.get('answers', [[ans] for ans in raw_answers])

            if isinstance(gt_answers_list, list) and len(gt_answers_list) > 0 and isinstance(gt_answers_list[0], (list, tuple)):
                gt_answers_list = list(zip(*gt_answers_list))

            batch_size = images.size(0)
            total_samples += batch_size
            output = model(images, questions)

            valid_mask = answers != -1
            if valid_mask.any():
                v_outputs = output[valid_mask]
                v_answers = answers[valid_mask]
                loss = loss_fn(v_outputs, v_answers)
                total_loss += loss.item() * valid_mask.sum().item()

                _, predicted = v_outputs.max(1)
                correct_top1 += predicted.eq(v_answers).sum().item()

                _, top5_preds = v_outputs.topk(min(5, v_outputs.size(1)), 1)
                correct_top5 += (top5_preds == v_answers.view(-1, 1)).sum().item()

                total_valid_vocab += valid_mask.sum().item()

            k = min(5, output.size(1))
            _, top5_all = output.topk(k, 1)

            for i in range(batch_size):
                pred_id = top5_all[i, 0].item()
                pred_text_top1 = idx_to_ans.get(pred_id, "")
                gt_list = gt_answers_list[i]

                sum_vqa_acc += compute_vqa_accuracy(pred_text_top1, gt_list)
                vqa_scores_top5 = [compute_vqa_accuracy(idx_to_ans.get(top5_all[i, j].item(), ""), gt_list) for j in range(k)]
                sum_vqa_top5 += max(vqa_scores_top5)
                sum_anls += compute_anls(raw_answers[i], pred_text_top1)

    avg_loss = total_loss / total_valid_vocab if total_valid_vocab > 0 else 0
    acc_top1 = 100.0 * correct_top1 / total_valid_vocab if total_valid_vocab > 0 else 0
    acc_top5 = 100.0 * correct_top5 / total_valid_vocab if total_valid_vocab > 0 else 0
    vqa_acc = 100.0 * sum_vqa_acc / total_samples if total_samples > 0 else 0
    vqa_top5 = 100.0 * sum_vqa_top5 / total_samples if total_samples > 0 else 0
    anls = sum_anls / total_samples if total_samples > 0 else 0

    log = {'loss': avg_loss, 'accuracy_top1': acc_top1, 'accuracy_top5': acc_top5,
           'vqa_accuracy': vqa_acc, 'vqa_accuracy_top5': vqa_top5, 'anls': anls}

    logger.info('Test Results:')
    for key, value in log.items():
        logger.info(f'    {key:20s}: {value:.4f}')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='VQA Testing')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path')
    args.add_argument('-r', '--resume', default=None, type=str, help='checkpoint path')
    args.add_argument('-d', '--device', default=None, type=str, help='GPU indices')

    config = ConfigParser.from_args(args)
    main(config)
