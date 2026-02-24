import torch
from base import BaseTrainer
from utils import MetricTracker
from model import compute_vqa_accuracy, compute_anls


class VQATrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, idx_to_ans=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(len(data_loader) * 0.1)
        self.idx_to_ans = idx_to_ans if idx_to_ans is not None else {}

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        
        total_samples = 0
        sum_vqa_acc = 0
        sum_anls = 0

        for batch_idx, batch in enumerate(self.data_loader):
            images = batch['image'].to(self.device)
            questions = batch['question']
            answers = batch['answer'].to(self.device)
            raw_answers = batch['raw_answer']
            gt_answers_list = batch.get('answers', [[ans] for ans in raw_answers])

            if isinstance(gt_answers_list, list) and len(gt_answers_list) > 0 and isinstance(gt_answers_list[0], (list, tuple)):
                gt_answers_list = list(zip(*gt_answers_list))

            valid_mask = answers != -1
            if not valid_mask.any():
                continue

            self.optimizer.zero_grad()
            output = self.model(images, questions)

            v_outputs = output[valid_mask]
            v_answers = answers[valid_mask]
            loss = self.criterion(v_outputs, v_answers)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.train_metrics.update('loss', loss.item())

            batch_size = images.size(0)
            total_samples += batch_size

            _, predicted = output.max(1)
            for i in range(batch_size):
                pred_id = predicted[i].item()
                pred_text = self.idx_to_ans.get(pred_id, "")
                gt_list = gt_answers_list[i]
                sum_vqa_acc += compute_vqa_accuracy(pred_text, gt_list)
                sum_anls += compute_anls(raw_answers[i], pred_text)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == len(self.data_loader):
                break

        vqa_acc = 100.0 * sum_vqa_acc / total_samples if total_samples > 0 else 0
        anls = sum_anls / total_samples if total_samples > 0 else 0

        log = self.train_metrics.result()
        log['vqa_accuracy'] = vqa_acc
        log['anls'] = anls

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        
        correct_top1 = 0
        correct_top5 = 0
        total_valid_vocab = 0
        sum_vqa_acc = 0
        sum_vqa_top5 = 0
        sum_anls = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                images = batch['image'].to(self.device)
                questions = batch['question']
                answers = batch['answer'].to(self.device)
                raw_answers = batch['raw_answer']
                gt_answers_list = batch.get('answers', [[ans] for ans in raw_answers])

                if isinstance(gt_answers_list, list) and len(gt_answers_list) > 0 and isinstance(gt_answers_list[0], (list, tuple)):
                    gt_answers_list = list(zip(*gt_answers_list))

                batch_size = images.size(0)
                total_samples += batch_size
                output = self.model(images, questions)

                valid_mask = answers != -1
                if valid_mask.any():
                    v_outputs = output[valid_mask]
                    v_answers = answers[valid_mask]
                    loss = self.criterion(v_outputs, v_answers)

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    self.valid_metrics.update('loss', loss.item())

                    _, predicted = v_outputs.max(1)
                    correct_top1 += predicted.eq(v_answers).sum().item()

                    _, top5_preds = v_outputs.topk(min(5, v_outputs.size(1)), 1)
                    correct_top5 += (top5_preds == v_answers.view(-1, 1)).sum().item()

                    total_valid_vocab += valid_mask.sum().item()

                k = min(5, output.size(1))
                _, top5_all = output.topk(k, 1)

                for i in range(batch_size):
                    pred_id = top5_all[i, 0].item()
                    pred_text_top1 = self.idx_to_ans.get(pred_id, "")
                    gt_list = gt_answers_list[i]

                    sum_vqa_acc += compute_vqa_accuracy(pred_text_top1, gt_list)
                    vqa_scores_top5 = [compute_vqa_accuracy(self.idx_to_ans.get(top5_all[i, j].item(), ""), gt_list) for j in range(k)]
                    sum_vqa_top5 += max(vqa_scores_top5)
                    sum_anls += compute_anls(raw_answers[i], pred_text_top1)

        acc_top1 = 100.0 * correct_top1 / total_valid_vocab if total_valid_vocab > 0 else 0
        acc_top5 = 100.0 * correct_top5 / total_valid_vocab if total_valid_vocab > 0 else 0
        vqa_acc = 100.0 * sum_vqa_acc / total_samples if total_samples > 0 else 0
        vqa_top5 = 100.0 * sum_vqa_top5 / total_samples if total_samples > 0 else 0
        anls = sum_anls / total_samples if total_samples > 0 else 0

        log = self.valid_metrics.result()
        log.update({'accuracy_top1': acc_top1, 'accuracy_top5': acc_top5,
                   'vqa_accuracy': vqa_acc, 'vqa_accuracy_top5': vqa_top5, 'anls': anls})
        return log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = len(self.data_loader)
        return base.format(current, total, 100.0 * current / total)
