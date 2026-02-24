import torch


def compute_vqa_accuracy(prediction, ground_truth_answers):
    """VQA accuracy: correct if at least 3 annotators agree"""
    if not ground_truth_answers:
        return 0.0
    count = 0
    pred = prediction.lower().strip()
    for ans in ground_truth_answers:
        if ans.lower().strip() == pred:
            count += 1
    return min(1.0, count / 3.0)


def levenshtein_distance(s1, s2):
    """Compute edit distance between two strings"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def compute_anls(target, prediction, threshold=0.5):
    """Compute Average Normalized Levenshtein Similarity"""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    if not target or not prediction:
        return 0.0 if target != prediction else 1.0
    dist = levenshtein_distance(target, prediction)
    max_len = max(len(target), len(prediction))
    score = 1.0 - float(dist) / float(max_len)
    return score if score >= threshold else 0.0


def accuracy_top1(output, target):
    """Compute top-1 accuracy"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        return correct_1.mul_(100.0 / batch_size).item()


def accuracy_top5(output, target):
    """Compute top-5 accuracy"""
    with torch.no_grad():
        batch_size = target.size(0)
        k = min(5, output.size(1))
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()
