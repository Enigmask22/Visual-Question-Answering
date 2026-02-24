import json
import os
from collections import Counter
from torch.utils.data import DataLoader
from torchvision import transforms

from base import BaseDataLoader
from .dataset import VQADataset


def build_answer_vocab(train_json, val_json=None, max_answers=10000):
    """
    Build answer vocabulary from training and validation data
    
    Args:
        train_json: path to training JSON file
        val_json: path to validation JSON file (optional)
        max_answers: maximum number of answers to keep in vocabulary
    
    Returns:
        vocab: dictionary mapping answer -> index
        idx_to_ans: dictionary mapping index -> answer
    """
    all_answers = []
    
    # Load training data
    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    for item in train_data:
        all_answers.extend(item.get('answers', [item['answer']]))
    
    # Load validation data if provided
    if val_json is not None:
        with open(val_json, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        for item in val_data:
            all_answers.extend(item.get('answers', [item['answer']]))
    
    # Count answer frequencies
    answer_counts = Counter(all_answers)
    unique_answers = list(answer_counts.keys())
    
    # Keep only top max_answers
    if len(unique_answers) > max_answers:
        unique_answers = [ans for ans, _ in answer_counts.most_common(max_answers)]
    
    # Create vocabulary
    vocab = {ans: idx for idx, ans in enumerate(sorted(unique_answers))}
    idx_to_ans = {idx: ans for ans, idx in vocab.items()}
    
    return vocab, idx_to_ans


def get_transforms(img_size=224):
    """
    Get image transforms for training and validation
    
    Args:
        img_size: size to resize images to
    
    Returns:
        train_transform: transform for training data
        val_transform: transform for validation/test data
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


class VQADataLoader(DataLoader):
    """
    VQA data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, 
                 num_workers=1, training=True, vocab=None, img_size=224):
        """
        Args:
            data_dir: directory containing train.json, val.json, test.json
            batch_size: batch size
            shuffle: whether to shuffle data
            validation_split: fraction of training data to use for validation (0.0 means use separate val set)
            num_workers: number of workers for data loading
            training: whether this is for training (uses train.json) or testing (uses test.json/val.json)
            vocab: answer vocabulary (if None, will be built from data)
            img_size: image size for resizing
        """
        self.data_dir = data_dir
        self.img_size = img_size
        
        # Get transforms
        train_transform, val_transform = get_transforms(img_size)
        
        # Build vocabulary if not provided
        if vocab is None and training:
            train_json = os.path.join(data_dir, 'train.json')
            val_json = os.path.join(data_dir, 'val.json')
            vocab, idx_to_ans = build_answer_vocab(train_json, val_json)
            self.vocab = vocab
            self.idx_to_ans = idx_to_ans
        else:
            self.vocab = vocab
            # Build reverse mapping
            if vocab is not None:
                self.idx_to_ans = {idx: ans for ans, idx in vocab.items()}
            else:
                self.idx_to_ans = {}
        
        # Determine which JSON file to use
        if training:
            json_file = 'train.json'
            transform = train_transform
        else:
            # For testing, try test.json first, then val.json
            if os.path.exists(os.path.join(data_dir, 'test.json')):
                json_file = 'test.json'
            else:
                json_file = 'val.json'
            transform = val_transform
        
        # Create dataset
        self.dataset = VQADataset(
            json_path=os.path.join(data_dir, json_file),
            data_dir=data_dir,
            transform=transform,
            vocab=self.vocab
        )
        
        # Initialize DataLoader
        super().__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
