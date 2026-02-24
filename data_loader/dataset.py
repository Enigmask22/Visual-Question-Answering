import json
import os
from PIL import Image
from torch.utils.data import Dataset


class VQADataset(Dataset):
    """
    Visual Question Answering Dataset
    """
    def __init__(self, json_path, data_dir, transform=None, vocab=None):
        """
        Args:
            json_path: path to JSON file containing annotations
            data_dir: directory containing images
            transform: optional transform to be applied on images
            vocab: answer vocabulary dictionary (answer -> index)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.data_dir = data_dir
        self.transform = transform
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        img_path = os.path.join(self.data_dir, item['image_path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Question
        question = item['question']

        # Answer
        answer = item['answer']
        answer_idx = self.vocab.get(answer, -1) if self.vocab else -1

        # Multiple answers for VQA Accuracy
        answers = item.get('answers', [answer])

        return {
            'image': image,
            'question': question,
            'answer': answer_idx,
            'raw_answer': answer,
            'answers': answers
        }
