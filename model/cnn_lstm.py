import torch
import torch.nn as nn
import timm

from base import BaseModel


class CNNLSTMBaseline(BaseModel):
    """
    CNN + LSTM baseline for VQA
    Uses ResNet50 for image encoding and LSTM for question encoding
    """
    def __init__(self, num_classes, vocab_size=None, word_to_idx=None, hidden_dim=512, 
                 lstm_layers=2, embed_dim=300, max_len=64, train_questions=None):
        """
        Args:
            num_classes: number of answer classes
            vocab_size: size of word vocabulary
            word_to_idx: dictionary mapping word -> index
            hidden_dim: hidden dimension for LSTM and fusion layer
            lstm_layers: number of LSTM layers
            embed_dim: embedding dimension
            max_len: maximum question length
            train_questions: list of training questions for building vocabulary
        """
        super().__init__()

        # Build vocabulary if not provided
        if word_to_idx is None and train_questions is not None:
            word_set = set()
            for q in train_questions:
                words = q.lower().split()
                word_set.update(words)

            word_to_idx = {'<PAD>': 0, '<UNK>': 1}
            for idx, word in enumerate(sorted(word_set), start=2):
                word_to_idx[word] = idx

            vocab_size = len(word_to_idx)

        self.word_to_idx = word_to_idx if word_to_idx is not None else {}
        self.vocab_size = vocab_size if vocab_size is not None else 2
        self.max_len = max_len

        # CNN encoder: ResNet pretrained
        self.cnn = timm.create_model('resnet50', pretrained=True, num_classes=0)
        # Freeze CNN backbone
        for param in self.cnn.parameters():
            param.requires_grad = False
        cnn_output_dim = self.cnn.num_features

        # Text encoder: Embedding + LSTM
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, lstm_layers,
                           batch_first=True, bidirectional=True)

        lstm_output_dim = hidden_dim * 2

        # Bottleneck: Fusion layer
        fusion_dim = cnn_output_dim + lstm_output_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Fully Connected (FC): Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def tokenize(self, questions):
        """
        Tokenize questions into indices
        
        Args:
            questions: list of question strings
        
        Returns:
            tensor of token indices (batch_size, max_len)
        """
        batch_tokens = []
        for q in questions:
            words = q.lower().split()
            tokens = [self.word_to_idx.get(w, 1) for w in words]  # 1 is <UNK>

            # Padding or truncation
            if len(tokens) < self.max_len:
                tokens = tokens + [0] * (self.max_len - len(tokens))
            else:
                tokens = tokens[:self.max_len]

            batch_tokens.append(tokens)

        return torch.LongTensor(batch_tokens)

    def forward(self, images, questions):
        """
        Forward pass
        
        Args:
            images: batch of images (batch_size, 3, img_size, img_size)
            questions: list of question strings
        
        Returns:
            logits (batch_size, num_classes)
        """
        # Feature Extraction (FE)
        # Image features: CNN encoding
        img_features = self.cnn(images)

        # Text features: LSTM encoding
        question_tokens = self.tokenize(questions).to(images.device)
        question_embeds = self.embedding(question_tokens)
        lstm_out, (hidden, cell) = self.lstm(question_embeds)
        
        # Concatenate last hidden states from both directions
        text_features = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # Bottleneck: Fusion
        combined = torch.cat([img_features, text_features], dim=1)
        fused_features = self.bottleneck(combined)

        # Fully Connected (FC): Classifier
        output = self.classifier(fused_features)

        return output
