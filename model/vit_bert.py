import torch
import torch.nn as nn
import timm
from transformers import AutoTokenizer, AutoModel

from base import BaseModel


class ViTBERTBaseline(BaseModel):
    """
    ViT + BERT baseline for VQA
    Uses Vision Transformer for image encoding and BERT for question encoding
    """
    def __init__(self, num_classes, bert_model='vinai/phobert-base', hidden_dim=512, 
                 img_size=224, max_len=64):
        """
        Args:
            num_classes: number of answer classes
            bert_model: name of pretrained BERT model
            hidden_dim: hidden dimension for fusion layer
            img_size: size of input images
            max_len: maximum question length
        """
        super().__init__()

        self.max_len = max_len

        # ViT encoder
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True,
                                     num_classes=0, img_size=img_size)
        vit_output_dim = self.vit.num_features

        # BERT encoder
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert = AutoModel.from_pretrained(bert_model)
        bert_output_dim = self.bert.config.hidden_size

        # Freeze encoders
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.bert.parameters():
            param.requires_grad = False

        # Bottleneck: Fusion layer
        fusion_dim = vit_output_dim + bert_output_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Fully Connected (FC): Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode_questions(self, questions, device):
        """
        Encode questions using BERT
        
        Args:
            questions: list of question strings
            device: device to put tensors on
        
        Returns:
            tensor of BERT [CLS] embeddings (batch_size, bert_dim)
        """
        encoding = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # BERT encoding
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Get [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding

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
        # Image features: ViT encoding
        img_features = self.vit(images)

        # Text features: BERT encoding
        text_features = self.encode_questions(questions, images.device)

        # Bottleneck: Fusion
        combined = torch.cat([img_features, text_features], dim=1)
        fused_features = self.bottleneck(combined)

        # Fully Connected (FC): Classifier
        output = self.classifier(fused_features)

        return output
