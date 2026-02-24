import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

from base import BaseModel


class MLPBaseline(BaseModel):
    """
    Simple MLP baseline for VQA
    Uses TF-IDF for question encoding and flattened image pixels
    """
    def __init__(self, num_classes, img_size=224, tfidf_dim=1000, hidden_dim=512, train_questions=None):
        """
        Args:
            num_classes: number of answer classes
            img_size: size of input images
            tfidf_dim: dimensionality of TF-IDF features
            hidden_dim: hidden dimension for fusion layer
            train_questions: list of training questions for fitting TF-IDF vectorizer
        """
        super().__init__()

        self.img_size = img_size
        self.tfidf_dim = tfidf_dim
        self.img_flatten_dim = 3 * img_size * img_size

        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=tfidf_dim)
        
        # Fit vectorizer if training questions provided
        if train_questions is not None:
            self.tfidf_vectorizer.fit(train_questions)

        # Bottleneck: Fusion layer
        fusion_dim = self.img_flatten_dim + tfidf_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Fully Connected (FC): Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def encode_questions(self, questions):
        """
        Encode questions using TF-IDF
        
        Args:
            questions: list of question strings
        
        Returns:
            tensor of TF-IDF features (batch_size, tfidf_dim)
        """
        tfidf_features = self.tfidf_vectorizer.transform(questions).toarray()
        return torch.FloatTensor(tfidf_features)

    def forward(self, images, questions):
        """
        Forward pass
        
        Args:
            images: batch of images (batch_size, 3, img_size, img_size)
            questions: list of question strings
        
        Returns:
            logits (batch_size, num_classes)
        """
        batch_size = images.size(0)

        # Feature Extraction (FE)
        # Flatten image
        img_features = images.view(batch_size, -1)
        
        # TF-IDF encoding for questions
        text_features = self.encode_questions(questions).to(images.device)

        # Bottleneck: Fusion
        combined = torch.cat([img_features, text_features], dim=1)
        fused_features = self.bottleneck(combined)

        # Fully Connected (FC): Classifier
        output = self.classifier(fused_features)

        return output
