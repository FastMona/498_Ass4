"""Fuzzy ART classifier for MNIST.

Why: complements CNN/FFN with a template-based model that still exposes
get_features for Mahalanobis OOD; changing vigilance or feature shape requires
matching OOD params and artifact regeneration.
"""

import torch
from torch import nn
import torch.nn.functional as F


class FuzzyARTClassifier(nn.Module):
    """
    Fuzzy Adaptive Resonance Theory (Fuzzy ART) Network for MNIST classification.
    
    ART networks are biologically-inspired neural architectures that perform
    stable incremental learning without catastrophic forgetting. This implementation
    uses Fuzzy ART which handles continuous-valued inputs.
    
    Key characteristics:
    - Self-organizing clustering with vigilance parameter
    - Stable learning (no catastrophic forgetting)
    - Fast online learning capability
    - Complement coding for better pattern matching
    
    Architecture:
    - Input layer: 28x28 = 784 dimensions (flattened MNIST images)
    - Category layer: Dynamically grown template nodes (max_categories)
    - Vigilance parameter controls specificity vs generalization
    """
    
    def __init__(self, input_dim=784, max_categories=100, vigilance=0.75, 
                 learning_rate=0.5, choice_alpha=0.001,
                 count_penalty_gamma=0.01, max_category_count=None):
        """
        Initialize Fuzzy ART network.
        
        Args:
            input_dim: Dimension of input patterns (default 784 for 28x28 MNIST)
            max_categories: Maximum number of category nodes to create
            vigilance: Vigilance parameter (0-1). Higher = more specific categories
            learning_rate: Learning rate for template updates (0-1)
            choice_alpha: Choice parameter for category selection (small positive)
            count_penalty_gamma: Penalty strength for overused categories
            max_category_count: Hard cap on patterns per category (None disables)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.max_categories = max_categories
        self.vigilance = vigilance
        self.learning_rate = learning_rate
        self.choice_alpha = choice_alpha
        self.count_penalty_gamma = count_penalty_gamma
        self.max_category_count = max_category_count
        
        # Complement coded input dimension (doubles input size)
        # Complement coding appends (1-x) to input x, creating [x, 1-x]
        # This prevents category proliferation and ensures symmetric pattern matching
        # e.g., [0.8, 0.2] becomes [0.8, 0.2, 0.2, 0.8] so both presence AND absence of features matter
        self.coded_dim = input_dim * 2
        
        # Category templates (weights) - initialized to 1 for uncommitted nodes
        # Shape: [max_categories, coded_dim]
        self.register_buffer('templates', torch.ones(max_categories, self.coded_dim))
        
        # Track which categories are committed (already have been used)
        self.register_buffer('committed', torch.zeros(max_categories, dtype=torch.bool))
        
        # Track category labels for classification (mapping category -> digit class)
        self.register_buffer('category_labels', torch.full((max_categories,), -1, dtype=torch.long))
        
        # Count patterns assigned to each category (for confidence/voting)
        self.register_buffer('category_counts', torch.zeros(max_categories, dtype=torch.long))
        
        # Track number of committed categories (must be a buffer to persist)
        self.register_buffer('_num_committed', torch.tensor(0, dtype=torch.long))
    
    @property
    def num_committed(self):
        """Property to access num_committed as an integer."""
        return self._num_committed.item()
    
    @num_committed.setter
    def num_committed(self, value):
        """Property setter to update num_committed buffer."""
        self._num_committed.fill_(value)
        
    def complement_code(self, x):
        """
        Apply complement coding to input patterns.
        
        Complement coding doubles the input dimension by appending (1 - x),
        which helps prevent category proliferation and improves matching.
        
        Args:
            x: Input tensor [batch_size, input_dim], values normalized to [0, 1]
        
        Returns:
            Complement coded tensor [batch_size, coded_dim]
        """
        return torch.cat([x, 1 - x], dim=-1)
    
    def category_choice(self, coded_input, committed_mask):
        """
        Compute category choice function T_j for all categories.
        
        T_j = |x ∧ w_j| / (α + |w_j| + γ * count_j)
        
        where ∧ is fuzzy AND (element-wise minimum), |·| is L1 norm,
        and γ * count_j penalizes mega-categories to prevent black holes.
        Higher values indicate better match.
        
        Args:
            coded_input: Complement coded input [batch_size, coded_dim]
            committed_mask: Boolean mask of committed categories [max_categories]
        
        Returns:
            Choice values [batch_size, max_categories]
        """
        # Fuzzy AND: element-wise minimum
        fuzzy_and = torch.minimum(
            coded_input.unsqueeze(1),  # [batch, 1, coded_dim]
            self.templates.unsqueeze(0)  # [1, max_categories, coded_dim]
        )
        
        # Numerator: |x ∧ w_j|
        numerator = fuzzy_and.sum(dim=-1)  # [batch, max_categories]
        
        # Denominator: α + |w_j| + γ * count_j
        # The count penalty discourages selecting overused categories (prevents black holes)
        count_penalty = self.count_penalty_gamma * self.category_counts.float()
        denominator = self.choice_alpha + self.templates.sum(dim=-1) + count_penalty  # [max_categories]
        
        # Choice function
        choice_values = numerator / denominator.unsqueeze(0)
        
        # Mask out uncommitted categories (set to -inf so they're not selected)
        choice_values = choice_values.masked_fill(~committed_mask.unsqueeze(0), float('-inf'))
        
        return choice_values
    
    def match_function(self, coded_input, category_idx):
        """
        Compute match function (vigilance test).
        
        Match = |x ∧ w_j| / |x|
        
        This measures how well the category template matches the input.
        Must be >= vigilance threshold to accept the category.
        
        Args:
            coded_input: Complement coded input [batch_size, coded_dim]
            category_idx: Category indices to test [batch_size]
        
        Returns:
            Match values [batch_size]
        """
        # Select templates for chosen categories
        selected_templates = self.templates[category_idx]  # [batch_size, coded_dim]
        
        # Fuzzy AND
        fuzzy_and = torch.minimum(coded_input, selected_templates)
        
        # Match = |x ∧ w_j| / |x|
        numerator = fuzzy_and.sum(dim=-1)
        denominator = coded_input.sum(dim=-1)
        
        return numerator / (denominator + 1e-10)
    
    def update_template(self, coded_input, category_idx):
        """
        Update category template using fast learning rule.
        
        w_j(new) = β * (x ∧ w_j(old)) + (1 - β) * w_j(old)
        
        Args:
            coded_input: Complement coded input [coded_dim]
            category_idx: Category index to update
        """
        old_template = self.templates[category_idx]
        fuzzy_and = torch.minimum(coded_input, old_template)
        new_template = self.learning_rate * fuzzy_and + (1 - self.learning_rate) * old_template
        self.templates[category_idx] = new_template
    
    def train_pattern(self, x, label):
        """
        Train on a single pattern using ART learning algorithm.
        
        Args:
            x: Input pattern [input_dim], normalized to [0, 1]
            label: Ground truth class label (0-9 for MNIST)
        
        Returns:
            Selected category index
        """
        # Ensure label is a plain int (avoid tensor truthiness issues)
        label_int = int(label.item()) if torch.is_tensor(label) else int(label)

        # Complement code the input
        coded_input = self.complement_code(x.unsqueeze(0)).squeeze(0)
        
        # Resonance search loop
        reset_categories = torch.zeros(self.max_categories, dtype=torch.bool, device=x.device)
        max_resets = min(self.num_committed, self.max_categories // 2)  # Prevent infinite loops
        
        while True:
            # Create committed mask excluding reset categories
            available_mask = self.committed & ~reset_categories
            
            # If no committed categories available, create new one
            if not available_mask.any():
                if self.num_committed < self.max_categories:
                    # Commit new category
                    category_idx = self.num_committed
                    self.committed[category_idx] = True
                    self.templates[category_idx] = coded_input
                    self.category_labels[category_idx] = label_int
                    self.category_counts[category_idx] = 1
                    self.num_committed += 1
                    return category_idx
                else:
                    # All categories exhausted - fall back to unsupervised best match (avoid infinite loop)
                    category_idx = torch.argmax(
                        self.category_choice(coded_input.unsqueeze(0), self.committed).squeeze(0)
                    ).item()
                    self.update_template(coded_input, category_idx)
                    self.category_counts[category_idx] += 1
                    return category_idx
            
            # Find best matching category among available
            choice_values = self.category_choice(coded_input.unsqueeze(0), available_mask).squeeze(0)
            category_idx = torch.argmax(choice_values).item()
            
            # Vigilance test (similarity-based)
            match_value = self.match_function(coded_input.unsqueeze(0), 
                                             torch.tensor([category_idx], device=x.device)).item()
            
            # CRITICAL: Supervised vigilance - reject if labels don't match
            # This prevents cross-class template contamination and mega-category collapse
            category_label = self.category_labels[category_idx].item()
            label_match = (category_label == -1) or (category_label == label_int)
            
            maxed_out = False
            if self.max_category_count is not None:
                maxed_out = self.category_counts[category_idx].item() >= self.max_category_count

            if match_value >= self.vigilance and label_match and not maxed_out:
                # Resonance achieved! Update template
                self.update_template(coded_input, category_idx)
                # IMPORTANT: Only update label on first assignment (initial commit)
                # Don't overwrite the label for subsequent pattern matches
                if self.category_labels[category_idx] == -1:
                    self.category_labels[category_idx] = label_int
                self.category_counts[category_idx] += 1
                return category_idx
            else:
                # Mismatch - reset this category and try again
                reset_categories[category_idx] = True
                
                # If we've rejected all committed categories, create a new one (supervised ART)
                if not (self.committed & ~reset_categories).any() and self.num_committed < self.max_categories:
                    category_idx = self.num_committed
                    self.committed[category_idx] = True
                    self.templates[category_idx] = coded_input
                    self.category_labels[category_idx] = label_int
                    self.category_counts[category_idx] = 1
                    self.num_committed += 1
                    return category_idx
                
                # Safety valve: if too many resets, fall back to unsupervised matching
                if reset_categories.sum().item() > max_resets:
                    category_idx = torch.argmax(
                        self.category_choice(coded_input.unsqueeze(0), self.committed).squeeze(0)
                    ).item()
                    self.update_template(coded_input, category_idx)
                    self.category_counts[category_idx] += 1
                    return category_idx
    
    def forward(self, x, training=False, labels=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input images [batch_size, 1, 28, 28] or [batch_size, 784]
            training: If True, perform learning on each pattern
            labels: Ground truth labels (required if training=True)
        
        Returns:
            If training: category indices [batch_size]
            If inference: class logits [batch_size, 10]
        """
        # Flatten if necessary
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        
        # Normalize to [0, 1]
        x = (x - x.min()) / (x.max() - x.min() + 1e-10)
        
        if training:
            if labels is None:
                raise ValueError("Labels required for training mode")
            
            # Train on each pattern sequentially
            selected_categories = []
            for i in range(x.size(0)):
                cat_idx = self.train_pattern(x[i], labels[i])
                selected_categories.append(cat_idx)
            
            return torch.tensor(selected_categories, device=x.device)
        else:
            # Inference: find best matching category and return its label
            return self.predict(x)
    
    def predict(self, x):
        """
        Predict class labels for input patterns.
        
        Args:
            x: Input patterns [batch_size, input_dim], normalized to [0, 1]
        
        Returns:
            Class logits [batch_size, 10] for compatibility with CrossEntropyLoss
        """
        batch_size = x.size(0)
        coded_input = self.complement_code(x)
        
        if self.num_committed == 0:
            # No trained categories, return uniform distribution
            return torch.zeros(batch_size, 10, device=x.device)
        
        # Compute choice values for all committed categories
        choice_values = self.category_choice(coded_input, self.committed)
        
        # For each input, find best matching category
        best_categories = torch.argmax(choice_values, dim=1)
        
        # Convert to class predictions with voting mechanism
        logits = torch.zeros(batch_size, 10, device=x.device)
        
        for i in range(batch_size):
            cat_idx = best_categories[i].item()
            
            # Use the choice value (match score) as confidence
            choice_score = choice_values[i, cat_idx].item()
            
            # Get all categories with same label (voting)
            pred_label = self.category_labels[cat_idx].item()
            
            if pred_label >= 0:  # Valid label
                # Voting: accumulate choice scores from all categories with same label
                for j in range(self.num_committed):
                    if self.category_labels[j] == pred_label:
                        logits[i, pred_label] += choice_values[i, j]
            
        return logits
    
    def get_features(self, x):
        """
        Extract feature representation for OOD detection.
        
        Returns the best matching category's template as the feature vector.
        
        Args:
            x: Input images [batch_size, 1, 28, 28] or [batch_size, 784]
        
        Returns:
            Feature vectors [batch_size, coded_dim]
        """
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        
        x = (x - x.min()) / (x.max() - x.min() + 1e-10)
        coded_input = self.complement_code(x)
        
        if self.num_committed == 0:
            return coded_input
        
        # Find best matching categories
        choice_values = self.category_choice(coded_input, self.committed)
        best_categories = torch.argmax(choice_values, dim=1)
        
        # Return matched template as features
        return self.templates[best_categories]


class FuzzyARTMAPClassifier(nn.Module):
    """
    Fuzzy ARTMAP classifier for supervised learning.

    Adds match tracking to enforce label-consistent category selection.
    """

    def __init__(
        self,
        input_dim=784,
        max_categories=100,
        vigilance=0.75,
        learning_rate=0.5,
        choice_alpha=0.001,
        count_penalty_gamma=0.01,
        max_category_count=6000,
        match_tracking_epsilon=1e-3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.max_categories = max_categories
        self.vigilance = vigilance
        self.learning_rate = learning_rate
        self.choice_alpha = choice_alpha
        self.count_penalty_gamma = count_penalty_gamma
        self.max_category_count = max_category_count
        self.match_tracking_epsilon = match_tracking_epsilon

        self.coded_dim = input_dim * 2

        self.register_buffer('templates', torch.ones(max_categories, self.coded_dim))
        self.register_buffer('committed', torch.zeros(max_categories, dtype=torch.bool))
        self.register_buffer('category_labels', torch.full((max_categories,), -1, dtype=torch.long))
        self.register_buffer('category_counts', torch.zeros(max_categories, dtype=torch.long))

        # Track number of committed categories (must be a buffer to persist)
        self.register_buffer('_num_committed', torch.tensor(0, dtype=torch.long))
    
    @property
    def num_committed(self):
        """Property to access num_committed as an integer."""
        return self._num_committed.item()
    
    @num_committed.setter
    def num_committed(self, value):
        """Property setter to update num_committed buffer."""
        self._num_committed.fill_(value)

    def complement_code(self, x):
        return torch.cat([x, 1 - x], dim=-1)

    def category_choice(self, coded_input, committed_mask):
        fuzzy_and = torch.minimum(
            coded_input.unsqueeze(1),
            self.templates.unsqueeze(0),
        )

        numerator = fuzzy_and.sum(dim=-1)
        count_penalty = self.count_penalty_gamma * self.category_counts.float()
        denominator = self.choice_alpha + self.templates.sum(dim=-1) + count_penalty
        choice_values = numerator / denominator.unsqueeze(0)
        choice_values = choice_values.masked_fill(~committed_mask.unsqueeze(0), float('-inf'))
        return choice_values

    def match_function(self, coded_input, category_idx):
        selected_templates = self.templates[category_idx]
        fuzzy_and = torch.minimum(coded_input, selected_templates)
        numerator = fuzzy_and.sum(dim=-1)
        denominator = coded_input.sum(dim=-1)
        return numerator / (denominator + 1e-10)

    def update_template(self, coded_input, category_idx):
        old_template = self.templates[category_idx]
        fuzzy_and = torch.minimum(coded_input, old_template)
        new_template = self.learning_rate * fuzzy_and + (1 - self.learning_rate) * old_template
        self.templates[category_idx] = new_template

    def train_pattern(self, x, label):
        label_int = int(label.item()) if torch.is_tensor(label) else int(label)
        coded_input = self.complement_code(x.unsqueeze(0)).squeeze(0)

        reset_categories = torch.zeros(self.max_categories, dtype=torch.bool, device=x.device)
        local_vigilance = self.vigilance
        max_resets = min(self.num_committed, self.max_categories // 2)

        while True:
            available_mask = self.committed & ~reset_categories

            if not available_mask.any():
                if self.num_committed < self.max_categories:
                    category_idx = self.num_committed
                    self.committed[category_idx] = True
                    self.templates[category_idx] = coded_input
                    self.category_labels[category_idx] = label_int
                    self.category_counts[category_idx] = 1
                    self.num_committed += 1
                    return category_idx
                category_idx = torch.argmax(
                    self.category_choice(coded_input.unsqueeze(0), self.committed).squeeze(0)
                ).item()
                self.update_template(coded_input, category_idx)
                self.category_counts[category_idx] += 1
                return category_idx

            choice_values = self.category_choice(coded_input.unsqueeze(0), available_mask).squeeze(0)
            category_idx = torch.argmax(choice_values).item()

            match_value = self.match_function(
                coded_input.unsqueeze(0),
                torch.tensor([category_idx], device=x.device),
            ).item()

            category_label = self.category_labels[category_idx].item()
            label_match = (category_label == -1) or (category_label == label_int)

            maxed_out = False
            if self.max_category_count is not None:
                maxed_out = self.category_counts[category_idx].item() >= self.max_category_count

            if match_value >= local_vigilance and label_match and not maxed_out:
                self.update_template(coded_input, category_idx)
                if self.category_labels[category_idx] == -1:
                    self.category_labels[category_idx] = label_int
                self.category_counts[category_idx] += 1
                return category_idx

            reset_categories[category_idx] = True

            # ARTMAP match tracking: small conservative vigilance increase on label mismatch
            # Avoid aggressive vigilance escalation that causes category proliferation
            if not label_match:
                local_vigilance = min(self.vigilance + 0.03, 1.0)

            if not (self.committed & ~reset_categories).any() and self.num_committed < self.max_categories:
                category_idx = self.num_committed
                self.committed[category_idx] = True
                self.templates[category_idx] = coded_input
                self.category_labels[category_idx] = label_int
                self.category_counts[category_idx] = 1
                self.num_committed += 1
                return category_idx

            if reset_categories.sum().item() > max_resets:
                category_idx = torch.argmax(
                    self.category_choice(coded_input.unsqueeze(0), self.committed).squeeze(0)
                ).item()
                self.update_template(coded_input, category_idx)
                self.category_counts[category_idx] += 1
                return category_idx

    def forward(self, x, training=False, labels=None):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)

        x = (x - x.min()) / (x.max() - x.min() + 1e-10)

        if training:
            if labels is None:
                raise ValueError("Labels required for training mode")
            selected_categories = []
            for i in range(x.size(0)):
                cat_idx = self.train_pattern(x[i], labels[i])
                selected_categories.append(cat_idx)
            return torch.tensor(selected_categories, device=x.device)

        return self.predict(x)

    def predict(self, x):
        batch_size = x.size(0)
        coded_input = self.complement_code(x)

        if self.num_committed == 0:
            return torch.zeros(batch_size, 10, device=x.device)

        choice_values = self.category_choice(coded_input, self.committed)
        best_categories = torch.argmax(choice_values, dim=1)

        logits = torch.zeros(batch_size, 10, device=x.device)
        for i in range(batch_size):
            cat_idx = best_categories[i].item()
            pred_label = self.category_labels[cat_idx].item()
            if pred_label >= 0:
                for j in range(self.num_committed):
                    if self.category_labels[j] == pred_label:
                        logits[i, pred_label] += choice_values[i, j]
        return logits

    def get_features(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)

        x = (x - x.min()) / (x.max() - x.min() + 1e-10)
        coded_input = self.complement_code(x)

        if self.num_committed == 0:
            return coded_input

        choice_values = self.category_choice(coded_input, self.committed)
        best_categories = torch.argmax(choice_values, dim=1)
        return self.templates[best_categories]
