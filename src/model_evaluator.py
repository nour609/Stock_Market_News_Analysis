from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List

class ModelEvaluator:
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate model performance metrics including AUC as specified in the dataset requirements
        """
        # Calculate AUC (primary metric as per requirements)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        # Convert probabilities to class predictions for other metrics
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate additional metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.metrics = {
            'auc': auc_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        return self.metrics
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """
        Plot ROC curve for visual evaluation of model performance
        """
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {self.metrics["auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('roc_curve.png')
        plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot confusion matrix for visual evaluation
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.colorbar()
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    def perform_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray, 
                               cv: int = 5) -> Tuple[float, float]:
        """
        Perform k-fold cross-validation and return mean and std of AUC scores
        """
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        return cv_scores.mean(), cv_scores.std()