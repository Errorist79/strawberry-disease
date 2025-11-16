"""
Risk calculation logic for strawberry diseases.

This module defines the risk scoring system based on disease type and model confidence.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class DiseaseRiskLevel(Enum):
    """
    Base risk levels for different disease types.

    Risk scores are on a 0-100 scale, where:
    - 0-30: Low risk
    - 31-60: Medium risk
    - 61-80: High risk
    - 81-100: Critical risk
    """

    # Healthy or no detection
    HEALTHY = (0, 10)

    # Mild leaf spots
    LEAF_SPOT = (30, 50)
    ANGULAR_LEAFSPOT = (30, 50)

    # Moderate powdery mildew
    POWDERY_MILDEW_LEAF = (40, 65)
    POWDERY_MILDEW_FRUIT = (50, 70)

    # Severe and rapidly spreading diseases
    GRAY_MOLD = (60, 90)
    ANTHRACNOSE_FRUIT_ROT = (65, 95)
    BLOSSOM_BLIGHT = (70, 95)


# Mapping from disease class labels to risk levels
DISEASE_RISK_MAPPING: Dict[str, DiseaseRiskLevel] = {
    "healthy": DiseaseRiskLevel.HEALTHY,
    "leaf_spot": DiseaseRiskLevel.LEAF_SPOT,
    "angular_leafspot": DiseaseRiskLevel.ANGULAR_LEAFSPOT,
    "powdery_mildew_leaf": DiseaseRiskLevel.POWDERY_MILDEW_LEAF,
    "powdery_mildew_fruit": DiseaseRiskLevel.POWDERY_MILDEW_FRUIT,
    "gray_mold": DiseaseRiskLevel.GRAY_MOLD,
    "anthracnose_fruit_rot": DiseaseRiskLevel.ANTHRACNOSE_FRUIT_ROT,
    "blossom_blight": DiseaseRiskLevel.BLOSSOM_BLIGHT,
}


@dataclass
class PredictionRisk:
    """Risk assessment for a single prediction."""

    class_label: str
    confidence: float
    risk_score: int
    base_risk_range: tuple[int, int]


class RiskCalculator:
    """
    Calculate risk scores for disease predictions.

    The risk score combines:
    1. Base risk level associated with the disease type
    2. Model confidence (higher confidence = higher risk)
    """

    def __init__(self, risk_mapping: Dict[str, DiseaseRiskLevel] = None):
        """
        Initialize risk calculator.

        Args:
            risk_mapping: Custom disease risk mapping. Uses default if None.
        """
        self.risk_mapping = risk_mapping or DISEASE_RISK_MAPPING

    def calculate_risk(self, class_label: str, confidence: float) -> int:
        """
        Calculate risk score for a single prediction.

        Args:
            class_label: Disease class label
            confidence: Model confidence score (0-1)

        Returns:
            Risk score (0-100)

        Raises:
            ValueError: If class_label is not recognized or confidence is invalid
        """
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

        # Get risk level for this disease class
        risk_level = self.risk_mapping.get(class_label.lower())

        if risk_level is None:
            # Unknown disease class - treat as medium risk
            min_risk, max_risk = 40, 60
        else:
            min_risk, max_risk = risk_level.value

        # Scale risk score based on confidence
        # Higher confidence = risk closer to max, lower confidence = closer to min
        risk_score = int(min_risk + (max_risk - min_risk) * confidence)

        # Ensure risk score stays within 0-100 bounds
        return max(0, min(100, risk_score))

    def calculate_detailed_risk(self, class_label: str, confidence: float) -> PredictionRisk:
        """
        Calculate detailed risk assessment.

        Args:
            class_label: Disease class label
            confidence: Model confidence score (0-1)

        Returns:
            PredictionRisk object with detailed risk information
        """
        risk_level = self.risk_mapping.get(class_label.lower())
        base_range = risk_level.value if risk_level else (40, 60)
        risk_score = self.calculate_risk(class_label, confidence)

        return PredictionRisk(
            class_label=class_label,
            confidence=confidence,
            risk_score=risk_score,
            base_risk_range=base_range,
        )

    def get_risk_category(self, risk_score: int) -> str:
        """
        Get risk category based on score.

        Args:
            risk_score: Risk score (0-100)

        Returns:
            Risk category string
        """
        if risk_score <= 30:
            return "low"
        elif risk_score <= 60:
            return "medium"
        elif risk_score <= 80:
            return "high"
        else:
            return "critical"

    def aggregate_image_risk(self, predictions: list[tuple[str, float]]) -> dict:
        """
        Calculate aggregated risk for an image with multiple predictions.

        Args:
            predictions: List of (class_label, confidence) tuples

        Returns:
            Dictionary with max_risk, dominant_disease, and prediction_count
        """
        if not predictions:
            # No detections = healthy
            return {
                "max_risk": self.calculate_risk("healthy", 1.0),
                "dominant_disease": "healthy",
                "prediction_count": 0,
            }

        # Calculate risk for each prediction
        risks = [
            (label, conf, self.calculate_risk(label, conf)) for label, conf in predictions
        ]

        # Find prediction with highest risk
        max_prediction = max(risks, key=lambda x: x[2])

        return {
            "max_risk": max_prediction[2],
            "dominant_disease": max_prediction[0],
            "prediction_count": len(predictions),
        }
