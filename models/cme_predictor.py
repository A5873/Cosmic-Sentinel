#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CME Trajectory Prediction Module

This module provides functionality for predicting coronal mass ejection (CME) trajectories
using deep learning models that combine CNN and LSTM architectures. It processes coronagraph
imagery, solar magnetogram data, and solar wind parameters to predict CME speed,
direction, arrival time, and impact probability.

The module includes:
1. Data preprocessing for coronagraph imagery and magnetogram data
2. CNN-LSTM model architecture for CME trajectory prediction
3. Prediction methods with confidence scores and arrival time estimation
4. Model evaluation metrics
5. Caching system for predictions
6. Integration with the space_weather module
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# For importing space_weather module
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from API.space_weather import SpaceWeatherMonitor, SpaceWeatherError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "models/cme_prediction/cme_predictor_v1.pth"
CACHE_DIR = "cache/cme_prediction"
CORONAGRAPH_DATA_PATH = "data/coronagraph"
MAGNETOGRAM_DATA_PATH = "data/magnetogram"
SOLAR_WIND_DATA_PATH = "data/solar_wind"
PREDICTION_HOURS = 72  # Predict CME arrival within 72 hours
CACHE_EXPIRATION = 6   # Cache expiration in hours

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CMEPredictionError(Exception):
    """Custom exception for CME prediction related errors."""
    pass


class CoronagraphPreprocessor:
    """
    Preprocessor for coronagraph imagery.
    
    This class handles loading, normalization, feature extraction, and 
    formatting of coronagraph imagery for input to CNN models.
    """
    
    def __init__(
        self, 
        data_path: str = CORONAGRAPH_DATA_PATH,
        image_size: Tuple[int, int] = (256, 256),
        instruments: List[str] = ["LASCO-C2", "LASCO-C3", "STEREO-A", "STEREO-B"]
    ):
        """
        Initialize the coronagraph preprocessor.
        
        Args:
                # Get magnetogram data
                magnetogram_tensor = self.magnetogram_sequences[idx] if idx < len(self.magnetogram_sequences) else torch.zeros(1, 1, 256, 256, dtype=torch.float32)
                
                # Get solar wind data
                solar_wind_tensor = self.solar_wind_sequences[idx] if idx < len(self.solar_wind_sequences) else torch.zeros(1, 24, 4, dtype=torch.float32)
                
                # Get CME properties (labels)
                speed = self.cme_properties.get("speed", [0])[idx] if idx < len(self.cme_properties.get("speed", [])) else 0
                direction = self.cme_properties.get("direction", [0])[idx] if idx < len(self.cme_properties.get("direction", [])) else 0
                arrival_time = self.cme_properties.get("arrival_time", [0])[idx] if idx < len(self.cme_properties.get("arrival_time", [])) else 0
                impact_prob = self.cme_properties.get("impact_prob", [0])[idx] if idx < len(self.cme_properties.get("impact_prob", [])) else 0
                
                # Create label tensor
                labels = torch.tensor([speed, direction, arrival_time, impact_prob], dtype=torch.float32)
                
                return {
                    "coronagraph": coronagraph_data,
                    "magnetogram": magnetogram_tensor,
                    "solar_wind": solar_wind_tensor,
                    "labels": labels
                }


class CNNLSTMCMEPredictor(nn.Module):
    """
    CNN-LSTM architecture for CME trajectory prediction.
    
    This model combines multiple CNN streams for processing coronagraph imagery
    and magnetogram data, with LSTM for solar wind time series, then fuses
    all features for trajectory prediction.
    """
    
    def __init__(
        self,
        coronagraph_channels: int = 4,     # Number of coronagraph instruments
        magnetogram_channels: int = 1,     # Single magnetogram channel
        solar_wind_features: int = 4,      # Solar wind feature dimensions
        sequence_length: int = 24,         # Time sequence length
        cnn_feature_size: int = 128,       # Size of CNN features
        lstm_hidden_size: int = 64,        # LSTM hidden state size
        lstm_num_layers: int = 2,          # Number of LSTM layers
        fc_hidden_size: int = 128,         # Hidden layer size
        dropout: float = 0.3               # Dropout probability
    ):
        """
        Initialize the CNN-LSTM CME prediction model.
        
        Args:
            coronagraph_channels: Number of coronagraph instruments
            magnetogram_channels: Number of magnetogram channels
            solar_wind_features: Number of solar wind features
            sequence_length: Length of time sequences
            cnn_feature_size: Size of features extracted by CNN
            lstm_hidden_size: LSTM hidden state size
            lstm_num_layers: Number of stacked LSTM layers
            fc_hidden_size: Size of hidden fully connected layer
            dropout: Dropout probability for regularization
        """
        super(CNNLSTMCMEPredictor, self).__init__()
        
        # Coronagraph CNN stream
        self.coronagraph_cnn = nn.Sequential(
            # Initial convolution block
            nn.Conv2d(coronagraph_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            # Second convolution block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            # Third convolution block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            # Fourth convolution block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        
        # Calculate CNN output size based on input size and pooling
        coronagraph_output_size = 256 * (256 // 16) * (256 // 16)  # Based on 256x256 input with 4 pooling layers
        
        # Coronagraph CNN to feature mapping
        self.coronagraph_to_feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(coronagraph_output_size, cnn_feature_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Magnetogram CNN stream
        self.magnetogram_cnn = nn.Sequential(
            # Initial convolution block
            nn.Conv2d(magnetogram_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            # Second convolution block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            # Third convolution block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            # Fourth convolution block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        
        # Calculate CNN output size for magnetogram stream
        magnetogram_output_size = 256 * (256 // 16) * (256 // 16)  # Based on 256x256 input with 4 pooling layers
        
        # Magnetogram CNN to feature mapping
        self.magnetogram_to_feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(magnetogram_output_size, cnn_feature_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for solar wind time series data
        self.solar_wind_lstm = nn.LSTM(
            input_size=solar_wind_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        # Combined feature size from all streams
        combined_size = cnn_feature_size * 2 + lstm_hidden_size
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(fc_hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout/2)  # Less dropout for shared layer
        )
        
        # Prediction heads
        
        # 1. Speed prediction (regression)
        self.speed_head = nn.Sequential(
            nn.Linear(fc_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Speed in km/s
        )
        
        # 2. Direction prediction (regression with circular loss)
        self.direction_head = nn.Sequential(
            nn.Linear(fc_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Sin and Cos of direction angle
        )
        
        # 3. Arrival time prediction (regression)
        self.arrival_head = nn.Sequential(
            nn.Linear(fc_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Hours until arrival
        )
        
        # 4. Earth impact probability (binary classification)
        self.impact_head = nn.Sequential(
            nn.Linear(fc_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Probability between 0 and 1
        )
        
        # 5. Confidence scores
        self.confidence_head = nn.Sequential(
            nn.Linear(fc_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # Confidence for each prediction
            nn.Sigmoid()  # Scale between 0 and 1
        )
    
    def forward(
        self, 
        coronagraph_batch: torch.Tensor,
        magnetogram_batch: torch.Tensor,
        solar_wind_sequence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            coronagraph_batch: Coronagraph imagery [batch_size, channels, height, width]
            magnetogram_batch: Magnetogram data [batch_size, 1, height, width]
            solar_wind_sequence: Solar wind time series [batch_size, seq_len, features]
            
        Returns:
            Dictionary with prediction outputs:
                - speed: CME speed prediction
                - direction: CME direction prediction (sin, cos)
                - arrival_time: Predicted arrival time
                - impact_prob: Earth impact probability
                - confidence: Confidence scores for each prediction
        """
        # Process coronagraph imagery through CNN
        coronagraph_out = self.coronagraph_cnn(coronagraph_batch)
        coronagraph_features = self.coronagraph_to_feature(coronagraph_out)
        
        # Process magnetogram data through CNN
        magnetogram_out = self.magnetogram_cnn(magnetogram_batch)
        magnetogram_features = self.magnetogram_to_feature(magnetogram_out)
        
        # Process solar wind data through LSTM
        lstm_out, (h_n, c_n) = self.solar_wind_lstm(solar_wind_sequence)
        solar_wind_features = h_n[-1]  # Use the final hidden state
        
        # Combine features from all streams
        combined_features = torch.cat(
            (coronagraph_features, magnetogram_features, solar_wind_features), 
            dim=1
        )
        
        # Fuse features
        fused_features = self.fusion_layer(combined_features)
        
        # Get shared representation
        shared_repr = self.shared_layer(fused_features)
        
        # Generate predictions from each head
        speed = self.speed_head(shared_repr)
        direction_vec = self.direction_head(shared_repr)  # Sin and Cos components
        arrival_time = self.arrival_head(shared_repr)
        impact_prob = self.impact_head(shared_repr)
        
        # Generate confidence scores
        confidence = self.confidence_head(shared_repr)
        
        # Return all predictions
        return {
            "speed": speed,
            "direction": direction_vec,
            "arrival_time": arrival_time,
            "impact_prob": impact_prob,
            "confidence": confidence
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        weights: Dict[str, float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for CME predictions.
        
        Args:
            predictions: Dictionary of model predictions
            targets: Target values [batch_size, 4] (speed, direction, arrival_time, impact_prob)
            weights: Optional dictionary of loss weights
            
        Returns:
            Dictionary of losses and total loss
        """
        # Default weights if not provided
        if weights is None:
            weights = {
                "speed": 1.0,
                "direction": 1.0,
                "arrival_time": 1.0,
                "impact_prob": 1.0
            }
        
        # Extract target components
        target_speed = targets[:, 0:1]           # Speed
        target_direction = targets[:, 1:2]       # Direction in radians
        target_arrival = targets[:, 2:3]         # Arrival time in hours
        target_impact = targets[:, 3:4]          # Impact probability
        
        # 1. Speed loss (Mean Squared Error)
        speed_loss = F.mse_loss(predictions["speed"], target_speed)
        
        # 2. Direction loss (Circular MSE: 1 - cos(pred - target))
        # Convert target direction to sin/cos representation
        target_dir_sin = torch.sin(target_direction)
        target_dir_cos = torch.cos(target_direction)
        target_dir_vec = torch.cat([target_dir_sin, target_dir_cos], dim=1)
        
        # Direction loss (cosine similarity based)
        dir_cos_sim = F.cosine_similarity(predictions["direction"], target_dir_vec, dim=1)
        direction_loss = torch.mean(1 - dir_cos_sim)
        
        # 3. Arrival time loss (Huber loss for robustness)
        arrival_loss = F.smooth_l1_loss(predictions["arrival_time"], target_arrival)
        
        # 4. Impact probability loss (Binary Cross Entropy)
        impact_loss = F.binary_cross_entropy(predictions["impact_prob"], target_impact)
        
        # Compute weighted total loss
        total_loss = (
            weights["speed"] * speed_loss +
            weights["direction"] * direction_loss +
            weights["arrival_time"] * arrival_loss +
            weights["impact_prob"] * impact_loss
        )
        
        # Return all losses
        return {
            "speed_loss": speed_loss,
            "direction_loss": direction_loss,
            "arrival_loss": arrival_loss,
            "impact_loss": impact_loss,
            "total_
                # Size of the active region
                size = np.random.uniform(0.1, 0.3)
                strength = np.random.uniform(50, 200)
                
                # Create bipolar region
                pos_pole = strength * np.exp(-((xx-x_pos)**2 + (yy-y_pos)**2) / (size**2))
                neg_pole = -strength * np.exp(-((xx-x_pos-size)**2 + (yy-y_pos)**2) / (size**2))
                
                # Add to magnetogram
                magnetogram += pos_pole + neg_pole
            
            # Add random noise
            magnetogram += np.random.normal(0, 5, (height, width))
            
            return magnetogram
            
        except Exception as e:
            logger.error(f"Error loading magnetogram data: {str(e)}")
            raise CMEPredictionError(f"Failed to load magnetogram data: {str(e)}")
    
    def normalize_magnetogram(
        self, 
        magnetogram: np.ndarray,
        instrument: str = "HMI"
    ) -> np.ndarray:
        """
        Normalize magnetogram data.
        
        Args:
            magnetogram: Magnetogram data
            instrument: Instrument that produced the data
            
        Returns:
            Normalized magnetogram data
            
        Raises:
            CMEPredictionError: If normalization fails
        """
        try:
            # Get normalization values for this instrument
            norm = self.norm_values.get(instrument, {"mean": 0.0, "std": 100.0})
            
            # Normalize
            normalized = (magnetogram - norm["mean"]) / norm["std"]
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing magnetogram data: {str(e)}")
            raise CMEPredictionError(f"Failed to normalize magnetogram data: {str(e)}")
    
    def extract_magnetogram_features(
        self,
        magnetogram: np.ndarray
    ) -> np.ndarray:
        """
        Extract features from magnetogram data.
        
        Args:
            magnetogram: Magnetogram data
            
        Returns:
            Array of magnetogram features
            
        Raises:
            CMEPredictionError: If feature extraction fails
        """
        try:
            # Extract basic statistical features
            features = np.zeros(10)
            
            # Mean and standard deviation
            features[0] = np.mean(magnetogram)
            features[1] = np.std(magnetogram)
            
            # Maximum absolute field strength
            features[2] = np.max(np.abs(magnetogram))
            
            # Total unsigned flux
            features[3] = np.sum(np.abs(magnetogram))
            
            # Total signed flux
            features[4] = np.sum(magnetogram)
            
            # Total positive flux
            features[5] = np.sum(magnetogram[magnetogram > 0])
            
            # Total negative flux
            features[6] = np.sum(magnetogram[magnetogram < 0])
            
            # Flux imbalance
            if features[3] > 0:
                features[7] = features[4] / features[3]
            else:
                features[7] = 0
            
            # Number of strong field regions (above 100 G)
            features[8] = np.sum(np.abs(magnetogram) > 100) / magnetogram.size
            
            # Polarity inversion line length (simplified approximation)
            # In a real implementation, this would use more sophisticated methods
            gradient_x = np.abs(np.diff(np.sign(magnetogram), axis=1))
            gradient_y = np.abs(np.diff(np.sign(magnetogram), axis=0))
            features[9] = (np.sum(gradient_x) + np.sum(gradient_y)) / magnetogram.size
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting magnetogram features: {str(e)}")
            raise CMEPredictionError(f"Failed to extract magnetogram features: {str(e)}")
    
    def prepare_magnetogram_batch(
        self, 
        magnetogram: np.ndarray
    ) -> torch.Tensor:
        """
        Prepare magnetogram as a batch for input to CNN.
        
        Args:
            magnetogram: Normalized magnetogram data
            
        Returns:
            PyTorch tensor with shape [1, 1, H, W] ready for CNN input
            
        Raises:
            CMEPredictionError: If batch preparation fails
        """
        try:
            # Add batch and channel dimensions
            batch = magnetogram.reshape(1, 1, *magnetogram.shape)
            
            # Convert to PyTorch tensor
            tensor = torch.from_numpy(batch).float()
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error preparing magnetogram batch: {str(e)}")
            raise CMEPredictionError(f"Failed to prepare magnetogram batch: {str(e)}")


class SolarWindPreprocessor:
    """
    Preprocessor for solar wind data.
    
    This class handles loading, normalization, feature extraction, and 
    formatting of solar wind data for input to LSTM models.
    """
    
    def __init__(
        self, 
        data_path: str = SOLAR_WIND_DATA_PATH,
        sequence_length: int = 24,  # 24 hours of data
        sampling_rate: int = 1      # 1-hour intervals
    ):
        """
        Initialize the solar wind data preprocessor.
        
        Args:
            data_path: Path to solar wind data files
            sequence_length: Number of time steps to include in sequence
            sampling_rate: Sampling interval in hours
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
        
        # Create data directory if it doesn't exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Feature statistics for normalization
        self.feature_stats = {
            "mean": np.array([400.0, 5.0, 5.0, 0.0]),  # Mean values for [speed, density, temp, Bz]
            "std": np.array([100.0, 3.0, 3.0, 5.0])    # Std values for [speed, density, temp, Bz]
        }
    
    def load_solar_wind_data(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> np.ndarray:
        """
        Load solar wind data for a specific time range.
        
        Args:
            start_date: Start date/time for the data
            end_date: End date/time for the data
            
        Returns:
            Numpy array of solar wind data with shape [time_steps, 4]
            
        Raises:
            CMEPredictionError: If data cannot be loaded
        """
        try:
            # In a real implementation, this would fetch data from files or API
            # For this project, we simulate the data
            
            # Calculate number of time steps
            delta = end_date - start_date
            hours = delta.total_seconds() / 3600
            steps = int(hours / self.sampling_rate)
            
            # Create time array
            times = np.array([start_date + timedelta(hours=i*self.sampling_rate) 
                              for i in range(steps)])
            
            # Simulate solar wind data
            # Features: [speed (km/s), density (p/cm^3), temperature (K), Bz (nT)]
            solar_wind = np.zeros((steps, 4))
            
            # Base values with random variations
            for i in range(steps):
                # Add some time-based trends and variations
                t = i / steps
                
                # Speed: typically 300-800 km/s
                solar_wind[i, 0] = 400 + 50 * np.sin(t * 2 * np.pi) + np.random.normal(0, 20)
                
                # Density: typically 1-10 p/cm^3
                solar_wind[i, 1] = 5 + 2 * np.sin(t * 4 * np.pi) + np.random.normal(0, 1)
                
                # Temperature: normalized units
                solar_wind[i, 2] = 5 + 1 * np.sin(t * 3 * np.pi) + np.random.normal(0, 0.5)
                
                # Bz: typically -10 to 10 nT
                solar_wind[i, 3] = 5 * np.sin(t * 5 * np.pi) + np.random.normal(0, 2)
            
            # Ensure positive values for physical parameters
            solar_wind[:, 0] = np.maximum(solar_wind[:, 0], 250)  # Minimum speed
            solar_wind[:, 1] = np.maximum(solar_wind[:, 1], 0.5)  # Minimum density
            solar_wind[:, 2] = np.maximum(solar_wind[:, 2], 1.0)  # Minimum temperature
            
            return solar_wind
            
        except Exception as e:
            logger.error(f"Error loading solar wind data: {str(e)}")
            raise CMEPredictionError(f"Failed to load solar wind data: {str(e)}")
    
    def normalize_features(
        self, 
        features: np.ndarray, 
        training: bool = False
    ) -> np.ndarray:
        """
        Normalize solar wind features to zero mean and unit variance.
        
        Args:
            features: Feature array to normalize
            training: Whether this is training data (to compute stats)
            
        Returns:
            Normalized feature array
            
        Raises:
            CMEPredictionError: If normalization fails
        """
        try:
            if training:
                # Compute statistics from this data
                mean = np.mean(features, axis=0)
                std = np.std(features, axis=0)
                
                # Update feature stats for future use
                self.feature_stats["mean"] = mean
                self.feature_stats["std"] = std
            else:
                # Use stored statistics
                mean = self.feature_stats["mean"]
                std = self.feature_stats["std"]
            
            # Avoid division by zero
            std = np.maximum(std, 1e-10)
            
            # Normalize
            normalized = (features - mean) / std
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing solar wind features: {str(e)}")
            raise CMEPredictionError(f"Failed to normalize solar wind features: {str(e)}")
    
    def prepare_sequence(
        self, 
        features: np.ndarray, 
        sequence_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Prepare features as sequence for input to LSTM.
        
        Args:
            features: Normalized feature array
            sequence_length: Length of sequence (default: self.sequence_length)
            
        Returns:
            Sequence array with shape [1, sequence_length, features]
            
        Raises:
            CMEPredictionError: If sequence preparation fails
        """
        try:
            if sequence_length is None:
                sequence_length = self.sequence_length
                
            # Get most recent data up to sequence_length
            if len(features) >= sequence_length:
                seq = features[-sequence_length:]
            else:
                # Pad with zeros if not enough data
                pad_length = sequence_length - len(features)
                padding = np.zeros((pad_length, features.shape[1]))
                seq = np.vstack((padding, features))
            
            # Add batch dimension for model input
            batch_seq = np.expand_dims(seq, 0)
            
            return batch_seq
            
        except Exception as e:
            logger.error(f"Error preparing solar wind sequence: {str(e)}")
            raise CMEPredictionError(f"Failed to prepare solar wind sequence: {str(e)}")


class CMEPredictionDataset(Dataset):
    """
    Dataset class for CME trajectory prediction.
    
    Combines coronagraph imagery, magnetogram data, and solar wind data
    for model training.
    """
    
    def __init__(
        self,
        coronagraph_sequences: Dict[str, List[np.ndarray]],
        magnetogram_sequences: List[np.ndarray],
        solar_wind_sequences: List[np.ndarray],
        cme_properties: Dict[str, np.ndarray]
    ):
        """
        Initialize the dataset.
        
        Args:
            coronagraph_sequences: Dictionary of coronagraph image sequences by instrument
            magnetogram_sequences: List of magnetogram image sequences
            solar_wind_sequences: List of solar wind data sequences
            cme_properties: Dictionary of CME properties (speed, direction, arrival_time, impact_prob)
        """
        self.coronagraph_sequences = coronagraph_sequences
        self.magnetogram_sequences = magnetogram_sequences
        self.solar_wind_sequences = solar_wind_sequences
        self.cme_properties = cme_properties
        
        # Determine dataset size from CME properties
        self.size = len(cme_properties.get("speed", []))
        
    def __len__(self) -> int:
        """Get the dataset size."""
        return self.size
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        # Prepare coronagraph data
        coronagraph_data = {}
        for instrument, sequences in self.coronagraph_sequences.items():
            if idx < len(sequences):
                coronagraph_data[instrument] = torch.tensor(sequences[idx], dtype=torch.float32)
        
        # Prepare magnetogram data
                    
                elif instrument.startswith("STEREO"):
                    # Process STEREO images for side view of CME
                    stereo_features = self._extract_stereo_view(images)
                    features[f"{instrument}_view"] = stereo_features
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting CME features: {str(e)}")
            raise CMEPredictionError(f"Failed to extract CME features: {str(e)}")
    
    def _extract_leading_edge(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract leading edge features from LASCO-C2 images.
        
        Args:
            images: List of C2 coronagraph images
            
        Returns:
            Array of leading edge features
        """
        try:
            height, width = self.image_size
            num_images = len(images)
            
            # Features: [time_step, [position, velocity, brightness, width]]
            features = np.zeros((num_images, 4))
            
            for i, img in enumerate(images):
                # Simplified feature extraction
                # In a real implementation, this would use advanced image processing
                
                # Find brightest pixel in each direction as proxy for CME front
                center_x, center_y = width // 2, height // 2
                max_dist = 0
                max_bright = 0
                
                # Sample angles
                angles = np.linspace(0, 2*np.pi, 36)
                angular_width = 0
                bright_angles = []
                
                for angle in angles:
                    # Check along this angle
                    dx, dy = np.cos(angle), np.sin(angle)
                    max_bright_angle = 0
                    max_dist_angle = 0
                    
                    # Check points along this angle
                    for dist in range(10, min(width, height) // 2 - 10):
                        x = int(center_x + dx * dist)
                        y = int(center_y + dy * dist)
                        
                        if 0 <= x < width and 0 <= y < height:
                            brightness = img[y, x]
                            if brightness > 0.3:  # Threshold for CME
                                if dist > max_dist_angle:
                                    max_dist_angle = dist
                                if brightness > max_bright_angle:
                                    max_bright_angle = brightness
                    
                    if max_bright_angle > 0.3:
                        bright_angles.append(angle)
                    
                    # Update maximum values
                    if max_dist_angle > max_dist:
                        max_dist = max_dist_angle
                    if max_bright_angle > max_bright:
                        max_bright = max_bright_angle
                
                # Calculate angular width from bright angles
                if bright_angles:
                    angular_width = self._calculate_angular_width(bright_angles)
                
                # Store features
                features[i, 0] = max_dist / (min(width, height) // 2)  # Normalized position
                if i > 0:
                    # Calculate velocity (change in position)
                    features[i, 1] = features[i, 0] - features[i-1, 0]
                else:
                    features[i, 1] = 0
                features[i, 2] = max_bright  # Brightness
                features[i, 3] = angular_width / (2 * np.pi)  # Normalized angular width
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting leading edge features: {str(e)}")
            return np.zeros((len(images), 4))
    
    def _extract_cme_expansion(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract CME expansion features from LASCO-C3 images.
        
        Args:
            images: List of C3 coronagraph images
            
        Returns:
            Array of expansion features
        """
        try:
            height, width = self.image_size
            num_images = len(images)
            
            # Features: [time_step, [area, expansion_rate, acceleration, brightness]]
            features = np.zeros((num_images, 4))
            
            for i, img in enumerate(images):
                # Simplified feature extraction
                # In a real implementation, this would use advanced image processing
                
                # Threshold image to find CME area
                area = np.sum(img > 0.3) / (width * height)
                
                # Average brightness of CME area
                if area > 0:
                    brightness = np.mean(img[img > 0.3])
                else:
                    brightness = 0
                
                # Store features
                features[i, 0] = area
                if i > 0:
                    # Calculate expansion rate (change in area)
                    features[i, 1] = features[i, 0] - features[i-1, 0]
                    if i > 1:
                        # Calculate acceleration (change in expansion rate)
                        features[i, 2] = features[i, 1] - features[i-1, 1]
                    else:
                        features[i, 2] = 0
                else:
                    features[i, 1] = 0
                    features[i, 2] = 0
                features[i, 3] = brightness
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting CME expansion features: {str(e)}")
            return np.zeros((len(images), 4))
    
    def _extract_stereo_view(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract CME features from STEREO coronagraph images.
        
        Args:
            images: List of STEREO coronagraph images
            
        Returns:
            Array of STEREO view features
        """
        try:
            height, width = self.image_size
            num_images = len(images)
            
            # Features: [time_step, [position, velocity, brightness, direction]]
            features = np.zeros((num_images, 4))
            
            for i, img in enumerate(images):
                # Simplified feature extraction
                # In a real implementation, this would use advanced image processing
                
                # Find brightest pixel in each direction as proxy for CME front
                center_x, center_y = width // 2, height // 2
                max_dist = 0
                max_bright = 0
                avg_angle = 0
                
                # Sample angles
                angles = np.linspace(0, 2*np.pi, 36)
                bright_dists = []
                
                for j, angle in enumerate(angles):
                    # Check along this angle
                    dx, dy = np.cos(angle), np.sin(angle)
                    max_dist_angle = 0
                    
                    # Check points along this angle
                    for dist in range(10, min(width, height) // 2 - 10):
                        x = int(center_x + dx * dist)
                        y = int(center_y + dy * dist)
                        
                        if 0 <= x < width and 0 <= y < height:
                            brightness = img[y, x]
                            if brightness > 0.3:  # Threshold for CME
                                if dist > max_dist_angle:
                                    max_dist_angle = dist
                    
                    if max_dist_angle > 0:
                        bright_dists.append((angle, max_dist_angle))
                
                # Calculate average direction
                if bright_dists:
                    # Calculate weighted average angle
                    sum_x = sum(np.cos(angle) * dist for angle, dist in bright_dists)
                    sum_y = sum(np.sin(angle) * dist for angle, dist in bright_dists)
                    avg_angle = np.arctan2(sum_y, sum_x) % (2 * np.pi)
                    
                    # Calculate maximum brightness
                    for angle, dist in bright_dists:
                        x = int(center_x + np.cos(angle) * dist)
                        y = int(center_y + np.sin(angle) * dist)
                        if 0 <= x < width and 0 <= y < height:
                            if img[y, x] > max_bright:
                                max_bright = img[y, x]
                    
                    # Calculate maximum distance
                    max_dist = max(dist for _, dist in bright_dists)
                
                # Store features
                features[i, 0] = max_dist / (min(width, height) // 2)  # Normalized position
                if i > 0:
                    # Calculate velocity (change in position)
                    features[i, 1] = features[i, 0] - features[i-1, 0]
                else:
                    features[i, 1] = 0
                features[i, 2] = max_bright  # Brightness
                features[i, 3] = avg_angle / (2 * np.pi)  # Normalized direction
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting STEREO view features: {str(e)}")
            return np.zeros((len(images), 4))
    
    def _calculate_angular_width(self, angles: List[float]) -> float:
        """
        Calculate the angular width of a CME from a list of angles.
        
        Args:
            angles: List of angles (in radians) where CME is detected
            
        Returns:
            Angular width in radians
        """
        if not angles:
            return 0
            
        # Sort angles
        sorted_angles = sorted(angles)
        
        # Unwrap angles for correct width calculation
        unwrapped = []
        for angle in sorted_angles:
            if unwrapped and angle - unwrapped[-1] > np.pi:
                unwrapped.append(angle - 2 * np.pi)
            elif unwrapped and unwrapped[-1] - angle > np.pi:
                unwrapped.append(angle + 2 * np.pi)
            else:
                unwrapped.append(angle)
        
        # Find the maximum angular width
        max_width = 0
        for i in range(len(unwrapped)):
            for j in range(i + 1, len(unwrapped)):
                width = abs(unwrapped[j] - unwrapped[i])
                if width > max_width:
                    max_width = width
        
        return max_width
    
    def prepare_image_batch(
        self, 
        images: Dict[str, np.ndarray]
    ) -> torch.Tensor:
        """
        Prepare images as a batch for input to CNN.
        
        Args:
            images: Dictionary of normalized images by instrument
            
        Returns:
            PyTorch tensor with shape [1, C, H, W] ready for CNN input
            
        Raises:
            CMEPredictionError: If batch preparation fails
        """
        try:
            # Stack channels along the channel dimension
            height, width = self.image_size
            channels = len(images)
            
            # Create a tensor to hold the batch
            batch = np.zeros((1, channels, height, width))
            
            # Fill the batch with images
            for i, instrument in enumerate(self.instruments):
                if instrument in images:
                    batch[0, i] = images[instrument]
            
            # Convert to PyTorch tensor
            tensor = torch.from_numpy(batch).float()
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error preparing image batch: {str(e)}")
            raise CMEPredictionError(f"Failed to prepare image batch: {str(e)}")


class MagnetogramPreprocessor:
    """
    Preprocessor for solar magnetogram data.
    
    This class handles loading, normalization, feature extraction, and 
    formatting of solar magnetogram data for input to CNN models.
    """
    
    def __init__(
        self, 
        data_path: str = MAGNETOGRAM_DATA_PATH,
        image_size: Tuple[int, int] = (256, 256),
        instruments: List[str] = ["HMI", "MDI"]
    ):
        """
        Initialize the magnetogram preprocessor.
        
        Args:
            data_path: Path to magnetogram data
            image_size: Size to resize images to (height, width)
            instruments: List of magnetogram instruments to use
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.instruments = instruments
        
        # Create data directory if it doesn't exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Instrument-specific normalization values
        self.norm_values = {
            "HMI": {"mean": 0.0, "std": 100.0},  # HMI typically in Gauss
            "MDI": {"mean": 0.0, "std": 100.0}   # MDI typically in Gauss
        }
    
    def load_magnetogram(
        self, 
        timestamp: datetime,
        instrument: str = "HMI"
    ) -> np.ndarray:
        """
        Load magnetogram data for a specific timestamp.
        
        Args:
            timestamp: Timestamp to load data for
            instrument: Instrument to load data from
            
        Returns:
            Magnetogram data as numpy array
            
        Raises:
            CMEPredictionError: If data cannot be loaded
        """
        try:
            # In a real implementation, this would load actual magnetogram files
            # For this project, we'll generate synthetic data
            
            height, width = self.image_size
            
            # Create a simulated magnetogram with active regions
            x = np.linspace(-1, 1, width)
            y = np.linspace(-1, 1, height)
            xx, yy = np.meshgrid(x, y)
            r = np.sqrt(xx**2 + yy**2)
            
            # Base quiet sun
            magnetogram = np.random.normal(0, 10, (height, width))
            
            # Add active regions (bipolar magnetic regions)
            num_active_regions = np.random.randint(1, 4)
            
            for _ in range(num_active_regions):
                # Random position for the active region
                x_pos = np.random.uniform(-0.7, 0.7)
                y_pos = np.random.uniform(-0.7, 0.7)
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.instruments = instruments
        
        # Create data directory if it doesn't exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Instrument-specific normalization values
        self.norm_values = {
            "LASCO-C2": {"mean": 0.5, "std": 0.25},
            "LASCO-C3": {"mean": 0.5, "std": 0.25},
            "STEREO-A": {"mean": 0.5, "std": 0.25},
            "STEREO-B": {"mean": 0.5, "std": 0.25}
        }
    
    def load_images(
        self, 
        timestamp: datetime,
        instruments: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Load coronagraph images for a specific timestamp.
        
        Args:
            timestamp: Timestamp to load images for
            instruments: Specific instruments to load (default: all configured instruments)
            
        Returns:
            Dictionary mapping instrument names to image arrays
            
        Raises:
            CMEPredictionError: If images cannot be loaded
        """
        try:
            # In a real implementation, this would load actual image files
            # For this project, we'll generate synthetic images
            
            if instruments is None:
                instruments = self.instruments
            
            # Generate dummy images for each instrument
            height, width = self.image_size
            images = {}
            
            for instrument in instruments:
                # Create a simulated coronagraph image
                if instrument == "LASCO-C2":
                    # Simulate C2 coronagraph: 2-6 solar radii
                    x = np.linspace(-1, 1, width)
                    y = np.linspace(-1, 1, height)
                    xx, yy = np.meshgrid(x, y)
                    r = np.sqrt(xx**2 + yy**2)
                    
                    # Create occulting disk and outer edge
                    img = np.zeros((height, width))
                    img[(r > 0.2) & (r < 0.6)] = 0.5  # Corona region
                    
                    # Add random noise and streaks to simulate solar wind features
                    noise = np.random.normal(0, 0.1, (height, width))
                    
                    # Simulate a CME as a bright region extended in one direction
                    theta = np.random.uniform(0, 2*np.pi)
                    cme_dir_x = np.cos(theta)
                    cme_dir_y = np.sin(theta)
                    
                    # Create CME streak
                    cme = np.zeros((height, width))
                    for i in range(height):
                        for j in range(width):
                            # Calculate angle from center to pixel
                            px = x[j]
                            py = y[i]
                            if r[i, j] > 0.2:  # Outside occulting disk
                                # Dot product to measure alignment with CME direction
                                alignment = (px * cme_dir_x + py * cme_dir_y) / r[i, j]
                                if alignment > 0.7:  # Within cone of CME
                                    # CME brightness decreases with distance
                                    cme[i, j] = 0.5 * np.exp(-(r[i, j] - 0.2) * 5)
                    
                    img = img + noise + cme
                    
                elif instrument == "LASCO-C3":
                    # Simulate C3 coronagraph: 3.7-30 solar radii
                    x = np.linspace(-1, 1, width)
                    y = np.linspace(-1, 1, height)
                    xx, yy = np.meshgrid(x, y)
                    r = np.sqrt(xx**2 + yy**2)
                    
                    # Create occulting disk and outer edge
                    img = np.zeros((height, width))
                    img[(r > 0.12) & (r < 1.0)] = 0.3  # Corona region
                    
                    # Add random noise and streaks to simulate solar wind features
                    noise = np.random.normal(0, 0.05, (height, width))
                    
                    # Simulate a CME as a bright region extended in one direction
                    theta = np.random.uniform(0, 2*np.pi)
                    cme_dir_x = np.cos(theta)
                    cme_dir_y = np.sin(theta)
                    
                    # Create CME streak
                    cme = np.zeros((height, width))
                    for i in range(height):
                        for j in range(width):
                            # Calculate angle from center to pixel
                            px = x[j]
                            py = y[i]
                            if r[i, j] > 0.12:  # Outside occulting disk
                                # Dot product to measure alignment with CME direction
                                alignment = (px * cme_dir_x + py * cme_dir_y) / r[i, j]
                                if alignment > 0.7:  # Within cone of CME
                                    # CME brightness decreases with distance
                                    cme[i, j] = 0.3 * np.exp(-(r[i, j] - 0.12) * 3)
                    
                    img = img + noise + cme
                    
                elif instrument.startswith("STEREO"):
                    # Simulate STEREO coronagraph
                    x = np.linspace(-1, 1, width)
                    y = np.linspace(-1, 1, height)
                    xx, yy = np.meshgrid(x, y)
                    r = np.sqrt(xx**2 + yy**2)
                    
                    # Create occulting disk and outer edge
                    img = np.zeros((height, width))
                    img[(r > 0.15) & (r < 0.8)] = 0.4  # Corona region
                    
                    # Add random noise and streaks to simulate solar wind features
                    noise = np.random.normal(0, 0.07, (height, width))
                    
                    # Simulate a CME as a bright region extended in one direction
                    theta = np.random.uniform(0, 2*np.pi)
                    cme_dir_x = np.cos(theta)
                    cme_dir_y = np.sin(theta)
                    
                    # Create CME streak
                    cme = np.zeros((height, width))
                    for i in range(height):
                        for j in range(width):
                            # Calculate angle from center to pixel
                            px = x[j]
                            py = y[i]
                            if r[i, j] > 0.15:  # Outside occulting disk
                                # Dot product to measure alignment with CME direction
                                alignment = (px * cme_dir_x + py * cme_dir_y) / r[i, j]
                                if alignment > 0.7:  # Within cone of CME
                                    # CME brightness decreases with distance
                                    cme[i, j] = 0.4 * np.exp(-(r[i, j] - 0.15) * 4)
                    
                    img = img + noise + cme
                
                # Ensure values are in reasonable range
                img = np.clip(img, 0, 1)
                
                # Store the image
                images[instrument] = img
            
            return images
            
        except Exception as e:
            logger.error(f"Error loading coronagraph images: {str(e)}")
            raise CMEPredictionError(f"Failed to load coronagraph images: {str(e)}")
    
    def load_image_sequence(
        self,
        start_time: datetime,
        end_time: datetime,
        interval_hours: int = 2,
        instruments: Optional[List[str]] = None
    ) -> Dict[str, List[np.ndarray]]:
        """
        Load a time sequence of coronagraph images.
        
        Args:
            start_time: Start time for the sequence
            end_time: End time for the sequence
            interval_hours: Hours between each image
            instruments: Specific instruments to load
            
        Returns:
            Dictionary mapping instruments to lists of image arrays
            
        Raises:
            CMEPredictionError: If image sequences cannot be loaded
        """
        try:
            # Calculate number of time steps
            delta = end_time - start_time
            hours = delta.total_seconds() / 3600
            steps = int(hours / interval_hours) + 1
            
            # Generate timestamps
            timestamps = [start_time + timedelta(hours=i*interval_hours) for i in range(steps)]
            
            # Initialize result
            image_sequences = {instrument: [] for instrument in (instruments or self.instruments)}
            
            # Load images for each timestamp
            for timestamp in timestamps:
                images = self.load_images(timestamp, instruments)
                for instrument, image in images.items():
                    image_sequences[instrument].append(image)
            
            return image_sequences
            
        except Exception as e:
            logger.error(f"Error loading coronagraph image sequence: {str(e)}")
            raise CMEPredictionError(f"Failed to load coronagraph image sequence: {str(e)}")
    
    def normalize_images(
        self, 
        images: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Normalize images using instrument-specific normalization values.
        
        Args:
            images: Dictionary of images by instrument
            
        Returns:
            Dictionary of normalized images
            
        Raises:
            CMEPredictionError: If normalization fails
        """
        try:
            normalized = {}
            
            for instrument, img in images.items():
                # Get normalization values for this instrument
                norm = self.norm_values.get(instrument, {"mean": 0.5, "std": 0.25})
                
                # Normalize
                norm_img = (img - norm["mean"]) / norm["std"]
                normalized[instrument] = norm_img
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing coronagraph images: {str(e)}")
            raise CMEPredictionError(f"Failed to normalize coronagraph images: {str(e)}")
    
    def extract_cme_features(
        self,
        image_sequence: Dict[str, List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Extract CME features from coronagraph image sequences.
        
        Args:
            image_sequence: Dictionary mapping instruments to lists of image arrays
            
        Returns:
            Dictionary of extracted features
            
        Raises:
            CMEPredictionError: If feature extraction fails
        """
        try:
            features = {}
            
            for instrument, images in image_sequence.items():
                if not images:
                    continue
                    
                # Define feature set based on instrument
                if instrument == "LASCO-C2":
                    # Process C2 images to extract CME leading edge and angular width
                    c2_features = self._extract_leading_edge(images)
                    features["c2_leading_edge"] = c2_features
                    
                elif instrument == "LASCO-C3":
                    # Process C3 images to extract CME expansion and acceleration
                    c3_features = self._extract_cme_expansion

