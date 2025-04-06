#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for CME prediction system.

Tests the following components:
1. CoronagraphPreprocessor
2. MagnetogramPreprocessor
3. SolarWindPreprocessor
4. CNNLSTMCMEPredictor
5. End-to-end prediction pipeline
"""

import os
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.cme_predictor import (
    CoronagraphPreprocessor,
    MagnetogramPreprocessor, 
    SolarWindPreprocessor,
    CNNLSTMCMEPredictor,
    CMEPredictionDataset,
    CMEPredictionError
)


# --- Fixture for temporary directories ---

@pytest.fixture
def temp_dirs():
    """Create temporary directories for test data."""
    base_dir = tempfile.mkdtemp()
    coronagraph_dir = os.path.join(base_dir, "coronagraph")
    magnetogram_dir = os.path.join(base_dir, "magnetogram")
    solar_wind_dir = os.path.join(base_dir, "solar_wind")
    cache_dir = os.path.join(base_dir, "cache")
    
    # Create directories
    for d in [coronagraph_dir, magnetogram_dir, solar_wind_dir, cache_dir]:
        os.makedirs(d, exist_ok=True)
    
    yield {
        "base": base_dir,
        "coronagraph": coronagraph_dir,
        "magnetogram": magnetogram_dir,
        "solar_wind": solar_wind_dir,
        "cache": cache_dir
    }
    
    # Cleanup
    shutil.rmtree(base_dir)


# --- Fixtures for test data ---

@pytest.fixture
def coronagraph_images():
    """Generate test coronagraph images."""
    height, width = 256, 256
    instruments = ["LASCO-C2", "LASCO-C3", "STEREO-A", "STEREO-B"]
    
    images = {}
    for instrument in instruments:
        # Create a simple test image with a gradient
        img = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                # Distance from center
                dist = np.sqrt((i - height/2)**2 + (j - width/2)**2) / (height/2)
                if 0.2 < dist < 0.8:
                    img[i, j] = 0.5 * (1 - dist)
        
        # Add some random noise
        img += np.random.normal(0, 0.05, (height, width))
        img = np.clip(img, 0, 1)
        
        images[instrument] = img
    
    return images


@pytest.fixture
def magnetogram_data():
    """Generate test magnetogram data."""
    height, width = 256, 256
    
    # Create a simple bipolar magnetogram
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    # Create positive and negative poles
    pos_pole = 100 * np.exp(-((xx-0.3)**2 + (yy-0.3)**2) / 0.1)
    neg_pole = -100 * np.exp(-((xx+0.3)**2 + (yy+0.3)**2) / 0.1)
    
    # Combine poles
    magnetogram = pos_pole + neg_pole
    
    # Add noise
    magnetogram += np.random.normal(0, 5, (height, width))
    
    return magnetogram


@pytest.fixture
def solar_wind_data():
    """Generate test solar wind data."""
    # Generate 48 hours of synthetic solar wind data
    # Features: [speed, density, temperature, Bz]
    time_steps = 48
    features = 4
    
    data = np.zeros((time_steps, features))
    
    # Fill with synthetic data
    for i in range(time_steps):
        t = i / time_steps
        # Speed (typically 300-800 km/s)
        data[i, 0] = 400 + 50 * np.sin(t * 2 * np.pi) + np.random.normal(0, 10)
        # Density (typically 1-10 p/cm^3)
        data[i, 1] = 5 + 2 * np.sin(t * 4 * np.pi) + np.random.normal(0, 0.5)
        # Temperature
        data[i, 2] = 5 + np.sin(t * 3 * np.pi) + np.random.normal(0, 0.3)
        # Bz
        data[i, 3] = 5 * np.sin(t * 5 * np.pi) + np.random.normal(0, 1)
    
    return data


@pytest.fixture
def model_inputs():
    """Generate test inputs for the CNN-LSTM model."""
    batch_size = 2
    coronagraph_channels = 4
    img_height, img_width = 256, 256
    solar_wind_seq_len = 24
    solar_wind_features = 4
    
    # Create random inputs
    coronagraph_batch = torch.rand(batch_size, coronagraph_channels, img_height, img_width)
    magnetogram_batch = torch.rand(batch_size, 1, img_height, img_width)
    solar_wind_sequence = torch.rand(batch_size, solar_wind_seq_len, solar_wind_features)
    
    return {
        "coronagraph_batch": coronagraph_batch,
        "magnetogram_batch": magnetogram_batch,
        "solar_wind_sequence": solar_wind_sequence
    }


@pytest.fixture
def test_timestamps():
    """Generate test timestamps."""
    now = datetime.now()
    timestamps = [now - timedelta(hours=i*2) for i in range(10)]
    return timestamps


# --- Tests for SolarWindPreprocessor ---

class TestSolarWindPreprocessor:
    
    def test_initialization(self, temp_dirs):
        """Test initialization of SolarWindPreprocessor."""
        preprocessor = SolarWindPreprocessor(data_path=temp_dirs["solar_wind"])
        
        # Check that the data path was set correctly
        assert preprocessor.data_path == Path(temp_dirs["solar_wind"])
        
        # Check default sequence length
        assert preprocessor.sequence_length == 24
        
        # Check default sampling rate
        assert preprocessor.sampling_rate == 1
    
    def test_load_solar_wind_data(self, temp_dirs, test_timestamps):
        """Test loading solar wind data."""
        preprocessor = SolarWindPreprocessor(data_path=temp_dirs["solar_wind"])
        
        # Load solar wind data for a time range
        start_date = test_timestamps[-1]
        end_date = test_timestamps[0]
        data = preprocessor.load_solar_wind_data(start_date, end_date)
        
        # Check data shape
        expected_hours = (end_date - start_date).total_seconds() / 3600
        expected_steps = int(expected_hours / preprocessor.sampling_rate)
        assert data.shape == (expected_steps, 4)  # 4 features
        
        # Check feature ranges
        speed = data[:, 0]  # Solar wind speed
        density = data[:, 1]  # Density
        temperature = data[:, 2]  # Temperature
        bz = data[:, 3]  # Bz component
        
        # Speed should be in reasonable range
        assert np.all(speed >= 250)  # Min speed check
        assert np.all(speed <= 800)  # Max speed check
        
        # Density and temperature should be positive
        assert np.all(density > 0)
        assert np.all(temperature > 0)
    
    def test_normalize_features(self, temp_dirs, solar_wind_data):
        """Test feature normalization."""
        preprocessor = SolarWindPreprocessor(data_path=temp_dirs["solar_wind"])
        
        # Normalize features
        normalized = preprocessor.normalize_features(solar_wind_data)
        
        # Check shape preserved
        assert normalized.shape == solar_wind_data.shape
        
        # Check normalization
        mean = preprocessor.feature_stats["mean"]
        std = preprocessor.feature_stats["std"]
        
        for i in range(4):  # For each feature
            # Sample a few time steps to verify normalization
            for t in [0, 10, 20, 30]:
                if t < solar_wind_data.shape[0]:
                    expected = (solar_wind_data[t, i] - mean[i]) / std[i]
                    assert np.isclose(normalized[t, i], expected, rtol=1e-5)
    
    def test_normalize_features_training(self, temp_dirs, solar_wind_data):
        """Test feature normalization in training mode."""
        preprocessor = SolarWindPreprocessor(data_path=temp_dirs["solar_wind"])
        
        # Save original stats
        original_mean = preprocessor.feature_stats["mean"].copy()
        original_std = preprocessor.feature_stats["std"].copy()
        
        # Normalize with training=True
        normalized = preprocessor.normalize_features(solar_wind_data, training=True)
        
        # Stats should be updated
        assert not np.array_equal(original_mean, preprocessor.feature_stats["mean"])
        assert not np.array_equal(original_std, preprocessor.feature_stats["std"])
        
        # New stats should match the data
        assert np.allclose(preprocessor.feature_stats["mean"], np.mean(solar_wind_data, axis=0))
        assert np.allclose(preprocessor.feature_stats["std"], np.std(solar_wind_data, axis=0))
    
    def test_prepare_sequence(self, temp_dirs, solar_wind_data):
        """Test preparing sequence for LSTM."""
        preprocessor = SolarWindPreprocessor(data_path=temp_dirs["solar_wind"])
        
        # Set sequence length
        seq_length = 12
        
        # Prepare sequence
        sequence = preprocessor.prepare_sequence(solar_wind_data, seq_length)
        
        # Check shape
        assert sequence.shape == (1, seq_length, 4)  # Batch, seq_length, features
    
    def test_prepare_sequence_padding(self, temp_dirs):
        """Test sequence preparation with padding."""
        preprocessor = SolarWindPreprocessor(data_path=temp_dirs["solar_wind"])
        
        # Create short data
        short_data = np.random.rand(5, 4)  # Only 5 time steps
        
        # Prepare sequence with longer length
        seq_length = 10
        sequence = preprocessor.prepare_sequence(short_data, seq_length)
        
        # Check shape
        assert sequence.shape == (1, seq_length, 4)
        
        # Check padding
        # First 5 should be zeros
        assert np.all(sequence[0, :5, :] == 0)
        # Last 5 should be from the data
        for i in range(5):
            assert np.all(sequence[0, 5+i, :] == short_data[i, :])


# --- Tests for CMEPredictionDataset ---

class TestCMEPredictionDataset:
    
    @pytest.fixture
    def sample_dataset(self, coronagraph_images, magnetogram_data, solar_wind_data):
        """Create a sample dataset for testing."""
        # Create sequences
        coronagraph_sequences = {
            instrument: [img, img] for instrument, img in coronagraph_images.items()
        }
        
        magnetogram_sequences = [magnetogram_data, magnetogram_data]
        
        solar_wind_sequences = [
            solar_wind_data[:24], 
            solar_wind_data[24:48] if len(solar_wind_data) >= 48 else solar_wind_data[:24]
        ]
        
        # Create properties
        cme_properties = {
            "speed": np.array([800.0, 650.0]),
            "direction": np.array([0.7, 1.2]),
            "arrival_time": np.array([36.0, 48.0]),
            "impact_prob": np.array([0.8, 0.3])
        }
        
        # Create dataset
        dataset = CMEPredictionDataset(
            coronagraph_sequences,
            magnetogram_sequences,
            solar_wind_sequences,
            cme_properties
        )
        
        return dataset
    
    def test_dataset_initialization(self, sample_dataset, coronagraph_images):
        """Test dataset initialization."""
        # Check size
        assert len(sample_dataset) == 2
        
        # Check stored sequences
        assert set(sample_dataset.coronagraph_sequences.keys()) == set(coronagraph_images.keys())
        assert len(sample_dataset.magnetogram_sequences) == 2
        assert len(sample_dataset.solar_wind_sequences) == 2
    
    def test_dataset_getitem(self, sample_dataset):
        """Test dataset __getitem__ method."""
        # Get first item
        item = sample_dataset[0]
        
        # Check returned dictionary keys
        assert set(item.keys()) == {"coronagraph", "magnetogram", "solar_wind", "labels"}
        
        # Check types and shapes
        assert isinstance(item["coronagraph"], torch.Tensor)
        assert isinstance(item["magnetogram"], torch.Tensor)
        assert isinstance(item["solar_wind"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
        
        # Check label values
        expected_labels = torch.tensor([800.0, 0.7, 36.0, 0.8], dtype=torch.float32)
        assert torch.allclose(item["labels"], expected_labels)


# --- Tests for CNNLSTMCMEPredictor ---

class TestCNNLSTMCMEPredictor:
    
    def test_initialization(self):
        """Test model initialization."""
        model = CNNLSTMCMEPredictor()
        
        # Check model components exist
        assert hasattr(model, 'coronagraph_cnn')
        assert hasattr(model, 'magnetogram_cnn')
        assert hasattr(model, 'solar_wind_lstm')
        assert hasattr(model, 'fusion_layer')
        assert hasattr(model, 'speed_head')
        assert hasattr(model, 'direction_head')
        assert hasattr(model, 'arrival_head')
        assert hasattr(model, 'impact_head')
        assert hasattr(model, 'confidence_head')
    
    def test_custom_initialization(self):
        """Test model initialization with custom parameters."""
        model = CNNLSTMCMEPredictor(
            coronagraph_channels=3,
            magnetogram_channels=2,
            solar_wind_features=5,
            sequence_length=12,
            cnn_feature_size=64,
            lstm_hidden_size=32,
            fc_hidden_size=50
        )
        
        # Check LSTM parameters
        assert model.solar_wind_lstm.input_size == 5
        assert model.solar_wind_lstm.hidden_size == 32
        
        # Check CNN feature size (indirectly through feature mapping layers)
        assert model.coronagraph_to_feature[-2].out_features == 64
        assert model.magnetogram_to_feature[-2].out_features == 64
    
    def test_forward_pass(self, model_inputs):
        """Test forward pass through the model."""
        model = CNNLSTMCMEPredictor()
        
        # Run forward pass
        outputs = model(
            model_inputs["coronagraph_batch"],
            model_inputs["magnetogram_batch"],
            model_inputs["solar_wind_sequence"]
        )
        
        # Check output types and shapes
        assert isinstance(outputs, dict)
        assert "speed" in outputs
        assert "direction" in outputs
        assert "arrival_time" in outputs
        assert "impact_prob" in outputs
        assert "confidence" in outputs
        
        batch_size = model_inputs["coronagraph_batch"].shape[0]
        
        # Check output shapes
        assert outputs["speed"].shape == (batch_size, 1)
        assert outputs["direction"].shape == (batch_size, 2)  # sin, cos
        assert outputs["arrival_time"].shape == (batch_size, 1)
        assert outputs["impact_prob"].shape == (batch_size, 1)
        assert outputs["confidence"].shape == (batch_size, 4)
        
        # Check probability outputs are in [0, 1]
        assert torch.all(outputs["impact_prob"] >= 0)
        assert torch.all(outputs["impact_prob"] <= 1)
        assert torch.all(outputs["confidence"] >= 0)
        assert torch.all(outputs["confidence"] <= 1)
    
    @pytest.mark.parametrize("train_step", [True, False])
    def test_compute_loss(self, model_inputs, train_step):
        """Test loss computation."""
        model = CNNLSTMCMEPredictor()
        
        # Create dummy targets
        batch_size = model_inputs["coronagraph_batch"].shape[0]
        targets = torch.rand(batch_size, 4)  # speed, direction, arrival, impact
        
        # Run forward pass
        outputs = model(
            model_inputs["coronagraph_batch"],
            model_inputs["magnetogram_batch"],
            model_inputs["solar_wind_sequence"]
        )
        
        # Compute loss
        losses = model.compute_loss(outputs, targets)
        
        # Check loss types and shapes
        assert isinstance(losses, dict)
        assert "speed_loss" in losses
        assert "direction_loss" in losses
        assert "arrival_loss" in losses
        assert "impact_loss" in losses
        
        # Check losses are positive
        for loss_name, loss in losses.items():
            if loss_name != "total_loss":  # Skip the total loss which isn't returned in current implementation
                assert loss >= 0
                assert not torch.isnan(loss)
                assert not torch.isinf(loss)
        
        # Test with custom weights
        custom_weights = {
            "speed": 2.0,
            "direction": 0.5,
            "arrival_time": 1.0,
            "impact_prob": 3.0
        }
        
        losses_weighted = model.compute_loss(outputs, targets, weights=custom_weights)
        
        # Check that weights were applied (this is a bit indirect)
        # We'd expect the weighted losses to be different from the unweighted ones
        assert not torch.isclose(losses["speed_loss"], losses_weighted["speed_loss"])
    
    def test_gradient_flow(self, model_inputs):
        """Test gradient flow through the model."""
        model = CNNLSTMCMEPredictor()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create dummy targets
        batch_size = model_inputs["coronagraph_batch"].shape[0]
        targets = torch.rand(batch_size, 4)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            model_inputs["coronagraph_batch"],
            model_inputs["magnetogram_batch"],
            model_inputs["solar_wind_sequence"]
        )
        
        # Compute loss
        losses = model.compute_loss(outputs, targets)
        
        # We need to manually compute the total loss since it's not in the returned dict
        total_loss = (
            losses["speed_loss"] + 
            losses["direction_loss"] + 
            losses["arrival_loss"] + 
            losses["impact_loss"]
        )
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients are computed
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                has_gradients = True
                break
        
        assert has_gradients, "No parameter gradients were computed"


# --- Tests for End-to-End Pipeline ---

class TestEndToEndPipeline:
    
    @pytest.fixture
    def mock_space_weather_monitor(self):
        """Create a mock space weather monitor."""
        mock_monitor = MagicMock()
        
        # Set up mock returns
        mock_monitor.get_coronagraph_data.return_value = {
            "LASCO-C2": np.random.rand(256, 256),
            "LASCO-C3": np.random.rand(256, 256),
            "
        """Test initialization of CoronagraphPreprocessor."""
        preprocessor = CoronagraphPreprocessor(data_path=temp_dirs["coronagraph"])
        
        # Check that the data path was set correctly
        assert preprocessor.data_path == Path(temp_dirs["coronagraph"])
        
        # Check default image size
        assert preprocessor.image_size == (256, 256)
        
        # Check default instruments
        assert set(preprocessor.instruments) == {"LASCO-C2", "LASCO-C3", "STEREO-A", "STEREO-B"}
    
    def test_load_images(self, temp_dirs, test_timestamps):
        """Test loading coronagraph images."""
        preprocessor = CoronagraphPreprocessor(data_path=temp_dirs["coronagraph"])
        
        # Load images for a timestamp
        images = preprocessor.load_images(test_timestamps[0])
        
        # Check that we have images for all instruments
        assert set(images.keys()) == set(preprocessor.instruments)
        
        # Check image shapes
        for instrument, img in images.items():
            assert img.shape == preprocessor.image_size
            
            # Check that image values are in [0, 1]
            assert np.min(img) >= 0
            assert np.max(img) <= 1
    
    def test_normalize_images(self, temp_dirs, coronagraph_images):
        """Test image normalization."""
        preprocessor = CoronagraphPreprocessor(data_path=temp_dirs["coronagraph"])
        
        # Normalize images
        normalized = preprocessor.normalize_images(coronagraph_images)
        
        # Check that we have normalized images for all instruments
        assert set(normalized.keys()) == set(coronagraph_images.keys())
        
        # Check normalization
        for instrument, img in normalized.items():
            norm_values = preprocessor.norm_values.get(
                instrument, {"mean": 0.5, "std": 0.25}
            )
            
            # Sample a few points to verify normalization
            orig_img = coronagraph_images[instrument]
            sample_points = [(100, 100), (150, 150), (200, 200)]
            
            for y, x in sample_points:
                expected = (orig_img[y, x] - norm_values["mean"]) / norm_values["std"]
                assert np.isclose(img[y, x], expected, rtol=1e-5)
    
    def test_load_image_sequence(self, temp_dirs, test_timestamps):
        """Test loading image sequences."""
        preprocessor = CoronagraphPreprocessor(data_path=temp_dirs["coronagraph"])
        
        # Load image sequence
        start_time = test_timestamps[0]
        end_time = test_timestamps[-1]
        sequence = preprocessor.load_image_sequence(start_time, end_time, interval_hours=6)
        
        # Check that we have sequences for all instruments
        assert set(sequence.keys()) == set(preprocessor.instruments)
        
        # Check sequence lengths
        expected_steps = (end_time - start_time).total_seconds() / 3600 / 6 + 1
        expected_steps = int(expected_steps)
        
        for instrument, images in sequence.items():
            assert len(images) == expected_steps
            
            # Check image shapes
            for img in images:
                assert img.shape == preprocessor.image_size
    
    def test_extract_cme_features(self, temp_dirs, coronagraph_images):
        """Test CME feature extraction."""
        preprocessor = CoronagraphPreprocessor(data_path=temp_dirs["coronagraph"])
        
        # Create a sequence of images
        sequence = {
            instrument: [img, img] for instrument, img in coronagraph_images.items()
        }
        
        # Extract features
        features = preprocessor.extract_cme_features(sequence)
        
        # Check that we have features for C2 and C3
        assert "c2_leading_edge" in features
        
        # Check feature shapes
        c2_features = features["c2_leading_edge"]
        assert c2_features.shape == (2, 4)  # 2 time steps, 4 features
        
    def test_prepare_image_batch(self, temp_dirs, coronagraph_images):
        """Test preparing image batch for CNN input."""
        preprocessor = CoronagraphPreprocessor(data_path=temp_dirs["coronagraph"])
        
        # Prepare batch
        batch = preprocessor.prepare_image_batch(coronagraph_images)
        
        # Check batch shape
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (1, len(coronagraph_images), *preprocessor.image_size)
        assert batch.dtype == torch.float32


# --- Tests for MagnetogramPreprocessor ---

class TestMagnetogramPreprocessor:
    
    def test_initialization(self, temp_dirs):
        """Test initialization of MagnetogramPreprocessor."""
        preprocessor = MagnetogramPreprocessor(data_path=temp_dirs["magnetogram"])
        
        # Check that the data path was set correctly
        assert preprocessor.data_path == Path(temp_dirs["magnetogram"])
        
        # Check default image size
        assert preprocessor.image_size == (256, 256)
        
        # Check default instruments
        assert set(preprocessor.instruments) == {"HMI", "MDI"}
    
    def test_load_magnetogram(self, temp_dirs, test_timestamps):
        """Test loading magnetogram data."""
        preprocessor = MagnetogramPreprocessor(data_path=temp_dirs["magnetogram"])
        
        # Load magnetogram for a timestamp
        magnetogram = preprocessor.load_magnetogram(test_timestamps[0])
        
        # Check magnetogram shape
        assert magnetogram.shape == preprocessor.image_size
        
        # Verify that magnetogram has both positive and negative values (bipolar region)
        assert np.min(magnetogram) < 0
        assert np.max(magnetogram) > 0
    
    def test_normalize_magnetogram(self, temp_dirs, magnetogram_data):
        """Test magnetogram normalization."""
        preprocessor = MagnetogramPreprocessor(data_path=temp_dirs["magnetogram"])
        
        # Normalize magnetogram
        normalized = preprocessor.normalize_magnetogram(magnetogram_data)
        
        # Check normalization
        instrument = "HMI"
        norm_values = preprocessor.norm_values.get(
            instrument, {"mean": 0.0, "std": 100.0}
        )
        
        # Sample a few points to verify normalization
        sample_points = [(100, 100), (150, 150), (200, 200)]
        
        for y, x in sample_points:
            expected = (magnetogram_data[y, x] - norm_values["mean"]) / norm_values["std"]
            assert np.isclose(normalized[y, x], expected, rtol=1e-5)
    
    def test_extract_magnetogram_features(self, temp_dirs, magnetogram_data):
        """Test extracting features from magnetogram data."""
        preprocessor = MagnetogramPreprocessor(data_path=temp_dirs["magnetogram"])
        
        # Extract features
        features = preprocessor.extract_magnetogram_features(magnetogram_data)
        
        # Check feature shape
        assert features.shape == (10,)  # 10 features
        
        # Check specific features
        assert np.isclose(features[0], np.mean(magnetogram_data), rtol=1e-5)  # mean
        assert np.isclose(features[1], np.std(magnetogram_data), rtol=1e-5)   # std
        assert np.isclose(features[2], np.max(np.abs(magnetogram_data)), rtol=1e-5)  # max abs
    
    def test_prepare_magnetogram_batch(self, temp_dirs, magnetogram_data):
        """Test preparing magnetogram batch for CNN input."""
        preprocessor = MagnetogramPreprocessor(data_path=temp_dirs["magnetogram"])
        
        # Prepare batch
        batch = preprocessor.prepare_magnetogram_batch(magnetogram_data)
        
        # Check batch shape
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (1, 1, *preprocessor.image_size)
        assert batch.dtype == torch.float32


# --- Tests for SolarWindPreprocessor ---

class TestSolarWindPreprocessor:
    
    def test_initialization(self, temp_

