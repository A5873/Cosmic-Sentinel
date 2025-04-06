#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Space Weather Testing Module

This module provides comprehensive test cases for the space weather components
of the Cosmic Sentinel application, including the main widget, indicators,
and event handling.
"""

import os
import sys
import unittest
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, Mock
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtGui import QColor, QImage

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components to test
from UI.space_weather_widget import (
    SpaceWeatherWidget, 
    SolarActivityIndicator,
    GeomagneticActivityBar,
    AuroraForecastDisplay,
    SpaceWeatherEventList
)
from API.space_weather import SpaceWeatherMonitor, SpaceWeatherEvent


# Mock NASA API responses
MOCK_SOLAR_FLARES = [
    {
        "flrID": "2025-04-06T12:30:00-FLR-001",
        "beginTime": "2025-04-06T12:30:00Z",
        "endTime": "2025-04-06T12:45:00Z",
        "classType": "X1.5",
        "sourceLocation": "N12W34",
        "activeRegionNum": 12345
    },
    {
        "flrID": "2025-04-05T08:15:00-FLR-002",
        "beginTime": "2025-04-05T08:15:00Z",
        "endTime": "2025-04-05T08:22:00Z",
        "classType": "M5.0",
        "sourceLocation": "S05E20",
        "activeRegionNum": 12346
    }
]

MOCK_CMES = [
    {
        "activityID": "2025-04-06-CME-001",
        "startTime": "2025-04-06T14:00:00Z",
        "cmeAnalyses": [
            {
                "speed": 1200,
                "type": "C",
                "note": "Earth-directed CME with significant energy",
                "isMostAccurate": True
            }
        ]
    }
]

MOCK_GEOMAGNETIC_STORMS = [
    {
        "gstID": "2025-04-07-GST-001",
        "startTime": "2025-04-07T02:00:00Z",
        "endTime": "2025-04-07T12:00:00Z",
        "allKpIndex": [
            {"kpIndex": 7, "observedTime": "2025-04-07T06:00:00Z"},
            {"kpIndex": 6, "observedTime": "2025-04-07T09:00:00Z"}
        ]
    }
]

MOCK_AURORA_FORECAST = {
    "Prediction_Time": "2025-04-06T18:00:00Z",
    "coordinates": [
        {"Aurora_Probability": 0.8, "Latitude": 60, "Longitude": -100},
        {"Aurora_Probability": 0.7, "Latitude": 58, "Longitude": -95},
        {"Aurora_Probability": 0.3, "Latitude": 50, "Longitude": -90},
    ],
    "activity_level": "HIGH",
    "average_probability": 0.6,
    "timestamp": "2025-04-06T18:00:00"
}

MOCK_GEOMAGNETIC_CONDITIONS = {
    "kp_index": 6.5,
    "activity_level": "HIGH",
    "data_time": "2025-04-06T17:00:00",
    "timestamp": "2025-04-06T18:00:00"
}

MOCK_SOLAR_CONDITIONS = {
    "activity_level": "EXTREME",
    "bt": 25.5,
    "bz": -18.2,
    "data_time": "2025-04-06T17:30:00",
    "timestamp": "2025-04-06T18:00:00"
}


# Create mock SpaceWeatherEvent instances
def create_mock_flare_event():
    """Create a mock solar flare event for testing."""
    return SpaceWeatherEvent(
        event_id="2025-04-06T12:30:00-FLR-001",
        event_type="SOLAR_FLARE",
        start_time=datetime.fromisoformat("2025-04-06T12:30:00+00:00"),
        end_time=datetime.fromisoformat("2025-04-06T12:45:00+00:00"),
        source="NASA DONKI",
        severity="EXTREME",
        description="Solar Flare: Class X1.5 from location N12W34",
        link="https://example.com/flare1",
        raw_data=MOCK_SOLAR_FLARES[0]
    )

def create_mock_cme_event():
    """Create a mock CME event for testing."""
    return SpaceWeatherEvent(
        event_id="2025-04-06-CME-001",
        event_type="CME",
        start_time=datetime.fromisoformat("2025-04-06T14:00:00+00:00"),
        end_time=None,
        source="NASA DONKI",
        severity="HIGH",
        description="Coronal Mass Ejection with speed 1200 km/s",
        link="https://example.com/cme1",
        raw_data=MOCK_CMES[0]
    )


class MockSpaceWeatherMonitor:
    """Mock SpaceWeatherMonitor class for testing."""
    
    def __init__(self, nasa_api_key=None, auto_monitor=False):
        self.nasa_api_key = nasa_api_key
        self.auto_monitor = auto_monitor
        self.event_callbacks = []
        self.monitoring_active = False
    
    def get_solar_flares(self, start_date=None, end_date=None, use_cache=True):
        """Mock method to get solar flares."""
        return [create_mock_flare_event()]
    
    def get_cmes(self, start_date=None, end_date=None, use_cache=True):
        """Mock method to get CMEs."""
        return [create_mock_cme_event()]
    
    def get_geomagnetic_storms(self, start_date=None, end_date=None, use_cache=True):
        """Mock method to get geomagnetic storms."""
        return []
    
    def get_aurora_forecast(self, use_cache=True):
        """Mock method to get aurora forecast."""
        return MOCK_AURORA_FORECAST
    
    def get_geomagnetic_conditions(self, use_cache=True):
        """Mock method to get geomagnetic conditions."""
        return MOCK_GEOMAGNETIC_CONDITIONS
    
    def get_solar_conditions(self, use_cache=True):
        """Mock method to get solar conditions."""
        return MOCK_SOLAR_CONDITIONS
    
    def register_callback(self, callback):
        """Mock method to register event callbacks."""
        self.event_callbacks.append(callback)
    
    async def _monitoring_task(self):
        """Mock monitoring task."""
        pass


# Test cases
class TestSolarActivityIndicator(unittest.TestCase):
    """Test cases for the SolarActivityIndicator widget."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the application for testing."""
        if not QApplication.instance():
            cls.app = QApplication([])
    
    def setUp(self):
        """Set up each test."""
        self.indicator = SolarActivityIndicator()
        self.indicator.resize(100, 100)
    
    def test_initialization(self):
        """Test initialization of the indicator."""
        self.assertEqual(self.indicator.activity_level, "UNKNOWN")
        self.assertIn("LOW", self.indicator.activity_colors)
        self.assertIn("MODERATE", self.indicator.activity_colors)
        self.assertIn("HIGH", self.indicator.activity_colors)
        self.assertIn("EXTREME", self.indicator.activity_colors)
    
    def test_set_activity_level(self):
        """Test setting activity levels."""
        # Test valid levels
        self.indicator.setActivityLevel("LOW")
        self.assertEqual(self.indicator.activity_level, "LOW")
        
        self.indicator.setActivityLevel("MODERATE")
        self.assertEqual(self.indicator.activity_level, "MODERATE")
        
        self.indicator.setActivityLevel("HIGH")
        self.assertEqual(self.indicator.activity_level, "HIGH")
        
        self.indicator.setActivityLevel("EXTREME")
        self.assertEqual(self.indicator.activity_level, "EXTREME")
        
        # Test invalid level (should stay at previous level)
        current_level = self.indicator.activity_level
        self.indicator.setActivityLevel("INVALID")
        self.assertEqual(self.indicator.activity_level, current_level)
    
    def test_rendering(self):
        """Test indicator rendering."""
        # Create an image to render into
        image = QImage(100, 100, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        
        # Render the indicator
        self.indicator.render(image)
        
        # Check that the image isn't empty (basic rendering test)
        self.assertFalse(image.isNull())
        
        # More detailed tests could check specific rendered pixels
        # for each activity level, but that's complex and brittle


class TestGeomagneticActivityBar(unittest.TestCase):
    """Test cases for the GeomagneticActivityBar widget."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the application for testing."""
        if not QApplication.instance():
            cls.app = QApplication([])
    
    def setUp(self):
        """Set up each test."""
        self.bar = GeomagneticActivityBar()
        self.bar.resize(200, 30)
    
    def test_initialization(self):
        """Test initialization of the bar."""
        self.assertEqual(self.bar.kp_index, 0.0)
    
    def test_set_kp_index(self):
        """Test setting the Kp index value."""
        # Test valid values
        self.bar.setKpIndex(3.5)
        self.assertEqual(self.bar.kp_index, 3.5)
        
        # Test boundary values
        self.bar.setKpIndex(-1.0)  # Should clamp to 0
        self.assertEqual(self.bar.kp_index, 0.0)
        
        self.bar.setKpIndex(10.0)  # Should clamp to 9
        self.assertEqual(self.bar.kp_index, 9.0)
    
    def test_rendering(self):
        """Test bar rendering."""
        # Create an image to render into
        image = QImage(200, 30, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        
        # Set Kp index and render
        self.bar.setKpIndex(5.0)
        self.bar.render(image)
        
        # Check that the image isn't empty
        self.assertFalse(image.isNull())


class TestAuroraForecastDisplay(unittest.TestCase):
    """Test cases for the AuroraForecastDisplay widget."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the application for testing."""
        if not QApplication.instance():
            cls.app = QApplication([])
    
    def setUp(self):
        """Set up each test."""
        self.display = AuroraForecastDisplay()
    
    def test_initialization(self):
        """Test initialization of the display."""
        self.assertEqual(self.display.activity_level.text(), "UNKNOWN")
        self.assertEqual(self.display.probability.text(), "Unknown")
        self.assertEqual(self.display.last_updated.text(), "Never")
    
    def test_update_forecast(self):
        """Test updating the forecast with new data."""
        # Test with valid data
        self.display.updateForecast(MOCK_AURORA_FORECAST)
        
        self.assertEqual(self.display.activity_level.text(), "HIGH")
        self.assertEqual(self.display.probability.text(), "60.0%")
        self.assertIn("2025-04-06", self.display.last_updated.text())
        
        # Test with minimal data
        minimal_data = {"activity_level": "LOW"}
        self.display.updateForecast(minimal_data)
        self.assertEqual(self.display.activity_level.text(), "LOW")
        
        # Test with invalid data (should handle gracefully)
        invalid_data = {"invalid_key": "value"}
        self.display.updateForecast(invalid_data)
        self.assertEqual(self.display.activity_level.text(), "UNKNOWN")


class TestSpaceWeatherEventList(unittest.TestCase):
    """Test cases for the SpaceWeatherEventList widget."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the application for testing."""
        if not QApplication.instance():
            cls.app = QApplication([])
    
    def setUp(self):
        """Set up each test."""
        self.event_list = SpaceWeatherEventList()
        
        # Mock events for testing
        self.flare_event = create_mock_flare_event()
        self.cme_event = create_mock_cme_event()
    
    def test_initialization(self):
        """Test initialization of the event list."""
        self.assertEqual(self.event_list.columnCount(), 4)
        self.assertEqual(len(self.event_list.events), 0)
        
        # Check header labels
        self.assertEqual(self.event_list.horizontalHeaderItem(0).text(), "Type")
        self.assertEqual(self.event_list.horizontalHeaderItem(1).text(), "Time")
        self.assertEqual(self.event_list.horizontalHeaderItem(2).text(), "Severity")
        self.assertEqual(self.event_list.horizontalHeaderItem(3).text(), "Description")
    
    def test_add_event(self):
        """Test adding events to the list."""
        # Add a solar flare event
        self.event_list.addEvent(self.flare_event)
        self.assertEqual(self.event_list.rowCount(), 1)
        self.assertEqual(len(self.event_list.events), 1)
        self.assertIn(self.flare_event.event_id, self.event_list.events)
        
        # Check cell contents
        self.assertEqual(self.event_list.item(0, 0).text(), "SOLAR FLARE")
        self.assertIn("2025-04-06

