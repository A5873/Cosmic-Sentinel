#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Settings Dialog for Cosmic Sentinel Application

This module provides a comprehensive settings interface for configuring
the Cosmic Sentinel application, including API keys, themes, data
management, and notification preferences.
"""

import os
import json
import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Set

from PyQt6.QtWidgets import (
    QDialog, QTabWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFormLayout, QGroupBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QFileDialog,
    QGridLayout, QListWidget, QListWidgetItem, QSlider,
    QMessageBox, QWidget, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import QSettings, Qt, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QIcon, QPixmap

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_DAYS = 7
DEFAULT_UPDATE_INTERVAL = 3600  # 1 hour in seconds
DEFAULT_NOTIFICATION_TYPES = ["Warnings", "Updates", "Critical Events"]
DEFAULT_DATA_PATH = os.path.expanduser("~/cosmic_sentinel_data")


class APIKeyManager(QWidget):
    """Manages API keys for external services like NASA NEO and space weather."""
    
    api_key_changed = pyqtSignal(str, str)  # service, new_key
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("CosmicSentinel", "Settings")
        self._init_ui()
        self._load_saved_keys()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # NASA API Key section
        nasa_group = QGroupBox("NASA API Keys")
        nasa_layout = QFormLayout()
        
        self.nasa_neo_key = QLineEdit()
        self.nasa_neo_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.nasa_neo_key.setPlaceholderText("Enter your NASA NEO API key")
        self.show_nasa_key = QPushButton("Show")
        self.show_nasa_key.setCheckable(True)
        self.show_nasa_key.toggled.connect(
            lambda checked: self.nasa_neo_key.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        
        nasa_key_layout = QHBoxLayout()
        nasa_key_layout.addWidget(self.nasa_neo_key)
        nasa_key_layout.addWidget(self.show_nasa_key)
        
        nasa_layout.addRow("NASA NEO API Key:", nasa_key_layout)
        self.nasa_key_status = QLabel("Not verified")
        nasa_layout.addRow("Status:", self.nasa_key_status)
        
        verify_nasa = QPushButton("Verify NASA Key")
        verify_nasa.clicked.connect(lambda: self._verify_api_key("nasa_neo"))
        nasa_layout.addRow("", verify_nasa)
        
        nasa_group.setLayout(nasa_layout)
        layout.addWidget(nasa_group)
        
        # Space Weather API Key section
        weather_group = QGroupBox("Space Weather API Keys")
        weather_layout = QFormLayout()
        
        self.space_weather_key = QLineEdit()
        self.space_weather_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.space_weather_key.setPlaceholderText("Enter your Space Weather API key")
        self.show_weather_key = QPushButton("Show")
        self.show_weather_key.setCheckable(True)
        self.show_weather_key.toggled.connect(
            lambda checked: self.space_weather_key.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        
        weather_key_layout = QHBoxLayout()
        weather_key_layout.addWidget(self.space_weather_key)
        weather_key_layout.addWidget(self.show_weather_key)
        
        weather_layout.addRow("Space Weather API Key:", weather_key_layout)
        self.weather_key_status = QLabel("Not verified")
        weather_layout.addRow("Status:", self.weather_key_status)
        
        verify_weather = QPushButton("Verify Weather Key")
        verify_weather.clicked.connect(lambda: self._verify_api_key("space_weather"))
        weather_layout.addRow("", verify_weather)
        
        weather_group.setLayout(weather_layout)
        layout.addWidget(weather_group)
        
        # Help information
        help_label = QLabel(
            "API keys are required for accessing NASA's NEO data and space weather alerts. "
            "You can obtain these keys for free by visiting their respective websites."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(help_label)
        
        # Key links
        nasa_link = QLabel("<a href='https://api.nasa.gov/'>Get NASA API Key</a>")
        nasa_link.setOpenExternalLinks(True)
        weather_link = QLabel("<a href='https://www.swpc.noaa.gov/'>Get Space Weather API Key</a>")
        weather_link.setOpenExternalLinks(True)
        
        links_layout = QHBoxLayout()
        links_layout.addWidget(nasa_link)
        links_layout.addWidget(weather_link)
        layout.addLayout(links_layout)
        
        layout.addStretch()
    
    def _load_saved_keys(self):
        """Load saved API keys from settings."""
        self.nasa_neo_key.setText(self.settings.value("api_keys/nasa_neo", ""))
        self.space_weather_key.setText(self.settings.value("api_keys/space_weather", ""))
        
        # Set verification status if keys exist
        if self.nasa_neo_key.text():
            self.nasa_key_status.setText("Saved (not verified)")
        if self.space_weather_key.text():
            self.weather_key_status.setText("Saved (not verified)")
    
    def _verify_api_key(self, service_name):
        """Verify that the API key is valid by making a test request."""
        key = ""
        status_label = None
        
        if service_name == "nasa_neo":
            key = self.nasa_neo_key.text().strip()
            status_label = self.nasa_key_status
        elif service_name == "space_weather":
            key = self.space_weather_key.text().strip()
            status_label = self.weather_key_status
        
        if not key:
            if status_label:
                status_label.setText("No key provided")
                status_label.setStyleSheet("color: red;")
            return
        
        try:
            # This would be replaced with actual API verification
            if service_name == "nasa_neo":
                # Placeholder for actual verification
                if len(key) < 8:  # Simple validation for demo
                    status_label.setText("Invalid key format")
                    status_label.setStyleSheet("color: red;")
                    return
                
                # Simulate successful verification
                status_label.setText("Verified ✓")
                status_label.setStyleSheet("color: green;")
                self.api_key_changed.emit(service_name, key)
            
            elif service_name == "space_weather":
                # Placeholder for actual verification
                if len(key) < 8:  # Simple validation for demo
                    status_label.setText("Invalid key format")
                    status_label.setStyleSheet("color: red;")
                    return
                
                # Simulate successful verification
                status_label.setText("Verified ✓")
                status_label.setStyleSheet("color: green;")
                self.api_key_changed.emit(service_name, key)
        
        except Exception as e:
            logger.error(f"API key verification failed: {e}")
            status_label.setText(f"Verification failed")
            status_label.setStyleSheet("color: red;")
    
    def save_keys(self):
        """Save API keys to settings."""
        # Store keys securely (in a real app, consider better encryption)
        self.settings.setValue("api_keys/nasa_neo", self.nasa_neo_key.text())
        self.settings.setValue("api_keys/space_weather", self.space_weather_key.text())
        self.settings.sync()
        
        # Emit signals for key changes
        self.api_key_changed.emit("nasa_neo", self.nasa_neo_key.text())
        self.api_key_changed.emit("space_weather", self.space_weather_key.text())


class ThemeManager(QWidget):
    """Manages application theme and visual preferences."""
    
    theme_changed = pyqtSignal(str)  # theme_name
    custom_colors_changed = pyqtSignal(dict)  # color_dict
    
    class Theme(Enum):
        LIGHT = "Light"
        DARK = "Dark"
        SYSTEM = "System"
        CUSTOM = "Custom"
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("CosmicSentinel", "Settings")
        self._init_ui()
        self._load_saved_theme()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Theme selection
        theme_group = QGroupBox("Application Theme")
        theme_layout = QVBoxLayout()
        
        self.theme_radio_group = QButtonGroup(self)
        
        # Create radio buttons for each theme
        for theme in ThemeManager.Theme:
            radio = QRadioButton(theme.value)
            self.theme_radio_group.addButton(radio)
            theme_layout.addWidget(radio)
            
            # Store reference to custom theme radio for enabling color picker
            if theme == ThemeManager.Theme.CUSTOM:
                self.custom_radio = radio
        
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)
        
        # Custom theme colors
        self.custom_colors_group = QGroupBox("Custom Theme Colors")
        custom_layout = QGridLayout()
        
        # Color pickers (simplified for demonstration)
        self.color_pickers = {}
        
        colors = [
            ("background", "Background"),
            ("foreground", "Foreground"),
            ("accent", "Accent"),
            ("warning", "Warning"),
            ("success", "Success"),
            ("error", "Error")
        ]
        
        row = 0
        for color_key, color_name in colors:
            label = QLabel(color_name)
            
            # Simplified color picker button (would be replaced with a proper color dialog)
            color_button = QPushButton()
            color_button.setFixedSize(24, 24)
            color_button.setProperty("color_key", color_key)
            color_button.clicked.connect(self._pick_color)
            
            self.color_pickers[color_key] = color_button
            
            custom_layout.addWidget(label, row, 0)
            custom_layout.addWidget(color_button, row, 1)
            row += 1
        
        self.custom_colors_group.setLayout(custom_layout)
        layout.addWidget(self.custom_colors_group)
        
        # Font size slider
        font_group = QGroupBox("Font Size")
        font_layout = QVBoxLayout()
        
        self.font_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_size_slider.setMinimum(8)
        self.font_size_slider.setMaximum(18)
        self.font_size_slider.setValue(10)
        self.font_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.font_size_slider.setTickInterval(2)
        
        self.font_size_label = QLabel("Font Size: 10pt")
        self.font_size_slider.valueChanged.connect(
            lambda size: self.font_size_label.setText(f"Font Size: {size}pt")
        )
        
        font_layout.addWidget(self.font_size_label)
        font_layout.addWidget(self.font_size_slider)
        font_group.setLayout(font_layout)
        layout.addWidget(font_group)
        
        # Preview (placeholder)
        preview_group = QGroupBox("Theme Preview")
        preview_layout = QVBoxLayout()
        preview_label = QLabel("Theme preview will appear here")
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_label.setMinimumHeight(100)
        preview_layout.addWidget(preview_label)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Connect signals
        self.theme_radio_group.buttonClicked.connect(self._on_theme_selected)
        
        # Initial state
        self._enable_custom_colors(False)
        
        layout.addStretch()
    
    def _load_saved_theme(self):
        """Load saved theme settings."""
        theme_name = self.settings.value("theme/name", ThemeManager.Theme.SYSTEM.value)
        
        # Select the corresponding radio button
        for button in self.theme_radio_group.buttons():
            if button.text() == theme_name:
                button.setChecked(True)
                if theme_name == ThemeManager.Theme.CUSTOM.value:
                    self._enable_custom_colors(True)
                break
        
        # Load saved font size
        font_size = self.settings.value("theme/font_size", 10, type=int)
        self.font_size_slider.setValue(font_size)
        self.font_size_label.setText(f"Font Size: {font_size}pt")
        
        # Load saved custom colors
        self._load_color_settings()
    
    def _load_color_settings(self):
        """Load custom color settings."""
        for color_key in self.color_pickers.keys():
            default_color = "#000000"  # Default black
            if color_key == "background":
                default_color = "#FFFFFF"  # Default white for background
            
            color_str = self.settings.value(f"theme/colors/{color_key}", default_color)
            
            # Set button background to the color
            self.color_pickers[color_key].setStyleSheet(f"background-color: {color_str};")
            self.color_pickers[color_key].setProperty("color_value", color_str)
    
    def _on_theme_selected(self, button):
        """Handle theme selection change."""
        theme_name = button.text()

import os
import json
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QGroupBox, 
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, 
    QCheckBox, QFileDialog, QMessageBox, QWidget, QFormLayout,
    QRadioButton, QButtonGroup, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings, QSize
from PyQt6.QtGui import QIcon, QColor

class APIKeyManager(QWidget):
    apiKeyChanged = pyqtSignal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # NASA API Key
        nasaGroup = QGroupBox("NASA API Keys")
        nasaLayout = QFormLayout()
        
        self.nasaKeyEdit = QLineEdit()
        self.nasaKeyEdit.setEchoMode(QLineEdit.EchoMode.Password)
        self.nasaKeyEdit.setPlaceholderText("Enter your NASA API key")
        
        self.showNasaKeyBtn = QPushButton("Show")
        self.showNasaKeyBtn.setCheckable(True)
        self.showNasaKeyBtn.toggled.connect(self._toggle_nasa_key_visibility)
        
        keyLayout = QHBoxLayout()
        keyLayout.addWidget(self.nasaKeyEdit)
        keyLayout.addWidget(self.showNasaKeyBtn)
        
        nasaLayout.addRow("API Key:", keyLayout)
        
        self.getNasaKeyBtn = QPushButton("Get NASA API Key")
        self.getNasaKeyBtn.clicked.connect(self._open_nasa_api_portal)
        nasaLayout.addRow("", self.getNasaKeyBtn)
        
        nasaGroup.setLayout(nasaLayout)
        layout.addWidget(nasaGroup)
        
        # Other API Keys could be added here
        
        layout.addStretch()
    
    def _toggle_nasa_key_visibility(self, checked):
        self.nasaKeyEdit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )
        self.showNasaKeyBtn.setText("Hide" if checked else "Show")
    
    def _open_nasa_api_portal(self):
        # In a real implementation, this would use QDesktopServices to open the URL
        QMessageBox.information(
            self, 
            "NASA API Portal", 
            "Please visit https://api.nasa.gov to get your free API key."
        )
    
    def get_nasa_api_key(self):
        return self.nasaKeyEdit.text()
    
    def set_nasa_api_key(self, key):
        self.nasaKeyEdit.setText(key)
    
    def save_settings(self, settings):
        settings.setValue("API/NASAKey", self.nasaKeyEdit.text())
    
    def load_settings(self, settings):
        self.nasaKeyEdit.setText(settings.value("API/NASAKey", ""))

class ThemeManager(QWidget):
    themeChanged = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Theme selection
        themeGroup = QGroupBox("Application Theme")
        themeLayout = QVBoxLayout()
        
        self.themeCombo = QComboBox()
        self.themeCombo.addItems(["System Default", "Light", "Dark", "Astronomy"])
        self.themeCombo.currentTextChanged.connect(lambda text: self.themeChanged.emit(text))
        
        themeLayout.addWidget(QLabel("Select Theme:"))
        themeLayout.addWidget(self.themeCombo)
        
        # Color Customization
        colorGroup = QGroupBox("Color Customization")
        colorLayout = QFormLayout()
        
        self.enableCustomColors = QCheckBox("Enable Custom Colors")
        self.enableCustomColors.toggled.connect(self._toggle_color_options)
        
        self.accentColorBtn = QPushButton()
        self.accentColorBtn.setEnabled(False)
        self.accentColorBtn.setFixedSize(80, 30)
        self.accentColorBtn.setStyleSheet("background-color: #0078D7;")
        
        self.bgColorBtn = QPushButton()
        self.bgColorBtn.setEnabled(False)
        self.bgColorBtn.setFixedSize(80, 30)
        self.bgColorBtn.setStyleSheet("background-color: #FFFFFF;")
        
        colorLayout.addRow("", self.enableCustomColors)
        colorLayout.addRow("Accent Color:", self.accentColorBtn)
        colorLayout.addRow("Background:", self.bgColorBtn)
        
        colorGroup.setLayout(colorLayout)
        
        # Font Options
        fontGroup = QGroupBox("Font Settings")
        fontLayout = QFormLayout()
        
        self.fontSizeSlider = QSlider(Qt.Orientation.Horizontal)
        self.fontSizeSlider.setRange(8, 16)
        self.fontSizeSlider.setValue(11)
        self.fontSizeSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.fontSizeSlider.setTickInterval(1)
        
        fontLayout.addRow("Font Size:", self.fontSizeSlider)
        fontGroup.setLayout(fontLayout)
        
        themeLayout.addWidget(colorGroup)
        themeLayout.addWidget(fontGroup)
        themeGroup.setLayout(themeLayout)
        
        layout.addWidget(themeGroup)
        layout.addStretch()
    
    def _toggle_color_options(self, enabled):
        self.accentColorBtn.setEnabled(enabled)
        self.bgColorBtn.setEnabled(enabled)
    
    def get_theme(self):
        return self.themeCombo.currentText()
    
    def set_theme(self, theme):
        index = self.themeCombo.findText(theme)
        if index >= 0:
            self.themeCombo.setCurrentIndex(index)
    
    def save_settings(self, settings):
        settings.setValue("Theme/CurrentTheme", self.themeCombo.currentText())
        settings.setValue("Theme/CustomColorsEnabled", self.enableCustomColors.isChecked())
        settings.setValue("Theme/FontSize", self.fontSizeSlider.value())
    
    def load_settings(self, settings):
        theme = settings.value("Theme/CurrentTheme", "System Default")
        index = self.themeCombo.findText(theme)
        if index >= 0:
            self.themeCombo.setCurrentIndex(index)
        
        custom_colors = settings.value("Theme/CustomColorsEnabled", False, type=bool)
        self.enableCustomColors.setChecked(custom_colors)
        
        font_size = settings.value("Theme/FontSize", 11, type=int)
        self.fontSizeSlider.setValue(font_size)

class DataManager(QWidget):
    dataSettingsChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Update intervals
        updateGroup = QGroupBox("Update Intervals")
        updateLayout = QFormLayout()
        
        self.planetUpdateSpin = QSpinBox()
        self.planetUpdateSpin.setRange(1, 60)
        self.planetUpdateSpin.setValue(5)
        self.planetUpdateSpin.setSuffix(" minutes")
        
        self.asteroidUpdateSpin = QSpinBox()
        self.asteroidUpdateSpin.setRange(1, 24)
        self.asteroidUpdateSpin.setValue(6)
        self.asteroidUpdateSpin.setSuffix(" hours")
        
        self.spaceWeatherUpdateSpin = QSpinBox()
        self.spaceWeatherUpdateSpin.setRange(5, 120)
        self.spaceWeatherUpdateSpin.setValue(30)
        self.spaceWeatherUpdateSpin.setSuffix(" minutes")
        
        updateLayout.addRow("Planetary Data:", self.planetUpdateSpin)
        updateLayout.addRow("Asteroid Data:", self.asteroidUpdateSpin)
        updateLayout.addRow("Space Weather:", self.spaceWeatherUpdateSpin)
        
        updateGroup.setLayout(updateLayout)
        layout.addWidget(updateGroup)
        
        # Storage settings
        storageGroup = QGroupBox("Data Storage")
        storageLayout = QFormLayout()
        
        self.cacheLocation = QLineEdit()
        self.cacheLocation.setReadOnly(True)
        
        cacheBrowseBtn = QPushButton("Browse...")
        cacheBrowseBtn.clicked.connect(self._browse_cache_location)
        
        cacheLayout = QHBoxLayout()
        cacheLayout.addWidget(self.cacheLocation)
        cacheLayout.addWidget(cacheBrowseBtn)
        
        self.maxCacheSizeSpin = QSpinBox()
        self.maxCacheSizeSpin.setRange(100, 10000)
        self.maxCacheSizeSpin.setValue(1000)
        self.maxCacheSizeSpin.setSuffix(" MB")
        
        self.clearCacheBtn = QPushButton("Clear Cache")
        self.clearCacheBtn.clicked.connect(self._clear_cache)
        
        self.autoClearCacheCheck = QCheckBox("Auto-clear cache on exit")
        
        storageLayout.addRow("Cache Location:", cacheLayout)
        storageLayout.addRow("Maximum Cache Size:", self.maxCacheSizeSpin)
        storageLayout.addRow("", self.autoClearCacheCheck)
        storageLayout.addRow("", self.clearCacheBtn)
        
        storageGroup.setLayout(storageLayout)
        layout.addWidget(storageGroup)
        
        # Data retention policy
        retentionGroup = QGroupBox("Data Retention")
        retentionLayout = QVBoxLayout()
        
        self.retentionButtonGroup = QButtonGroup(self)
        
        self.retain7Days = QRadioButton("Keep data for 7 days")
        self.retain30Days = QRadioButton("Keep data for 30 days")
        self.retainIndefinite = QRadioButton("Keep data indefinitely")
        
        self.retentionButtonGroup.addButton(self.retain7Days, 7)
        self.retentionButtonGroup.addButton(self.retain30Days, 30)
        self.retentionButtonGroup.addButton(self.retainIndefinite, 0)
        
        self.retain30Days.setChecked(True)
        
        retentionLayout.addWidget(self.retain7Days)
        retentionLayout.addWidget(self.retain30Days)
        retentionLayout.addWidget(self.retainIndefinite)
        
        retentionGroup.setLayout(retentionLayout)
        layout.addWidget(retentionGroup)
        
        # Download settings
        downloadGroup = QGroupBox("Downloads")
        downloadLayout = QFormLayout()
        
        self.downloadLocation = QLineEdit()
        self.downloadLocation.setReadOnly(True)
        
        downloadBrowseBtn = QPushButton("Browse...")
        downloadBrowseBtn.clicked.connect(self._browse_download_location)
        
        downloadLayout = QHBoxLayout()
        downloadLayout.addWidget(self.downloadLocation)
        downloadLayout.addWidget(downloadBrowseBtn)
        
        self.autoExportCheck = QCheckBox("Automatically export reports")
        self.exportFormatCombo = QComboBox()
        self.exportFormatCombo.addItems(["PDF", "HTML", "CSV", "JSON"])
        
        downloadFormLayout = QFormLayout()
        downloadFormLayout.addRow("Save Location:", downloadLayout)
        downloadFormLayout.addRow("", self.autoExportCheck)
        downloadFormLayout.addRow("Export Format:", self.exportFormatCombo)
        
        downloadGroup.setLayout(downloadFormLayout)
        layout.addWidget(downloadGroup)
        
        layout.addStretch()
    
    def _browse_cache_location(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Cache Directory", self.cacheLocation.text()
        )
        if directory:
            self.cacheLocation.setText(directory)
    
    def _browse_download_location(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Download Directory", self.downloadLocation.text()
        )
        if directory:
            self.downloadLocation.setText(directory)
    
    def _clear_cache(self):
        reply = QMessageBox.question(
            self, 
            "Clear Cache", 
            "Are you sure you want to clear all cached data? This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # In a real implementation, this would clear the cache directory
            QMessageBox.information(self, "Cache Cleared", "All cached data has been cleared.")
    
    def get_data_settings(self):
        return {
            "planet_update_interval": self.planetUpdateSpin.value(),
            "asteroid_update_interval": self.asteroidUpdateSpin.value(),
            "space_weather_update_interval": self.spaceWeatherUpdateSpin.value(),
            "cache_location": self.cacheLocation.text(),
            "max_cache_size": self.maxCacheSizeSpin.value(),
            "auto_clear_cache": self.autoClearCacheCheck.isChecked(),
            "retention_days": self.retentionButtonGroup.checkedId(),
            "download_location": self.downloadLocation.text(),
            "auto_export": self.autoExportCheck.isChecked(),
            "export_format": self.exportFormatCombo.currentText()
        }
    
    def save_settings(self, settings):
        settings.setValue("Data/PlanetUpdateInterval", self.planetUpdateSpin.value())
        settings.setValue("Data/AsteroidUpdateInterval", self.asteroidUpdateSpin.value())
        settings.setValue("Data/SpaceWeatherUpdateInterval", self.spaceWeatherUpdateSpin.value())
        settings.setValue("Data/CacheLocation", self.cacheLocation.text())
        settings.setValue("Data/MaxCacheSize", self.maxCacheSizeSpin.value())
        settings.setValue("Data/AutoClearCache", self.autoClearCacheCheck.isChecked())
        settings.setValue("Data/RetentionDays", self.retentionButtonGroup.checkedId())
        settings.setValue("Data/DownloadLocation", self.downloadLocation.text())
        settings.setValue("Data/AutoExport", self.autoExportCheck.isChecked())
        settings.setValue("Data/ExportFormat", self.exportFormatCombo

import os
import json
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (
    QDialog, QTabWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QWidget, QFormLayout, QLineEdit, QLabel, QComboBox, QCheckBox,
    QSpinBox, QGroupBox, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings, QSize
from PyQt6.QtGui import QColor, QPalette

class APIKeyManager(QWidget):
    """Widget for managing API keys configuration."""
    
    api_keys_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("CosmicSentinel", "CosmicSentinel")
        self._api_keys = {}
        self._init_ui()
        self._load_api_keys()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # NASA API Key
        nasa_group = QGroupBox("NASA API Key")
        nasa_layout = QFormLayout()
        
        self.nasa_key_edit = QLineEdit()
        self.nasa_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.nasa_key_edit.setPlaceholderText("Enter your NASA API key")
        
        nasa_layout.addRow("API Key:", self.nasa_key_edit)
        
        show_key_btn = QPushButton("Show Key")
        show_key_btn.setCheckable(True)
        show_key_btn.toggled.connect(self._toggle_nasa_key_visibility)
        
        get_key_btn = QPushButton("Get NASA API Key")
        get_key_btn.clicked.connect(lambda: self._open_api_key_website("https://api.nasa.gov/"))
        
        key_btns_layout = QHBoxLayout()
        key_btns_layout.addWidget(show_key_btn)
        key_btns_layout.addWidget(get_key_btn)
        
        nasa_layout.addRow("", key_btns_layout)
        nasa_group.setLayout(nasa_layout)
        layout.addWidget(nasa_group)
        
        # NOAA Space Weather API
        noaa_group = QGroupBox("NOAA Space Weather API")
        noaa_layout = QFormLayout()
        
        self.noaa_key_edit = QLineEdit()
        self.noaa_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.noaa_key_edit.setPlaceholderText("Enter your NOAA API key (if required)")
        
        noaa_layout.addRow("API Key:", self.noaa_key_edit)
        
        show_noaa_key_btn = QPushButton("Show Key")
        show_noaa_key_btn.setCheckable(True)
        show_noaa_key_btn.toggled.connect(self._toggle_noaa_key_visibility)
        
        get_noaa_key_btn = QPushButton("Get NOAA API Key")
        get_noaa_key_btn.clicked.connect(lambda: self._open_api_key_website("https://www.weather.gov/documentation/services-web-api"))
        
        noaa_btns_layout = QHBoxLayout()
        noaa_btns_layout.addWidget(show_noaa_key_btn)
        noaa_btns_layout.addWidget(get_noaa_key_btn)
        
        noaa_layout.addRow("", noaa_btns_layout)
        noaa_group.setLayout(noaa_layout)
        layout.addWidget(noaa_group)
        
        # API Usage information
        usage_label = QLabel(
            "API keys are required for accessing data from external services. "
            "Each key has a request limit - please check the provider's documentation "
            "for details. Your keys are stored securely on your local machine."
        )
        usage_label.setWordWrap(True)
        layout.addWidget(usage_label)
        
        layout.addStretch()
    
    def _toggle_nasa_key_visibility(self, show: bool):
        self.nasa_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if show else QLineEdit.EchoMode.Password
        )
    
    def _toggle_noaa_key_visibility(self, show: bool):
        self.noaa_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if show else QLineEdit.EchoMode.Password
        )
    
    def _open_api_key_website(self, url: str):
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl(url))
    
    def _load_api_keys(self):
        """Load API keys from settings."""
        if self.settings.contains("api_keys"):
            encrypted_keys = self.settings.value("api_keys")
            try:
                # In a real app, this would use proper encryption/decryption
                # This is a simplified version for demonstration
                self._api_keys = json.loads(encrypted_keys)
                
                if "nasa" in self._api_keys:
                    self.nasa_key_edit.setText(self._api_keys["nasa"])
                
                if "noaa" in self._api_keys:
                    self.noaa_key_edit.setText(self._api_keys["noaa"])
            except Exception as e:
                print(f"Error loading API keys: {e}")
    
    def save_api_keys(self):
        """Save API keys to settings."""
        self._api_keys = {
            "nasa": self.nasa_key_edit.text().strip(),
            "noaa": self.noaa_key_edit.text().strip()
        }
        
        # In a real app, this would use proper encryption
        # This is a simplified version for demonstration
        encrypted_keys = json.dumps(self._api_keys)
        self.settings.setValue("api_keys", encrypted_keys)
        
        self.api_keys_changed.emit(self._api_keys)
        return self._api_keys
    
    def get_api_keys(self) -> Dict[str, str]:
        """Get the current API keys."""
        return self._api_keys


class ThemeManager(QWidget):
    """Widget for managing application theme and appearance."""
    
    theme_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("CosmicSentinel", "CosmicSentinel")
        self._init_ui()
        self._load_theme_settings()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Theme Selection
        theme_group = QGroupBox("Application Theme")
        theme_layout = QFormLayout()
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["System Default", "Light", "Dark", "Astronomy"])
        theme_layout.addRow("Theme:", self.theme_combo)
        
        # Font Size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 16)
        self.font_size_spin.setValue(10)
        theme_layout.addRow("Font Size:", self.font_size_spin)
        
        # Icon Size
        self.icon_size_combo = QComboBox()
        self.icon_size_combo.addItems(["Small", "Medium", "Large"])
        theme_layout.addRow("Icon Size:", self.icon_size_combo)
        
        # High contrast mode
        self.high_contrast_check = QCheckBox("High Contrast Mode")
        self.high_contrast_check.setToolTip("Enhances visibility with higher contrast colors")
        theme_layout.addRow("", self.high_contrast_check)
        
        # Preview section could be added here
        
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)
        
        # Visualization preferences
        viz_group = QGroupBox("Visualization Preferences")
        viz_layout = QFormLayout()
        
        self.grid_check = QCheckBox("Show Grid")
        self.grid_check.setChecked(True)
        viz_layout.addRow("", self.grid_check)
        
        self.labels_check = QCheckBox("Show Labels")
        self.labels_check.setChecked(True)
        viz_layout.addRow("", self.labels_check)
        
        # Color scheme for visualizations
        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems([
            "Standard", "Colorblind-friendly", "Print-optimized", "Night Vision"
        ])
        viz_layout.addRow("Color Scheme:", self.color_scheme_combo)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        layout.addStretch()
    
    def _load_theme_settings(self):
        """Load theme settings from QSettings."""
        theme = self.settings.value("theme/current", "System Default")
        font_size = int(self.settings.value("theme/font_size", 10))
        icon_size = self.settings.value("theme/icon_size", "Medium")
        high_contrast = self.settings.value("theme/high_contrast", False, type=bool)
        
        # Visualization settings
        show_grid = self.settings.value("visualization/show_grid", True, type=bool)
        show_labels = self.settings.value("visualization/show_labels", True, type=bool)
        color_scheme = self.settings.value("visualization/color_scheme", "Standard")
        
        # Apply loaded settings to UI
        self.theme_combo.setCurrentText(theme)
        self.font_size_spin.setValue(font_size)
        self.icon_size_combo.setCurrentText(icon_size)
        self.high_contrast_check.setChecked(high_contrast)
        
        self.grid_check.setChecked(show_grid)
        self.labels_check.setChecked(show_labels)
        self.color_scheme_combo.setCurrentText(color_scheme)
    
    def save_theme_settings(self):
        """Save theme settings to QSettings."""
        theme = self.theme_combo.currentText()
        font_size = self.font_size_spin.value()
        icon_size = self.icon_size_combo.currentText()
        high_contrast = self.high_contrast_check.isChecked()
        
        # Visualization settings
        show_grid = self.grid_check.isChecked()
        show_labels = self.labels_check.isChecked()
        color_scheme = self.color_scheme_combo.currentText()
        
        # Save to QSettings
        self.settings.setValue("theme/current", theme)
        self.settings.setValue("theme/font_size", font_size)
        self.settings.setValue("theme/icon_size", icon_size)
        self.settings.setValue("theme/high_contrast", high_contrast)
        
        self.settings.setValue("visualization/show_grid", show_grid)
        self.settings.setValue("visualization/show_labels", show_labels)
        self.settings.setValue("visualization/color_scheme", color_scheme)
        
        # Emit signal
        self.theme_changed.emit(theme)
        
        return {
            "theme": theme,
            "font_size": font_size,
            "icon_size": icon_size,
            "high_contrast": high_contrast,
            "show_grid": show_grid,
            "show_labels": show_labels,
            "color_scheme": color_scheme
        }


class DataManager(QWidget):
    """Widget for managing data settings."""
    
    data_settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("CosmicSentinel", "CosmicSentinel")
        self._init_ui()
        self._load_data_settings()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Cache settings
        cache_group = QGroupBox("Data Cache Settings")
        cache_layout = QFormLayout()
        
        self.cache_enabled_check = QCheckBox("Enable Data Caching")
        self.cache_enabled_check.setChecked(True)
        cache_layout.addRow("", self.cache_enabled_check)
        
        self.cache_size_spin = QSpinBox()
        self.cache_size_spin.setRange(100, 5000)
        self.cache_size_spin.setValue(500)
        self.cache_size_spin.setSuffix(" MB")
        cache_layout.addRow("Maximum Cache Size:", self.cache_size_spin)
        
        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.clicked.connect(self._clear_cache)
        cache_layout.addRow("", self.clear_cache_btn)
        
        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)
        
        # Data update settings
        update_group = QGroupBox("Data Update Settings")
        update_layout = QFormLayout()
        
        self.auto_update_check = QCheckBox("Enable Automatic Updates")
        self.auto_update_check.setChecked(True)
        update_layout.addRow("", self.auto_update_check)
        
        self.update_interval_spin = QSpinBox()
        self.update_interval_spin.setRange(1, 60)
        self.update_interval_spin.setValue(15)
        self.update_interval_spin.setSuffix(" minutes")
        update_layout.addRow("Update Interval:", self.update_interval_spin)
        
        # Update options
        update_options_group = QGroupBox("Update Data Types")
        update_options_layout = QVBoxLayout()
        
        self.update_planets_check = QCheckBox("Planetary Data")
        self.update_planets_check.setChecked(True)
        
        self.update_asteroids_check = QCheckBox("Asteroid Data")
        self.update_asteroids_check.setChecked(True)
        
        self.update_weather_check = QCheckBox("Space Weather")
        self.update_weather_check.setChecked(True)
        
        update_options_layout.addWidget(self.update_planets_check)
        update_options_layout.addWidget(self.update_asteroids_check)
        update_options_layout.addWidget(self.update_weather_check)
        
        update_options_group.setLayout(update_options_layout)
        update_layout.addRow("", update_options_group)
        
        update_group.setLayout(update_layout)
        layout.addWidget(update_group)
        
        # Data storage location
        storage_group = QGroupBox("Data Storage Location")

        self.email_notify.setChecked(self.config_manager.get_notification_setting("email"))
        notify_layout.addRow("Email Notifications:", self.email_notify)
        
        self.sound_notify = QCheckBox()
        self.sound_notify.setChecked(self.config_manager.get_notification_setting("sound"))
        notify_layout.addRow("Sound Alerts:", self.sound_notify)
        
        notify_group.setLayout(notify_layout)
        
        # Email settings
        email_group = QGroupBox("Email Settings")
        email_group.setEnabled(self.email_notify.isChecked())
        self.email_notify.toggled.connect(email_group.setEnabled)
        
        email_layout = QFormLayout()
        
        self.email_address = QLineEdit(self.config_manager.get_email_address())
        email_layout.addRow("Email Address:", self.email_address)
        
        self.email_freq = QComboBox()
        self.email_freq.addItems(["Immediate", "Hourly Digest", "Daily Digest"])
        self.email_freq.setCurrentText(self.config_manager.get_email_frequency())
        email_layout.addRow("Email Frequency:", self.email_freq)
        
        email_group.setLayout(email_layout)
        
        # Alert thresholds
        threshold_group = QGroupBox("Alert Thresholds")
        threshold_layout = QFormLayout()
        
        self.asteroid_proximity = QSpinBox()
        self.asteroid_proximity.setRange(1, 10000)
        self.asteroid_proximity.setSuffix(" LD")  # Lunar Distance
        self.asteroid_proximity.setValue(self.config_manager.get_threshold("asteroid_proximity"))
        threshold_layout.addRow("Asteroid Proximity Alert:", self.asteroid_proximity)
        
        self.planet_events = QCheckBox()
        self.planet_events.setChecked(self.config_manager.get_threshold("planet_events"))
        threshold_layout.addRow("Planetary Events:", self.planet_events)
        
        self.urgent_only = QCheckBox()
        self.urgent_only.setChecked(self.config_manager.get_notification_setting("urgent_only"))
        threshold_layout.addRow("Urgent Alerts Only:", self.urgent_only)
        
        threshold_group.setLayout(threshold_layout)
        
        # Add all groups to main layout

import os
import json
from typing import Dict, Any, Optional, List, Tuple

from PyQt6.QtWidgets import (
    QDialog, QTabWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QCheckBox, QGroupBox, QMessageBox, QFileDialog, QRadioButton,
    QButtonGroup, QSlider, QColorDialog, QWidget
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal, QUrl
from PyQt6.QtGui import QColor, QIcon

class APIKeyManager(QWidget):
    """Widget for managing API keys for various services used by the application."""
    
    keys_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.api_keys = {}
        self._load_api_keys()
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # NASA API key section
        nasa_group = QGroupBox("NASA API Keys")
        nasa_layout = QFormLayout()
        
        self.nasa_api_key = QLineEdit()
        self.nasa_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.nasa_api_key.setText(self.api_keys.get("nasa", ""))
        self.show_nasa_key = QPushButton("Show")
        self.show_nasa_key.setCheckable(True)
        self.show_nasa_key.toggled.connect(
            lambda checked: self.nasa_api_key.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        
        nasa_key_layout = QHBoxLayout()
        nasa_key_layout.addWidget(self.nasa_api_key)
        nasa_key_layout.addWidget(self.show_nasa_key)
        
        nasa_layout.addRow("NASA API Key:", nasa_key_layout)
        self.nasa_api_status = QLabel("Status: Not verified")
        nasa_layout.addRow("", self.nasa_api_status)
        
        verify_nasa = QPushButton("Verify Key")
        verify_nasa.clicked.connect(lambda: self._verify_api_key("nasa", self.nasa_api_key.text()))
        nasa_layout.addRow("", verify_nasa)
        
        nasa_group.setLayout(nasa_layout)
        layout.addWidget(nasa_group)
        
        # Space-Track.org API key section
        spacetrack_group = QGroupBox("Space-Track.org Credentials")
        spacetrack_layout = QFormLayout()
        
        self.spacetrack_username = QLineEdit()
        self.spacetrack_username.setText(self.api_keys.get("spacetrack_username", ""))
        spacetrack_layout.addRow("Username:", self.spacetrack_username)
        
        self.spacetrack_password = QLineEdit()
        self.spacetrack_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.spacetrack_password.setText(self.api_keys.get("spacetrack_password", ""))
        
        self.show_spacetrack_pwd = QPushButton("Show")
        self.show_spacetrack_pwd.setCheckable(True)
        self.show_spacetrack_pwd.toggled.connect(
            lambda checked: self.spacetrack_password.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        
        spacetrack_pwd_layout = QHBoxLayout()
        spacetrack_pwd_layout.addWidget(self.spacetrack_password)
        spacetrack_pwd_layout.addWidget(self.show_spacetrack_pwd)
        
        spacetrack_layout.addRow("Password:", spacetrack_pwd_layout)
        
        self.spacetrack_status = QLabel("Status: Not verified")
        spacetrack_layout.addRow("", self.spacetrack_status)
        
        verify_spacetrack = QPushButton("Verify Credentials")
        verify_spacetrack.clicked.connect(self._verify_spacetrack)
        spacetrack_layout.addRow("", verify_spacetrack)
        
        spacetrack_group.setLayout(spacetrack_layout)
        layout.addWidget(spacetrack_group)
        
        # Weather service API key (for space weather alerts)
        weather_group = QGroupBox("Weather Services")
        weather_layout = QFormLayout()
        
        self.weather_api_key = QLineEdit()
        self.weather_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.weather_api_key.setText(self.api_keys.get("weather", ""))
        
        self.show_weather_key = QPushButton("Show")
        self.show_weather_key.setCheckable(True)
        self.show_weather_key.toggled.connect(
            lambda checked: self.weather_api_key.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        
        weather_key_layout = QHBoxLayout()
        weather_key_layout.addWidget(self.weather_api_key)
        weather_key_layout.addWidget(self.show_weather_key)
        
        weather_layout.addRow("Space Weather API Key:", weather_key_layout)
        weather_group.setLayout(weather_layout)
        layout.addWidget(weather_group)
        
        # Save button
        save_btn = QPushButton("Save API Keys")
        save_btn.clicked.connect(self._save_api_keys)
        layout.addWidget(save_btn)
        
        # Add spacer at the end
        layout.addStretch()
    
    def _load_api_keys(self):
        """Load API keys from secure storage."""
        try:
            settings = QSettings("CosmicSentinel", "ApiKeys")
            self.api_keys = {
                "nasa": settings.value("nasa", ""),
                "spacetrack_username": settings.value("spacetrack_username", ""),
                "spacetrack_password": settings.value("spacetrack_password", ""),
                "weather": settings.value("weather", "")
            }
        except Exception as e:
            QMessageBox.warning(
                self, 
                "API Key Loading Error",
                f"Could not load API keys: {str(e)}"
            )
    
    def _save_api_keys(self):
        """Save API keys to secure storage."""
        try:
            # Update the keys dictionary
            self.api_keys = {
                "nasa": self.nasa_api_key.text(),
                "spacetrack_username": self.spacetrack_username.text(),
                "spacetrack_password": self.spacetrack_password.text(),
                "weather": self.weather_api_key.text()
            }
            
            # Save to QSettings
            settings = QSettings("CosmicSentinel", "ApiKeys")
            for key, value in self.api_keys.items():
                settings.setValue(key, value)
            
            # Emit signal that keys have been updated
            self.keys_updated.emit(self.api_keys)
            
            QMessageBox.information(
                self,
                "API Keys Saved",
                "Your API keys have been saved successfully."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "API Key Save Error",
                f"Could not save API keys: {str(e)}"
            )
    
    def _verify_api_key(self, service: str, key: str):
        """Verify API key for the specified service."""
        # This would be implemented with actual API validation
        # For now, just check if key is not empty
        if not key:
            self._update_key_status(service, False, "API key cannot be empty")
            return
        
        # Here you would make an actual API call to verify the key
        # Simulating a successful verification for now
        self._update_key_status(service, True, "API key verified successfully")
    
    def _verify_spacetrack(self):
        """Verify Space-Track.org credentials."""
        username = self.spacetrack_username.text()
        password = self.spacetrack_password.text()
        
        if not username or not password:
            self.spacetrack_status.setText("Status: Missing credentials")
            self.spacetrack_status.setStyleSheet("color: red;")
            return
        
        # Here you would make an actual API call to verify the credentials
        # Simulating a successful verification for now
        self.spacetrack_status.setText("Status: Credentials verified")
        self.spacetrack_status.setStyleSheet("color: green;")
    
    def _update_key_status(self, service: str, is_valid: bool, message: str):
        """Update the status label for an API key verification attempt."""
        if service == "nasa":
            self.nasa_api_status.setText(f"Status: {message}")
            self.nasa_api_status.setStyleSheet(
                "color: green;" if is_valid else "color: red;"
            )


class ThemeManager(QWidget):
    """Widget for managing UI theme settings."""
    
    theme_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_settings = self._load_theme_settings()
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Theme selection
        theme_group = QGroupBox("UI Theme")
        theme_layout = QVBoxLayout()
        
        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["Light", "Dark", "System Default", "Custom"])
        self.theme_selector.setCurrentText(self.theme_settings.get("theme", "System Default"))
        self.theme_selector.currentTextChanged.connect(self._on_theme_changed)
        
        theme_layout.addWidget(QLabel("Select Theme:"))
        theme_layout.addWidget(self.theme_selector)
        
        # Custom theme color pickers (initially hidden)
        self.custom_colors_group = QGroupBox("Custom Theme Colors")
        self.custom_colors_group.setVisible(self.theme_selector.currentText() == "Custom")
        
        custom_colors_layout = QFormLayout()
        
        # Primary color
        self.primary_color = QColor(self.theme_settings.get("primary_color", "#1976D2"))
        self.primary_color_btn = QPushButton()
        self.primary_color_btn.setStyleSheet(
            f"background-color: {self.primary_color.name()}; min-width: 50px; min-height: 30px;"
        )
        self.primary_color_btn.clicked.connect(lambda: self._choose_color("primary"))
        custom_colors_layout.addRow("Primary Color:", self.primary_color_btn)
        
        # Secondary color
        self.secondary_color = QColor(self.theme_settings.get("secondary_color", "#26A69A"))
        self.secondary_color_btn = QPushButton()
        self.secondary_color_btn.setStyleSheet(
            f"background-color: {self.secondary_color.name()}; min-width: 50px; min-height: 30px;"
        )
        self.secondary_color_btn.clicked.connect(lambda: self._choose_color("secondary"))
        custom_colors_layout.addRow("Secondary Color:", self.secondary_color_btn)
        
        # Accent color
        self.accent_color = QColor(self.theme_settings.get("accent_color", "#FF9800"))
        self.accent_color_btn = QPushButton()
        self.accent_color_btn.setStyleSheet(
            f"background-color: {self.accent_color.name()}; min-width: 50px; min-height: 30px;"
        )
        self.accent_color_btn.clicked.connect(lambda: self._choose_color("accent"))
        custom_colors_layout.addRow("Accent Color:", self.accent_color_btn)
        
        # Background color
        self.background_color = QColor(self.theme_settings.get("background_color", "#F5F5F5"))
        self.background_color_btn = QPushButton()
        self.background_color_btn.setStyleSheet(
            f"background-color: {self.background_color.name()}; min-width: 50px; min-height: 30px;"
        )
        self.background_color_btn.clicked.connect(lambda: self._choose_color("background"))
        custom_colors_layout.addRow("Background Color:", self.background_color_btn)
        
        # Text color
        self.text_color = QColor(self.theme_settings.get("text_color", "#212121"))
        self.text_color_btn = QPushButton()
        self.text_color_btn.setStyleSheet(
            f"background-color: {self.text_color.name()}; min-width: 50px; min-height: 30px;"
        )
        self.text_color_btn.clicked.connect(lambda: self._choose_color("text"))
        custom_colors_layout.addRow("Text Color:", self.text_color_btn)
        
        self.custom_colors_group.setLayout(custom_colors_layout)
        
        # Font settings
        font_group = QGroupBox("Font Settings")
        font_layout = QFormLayout()
        
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 20)
        self.font_size_spin.setValue(int(self.theme_settings.get("font_size", 12)))
        font_layout.addRow("Font Size:", self.font_size_spin)
        
        font_group.setLayout(font_layout)
        
        # Add all widgets to the main layout
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)
        layout.addWidget(self.custom_colors_group)
        layout.addWidget(font_group)
        
        # Animation settings
        animation_group = QGroupBox("UI Animation")
        animation_layout = QFormLayout()
        
        self.enable_animations = QCheckBox()
        self.enable_animations.setChecked(self.theme_settings.get("enable_animations", True))
        animation_layout.addRow("Enable UI Animations:", self.enable_animations)
        
        animation_speed_label = QLabel("Animation Speed:")
        self.animation_speed = QSlider(Qt.Orientation.Horizontal)
        self.animation_speed.setRange(1, 5)
        self.animation_speed.setValue(int(self.theme_settings.get("animation_speed", 3)))
        self.animation_speed.setEnabled(self.enable_animations.isChecked())
        self.enable_animations.toggled.connect(self.animation_speed.setEnabled)
        
        animation_layout.addRow(animation_speed_label, self.animation_speed)
        animation_group.setLayout(animation_layout)
        layout.addWidget(animation_group)
        
        #

class NotificationManager(QWidget):
    """Widget for managing notification settings."""
    
    notification_settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("CosmicSentinel", "CosmicSentinel")
        self._init_ui()
    
    def _init_ui(self):
        self.layout = QVBoxLayout(self)
        
        # Enable/disable notifications
        self.enable_notifications = QCheckBox("Enable Notifications")
        self.enable_notifications.setChecked(True)
        self.enable_notifications.toggled.connect(self._toggle_notifications)
        self.layout.addWidget(self.enable_notifications)
        
        # Notification types
        self.notification_types = QGroupBox("Notification Types")
        types_layout = QVBoxLayout()
        
        self.asteroid_alerts = QCheckBox("Asteroid Approach Alerts")
        self.asteroid_alerts.setChecked(True)
        
        self.planetary_events = QCheckBox("Planetary Events")
        self.planetary_events.setChecked(True)
        
        self.space_weather = QCheckBox("Space Weather Alerts")
        self.space_weather.setChecked(True)
        
        self.system_notifications = QCheckBox("System Notifications")
        self.system_notifications.setChecked(True)
        
        types_layout.addWidget(self.asteroid_alerts)
        types_layout.addWidget(self.planetary_events)
        types_layout.addWidget(self.space_weather)
        types_layout.addWidget(self.system_notifications)
        
        self.notification_types.setLayout(types_layout)
        self.layout.addWidget(self.notification_types)
        
        # Notification methods
        self.notification_methods = QGroupBox("Notification Methods")
        methods_layout = QFormLayout()
        
        self.desktop_notifications = QCheckBox()
        self.desktop_notifications.setChecked(True)
        methods_layout.addRow("Desktop Notifications:", self.desktop_notifications)
        
        self.email_notifications = QCheckBox()
        self.email_notifications.setChecked(False)
        self.email_notifications.toggled.connect(self._toggle_email_options)
        methods_layout.addRow("Email Notifications:", self.email_notifications)
        
        self.email_address = QLineEdit()
        self.email_address.setPlaceholderText("Enter your email address")
        self.email_address.setEnabled(False)
        methods_layout.addRow("Email Address:", self.email_address)
        
        self.notification_methods.setLayout(methods_layout)
        self.layout.addWidget(self.notification_methods)
        
        # Alert levels
        self.alert_levels = QGroupBox("Alert Thresholds")
        alert_layout = QFormLayout()
        
        # Proximity threshold
        self.proximity_threshold = QSpinBox()
        self.proximity_threshold.setRange(1, 50)
        self.proximity_threshold.setValue(5)
        self.proximity_threshold.setSingleStep(1)
        self.proximity_label = QLabel("5 lunar distances")
        self.proximity_threshold.valueChanged.connect(self._update_proximity_label)
        
        proximity_layout = QHBoxLayout()
        proximity_layout.addWidget(self.proximity_threshold)
        proximity_layout.addWidget(self.proximity_label)
        
        alert_layout.addRow("NEO Proximity Alert:", proximity_layout)
        
        # Solar activity level
        self.solar_activity = QComboBox()
        self.solar_activity.addItems(["Low (C-class)", "Medium (M-class)", "High (X-class)"])
        alert_layout.addRow("Solar Activity Alert:", self.solar_activity)
        
        self.alert_levels.setLayout(alert_layout)
        self.layout.addWidget(self.alert_levels)
    
    def _toggle_notifications(self, enabled):
        self.notification_types.setEnabled(enabled)
        self.notification_methods.setEnabled(enabled)
        self.alert_levels.setEnabled(enabled)
    
    def _toggle_email_options(self, enabled):
        self.email_address.setEnabled(enabled)
    
    def _update_proximity_label(self, value):
        self.proximity_label.setText(f"{value} lunar distances")
    
    def get_notification_settings(self):
        return {
            "enabled": self.enable_notifications.isChecked(),
            "types": {
                "asteroid_alerts": self.asteroid_alerts.isChecked(),
                "planetary_events": self.planetary_events.isChecked(),
                "space_weather": self.space_weather.isChecked(),
                "system": self.system_notifications.isChecked()
            },
            "methods": {
                "desktop": self.desktop_notifications.isChecked(),
                "email": self.email_notifications.isChecked(),
                "email_address": self.email_address.text()
            },
            "thresholds": {
                "neo_proximity": self.proximity_threshold.value(),
                "solar_activity": self.solar_activity.currentText()
            }
        }
    
    def set_notification_settings(self, settings):
        if not settings:
            return
            
        self.enable_notifications.setChecked(settings.get("enabled", True))
        
        types = settings.get("types", {})
        self.asteroid_alerts.setChecked(types.get("asteroid_alerts", True))
        self.planetary_events.setChecked(types.get("planetary_events", True))
        self.space_weather.setChecked(types.get("space_weather", True))
        self.system_notifications.setChecked(types.get("system", True))
        
        methods = settings.get("methods", {})
        self.desktop_notifications.setChecked(methods.get("desktop", True))
        self.email_notifications.setChecked(methods.get("email", False))
        self.email_address.setText(methods.get("email_address", ""))
        
        thresholds = settings.get("thresholds", {})
        self.proximity_threshold.setValue(thresholds.get("neo_proximity", 5))
        
        solar_index = self.solar_activity.findText(thresholds.get("solar_activity", "Medium (M-class)"))
        if solar_index >= 0:
            self.solar_activity.setCurrentIndex(solar_index)


# Main SettingsDialog class
class SettingsDialog(QDialog):
    settings_applied = pyqtSignal(dict)  # Emitted when settings are applied
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("CosmicSentinel", "Settings")
        self._init_ui()
        self._load_settings()
    
    def _init_ui(self):
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # API Keys tab
        self.api_manager = APIKeyManager()
        tabs.addTab(self.api_manager, "API Keys")
        
        # Theme tab
        self.theme_manager = ThemeManager()
        tabs.addTab(self.theme_manager, "Theme")
        
        # Data Management tab
        self.data_manager = DataManager()
        tabs.addTab(self.data_manager, "Data")
        
        # Notifications tab
        self.notification_manager = NotificationManager()
        tabs.addTab(self.notification_manager, "Notifications")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        restore_defaults = QPushButton("Restore Defaults")
        restore_defaults.clicked.connect(self._restore_defaults)
        button_layout.addWidget(restore_defaults)
        
        button_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self._apply_settings)
        button_layout.addWidget(apply_button)
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self._ok_clicked)
        ok_button.setDefault(True)
        button_layout.addWidget(ok_button)
        
        layout.addLayout(button_layout)
    
    def _load_settings(self):
        # Load all saved settings and apply them to the UI
        self.api_manager.load_settings(self.settings)
        self.theme_manager.set_theme_settings(self._load_theme_settings())
        self.data_manager.set_data_settings(self._load_data_settings())
        self.notification_manager.set_notification_settings(self._load_notification_settings())
    
    def _save_settings(self):
        # Save all current settings
        self.api_manager.save_settings(self.settings)
        
        # Save theme settings
        theme_settings = self.theme_manager.get_theme_settings()
        for key, value in theme_settings.items():
            self.settings.setValue(f"theme/{key}", value)
        
        # Save data settings
        data_settings = self.data_manager.get_data_settings()
        for key, value in data_settings.items():
            self.settings.setValue(f"data/{key}", value)
        
        # Save notification settings
        notification_settings = self.notification_manager.get_notification_settings()
        self.settings.setValue("notifications", json.dumps(notification_settings))
        
        self.settings.sync()
    
    def _apply_settings(self):
        settings = {
            "api": {
                "nasa": self.api_manager.get_nasa_key()
            },
            "theme": self.theme_manager.get_theme_settings(),
            "data": self.data_manager.get_data_settings(),
            "notifications": self.notification_manager.get_notification_settings()
        }
        
        self._save_settings()
        self.settings_applied.emit(settings)
    
    def _ok_clicked(self):
        self._apply_settings()
        self.accept()
    
    def _restore_defaults(self):
        reply = QMessageBox.question(
            self,
            "Restore Defaults",
            "Are you sure you want to restore all settings to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.settings.clear()
            self._load_settings()
    
    def _load_theme_settings(self):
        return {
            "dark_mode": self.settings.value("theme/dark_mode", False, type=bool),
            "primary_color": self.settings.value("theme/primary_color", "#0078D7"),
            "accent_color": self.settings.value("theme/accent_color", "#00CC00"),
            "font_size": self.settings.value("theme/font_size", 10, type=int)
        }
    
    def _load_data_settings(self):
        return {
            "planet_update_interval": self.settings.value("data/planet_update_interval", 5, type=int),
            "asteroid_update_interval": self.settings.value("data/asteroid_update_interval", 60, type=int),
            "cache_directory": self.settings.value("data/cache_directory", ""),
            "cache_size_limit": self.settings.value("data/cache_size_limit", 
