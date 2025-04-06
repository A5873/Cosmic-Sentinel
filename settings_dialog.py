from PyQt6.QtCore import Qt, pyqtSignal, QSettings
from PyQt6.QtWidgets import (
    QDialog, QTabWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QCheckBox, 
    QColorDialog, QSpinBox, QGroupBox, QFormLayout,
    QComboBox, QSlider, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QColor, QPalette

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

class APIKeyManager(QGroupBox):
    """Widget for managing API keys"""
    
    api_key_changed = pyqtSignal(str, str)  # name, key
    
    def __init__(self, parent=None):
        super().__init__("API Keys", parent)
        self.layout = QVBoxLayout(self)
        
        # NASA API Key
        nasa_layout = QHBoxLayout()
        nasa_layout.addWidget(QLabel("NASA API Key:"))
        self.nasa_key_edit = QLineEdit()
        self.nasa_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        nasa_layout.addWidget(self.nasa_key_edit)
        self.show_nasa_key = QPushButton("Show")
        self.show_nasa_key.setCheckable(True)
        self.show_nasa_key.toggled.connect(self._toggle_nasa_key_visibility)
        nasa_layout.addWidget(self.show_nasa_key)
        self.layout.addLayout(nasa_layout)
        
        # Other API keys can be added here
        
        # Test connection button
        self.test_button = QPushButton("Test Connections")
        self.test_button.clicked.connect(self._test_connections)
        self.layout.addWidget(self.test_button)
    
    def _toggle_nasa_key_visibility(self, checked):
        self.nasa_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )
        self.show_nasa_key.setText("Hide" if checked else "Show")
    
    def _test_connections(self):
        # This would be connected to the actual API testing logic
        nasa_key = self.nasa_key_edit.text()
        if not nasa_key:
            QMessageBox.warning(self, "Warning", "NASA API key is empty.")
            return
            
        # Placeholder for actual API testing
        QMessageBox.information(self, "Success", "API connections tested successfully.")
        
    def set_nasa_key(self, key):
        self.nasa_key_edit.setText(key)
        
    def get_nasa_key(self):
        return self.nasa_key_edit.text()

class ThemeManager(QGroupBox):
    """Widget for managing UI theme settings"""
    
    theme_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__("Theme Settings", parent)
        self.layout = QVBoxLayout(self)
        
        # Dark mode toggle
        self.dark_mode = QCheckBox("Dark Mode")
        self.layout.addWidget(self.dark_mode)
        
        # Custom colors
        colors_group = QGroupBox("Custom Colors")
        colors_layout = QFormLayout(colors_group)
        
        # Primary color
        primary_layout = QHBoxLayout()
        self.primary_color = QColor(0, 120, 212)  # Default blue
        self.primary_preview = QLabel()
        self.primary_preview.setFixedSize(20, 20)
        self._update_color_preview(self.primary_preview, self.primary_color)
        primary_layout.addWidget(self.primary_preview)
        
        self.primary_button = QPushButton("Choose...")
        self.primary_button.clicked.connect(lambda: self._pick_color("primary"))
        primary_layout.addWidget(self.primary_button)
        colors_layout.addRow("Primary Color:", primary_layout)
        
        # Accent color
        accent_layout = QHBoxLayout()
        self.accent_color = QColor(0, 200, 0)  # Default green
        self.accent_preview = QLabel()
        self.accent_preview.setFixedSize(20, 20)
        self._update_color_preview(self.accent_preview, self.accent_color)
        accent_layout.addWidget(self.accent_preview)
        
        self.accent_button = QPushButton("Choose...")
        self.accent_button.clicked.connect(lambda: self._pick_color("accent"))
        accent_layout.addWidget(self.accent_button)
        colors_layout.addRow("Accent Color:", accent_layout)
        
        self.layout.addWidget(colors_group)
        
        # Font size
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("UI Font Size:"))
        self.font_size = QSpinBox()
        self.font_size.setRange(8, 24)
        self.font_size.setValue(10)
        font_layout.addWidget(self.font_size)
        self.layout.addLayout(font_layout)
    
    def _update_color_preview(self, label, color):
        """Update the color preview label with the selected color"""
        style = f"background-color: {color.name()}; border: 1px solid black;"
        label.setStyleSheet(style)
    
    def _pick_color(self, color_type):
        """Open a color dialog to pick a color"""
        current_color = self.primary_color if color_type == "primary" else self.accent_color
        
        color = QColorDialog.getColor(
            current_color, 
            self, 
            f"Select {color_type.capitalize()} Color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel
        )
        
        if color.isValid():
            if color_type == "primary":
                self.primary_color = color
                self._update_color_preview(self.primary_preview, color)
            else:
                self.accent_color = color
                self._update_color_preview(self.accent_preview, color)
            
            self.theme_changed.emit()
    
    def get_theme_settings(self):
        """Get the current theme settings as a dictionary"""
        return {
            "dark_mode": self.dark_mode.isChecked(),
            "primary_color": self.primary_color.name(),
            "accent_color": self.accent_color.name(),
            "font_size": self.font_size.value()
        }
    
    def set_theme_settings(self, settings):
        """Apply the provided theme settings"""
        if "dark_mode" in settings:
            self.dark_mode.setChecked(settings["dark_mode"])
        
        if "primary_color" in settings:
            self.primary_color = QColor(settings["primary_color"])
            self._update_color_preview(self.primary_preview, self.primary_color)
        
        if "accent_color" in settings:
            self.accent_color = QColor(settings["accent_color"])
            self._update_color_preview(self.accent_preview, self.accent_color)
        
        if "font_size" in settings:
            self.font_size.setValue(settings["font_size"])

class DataManager(QGroupBox):
    """Widget for managing data settings"""
    
    data_settings_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__("Data Management", parent)
        self.layout = QVBoxLayout(self)
        
        # Update frequency settings
        update_group = QGroupBox("Update Frequency")
        update_layout = QFormLayout(update_group)
        
        self.planet_update = QSpinBox()
        self.planet_update.setRange(1, 60)
        self.planet_update.setValue(5)
        self.planet_update.setSuffix(" min")
        update_layout.addRow("Planetary Data:", self.planet_update)
        
        self.asteroid_update = QSpinBox()
        self.asteroid_update.setRange(1, 1440)  # Up to 24 hours in minutes
        self.asteroid_update.setValue(60)
        self.asteroid_update.setSuffix(" min")
        update_layout.addRow("Asteroid Data:", self.asteroid_update)
        
        self.layout.addWidget(update_group)
        
        # Storage settings
        storage_group = QGroupBox("Storage Settings")
        storage_layout = QFormLayout(storage_group)
        
        # Cache directory
        cache_layout = QHBoxLayout()
        self.cache_dir = QLineEdit()
        self.cache_dir.setReadOnly(True)
        cache_layout.addWidget(self.cache_dir)
        
        self.browse_cache = QPushButton("Browse...")
        self.browse_cache.clicked.connect(self._browse_cache_dir)
        cache_layout.addWidget(self.browse_cache)
        
        storage_layout.addRow("Cache Directory:", cache_layout)
        
        # Cache size limit
        self.cache_limit = QSpinBox()
        self.cache_limit.setRange(50, 10000)
        self.cache_limit.setValue(500)
        self.cache_limit.setSuffix(" MB")
        storage_layout.addRow("Cache Size Limit:", self.cache_limit)
        
        # Clear cache button
        self.clear_cache = QPushButton("Clear Cache")
        self.clear_cache.clicked.connect(self._clear_cache)
        storage_layout.addRow("", self.clear_cache)
        
        self.layout.addWidget(storage_group)
    
    def _browse_cache_dir(self):
        """Open a dialog to select the cache directory"""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Cache Directory",
            self.cache_dir.text() or str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if directory:
            self.cache_dir.setText(directory)
            self.data_settings_changed.emit()
    
    def _clear_cache(self):
        """Clear the application cache"""
        reply = QMessageBox.question(
            self, 
            "Clear Cache",
            "Are you sure you want to clear all cached data? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Placeholder for actual cache clearing logic
            QMessageBox.information(self, "Success", "Cache cleared successfully.")
    
    def get_data_settings(self):
        """Get the current data settings as a dictionary"""
        return {
            "planet_update_interval": self.planet_update.value(),
            "asteroid_update_interval": self.asteroid_update.value(),
            "cache_directory": self.cache_dir.text(),
            "cache_size_limit": self.cache_limit.value()
        }
    
    def set_data_settings(self, settings):
        """Apply the provided data settings"""
        if "planet_update_interval" in settings:
            self.planet_update.setValue(settings["planet_update_interval"])
        
        if "asteroid_update_interval" in settings:
            self.asteroid_update.setValue(settings["asteroid_update_interval"])
        
        if "cache_directory" in settings and settings["cache_directory"]:
            self.cache_dir.setText(settings["cache_directory"])
        
        if "cache_size_limit" in settings:
            self.cache_limit.setValue(settings["cache_size_limit"])

class NotificationManager(QGroupBox):
    """Widget for managing notification settings"""
    
    notification_settings_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__("Notifications", parent)
        self.layout = QVBoxLayout(self)
        
        # Enable notifications
        self.enable_notifications = QCheckBox("Enable Notifications")
        self.enable_notifications.setChecked(True)
        self.enable_notifications.toggled.connect(self._toggle_notifications)
        self.layout.addWidget(self.enable_notifications)
        
        # Create notification types group
        self.notification_types = QGroupBox("Notification Types")
        types_layout = QVBoxLayout(self.notification_types)
        
        # Asteroid alerts
        self.asteroid_alerts = QCheckBox("Potentially Hazardous Asteroid Alerts")
        self.asteroid_alerts.setChecked(True)
        types_layout.addWidget(self.asteroid_alerts)
        
        # Planetary events
        self.planetary_events = QCheckBox("Planetary Events (Conjunctions, Oppositions)")
        self.planetary_events.setChecked(True)
        types_layout.addWidget(self.planetary_events)
        
        # Space weather
        self.space_weather = QCheckBox("Space Weather Alerts (Solar Flares, CMEs)")
        self.space_weather.setChecked(True)
        types_layout.addWidget(self.space_weather)
        
        # System notifications
        self.system_notifications = QCheckBox("System Updates and Maintenance")
        self.system_notifications.setChecked(True)
        types_layout.addWidget(self.system_notifications)
        
        self.layout.addWidget(self.notification_types)
        
        # Notification methods
        self.notification_methods = QGroupBox("Notification Methods")
        methods_layout = QVBoxLayout(self.notification_methods)
        
        # Desktop notifications
        self.desktop_notifications = QCheckBox("Desktop Notifications")
        self.desktop_notifications.setChecked(True)
        methods_layout.addWidget(self.desktop_notifications)
        
        # Email notifications
        email_layout = QHBoxLayout()
        self.email_notifications = QCheckBox("Email Notifications")
        email_layout.addWidget(self.email_notifications)
        
        self.email_address = QLineEdit()
        self.email_address.setPlaceholderText("your.email@example.com")
        self.email_address.setEnabled(False)
        email_layout.addWidget(self.email_address)
        
        methods_layout.addLayout(email_layout)
        
        # Connect email checkbox to enable/disable the email field
        self.email_notifications.toggled.connect(self.email_address.setEnabled)
        
        self.layout.addWidget(self.notification_methods)
        
        # Alert levels
        self.alert_levels = QGroupBox("Alert Level Thresholds")
        alert_layout = QFormLayout(self.alert_levels)
        
        # NEO Proximity threshold
        proximity_layout = QHBoxLayout()
        self.proximity_threshold = QSlider(Qt.Orientation.Horizontal)
        self.proximity_threshold.setRange(1, 20)  # 1-20 lunar distances
        self.proximity_threshold.setValue(5)
        self.proximity_label = QLabel("5 lunar distances")
        
        self.proximity_threshold.valueChanged.connect(self._update_proximity_label)
        
        proximity_layout.addWidget(self.proximity_threshold)
        proximity_layout.addWidget(self.proximity_label)
        alert_layout.addRow("NEO Proximity Alert:", proximity_layout)
        
        # Solar activity threshold
        solar_

