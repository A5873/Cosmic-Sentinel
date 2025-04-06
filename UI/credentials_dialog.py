#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NASA API Credentials Dialog

This module provides a dialog for configuring NASA API credentials required
for various features of the application, including asteroid monitoring and
space weather data.
"""

import re
import logging
import requests
import threading
from typing import Optional, Callable

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QFont, QDesktopServices, QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QProgressBar, QMessageBox, QGroupBox, 
    QFormLayout, QTextBrowser, QDialogButtonBox, QCheckBox
)

from UI.settings import SettingsManager

# Configure logging
logger = logging.getLogger(__name__)


class NASAApiCredentialsDialog(QDialog):
    """
    Dialog for configuring NASA API credentials.
    
    This dialog guides users in obtaining and configuring a NASA API key,
    which is required for various features of the application.
    """
    
    # Signal emitted when API key is successfully configured
    apiKeyConfigured = pyqtSignal(str)  # Emits the configured API key
    
    def __init__(self, settings_manager: SettingsManager, parent=None):
        """
        Initialize the NASA API credentials dialog.
        
        Args:
            settings_manager: The application settings manager
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        
        self.settings_manager = settings_manager
        self.api_key = self.settings_manager.get_api_key("nasa", "")
        self.api_test_in_progress = False
        
        self.setWindowTitle("NASA API Credentials")
        self.setMinimumWidth(600)
        
        self.initUI()
        
    def initUI(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        
        # === Header section ===
        header_label = QLabel("NASA API Credentials")
        header_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header_label)
        
        info_label = QLabel(
            "Cosmic Sentinel requires a NASA API key to access various data sources. "
            "Please enter your NASA API key below or register for a free key at the NASA API Portal."
        )
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)
        
        # === Credentials input ===
        credentials_group = QGroupBox("API Key")
        credentials_layout = QFormLayout(credentials_group)
        
        self.api_key_input = QLineEdit(self.api_key)
        self.api_key_input.setMinimumWidth(300)
        self.api_key_input.setPlaceholderText("Enter your NASA API key")
        credentials_layout.addRow("NASA API Key:", self.api_key_input)
        
        # Horizontal layout for test button and status
        test_layout = QHBoxLayout()
        
        self.test_button = QPushButton("Test Key")
        self.test_button.clicked.connect(self.test_api_key)
        test_layout.addWidget(self.test_button)
        
        self.test_status = QLabel("")
        test_layout.addWidget(self.test_status)
        
        self.test_progress = QProgressBar()
        self.test_progress.setRange(0, 0)  # Indeterminate progress
        self.test_progress.setVisible(False)
        test_layout.addWidget(self.test_progress)
        
        test_layout.addStretch()
        
        credentials_layout.addRow("Validate:", test_layout)
        
        # Remember API key option
        self.remember_key = QCheckBox("Remember API key")
        self.remember_key.setChecked(True)
        credentials_layout.addRow("", self.remember_key)
        
        main_layout.addWidget(credentials_group)
        
        # === Instructions ===
        instructions_group = QGroupBox("How to Get a NASA API Key")
        instructions_layout = QVBoxLayout(instructions_group)
        
        instructions_text = QTextBrowser()
        instructions_text.setOpenExternalLinks(True)
        instructions_text.setHtml("""
            <p>Follow these steps to obtain a free NASA API key:</p>
            <ol>
                <li>Visit the <a href="https://api.nasa.gov/">NASA API Portal</a></li>
                <li>Fill out the signup form with your information</li>
                <li>Check your email for the API key (it will also appear on the website after signing up)</li>
                <li>Copy the API key and paste it in the field above</li>
            </ol>
            <p><b>Note:</b> The free API key has a limit of 1,000 requests per hour, which is more than sufficient for normal use.</p>
            <p>For more information, see the <a href="https://api.nasa.gov/documentation">NASA API Documentation</a>.</p>
        """)
        instructions_layout.addWidget(instructions_text)
        
        # Button to open NASA API portal in browser
        portal_button = QPushButton("Open NASA API Portal")
        portal_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://api.nasa.gov/")))
        instructions_layout.addWidget(portal_button)
        
        main_layout.addWidget(instructions_group)
        
        # === Dialog buttons ===
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        
        # Adjust the OK button to be disabled until we have a valid API key
        self.ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("Save API Key")
        self.ok_button.setEnabled(self.validate_api_key_format(self.api_key))
        
        # Connect input changes to validation
        self.api_key_input.textChanged.connect(self.on_api_key_changed)
        
    def on_api_key_changed(self, text):
        """
        Handle API key input changes.
        
        Args:
            text: Current text in the API key input field
        """
        valid = self.validate_api_key_format(text)
        self.ok_button.setEnabled(valid)
        
        # Clear the test status when input changes
        self.test_status.setText("")
        self.test_status.setStyleSheet("")
        
    def validate_api_key_format(self, api_key: str) -> bool:
        """
        Validate the format of the NASA API key.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            True if the key format is valid, False otherwise
        """
        # NASA API keys are typically 40 characters
        # Demo key is 'DEMO_KEY'
        if api_key == "DEMO_KEY":
            return True
            
        # Regular API keys should be alphanumeric and at least 30 characters
        pattern = r'^[a-zA-Z0-9_-]{30,}$'
        return bool(re.match(pattern, api_key))
        
    def test_api_key(self):
        """Test the API key by making a sample API request."""
        api_key = self.api_key_input.text().strip()
        
        if not api_key:
            self.test_status.setText("Please enter an API key")
            self.test_status.setStyleSheet("color: #e74c3c;")  # Red color
            return
            
        if not self.validate_api_key_format(api_key):
            self.test_status.setText("Invalid key format")
            self.test_status.setStyleSheet("color: #e74c3c;")  # Red color
            return
            
        # Disable UI during test
        self.api_test_in_progress = True
        self.test_button.setEnabled(False)
        self.api_key_input.setEnabled(False)
        self.test_progress.setVisible(True)
        self.test_status.setText("Testing...")
        self.test_status.setStyleSheet("")
        
        # Run the test in a separate thread to prevent UI freezing
        threading.Thread(target=self._test_api_key_thread, args=(api_key,), daemon=True).start()
        
    def _test_api_key_thread(self, api_key: str):
        """
        Test the API key in a separate thread.
        
        Args:
            api_key: The API key to test
        """
        try:
            # Use a simple APOD request to test the API key
            url = f"https://api.nasa.gov/planetary/apod?api_key={api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Success - key works
                self._update_test_ui(True, "API key is valid")
            elif response.status_code == 403:
                # Invalid key
                self._update_test_ui(False, "Invalid API key")
            elif response.status_code == 429:
                # Rate limit exceeded
                self._update_test_ui(False, "Rate limit exceeded")
            else:
                # Other error
                self._update_test_ui(False, f"Error: HTTP {response.status_code}")
                
        except requests.RequestException as e:
            # Network or other request error
            logger.error(f"API key test failed: {e}")
            self._update_test_ui(False, "Connection error")
        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error during API key test: {e}")
            self._update_test_ui(False, "Unknown error")
            
    def _update_test_ui(self, success: bool, message: str):
        """
        Update the UI after API test completes.
        
        Args:
            success: Whether the test was successful
            message: Message to display
        """
        # We need to update UI in the main thread
        from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
        
        QMetaObject.invokeMethod(
            self, 
            "_update_test_ui_main_thread", 
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(bool, success),
            Q_ARG(str, message)
        )
        
    def _update_test_ui_main_thread(self, success: bool, message: str):
        """
        Update the UI in the main thread.
        
        Args:
            success: Whether the test was successful
            message: Message to display
        """
        # Update UI elements
        self.test_progress.setVisible(False)
        self.test_button.setEnabled(True)
        self.api_key_input.setEnabled(True)
        self.api_test_in_progress = False
        
        # Update status message with appropriate color
        self.test_status.setText(message)
        if success:
            self.test_status.setStyleSheet("color: #2ecc71;")  # Green color
            # If test passed, enable the OK button regardless of format validation
            self.ok_button.setEnabled(True)
        else:
            self.test_status.setStyleSheet("color: #e74c3c;")  # Red color
            # On failure, re-check format validation
            self.ok_button.setEnabled(self.validate_api_key_format(self.api_key_input.text()))
            
    def accept(self):
        """Handle dialog acceptance (OK button)."""
        api_key = self.api_key_input.text().strip()
        
        # Store API key if needed
        if self.remember_key.isChecked():
            try:
                self.settings_manager.store_api_key("nasa", api_key)
                logger.info("NASA API key stored successfully")
            except Exception as e:
                logger.error(f"Failed to store NASA API key: {e}")
                QMessageBox.warning(
                    self,
                    "API Key Storage Error",
                    f"Failed to store the API key: {str(e)}\n\n"
                    "The application will still use the key for this session."
                )
        
        # Emit signal with the new API key
        self.apiKeyConfigured.emit(api_key)
        
        # Call the parent class accept method
        super().accept()
        
    def reject(self):
        """Handle dialog rejection (Cancel button)."""
        # If API test is in progress, warn the user
        if self.api_test_in_progress:
            confirm = QMessageBox.question(
                self,
                "Cancel API Test",
                "An API test is in progress. Are you sure you want to cancel?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
                
        # Call the parent class reject method
        super().reject()


def main():
    """Run a standalone demo of the credentials dialog."""
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    settings = SettingsManager()
    dialog = NASAApiCredentialsDialog(settings)
    
    def on_key_configured(api_key):
        print(f"API key configured: {api_key}")
        
    dialog.apiKeyConfigured.connect(on_key_configured)
    dialog.exec()


if __name__ == "__main__":
    main()

