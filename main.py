#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cosmic Sentinel - Main Application Module

This module contains the main application window and entry point for the
Cosmic Sentinel space observation and monitoring platform.
"""

# Standard library imports
import logging
import sys
from datetime import datetime
from typing import Optional, Tuple

# Third-party imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QStatusBar, QMenuBar, QMenu, QLabel, QFrame, 
    QDockWidget, QMessageBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QFont

# Local application imports
from UI.planetary_view import PlanetaryView
from UI.asteroid_view import AsteroidView
from UI.space_weather_widget import SpaceWeatherWidget
from UI.settings import SettingsManager
from UI.credentials_dialog import NASAApiCredentialsDialog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CosmicSentinelApp(QMainWindow):
    """
    Main application window for the Cosmic Sentinel platform.
    
    This class manages the main UI components, views, and provides methods
    for interacting with various space monitoring features.
    """
    
    def __init__(self):
        """Initialize the main application window."""
        super().__init__()
        self.planetary_view: Optional[PlanetaryView] = None
        self.asteroid_view: Optional[AsteroidView] = None
        self.space_weather_view: Optional[SpaceWeatherWidget] = None
        self.settings_manager = SettingsManager()
        self.status_bar: Optional[QStatusBar] = None
        self.planetary_dock: Optional[QDockWidget] = None
        self.asteroid_dock: Optional[QDockWidget] = None
        self.space_weather_dock: Optional[QDockWidget] = None
        
        self.initUI()
        
        logger.info("Main application UI initialized")

    def initUI(self) -> None:
        """
        Initialize and configure the user interface components.
        
        Sets up the main layout, buttons, frames, and menu structure.
        """
        # Set window properties
        self.setWindowTitle("Cosmic Sentinel")
        self.resize(1024, 768)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Add title
        title_label = QLabel("Cosmic Sentinel")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create button areas
        button_layout = QHBoxLayout()
        
        # Planetary Tracking section
        planetary_frame = QFrame()
        planetary_frame.setFrameShape(QFrame.Shape.StyledPanel)
        planetary_layout = QVBoxLayout(planetary_frame)
        planetary_label = QLabel("Planetary Tracking")
        planetary_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        planetary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        planetary_button = QPushButton("Launch Planetary Tracker")
        planetary_button.setMinimumHeight(50)
        planetary_button.clicked.connect(self.open_planetary_view)
        planetary_layout.addWidget(planetary_label)
        planetary_layout.addWidget(planetary_button)
        planetary_layout.addStretch()
        button_layout.addWidget(planetary_frame)
        
        # Asteroid Monitoring section
        asteroid_frame = QFrame()
        asteroid_frame.setFrameShape(QFrame.Shape.StyledPanel)
        asteroid_layout = QVBoxLayout(asteroid_frame)
        asteroid_label = QLabel("Asteroid Monitoring")
        asteroid_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        asteroid_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        asteroid_button = QPushButton("Launch Asteroid Monitor")
        asteroid_button.setMinimumHeight(50)
        asteroid_button.clicked.connect(self.open_asteroid_view)
        asteroid_layout.addWidget(asteroid_label)
        asteroid_layout.addWidget(asteroid_button)
        asteroid_layout.addStretch()
        button_layout.addWidget(asteroid_frame)
        
        # Reports section
        reports_frame = QFrame()
        reports_frame.setFrameShape(QFrame.Shape.StyledPanel)
        reports_layout = QVBoxLayout(reports_frame)
        reports_label = QLabel("Reports")
        reports_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        reports_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reports_button = QPushButton("Generate Reports")
        reports_button.setMinimumHeight(50)
        reports_layout.addWidget(reports_label)
        reports_layout.addWidget(reports_button)
        reports_layout.addStretch()
        button_layout.addWidget(reports_frame)
        
        # Space Weather section
        weather_frame = QFrame()
        weather_frame.setFrameShape(QFrame.Shape.StyledPanel)
        weather_layout = QVBoxLayout(weather_frame)
        weather_label = QLabel("Space Weather")
        weather_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        weather_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        weather_button = QPushButton("Launch Space Weather Monitor")
        weather_button.setMinimumHeight(50)
        weather_button.clicked.connect(self.open_space_weather_view)
        weather_layout.addWidget(weather_label)
        weather_layout.addWidget(weather_button)
        weather_layout.addStretch()
        button_layout.addWidget(weather_frame)
        
        main_layout.addLayout(button_layout)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System ready")
        
        # Create menu bar
        self.create_menus()

    def create_menus(self) -> None:
        """
        Create and configure application menus.
        
        Sets up File, View, Tools, and Help menus with their respective actions.
        """
        # File menu
        file_menu = self.menuBar().addMenu("&File")
        
        new_action = QAction("&New", self)
        new_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open", self)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = self.menuBar().addMenu("&View")
        
        refresh_action = QAction("&Refresh", self)
        refresh_action.setShortcut("F5")
        view_menu.addAction(refresh_action)
        
        # Space Weather submenu
        view_menu.addSeparator()
        space_weather_action = QAction("Space &Weather Monitor", self)
        space_weather_action.triggered.connect(self.open_space_weather_view)
        view_menu.addAction(space_weather_action)
        
        # Tools menu
        tools_menu = self.menuBar().addMenu("&Tools")
        
        settings_action = QAction("&Settings", self)
        tools_menu.addAction(settings_action)
        settings_action.triggered.connect(self.open_settings)
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help")
        
        about_action = QAction("&About", self)
        help_menu.addAction(about_action)
        help_action = QAction("&Help Contents", self)
        help_action.setShortcut("F1")
        help_menu.addAction(help_action)

    def open_planetary_view(self) -> None:
        """
        Open the planetary tracking view in a dock widget.
        
        This method is called when the planetary tracking button is clicked.
        """
        try:
            # If the planetary view is already open, just activate it
            if self.planetary_view and hasattr(self, 'planetary_dock'):
                self.planetary_dock.setVisible(True)
                self.planetary_dock.raise_()
                logger.info("Activated existing Planetary Tracker view")
                return
                
            # Create the planetary view widget
            self.planetary_view = PlanetaryView(self)
            
            # Create a dock widget to contain the planetary view
            self.planetary_dock = QDockWidget("Planetary Tracker", self)
            self.planetary_dock.setWidget(self.planetary_view)
            self.planetary_dock.setFloating(False)
            self.planetary_dock.setAllowedAreas(
                Qt.DockWidgetArea.TopDockWidgetArea | 
                Qt.DockWidgetArea.BottomDockWidgetArea | 
                Qt.DockWidgetArea.LeftDockWidgetArea | 
                Qt.DockWidgetArea.RightDockWidgetArea
            )
            
            # Add the dock widget to the main window
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.planetary_dock)
            
            # Connect signals from the planetary view to update status
            self.planetary_view.view_updated.connect(self.update_status_from_view)
            self.planetary_view.event_detected.connect(self.handle_planetary_event)
            
            # Update status bar
            self.status_bar.showMessage("Planetary Tracker launched")
            logger.info("Planetary Tracker view opened successfully")
            
        except Exception as e:
            error_msg = f"Error opening Planetary Tracker: {str(e)}"
            self.status_bar.showMessage(error_msg)
            logger.error(error_msg, exc_info=True)


    def update_status_from_view(self, timestamp: datetime) -> None:
        """
        Update status bar when the planetary view is updated.
        
        Args:
            timestamp: Current timestamp of the planetary view update
        """
        self.status_bar.showMessage(f"Planetary view updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
    def handle_planetary_event(self, event_type: str, description: str) -> None:
        """
        Handle events detected by the planetary tracker.
        
        Args:
            event_type: Type of planetary event detected
            description: Detailed description of the event
        """
        event_msg = f"{event_type}: {description}"
        self.status_bar.showMessage(event_msg)
        logger.info(f"Planetary event detected - {event_msg}")
        
    def update_status_from_asteroid_view(self, timestamp: datetime) -> None:
        """
        Update status bar when the asteroid view is updated.
        
        Args:
            timestamp: Current timestamp of the asteroid view update
        """
        self.status_bar.showMessage(f"Asteroid monitor updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
    def handle_asteroid_hazard(self, asteroid_id: str, hazard_level: str, approach_date: str) -> None:
        """
        Handle asteroid hazard alerts.
        
        Args:
            asteroid_id: ID of the potentially hazardous asteroid
            hazard_level: Severity level of the hazard
            approach_date: Date when the asteroid will make a close approach
        """
        hazard_msg = f"ALERT: Potentially hazardous asteroid {asteroid_id} detected - {hazard_level} risk on {approach_date}"
        self.status_bar.showMessage(hazard_msg)
        logger.warning(f"Asteroid hazard detected - {hazard_msg}")
        
        # Depending on the hazard level, we might want to do more here like show a warning dialog
        if hazard_level.lower() == "high":
            QMessageBox.warning(
                self,
                "High Hazard Asteroid Alert",
                f"A high-risk asteroid {asteroid_id} will make a close approach on {approach_date}.\n\n"
                f"Please check the Asteroid Monitor for details."
            )

    
    def open_asteroid_view(self) -> None:
        """
        Open the asteroid monitoring view in a dock widget.
        
        This method is called when the asteroid monitoring button is clicked.
        """
        try:
            # If the asteroid view is already open, just activate it
            if self.asteroid_view and hasattr(self, 'asteroid_dock'):
                self.asteroid_dock.setVisible(True)
                self.asteroid_dock.raise_()
                logger.info("Activated existing Asteroid Monitor view")
                return
                
            # Check NASA API key
            nasa_api_key = self.settings_manager.get_api_key("nasa")
            if not nasa_api_key:
                self.status_bar.showMessage("NASA API key not configured. Please set it in Settings.")
                logger.warning("NASA API key not configured")
                # Open settings to configure API key
                self._show_nasa_credentials_dialog()
                return
            
            # Create the asteroid view widget with the API key
            self.asteroid_view = AsteroidView(self, api_key=nasa_api_key)
            
            # Create a dock widget to contain the asteroid view
            self.asteroid_dock = QDockWidget("Asteroid Monitor", self)
            self.asteroid_dock.setWidget(self.asteroid_view)
            self.asteroid_dock.setFloating(False)
            self.asteroid_dock.setAllowedAreas(
                Qt.DockWidgetArea.TopDockWidgetArea | 
                Qt.DockWidgetArea.BottomDockWidgetArea | 
                Qt.DockWidgetArea.LeftDockWidgetArea | 
                Qt.DockWidgetArea.RightDockWidgetArea
            )
            
            # Add the dock widget to the main window
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.asteroid_dock)
            
            # Connect signals from the asteroid view to update status
            self.asteroid_view.view_updated.connect(self.update_status_from_asteroid_view)
            self.asteroid_view.hazard_detected.connect(self.handle_asteroid_hazard)
            
            # Update status bar
            self.status_bar.showMessage("Asteroid Monitor launched")
            logger.info("Asteroid Monitor view opened successfully")
            
        except Exception as e:
            error_msg = f"Error opening Asteroid Monitor: {str(e)}"
            self.status_bar.showMessage(error_msg)
            logger.error(error_msg, exc_info=True)

    
    def open_settings(self) -> None:
        """
        Open the settings dialog.
        
        Displays the application settings dialog and handles any changes.
        """
        try:
            from UI.settings_dialog import SettingsDialog
            settings_dialog = SettingsDialog(self.settings_manager, self)
            settings_dialog.exec()
            
            # Reload views with new settings if needed
            if self.planetary_view:
                self.planetary_view.apply_settings()
                
            if self.asteroid_view:
                self.asteroid_view.apply_settings()
                
            # Reload space weather view with new settings if needed
            if self.space_weather_view:
                # Force a data update
                self.space_weather_view.update_data()
                
        except Exception as e:
            error_msg = f"Error opening Settings: {str(e)}"
            self.status_bar.showMessage(error_msg)
            logger.error(error_msg, exc_info=True)


    def open_space_weather_view(self) -> None:
        """
        Open the space weather monitoring view in a dock widget.
        
        This method is called when the space weather button is clicked.
        """
        try:
            # If the space weather view is already open, just activate it
            if self.space_weather_view and hasattr(self, 'space_weather_dock'):
                self.space_weather_dock.setVisible(True)
                self.space_weather_dock.raise_()
                logger.info("Activated existing Space Weather Monitor view")
                return
                
            # Check NASA API key
            nasa_api_key = self.settings_manager.get_api_key("nasa")
            if not nasa_api_key:
                self.status_bar.showMessage("NASA API key required for Space Weather Monitor")
                logger.info("NASA API key not configured, showing credentials dialog")
                
                # Show the credentials dialog
                self._show_nasa_credentials_dialog()
                return
            
            # Create the space weather monitor with the API key
            self._create_space_weather_view(nasa_api_key)
            
        except Exception as e:
            error_msg = f"Error opening Space Weather Monitor: {str(e)}"
            self.status_bar.showMessage(error_msg)
            logger.error(error_msg, exc_info=True)

    
    def _show_nasa_credentials_dialog(self) -> None:
        """
        Show the NASA API credentials dialog to configure an API key.
        
        This method displays a dialog that allows the user to enter and save
        their NASA API key for access to NASA data services.
        """
        try:
            credentials_dialog = NASAApiCredentialsDialog(self.settings_manager, self)
            
            # Connect the signal to receive the configured API key
            credentials_dialog.apiKeyConfigured.connect(self._on_api_key_configured)
            
            # Show the dialog
            credentials_dialog.exec()
            
        except Exception as e:
            error_msg = f"Error showing NASA API credentials dialog: {str(e)}"
            self.status_bar.showMessage(error_msg)
            logger.error(error_msg, exc_info=True)

    
    def _on_api_key_configured(self, api_key: str) -> None:
        """
        Handle the API key configuration from the credentials dialog.
        
        Args:
            api_key: The newly configured NASA API key
        """
        if api_key:
            self.status_bar.showMessage("NASA API key configured successfully")
            logger.info("NASA API key configured, creating Space Weather Monitor")
            
            # Try to create the space weather view with the new key
            self._create_space_weather_view(api_key)
        else:
            self.status_bar.showMessage("NASA API key configuration cancelled")
            logger.info("NASA API key configuration was cancelled")

    
    def _create_space_weather_view(self, nasa_api_key: str) -> None:
        """
        Create and initialize the space weather view with the provided API key.
        
        Args:
            nasa_api_key: NASA API key to use for the space weather monitor
        """
        try:
            # Update status bar
            self.status_bar.showMessage("Initializing Space Weather Monitor...")
            
            # Create the space weather view widget with the API key
            self.space_weather_view = SpaceWeatherWidget(nasa_api_key, self)
            
            # Create a dock widget to contain the space weather view
            self.space_weather_dock = QDockWidget("Space Weather Monitor", self)
            self.space_weather_dock.setWidget(self.space_weather_view)
            self.space_weather_dock.setFloating(False)
            self.space_weather_dock.setAllowedAreas(
                Qt.DockWidgetArea.TopDockWidgetArea | 
                Qt.DockWidgetArea.BottomDockWidgetArea | 
                Qt.DockWidgetArea.LeftDockWidgetArea | 
                Qt.DockWidgetArea.RightDockWidgetArea
            )
            
            # Add the dock widget to the main window
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.space_weather_dock)
            
            # Connect alert signal to handle space weather alerts
            self.space_weather_view.alertSignal.connect(self.handle_space_weather_alert)
            
            # Start space weather monitoring
            self.space_weather_view.start_monitoring()
            
            # Update status bar
            self.status_bar.showMessage("Space Weather Monitor launched successfully")
            logger.info("Space Weather Monitor view opened successfully")
            
        except Exception as e:
            error_msg = f"Error creating Space Weather Monitor: {str(e)}"
            self.status_bar.showMessage(error_msg)
            logger.error(error_msg, exc_info=True)


    def handle_space_weather_alert(self, title: str, message: str) -> None:
        """
        Handle space weather alerts.
        
        Args:
            title: Alert title
            message: Alert message
        """
        alert_msg = f"SPACE WEATHER ALERT: {title}"
        self.status_bar.showMessage(alert_msg)
        logger.warning(f"Space weather alert: {title} - {message}")
        
        # Show a warning dialog for important alerts
        QMessageBox.warning(
            self,
            f"Space Weather Alert: {title}",
            message
        )

    
    def closeEvent(self, event) -> None:
        """
        Handle application close event to clean up resources.
        
        Args:
            event: The close event object
        """
        try:
            # Shut down the space weather monitor if it's running
            if self.space_weather_view:
                self.space_weather_view.shutdown()
                
            # Add other cleanup tasks here as needed
            
            logger.info("Application shutting down cleanly")
            event.accept()
        except Exception as e:
            logger.error(f"Error during application shutdown: {str(e)}")
            event.accept()  # Still allow closing even if errors occur


def main() -> None:
    """
    Main entry point for the Cosmic Sentinel application.
    
    Initializes the application, shows the main window, and starts the event loop.
    """
    try:
        app = QApplication(sys.argv)
        window = CosmicSentinelApp()
        window.show()
        logger.info("Cosmic Sentinel application started")
        sys.exit(app.exec())
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()

