#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Space Weather Widget Module

This module provides a PyQt6-based widget for displaying space weather information,
including solar activity, geomagnetic conditions, aurora forecasts, and recent space 
weather events that may affect astronomical observations.
"""

import sys
import logging
import asyncio
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

from PyQt6.QtCore import (Qt, QTimer, pyqtSignal, pyqtSlot, QDateTime, 
                         QUrl)
from PyQt6.QtGui import (QColor, QPainter, QFont, QBrush, QPen, QIcon,
                         QRadialGradient, QLinearGradient)
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QTabWidget, QFrame, QGroupBox,
                            QGridLayout, QSizePolicy, QTableWidget, QTableWidgetItem,
                            QHeaderView, QProgressBar, QMenu, QToolButton, 
                            QMessageBox, QSplitter, QScrollArea)

# Import for            self._update_observation_impact()
            
        except Exception as e:
            logger.error(f"Error updating space weather data: {e}")
            QMessageBox.warning(
                self, 
                "Update Error", 
                f"Error updating space weather data: {str(e)}"
            )
        finally:
            # Re-enable update button
            self.update_button.setEnabled(True)
            self.update_button.setText("Update Now")
    
    def _update_observation_impact(self):
        """Update the observation impact text based on current conditions."""
        # Get current conditions
        solar_activity = self.solar_activity_label.text()
        geo_activity = self.geo_activity_label.text()
        aurora_activity = self.aurora_activity_label.text()
        
        impact_text = []
        
        # Solar activity impacts
        if solar_activity == "EXTREME":
            impact_text.append("Extreme solar activity may cause radio blackouts and disrupt satellite communications.")
        elif solar_activity == "HIGH":
            impact_text.append("High solar activity may cause temporary radio signal degradation.")
        
        # Geomagnetic impacts
        if geo_activity == "EXTREME" or geo_activity == "HIGH":
            impact_text.append("Strong geomagnetic activity may create auroras visible at lower latitudes, but may interfere with observation equipment.")
        elif geo_activity == "MODERATE":
            impact_text.append("Moderate geomagnetic activity may create auroras at higher latitudes.")
        
        # Aurora impacts
        if aurora_activity == "HIGH" or aurora_activity == "EXTREME":
            impact_text.append("High aurora probability may provide opportunities for aurora photography but may reduce visibility for deep-sky objects.")
        
        # Overall observing conditions
        if (solar_activity in ["LOW", "MODERATE"] and 
            geo_activity in ["LOW", "MODERATE"]):
            impact_text.append("Overall favorable conditions for astronomical observations.")
        elif (solar_activity in ["HIGH", "EXTREME"] or 
              geo_activity in ["HIGH", "EXTREME"]):
            impact_text.append("Use caution with sensitive equipment due to potential electromagnetic interference.")
        
        # Set the text
        if impact_text:
            self.observation_impact.setText(" ".join(impact_text))
        else:
            self.observation_impact.setText("No significant impact on astronomical observations expected.")
    
    def _handle_space_weather_event(self, event: SpaceWeatherEvent):
        """
        Handle space weather event notifications from the monitor.
        
        Args:
            event: The SpaceWeatherEvent that triggered the notification
        """
        try:
            # Add or update the event in our list
            if event.event_id in self.event_list.events:
                self.event_list.updateEvent(event)
            else:
                self.event_list.addEvent(event)
            
            # Check if this is a high-priority event that needs an alert
            if event.severity in ["HIGH", "EXTREME"]:
                title = f"{event.severity} {event.event_type}"
                message = f"{event.description}"
                
                if event.start_time:
                    message += f"\nTime: {event.start_time.strftime('%Y-%m-%d %H:%M')}"
                    
                # Emit the alert signal
                self.alertSignal.emit(title, message)
                
                # Update observation impact immediately
                self._update_observation_impact()
                
        except Exception as e:
            logger.error(f"Error handling space weather event: {e}")
    
    def _show_detailed_report(self):
        """Show a detailed space weather report dialog."""
        try:
            # Create a simple message box with detailed information
            report = "Space Weather Detailed Report\n"
            report += "=" * 40 + "\n\n"
            
            # Add current conditions
            report += "CURRENT CONDITIONS:\n"
            report += f"Solar Activity: {self.solar_activity_label.text()}\n"
            report += f"Geomagnetic Activity: {self.geo_activity_label.text()}\n"
            report += f"Aurora Activity: {self.aurora_activity_label.text()}\n\n"
            
            # Add impact on observations
            report += "IMPACT ON OBSERVATIONS:\n"
            report += f"{self.observation_impact.text()}\n\n"
            
            # Add recent events
            report += "RECENT EVENTS:\n"
            event_count = self.event_list.rowCount()
            if event_count > 0:
                for i in range(min(5, event_count)):
                    event_type = self.event_list.item(i, 0).text()
                    event_time = self.event_list.item(i, 1).text()
                    event_severity = self.event_list.item(i, 2).text()
                    event_desc = self.event_list.item(i, 3).text()
                    
                    report += f"- {event_type} ({event_severity}) at {event_time}\n"
                    report += f"  {event_desc}\n"
            else:
                report += "No recent significant events.\n"
            
            # Add last update time
            if self.last_update_time:
                report += f"\nLast updated: {self.last_update_time.strftime('%Y-%m-%d %H:%M')}\n"
            
            # Show the report
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Detailed Space Weather Report")
            msg_box.setText(report)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            
        except Exception as e:
            logger.error(f"Error showing detailed report: {e}")
            QMessageBox.warning(
                self,
                "Report Error",
                f"Error generating space weather report: {str(e)}"
            )
    
    def start_monitoring(self):
        """Start the space weather monitoring background task."""
        try:
            # Start the monitoring task in the monitor
            asyncio.create_task(self.monitor._monitoring_task())
            logger.info("Space weather monitoring started")
        except Exception as e:
            logger.error(f"Error starting space weather monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop the space weather monitoring background task."""
        try:
            # Stop the monitoring in the monitor
            self.monitor.monitoring_active = False
            
            # Stop the update timer
            self.update_timer.stop()
            
            logger.info("Space weather monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping space weather monitoring: {e}")
    
    def shutdown(self):
        """Clean up resources before the widget is destroyed."""
        try:
            # Stop monitoring
            self.stop_monitoring()
            
            # Clear any callbacks
            self.monitor.event_callbacks.clear()
            
            logger.info("Space weather monitor shutdown complete")
        except Exception as e:
            logger.error(f"Error during space weather monitor shutdown: {e}")
    
    def closeEvent(self, event):
        """Handle the widget close event."""
        self.shutdown()
        super().closeEvent(event)


# Demo function to test the widget
def main():
    """Run a standalone demo of the SpaceWeatherWidget."""
    app = QApplication(sys.argv)
    
    # You would need a NASA API key here
    nasa_api_key = "YOUR_NASA_API_KEY"
    
    widget = SpaceWeatherWidget(nasa_api_key)
    widget.setWindowTitle("Space Weather Monitor Demo")
    widget.resize(800, 600)
    widget.show()
    
    # Handle alert signals
    def alert_handler(title, message):
        QMessageBox.warning(widget, title, message)
    
    widget.alertSignal.connect(alert_handler)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()



class SolarActivityIndicator(QWidget):
    """
    Widget for displaying current solar activity level with visual indicator.
    
    This widget shows the current solar activity level (LOW/MODERATE/HIGH/EXTREME)
    with an appropriate color-coded visualization.
    """
    
    def __init__(self, parent=None):
        """Initialize the solar activity indicator widget."""
        super().__init__(parent)
        
        # Set minimum size


class GeomagneticActivityBar(QWidget):
    """Widget for displaying geomagnetic activity levels as a horizontal bar."""
    
    def __init__(self, parent=None):
        """Initialize the geomagnetic activity bar widget."""
        super().__init__(parent)
        
        self.setMinimumHeight(30)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )
        
        # Current Kp index value
        self.kp_index = 0.0
        
    def setKpIndex(self, kp: float):
        """
        Set the current Kp index value.
        
        Args:
            kp: The Kp index value (0-9)
        """
        self.kp_index = max(0, min(9, kp))  # Clamp between 0-9
        self.update()
        
    def paintEvent(self, event):
        """Paint the geomagnetic activity bar."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width, height = self.width(), self.height()
        
        # Draw the background bar
        painter.setPen(Qt.PenStyle.NoPen)
        
        # Create a gradient for the background
        gradient = QLinearGradient(0, 0, width, 0)
        gradient.setColorAt(0.0, QColor(0, 180, 0))      # Green (low activity)
        gradient.setColorAt(0.4, QColor(255, 230, 0))    # Yellow (moderate)
        gradient.setColorAt(0.7, QColor(255, 130, 0))    # Orange (high)
        gradient.setColorAt(1.0, QColor(255, 0, 0))      # Red (severe)
        
        # Draw the full background
        painter.setBrush(QBrush(QColor(240, 240, 240)))
        painter.drawRoundedRect(0, 0, width, height, 5, 5)
        
        # Draw the active portion based on Kp value
        active_width = int(width * (self.kp_index / 9.0))
        if active_width > 0:
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(0, 0, active_width, height, 5, 5)
        
        # Draw scale markers
        painter.setPen(QPen(QColor(80, 80, 80), 1, Qt.PenStyle.SolidLine))
        for i in range(1, 9):
            x = width * i / 9
            painter.drawLine(int(x), 0, int(x), height // 3)
        
        # Draw Kp value text
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        
        # Choose text color based on position
        if self.kp_index > 6:
            text_color = QColor(255, 255, 255)
        else:
            text_color = QColor(0, 0, 0)
            
        painter.setPen(text_color)
        kp_text = f"Kp: {self.kp_index:.1f}"
        painter.drawText(10, height // 2 + 5, kp_text)
        
        # Draw labels for the scale
        painter.setPen(QColor(80, 80, 80))
        painter.drawText(5, height - 2, "Low")
        painter.drawText(width - 50, height - 2, "Extreme")


class AuroraForecastDisplay(QWidget):
    """Widget for displaying aurora forecast information."""
    
    def __init__(self, parent=None):
        """Initialize the aurora forecast display widget."""
        super().__init__(parent)
        
        # Set up layout
        layout = QVBoxLayout(self)
        
        # Create header
        header = QLabel("Aurora Forecast")
        header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Activity level display
        activity_layout = QHBoxLayout()
        activity_layout.addWidget(QLabel("Activity Level:"))
        
        self.activity_level = QLabel("UNKNOWN")
        self.activity_level.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        activity_layout.addWidget(self.activity_level)
        activity_layout.addStretch()
        
        layout.addLayout(activity_layout)
        
        # Probability display
        prob_layout = QHBoxLayout()
        prob_layout.addWidget(QLabel("Average Probability:"))
        
        self.probability = QLabel("Unknown")
        prob_layout.addWidget(self.probability)
        prob_layout.addStretch()
        
        layout.addLayout(prob_layout)
        
        # Last updated
        update_layout = QHBoxLayout()
        update_layout.addWidget(QLabel("Last Updated:"))
        
        self.last_updated = QLabel("Never")
        update_layout.addWidget(self.last_updated)
        update_layout.addStretch()
        
        layout.addLayout(update_layout)
        
        # Add a visualization if web engine is available
        if HAS_WEBENGINE:
            # Web view for NOAA aurora map
            self.web_view = QWebEngineView()
            self.web_view.setMinimumHeight(200)
            self.web_view.load(QUrl("https://services.swpc.noaa.gov/images/aurora-forecast-northern-hemisphere.jpg"))
            layout.addWidget(self.web_view)
        else:
            # Fallback text display
            self.web_view = QLabel("Aurora visualization unavailable.\nQt WebEngine is required for this feature.")
            self.web_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.web_view.setStyleSheet("background-color: #f0f0f0; padding: 20px;")
            layout.addWidget(self.web_view)
        
        # Add stretch to keep items at the top
        layout.addStretch()
        
    def updateForecast(self, forecast_data):
        """
        Update the aurora forecast display with new data.
        
        Args:
            forecast_data: Dictionary with forecast information
        """
        try:
            # Update activity level with color
            activity = forecast_data.get("activity_level", "UNKNOWN")
            self.activity_level.setText(activity)
            
            # Set text color based on activity level
            if activity == "EXTREME":
                self.activity_level.setStyleSheet("color: red;")
            elif activity == "HIGH":
                self.activity_level.setStyleSheet("color: orange;")
            elif activity == "MODERATE":
                self.activity_level.setStyleSheet("color: #DAA520;")  # Goldenrod
            elif activity == "LOW":
                self.activity_level.setStyleSheet("color: green;")
            else:
                self.activity_level.setStyleSheet("")  # Default color
                
            # Update probability if available
            if "average_probability" in forecast_data:
                prob = forecast_data["average_probability"]
                self.probability.setText(f"{prob:.1%}")
            else:
                self.probability.setText("Unknown")
                
            # Update last updated timestamp
            if "timestamp" in forecast_data:
                try:
                    timestamp = datetime.fromisoformat(forecast_data["timestamp"])
                    self.last_updated.setText(timestamp.strftime("%Y-%m-%d %H:%M"))
                except (ValueError, TypeError):
                    self.last_updated.setText(str(forecast_data["timestamp"]))
            else:
                self.last_updated.setText("Unknown")
                
        except Exception as e:
            logger.error(f"Error updating aurora forecast display: {e}")
            self.activity_level.setText("ERROR")
            self.probability.setText("Error displaying forecast")


class SpaceWeatherWidget(QWidget):
    """
    Widget for displaying space weather information.
    
    This widget combines various components to provide a comprehensive
    view of current space weather conditions that may affect astronomical
    observations. It integrates with the SpaceWeatherMonitor to fetch
    and display real-time data.
    """
    
    # Signal emitted when significant space weather events occur
    alertSignal = pyqtSignal(str, str)  # (title, message)
    
    def __init__(self, nasa_api_key, parent=None):
        """
        Initialize the space weather widget.
        
        Args:
            nasa_api_key: NASA API key for the SpaceWeatherMonitor
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create space weather monitor
        self.monitor = SpaceWeatherMonitor(
            nasa_api_key=nasa_api_key,
            auto_monitor=False  # We'll start it manually
        )
        
        # Register for event callbacks
        self.monitor.register_callback(self._handle_space_weather_event)
        
        # Setup timer for periodic updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_data)
        
        # Track when data was last updated
        self.last_update_time = None
        
        # Setup UI
        self._setup_ui()
        
        # Initial data update
        self.update_data()
        
        # Start automatic updates every 15 minutes
        self.update_timer.start(15 * 60 * 1000)  # 15 minutes
        
    def _setup_ui(self):
        """Set up the user interface components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Title and update button
        title_layout = QHBoxLayout()
        
        title = QLabel("Space Weather Monitor")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_layout.addWidget(title)
        
        title_layout.addStretch()
        
        self.last_update_label = QLabel("Last update: Never")
        title_layout.addWidget(self.last_update_label)
        
        self.update_button = QPushButton("Update Now")
        self.update_button.clicked.connect(self.update_data)
        title_layout.addWidget(self.update_button)
        
        main_layout.addLayout(title_layout)
        
        # Create tab widget for different aspects of space weather
        self.tab_widget = QTabWidget()
        
        # Overview tab
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        
        # Current conditions section
        conditions_group = QGroupBox("Current Space Weather Conditions")
        conditions_layout = QGridLayout(conditions_group)
        
        # Solar activity display
        solar_layout = QVBoxLayout()
        solar_label = QLabel("Solar Activity")
        solar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        solar_layout.addWidget(solar_label)
        
        self.solar_indicator = SolarActivityIndicator()
        solar_layout.addWidget(self.solar_indicator)
        
        self.solar_activity_label = QLabel("UNKNOWN")
        self.solar_activity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.solar_activity_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        solar_layout.addWidget(self.solar_activity_label)
        
        conditions_layout.addLayout(solar_layout, 0, 0)
        
        # Geomagnetic activity display
        geo_layout = QVBoxLayout()
        geo_label = QLabel("Geomagnetic Activity")
        geo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        geo_layout.addWidget(geo_label)
        
        self.geo_bar = GeomagneticActivityBar()
        geo_layout.addWidget(self.geo_bar)
        
        self.geo_activity_label = QLabel("UNKNOWN")
        self.geo_activity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.geo_activity_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        geo_layout.addWidget(self.geo_activity_label)
        
        conditions_layout.addLayout(geo_layout, 0, 1)
        
        # Aurora activity display
        aurora_layout = QVBoxLayout()
        aurora_label = QLabel("Aurora Activity")
        aurora_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        aurora_layout.addWidget(aurora_label)
        
        # Use a progress bar to show aurora probability
        self.aurora_bar = QProgressBar()
        self.aurora_bar.setRange(0, 100)
        self.aurora_bar.setValue(0)
        self.aurora_bar.setFormat("%v%")
        aurora_layout.addWidget(self.aurora_bar)
        
        self.aurora_activity_label = QLabel("UNKNOWN")
        self.aurora_activity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.aurora_activity_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        aurora_layout.addWidget(self.aurora_activity_label)
        
        conditions_layout.addLayout(aurora_layout, 0, 2)
        
        # Add observation impact information
        impact_label = QLabel("<b>Impact on Astronomical Observations:</b>")
        conditions_layout.addWidget(impact_label, 1, 0, 1, 3)
        
        self.observation_impact = QLabel("No significant impact expected.")
        self.observation_impact.setWordWrap(True)
        conditions_layout.addWidget(self.observation_impact, 2, 0, 1, 3)
        
        overview_layout.addWidget(conditions_group)
        
        # Recent events section
        events_group = QGroupBox("Recent Space Weather Events")
        events_layout = QVBoxLayout(events_group)
        
        self.event_list = SpaceWeatherEventList()
        self.event_list.setMaximumHeight(200)
        events_layout.addWidget(self.event_list)
        
        overview_layout.addWidget(events_group)
        
        # Add overview tab
        self.tab_widget.addTab(overview_tab, "Overview")
        
        # Aurora Forecast tab
        aurora_tab = QWidget()
        aurora_layout = QVBoxLayout(aurora_tab)
        
        self.aurora_forecast = AuroraForecastDisplay()
        aurora_layout.addWidget(self.aurora_forecast)
        
        self.tab_widget.addTab(aurora_tab, "Aurora Forecast")
        
        # Add the tab widget to the main layout
        main_layout.addWidget(self.tab_widget)
        
        # Add detailed report button
        self.report_button = QPushButton("View Detailed Space Weather Report")
        self.report_button.clicked.connect(self._show_detailed_report)
        main_layout.addWidget(self.report_button)
    
    def update_data(self):
        """Update all space weather data from the monitor."""
        try:
            # Disable update button while updating
            self.update_button.setEnabled(False)
            self.update_button.setText("Updating...")
            
            # Update timestamp
            self.last_update_time = datetime.now()
            self.last_update_label.setText(f"Last update: {self.last_update_time.strftime('%Y-%m-%d %H:%M')}")
            
            # Fetch space weather data
            # Note: In a real implementation, these would use async methods
            # to avoid blocking the UI, but we'll use synchronous for simplicity
            
            # Get solar flares
            try:
                solar_flares = self.monitor.get_solar_flares(
                    start_date=datetime.now() - timedelta(days=3)
                )
                
                # Add flares to the event list
                for flare in solar_flares:
                    self.event_list.addEvent(flare)
                    
            except Exception as e:
                logger.error(f"Error fetching solar flares: {e}")
            
            # Get CMEs
            try:
                cmes = self.monitor.get_cmes(
                    start_date=datetime.now() - timedelta(days=3)
                )
                
                # Add CMEs to the event list
                for cme in cmes:
                    self.event_list.addEvent(cme)
                    
            except Exception as e:
                logger.error(f"Error fetching CMEs: {e}")
            
        # Activity level and colors
        self.activity_level = "UNKNOWN"
        self.activity_colors = {
            "LOW": QColor(0, 200, 0),      # Green
            "MODERATE": QColor(255, 180, 0),  # Yellow-Orange
            "HIGH": QColor(255, 100, 0),    # Orange
            "EXTREME": QColor(255, 0, 0),    # Red
            "UNKNOWN": QColor(150, 150, 150)  # Gray
        }
        
    def setActivityLevel(self, level: str):
        """
        Set the current solar activity level.
        
        Args:
            level: Activity level (LOW, MODERATE, HIGH, EXTREME, UNKNOWN)
        """
        if level in self.activity_colors:
            self.activity_level = level
            self.update()  # Trigger a repaint
        
    def paintEvent(self, event):
        """Paint the activity indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get color for current level
        color = self.activity_colors.get(self.activity_level, self.activity_colors["UNKNOWN"])
        
        # Calculate dimensions
        width, height = self.width(), self.height()
        size = min(width, height) - 10
        x = (width - size) // 2
        y = (height - size) // 2
        
        # Create a radial gradient for solar appearance
        gradient = QRadialGradient(x + size//2, y + size//2, size//2)
        
        # Set gradient colors based on activity level
        if self.activity_level == "EXTREME":
            gradient.setColorAt(0.0, QColor(255, 255, 200))
            gradient.setColorAt(0.5, QColor(255, 200, 0))
            gradient.setColorAt(0.7, QColor(255, 100, 0))
            gradient.setColorAt(1.0, QColor(255, 0, 0))
        elif self.activity_level == "HIGH":
            gradient.setColorAt(0.0, QColor(255, 255, 200))
            gradient.setColorAt(0.5, QColor(255, 200, 0))
            gradient.setColorAt(1.0, QColor(255, 100, 0))
        elif self.activity_level == "MODERATE":
            gradient.setColorAt(0.0, QColor(255, 255, 200))
            gradient.setColorAt(0.7, QColor(255, 220, 0))
            gradient.setColorAt(1.0, QColor(255, 180, 0))
        elif self.activity_level == "LOW":
            gradient.setColorAt(0.0, QColor(255, 255, 200))
            gradient.setColorAt(0.7, QColor(255, 240, 0))
            gradient.setColorAt(1.0, QColor(180, 255, 0))
        else:  # UNKNOWN
            gradient.setColorAt(0.0, QColor(200, 200, 200))
            gradient.setColorAt(1.0, QColor(120, 120, 120))
        
        # Fill the sun circle
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(x, y, size, size)
        
        # Draw some solar features for visual interest
        if self.activity_level != "UNKNOWN":
            # Draw solar prominence effects around the edge
            painter.setPen(QPen(color.lighter(120), 2, Qt.PenStyle.SolidLine))
            num_prominences = {
                "LOW": 3,
                "MODERATE": 5,
                "HIGH": 8,
                "EXTREME": 12
            }.get(self.activity_level, 0)
            
            import random
            import math
            
            # Seed with consistent value for the same level
            random.seed(hash(self.activity_level))
            
            # Draw some prominences
            for i in range(num_prominences):
                angle = random.uniform(0, 2 * math.pi)
                length = random.uniform(size * 0.05, size * 0.15)
                arc_length = random.uniform(math.pi / 16, math.pi / 8)
                
                # Calculate points
                cx, cy = x + size//2, y + size//2
                radius = size // 2
                
                # Draw a small arc at the given angle
                start_angle = (angle - arc_length/2) * 16 * 180 / math.pi
                end_angle = arc_length * 16 * 180 / math.pi
                
                # Draw prominence 
                painter.drawArc(
                    int(cx - radius - length), 
                    int(cy - radius - length),
                    int(size + length * 2),
                    int(size + length * 2),
                    int(start_angle),
                    int(end_angle)
                )


        
        # Create a gradient for the background
        gradient = QLinearGradient(0, 0, width, 0)
        gradient.setColorAt(0.0, QColor(0, 180, 0))      # Green (low activity)
        gradient.setColorAt(0.4, QColor(255, 230, 0))    # Yellow (moderate)
        gradient.setColorAt(0.7, QColor(255, 130, 0))    # Orange (high)
        gradient.setColorAt(1.0, QColor(255, 0, 0))      # Red (severe)
        
        # Draw the full background
        painter.setBrush(QBrush(QColor(240, 240, 240)))
        painter.drawRoundedRect(0, 0, width, height, 5, 5)
        
        # Draw the active portion based on Kp value
        active_width = int(width * (self.kp_index / 9.0))
        if active_width > 0:
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(0, 0, active_width, height, 5, 5)
        
        # Draw scale markers
        painter.setPen(QPen(QColor(80, 80, 80), 1, Qt.PenStyle.SolidLine))
        for i in range(1, 9):
            x = width * i / 9
            painter.drawLine(int(x), 0, int(x), height // 3)
        
        # Draw Kp value text
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        
        # Choose text color based on position
        if self.kp_index > 6:
            text_color = QColor(255, 255, 255)
        else:
            text_color = QColor(0, 0, 0)
            
        painter.setPen(text_color)
        kp_text = f"Kp: {self.kp_index:.1f}"
        painter.drawText(10, height // 2 + 5, kp_text)
        
        # Draw labels for the scale
        painter.setPen(QColor(80, 80, 80))
        painter.drawText(5, height - 2, "Low")
        painter.drawText(width - 50, height - 2, "Extreme")


class SpaceWeatherEventList(QTableWidget):
    """Widget for displaying a list of space weather events."""
    
    # Signal emitted when an event is selected
    eventSelected = pyqtSignal(SpaceWeatherEvent)
    
    def __init__(self, parent=None):
        """Initialize the space weather event list widget."""
        super().__init__(parent)
        
        # Configure table
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Type", "Time", "Severity", "Description"])
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        # Store events for reference
        self.events = {}
        
        # Connect signals
        self.itemSelectionChanged.connect(self._handle_selection)
        
    def _handle_selection(self):
        """Handle selection changes in the event list."""
        selected_rows = self.selectedItems()
        if not selected_rows:
            return
            
        row = selected_rows[0].row()
        event_id = self.item(row, 0).data(Qt.ItemDataRole.UserRole)
        
        if event_id in self.events:
            self.eventSelected.emit(self.events[event_id])
            
    def addEvent(self, event: SpaceWeatherEvent):
        """
        Add a space weather event to the list.
        
        Args:
            event: The SpaceWeatherEvent to add
        """
        # Check if event already exists
        if event.event_id in self.events:
            return
            
        # Store the event
        self.events[event.event_id] = event
        
        # Add to table
        row = self.rowCount()
        self.insertRow(row)
        
        # Type column with icon
        type_item = QTableWidgetItem(event.event_type.replace("_", " "))
        type_item.setData(Qt.ItemDataRole.UserRole, event.event_id)
        
        # Set icon based on event type
        if event.event_type == "SOLAR_FLARE":
            type_item.setIcon(QIcon.fromTheme("weather-sunny"))
        elif event.event_type == "CME":
            type_item.setIcon(QIcon.fromTheme("weather-storm"))
        elif event.event_type == "GEOMAGNETIC_STORM":
            type_item.setIcon(QIcon.fromTheme("weather-severe-alert"))
        else:
            type_item.setIcon(QIcon.fromTheme("dialog-information"))
            
        self.setItem(row, 0, type_item)
        
        # Time column
        time_str = event.start_time.strftime("%Y-%m-%d %H:%M") if event.start_time else "Unknown"
        time_item = QTableWidgetItem(time_str)
        self.setItem(row, 1, time_item)
        
        # Severity column with color
        severity_item = QTableWidgetItem(event.severity)
        
        # Set background color based on severity
        if event.severity == "EXTREME":
            severity_item.setBackground(QBrush(QColor(255, 0, 0, 100)))
        elif event.severity == "HIGH":
            severity_item.setBackground(QBrush(QColor(255, 100, 0, 100)))
        elif event.severity == "MODERATE":
            severity_item.setBackground(QBrush(QColor(255, 200, 0, 100)))
        elif event.severity == "LOW":
            severity_item.setBackground(QBrush(QColor(0, 200, 0, 100)))
            
        self.setItem(row, 2, severity_item)
        
        #

