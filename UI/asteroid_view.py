#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Asteroid View Module

This module provides a PyQt6-based visualization interface for tracking and 
displaying Near-Earth Objects (NEOs). It includes interactive orbital visualization,
threat assessment indicators, and detailed information panels.
"""

import sys
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set

from PyQt6.QtCore import (Qt, QTimer, QPointF, QRectF, pyqtSignal, pyqtSlot, 
                         QSize, QThread)
from PyQt6.QtGui import (QColor, QPainter, QPen, QBrush, QFont, 
                        QLinearGradient, QPainterPath, QPixmap)
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                            QSplitter, QComboBox, QTabWidget, QFrame, QGroupBox,
                            QSlider, QGridLayout, QSizePolicy, QScrollArea,
                            QProgressBar, QTextEdit)

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# Import local modules
from API.neo_tracking import NEOTracker, NEO, CloseApproach
from models.planet_data import PlanetData, OrbitData


class OrbitVisualization(QWidget):
    """
    Widget for rendering 2D visualization of asteroid orbits and paths.
    
    This widget renders the orbits of Earth and NEOs in a 2D plane,
    showing the current positions and approach vectors.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the orbit visualization widget."""
        super().__init__(parent)
        self.setMinimumSize(600, 600)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Visual settings
        self.bg_color = QColor(10, 10, 30)
        self.grid_color = QColor(40, 40, 60)
        self.earth_color = QColor(30, 100, 255)
        self.sun_color = QColor(255, 240, 200)
        self.orbit_color = QColor(70, 70, 90)
        
        # Asteroid colors based on threat level
        self.asteroid_colors = {
            "safe": QColor(50, 200, 50),
            "attention": QColor(240, 200, 50),
            "warning": QColor(240, 130, 20),
            "danger": QColor(255, 30, 30)
        }
        
        # Scale and offset for rendering
        self.scale = 50.0  # AU to pixels
        self.offset_x = 0
        self.offset_y = 0
        self.center_x = 0
        self.center_y = 0
        
        # Data
        self.asteroids = {}  # Dictionary of NEO objects
        self.selected_asteroid_id = None
        
        # Initialize for tracking mouse movements
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.dragging = False
        
    def set_asteroids(self, asteroids: Dict[str, NEO]) -> None:
        """
        Set the asteroid data to be visualized.
        
        Args:
            asteroids: Dictionary of NEO objects keyed by ID
        """
        self.asteroids = asteroids
        self.update()
        
    def select_asteroid(self, asteroid_id: str) -> None:
        """
        Select an asteroid to highlight.
        
        Args:
            asteroid_id: ID of the asteroid to highlight
        """
        self.selected_asteroid_id = asteroid_id
        self.update()
    
    def resizeEvent(self, event) -> None:
        """Handle widget resize events to adjust the visualization scaling."""
        super().resizeEvent(event)
        self.center_x = self.width() // 2
        self.center_y = self.height() // 2
        
    def paintEvent(self, event) -> None:
        """Render the orbital visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), self.bg_color)
        
        # Draw coordinate grid
        self._draw_grid(painter)
        
        # Draw orbits
        self._draw_earth_orbit(painter)
        
        # Draw sun
        self._draw_sun(painter)
        
        # Draw Earth
        self._draw_earth(painter)
        
        # Draw asteroids and their orbits
        self._draw_asteroids(painter)
        
    def _draw_grid(self, painter: QPainter) -> None:
        """Draw the coordinate grid with AU markers."""
        painter.setPen(QPen(self.grid_color, 1, Qt.PenStyle.DotLine))
        
        # Draw concentric circles for AU distances
        for au in range(1, 6):
            radius = au * self.scale
            painter.drawEllipse(QPointF(self.center_x + self.offset_x, 
                                       self.center_y + self.offset_y), 
                               radius, radius)
            
            # Draw AU distance label
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawText(
                self.center_x + self.offset_x + radius - 10,
                self.center_y + self.offset_y,
                f"{au} AU"
            )
            painter.setPen(QPen(self.grid_color, 1, Qt.PenStyle.DotLine))
        
        # Draw the coordinate axes
        painter.setPen(QPen(self.grid_color, 1, Qt.PenStyle.SolidLine))
        painter.drawLine(0, self.center_y + self.offset_y, 
                        self.width(), self.center_y + self.offset_y)
        painter.drawLine(self.center_x + self.offset_x, 0, 
                        self.center_x + self.offset_x, self.height())
                        
    def _draw_sun(self, painter: QPainter) -> None:
        """Draw the Sun at the center of the coordinate system."""
        sun_radius = 10
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.sun_color))
        painter.drawEllipse(
            QPointF(self.center_x + self.offset_x, self.center_y + self.offset_y),
            sun_radius, sun_radius
        )
        
        # Add a glow effect
        for i in range(5):
            glow_alpha = 100 - i * 20
            glow_color = QColor(self.sun_color)
            glow_color.setAlpha(glow_alpha)
            glow_radius = sun_radius + i * 3
            
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(glow_color, 2))
            painter.drawEllipse(
                QPointF(self.center_x + self.offset_x, self.center_y + self.offset_y),
                glow_radius, glow_radius
            )
    
    def _draw_earth_orbit(self, painter: QPainter) -> None:
        """Draw Earth's orbit around the Sun."""
        painter.setPen(QPen(self.orbit_color, 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        earth_orbit_radius = 1.0 * self.scale  # 1 AU
        painter.drawEllipse(
            QPointF(self.center_x + self.offset_x, self.center_y + self.offset_y),
            earth_orbit_radius, earth_orbit_radius
        )
    
    def _draw_earth(self, painter: QPainter) -> None:
        """Draw Earth at its current position."""
        # Calculate Earth's position (assume circular orbit for simplicity)
        # In a real application, this would use the actual Earth position from PlanetData
        angle = (datetime.now().timestamp() % 31536000) / 31536000 * 2 * math.pi
        earth_x = self.center_x + self.offset_x + math.cos(angle) * self.scale
        earth_y = self.center_y + self.offset_y + math.sin(angle) * self.scale
        
        # Draw Earth
        earth_radius = 5
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.earth_color))
        painter.drawEllipse(QPointF(earth_x, earth_y), earth_radius, earth_radius)
        
        # Label Earth
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.drawText(earth_x + 8, earth_y, "Earth")
    
    def _draw_asteroids(self, painter: QPainter) -> None:
        """Draw all asteroids and their orbits."""
        for asteroid_id, asteroid in self.asteroids.items():
            # Determine color based on threat level
            if asteroid.hazardous:
                if asteroid.close_approaches and asteroid.close_approaches[0].distance_au < 0.05:
                    color = self.asteroid_colors["danger"]
                else:
                    color = self.asteroid_colors["warning"]
            elif asteroid.close_approaches:
                color = self.asteroid_colors["attention"]
            else:
                color = self.asteroid_colors["safe"]
                
            # Highlight selected asteroid
            if asteroid_id == self.selected_asteroid_id:
                pen_width = 2
                asteroid_radius = 4
            else:
                pen_width = 1
                asteroid_radius = 3
            
            # Draw orbit if we have orbital elements
            if hasattr(asteroid, 'orbit') and asteroid.orbit:
                self._draw_asteroid_orbit(painter, asteroid, color, pen_width)
            
            # Draw asteroid position
            # In a real application, we would calculate this from orbital elements
            # Here we're using a placeholder position based on asteroid index
            idx = list(self.asteroids.keys()).index(asteroid_id)
            angle = (idx / len(self.asteroids)) * 2 * math.pi
            distance = (1.5 + (idx % 4) * 0.5) * self.scale
            
            ast_x = self.center_x + self.offset_x + math.cos(angle) * distance
            ast_y = self.center_y + self.offset_y + math.sin(angle) * distance
            
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(ast_x, ast_y), asteroid_radius, asteroid_radius)
            
            # Draw name label for the selected asteroid
            if asteroid_id == self.selected_asteroid_id:
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.drawText(ast_x + 8, ast_y, asteroid.name)
                
                # Draw close approach vector if applicable
                if asteroid.close_approaches:
                    self._draw_approach_vector(painter, asteroid, ast_x, ast_y)
    
    def _draw_asteroid_orbit(self, painter: QPainter, asteroid: NEO, 
                            color: QColor, pen_width: int) -> None:
        """
        Draw an asteroid's orbit based on orbital elements.
        
        Args:
            painter: QPainter object
            asteroid: NEO object with orbital information
            color: Color to use for drawing
            pen_width: Width of the drawing pen
        """
        # This is a simplified placeholder
        # In a real application, we would use actual orbital elements to draw an ellipse
        
        # Here we're just drawing a simple ellipse as placeholder
        semi_major = (1.5 + asteroid.absolute_magnitude / 30) * self.scale
        eccentricity = 0.2 + (hash(asteroid.id) % 100) / 200
        semi_minor = semi_major * math.sqrt(1 - eccentricity * eccentricity)
        
        orbit_color = QColor(color)
        orbit_color.setAlpha(120)
        painter.setPen(QPen(orbit_color, pen_width, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        # Draw tilted ellipse
        angle = (hash(asteroid.id) % 100) / 100 * math.pi
        painter.save()
        painter.translate(self.center_x + self.offset_x, self.center_y + self.offset_y)
        painter.rotate(angle * 180 / math.pi)
        painter.drawEllipse(QPointF(0, 0), semi_major, semi_minor)
        painter.restore()
    
    def _draw_approach_vector(self, painter: QPainter, asteroid: NEO, 
                             ast_x: float, ast_y: float) -> None:
        """
        Draw vector showing the closest approach trajectory.
        
        Args:
            painter: QPainter object
            asteroid: NEO object with close approach data
            ast_x: Asteroid x position on the canvas
            ast_y: Asteroid y position on the canvas
        """
        approach = asteroid.close_approaches[0]
        
        # Calculate Earth's position at approach time
        # This is a simplified calculation for illustration
        earth_angle = (approach.approach_time.timestamp() % 31536000) / 31536000 * 2 * math.pi
        earth_x = self.center_x + self.offset_x + math.cos(earth_angle) * self.scale
        earth_y = self.center_y + self.offset_y + math.sin(earth_angle) * self.scale
        
        # Draw approach line
        approach_color = QColor(255, 50, 50, 150)
        painter.setPen(QPen(approach_color, 2, Qt.PenStyle.DashLine))
        painter.drawLine(ast_x, ast_y, earth_x, earth_y)
        
        # Draw distance annotation
        mid_x = (ast_x + earth_x) / 2
        mid_y = (ast_y + earth_y) / 2
        distance_text = f"{approach.distance_au:.4f} AU"
        painter.setPen(QPen(Qt.GlobalColor.white))
        
        # Draw distance text with a background for better visibility
        text_rect = painter.fontMetrics().boundingRect(distance_text)
        bg_rect = QRectF(mid_x - 5, mid_y - text_rect.height() / 2 - 2, 
                         text_rect.width() + 10, text_rect.height() + 4)
        painter.fillRect(bg_rect, QColor(0, 0, 0, 150))
        painter.drawText(mid_x, mid_y + text_rect.height() / 2 - 2, distance_text)
        
        # Draw approach date near the line
        date_text = approach.approach_time.strftime("%Y-%m-%d")
        date_rect = painter.fontMetrics().boundingRect(date_text)
        date_x = mid_x
        date_y = mid_y + 20
        bg_rect = QRectF(date_x - 5, date_y - date_rect.height() / 2 - 2, 
                         date_rect.width() + 10, date_rect.height() + 4)
        painter.fillRect(bg_rect, QColor(0, 0, 0, 150))
        painter.drawText(date_x, date_y + date_rect.height() / 2 - 2, date_text)
        
        # Visual indicator of threat level
        if approach.distance_au < 0.05:  # Threshold for potentially hazardous
            threat_radius = 15
            threat_color = QColor(255, 30, 30, 180)  # Red with transparency
            painter.setBrush(QBrush(threat_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(mid_x, mid_y), threat_radius, threat_radius)


class ThreatAnalysisPanel(QWidget):
    """
    Panel for displaying detailed asteroid threat analysis information.
    
    This panel provides information about potentially hazardous asteroids,
    including approach dates, distances, and risk assessments.
    """
    
    def __init__(self, parent=None):
        """Initialize the threat analysis panel."""
        super().__init__(parent)
        self.asteroid = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components for the threat analysis panel."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Asteroid Threat Analysis")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title_label)
        
        # Asteroid information section
        self.info_group = QGroupBox("Asteroid Information")
        info_layout = QGridLayout(self.info_group)
        
        # Basic asteroid information
        self.name_label = QLabel("Name: N/A")
        self.id_label = QLabel("ID: N/A")
        self.magnitude_label = QLabel("Absolute Magnitude: N/A")
        self.diameter_label = QLabel("Estimated Diameter: N/A")
        self.hazardous_label = QLabel("Potentially Hazardous: N/A")
        
        info_layout.addWidget(self.name_label, 0, 0)
        info_layout.addWidget(self.id_label, 1, 0)
        info_layout.addWidget(self.magnitude_label, 2, 0)
        info_layout.addWidget(self.diameter_label, 3, 0)
        info_layout.addWidget(self.hazardous_label, 4, 0)
        
        # Set hazardous label font and initial color
        hazardous_font = QFont("Arial", 10, QFont.Weight.Bold)
        self.hazardous_label.setFont(hazardous_font)
        
        layout.addWidget(self.info_group)
        
        # Closest approach section
        self.approach_group = QGroupBox("Closest Approach")
        approach_layout = QGridLayout(self.approach_group)
        
        self.date_label = QLabel("Date: N/A")
        self.distance_label = QLabel("Miss Distance: N/A")
        self.velocity_label = QLabel("Relative Velocity: N/A")
        self.orbiting_body_label = QLabel("Orbiting Body: N/A")
        
        approach_layout.addWidget(self.date_label, 0, 0)
        approach_layout.addWidget(self.distance_label, 1, 0)
        approach_layout.addWidget(self.velocity_label, 2, 0)
        approach_layout.addWidget(self.orbiting_body_label, 3, 0)
        
        layout.addWidget(self.approach_group)
        
        # Threat assessment section
        self.threat_group = QGroupBox("Threat Assessment")
        threat_layout = QGridLayout(self.threat_group)
        
        self.risk_label = QLabel("Risk Level: N/A")
        self.risk_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        
        self.impact_probability_label = QLabel("Impact Probability: N/A")
        self.energy_label = QLabel("Impact Energy: N/A")
        self.torino_label = QLabel("Torino Scale: N/A")
        
        threat_layout.addWidget(self.risk_label, 0, 0, 1, 2)
        threat_layout.addWidget(self.impact_probability_label, 1, 0)
        threat_layout.addWidget(self.energy_label, 2, 0)
        threat_layout.addWidget(self.torino_label, 3, 0)
        
        # Progress bar for risk level visualization
        self.risk_progress = QProgressBar()
        self.risk_progress.setRange(0, 100)
        self.risk_progress.setValue(0)
        self.risk_progress.setTextVisible(False)
        threat_layout.addWidget(self.risk_progress, 1, 1, 3, 1)
        
        layout.addWidget(self.threat_group)
        
        # Additional notes text area
        self.notes_group = QGroupBox("Analysis Notes")
        notes_layout = QVBoxLayout(self.notes_group)
        
        self.notes_text = QTextEdit()
        self.notes_text.setReadOnly(True)
        self.notes_text.setMaximumHeight(100)
        notes_layout.addWidget(self.notes_text)
        
        layout.addWidget(self.notes_group)
        
        # Add stretch to keep UI components at the top
        layout.addStretch()
        
    def set_asteroid(self, asteroid):
        """
        Update the panel with information about the selected asteroid.
        
        Args:
            asteroid: NEO object with asteroid information
        """
        if not asteroid:
            self.clear_data()
            return
        
        self.asteroid = asteroid
        
        # Update basic information
        self.name_label.setText(f"Name: {asteroid.name}")
        self.id_label.setText(f"ID: {asteroid.id}")
        self.magnitude_label.setText(f"Absolute Magnitude: {asteroid.absolute_magnitude:.1f} H")
        
        # Estimated diameter (from absolute magnitude)
        if hasattr(asteroid, 'diameter_min') and hasattr(asteroid, 'diameter_max'):
            diameter_text = f"Estimated Diameter: {asteroid.diameter_min:.2f} - {asteroid.diameter_max:.2f} km"
        else:
            # Rough estimation based on absolute magnitude
            est_diameter = 1329 * 10**(-asteroid.absolute_magnitude/5) / 2
            diameter_text = f"Estimated Diameter: ~{est_diameter:.2f} km"
        self.diameter_label.setText(diameter_text)
        
        # Hazardous status
        hazardous_text = "Potentially Hazardous: Yes" if asteroid.hazardous else "Potentially Hazardous: No"
        self.hazardous_label.setText(hazardous_text)
        
        if asteroid.hazardous:
            self.hazardous_label.setStyleSheet("color: red;")
        else:
            self.hazardous_label.setStyleSheet("color: green;")
        
        # Close approach information
        if asteroid.close_approaches and len(asteroid.close_approaches) > 0:
            approach = asteroid.close_approaches[0]  # Most recent/relevant approach
            
            # Format approach date
            date_str = approach.approach_time.strftime("%Y-%m-%d %H:%M:%S")
            self.date_label.setText(f"Date: {date_str}")
            
            # Format distance
            au_distance = approach.distance_au
            km_distance = au_distance * 149597870.7  # Convert AU to km
            ld_distance = au_distance * 389.17  # Convert AU to lunar distances
            self.distance_label.setText(f"Miss Distance: {au_distance:.6f} AU ({ld_distance:.1f} LD)")
            
            # Format velocity
            if hasattr(approach, 'velocity_kps'):
                velocity = approach.velocity_kps
                self.velocity_label.setText(f"Relative Velocity: {velocity:.2f} km/s")
            else:
                self.velocity_label.setText("Relative Velocity: N/A")
            
            # Orbiting body
            if hasattr(approach, 'orbiting_body'):
                self.orbiting_body_label.setText(f"Orbiting Body: {approach.orbiting_body}")
            else:
                self.orbiting_body_label.setText("Orbiting Body: Earth")
            
            # Risk assessment
            if au_distance < 0.05:
                if au_distance < 0.002:  # ~ 0.8 lunar distances
                    risk_level = "HIGH"
                    risk_color = "darkred"
                    risk_value = 90
                    torino = "Elevated (4-7)"
                elif au_distance < 0.025:  # ~ 10 lunar distances
                    risk_level = "MODERATE"
                    risk_color = "orange"
                    risk_value = 60
                    torino = "Low (2-3)"
                else:
                    risk_level = "LOW"
                    risk_color = "yellow"
                    risk_value = 30
                    torino = "Normal (1)"
            else:
                risk_level = "NEGLIGIBLE"
                risk_color = "green"
                risk_value = 10
                torino = "None (0)"
            
            self.risk_label.setText(f"Risk Level: {risk_level}")
            self.risk_label.setStyleSheet(f"color: {risk_color};")
            self.risk_progress.setValue(risk_value)
            self.torino_label.setText(f"Torino Scale: {torino}")
            
            # Set progress bar color
            progress_style = f"QProgressBar::chunk {{ background-color: {risk_color}; }}"
            self.risk_progress.setStyleSheet(progress_style)
            
            # Very rough impact probability calculation (simplified)
            # In reality, this would involve complex orbital calculations
            impact_prob = 0
            if au_distance < 0.1:
                # Very simplistic model - just for demonstration
                impact_prob = (1 / (au_distance * 100)) * 0.01
                impact_prob = min(impact_prob, 0.1)  # Cap at 10%
            
            self.impact_probability_label.setText(f"Impact Probability: {impact_prob:.6f}%")
            
            # Simple impact energy calculation (simplified)
            # E = 1/2 * m * v^2, assuming density of 3000 kg/m^3 for asteroid
            if hasattr(approach, 'velocity_kps') and 'diameter' in diameter_text:
                # Extract diameter from the text
                import re
                diameter_match = re.search(r"~(\d+\.\d+)", diameter_text)
                if diameter_match:
                    diameter_km = float(diameter_match.group(1))
                    radius_m = diameter_km * 500  # Convert to radius in meters
                    
                    # Approximate mass (assuming spherical shape, density 3000 kg/m^3)
                    # Approximate mass (assuming spherical shape, density 3000 kg/m^3)
                    volume = (4/3) * math.pi * (radius_m**3)
                    mass = volume * 3000  # kg
                    
                    # Calculate impact energy (KE = 0.5 * m * v^2)
                    velocity_ms = approach.velocity_kps * 1000  # Convert km/s to m/s
                    energy_joules = 0.5 * mass * (velocity_ms**2)
                    
                    # Convert to megatons of TNT (1 megaton = 4.184e15 joules)
                    energy_megatons = energy_joules / 4.184e15
                    
                    self.energy_label.setText(f"Impact Energy: {energy_megatons:.4f} Mt")
                else:
                    # Can't calculate energy without diameter and velocity
                    self.energy_label.setText("Impact Energy: Insufficient data")
            else:
                # Clear approach information when no close approaches are available
                self.date_label.setText("Date: N/A")
                self.distance_label.setText("Miss Distance: N/A")
                self.velocity_label.setText("Relative Velocity: N/A")
                self.orbiting_body_label.setText("Orbiting Body: N/A")
                self.risk_label.setText("Risk Level: N/A")
                self.impact_probability_label.setText("Impact Probability: N/A")
                self.energy_label.setText("Impact Energy: N/A")
                self.torino_label.setText("Torino Scale: N/A")
                self.risk_progress.setValue(0)
            
            # Generate analysis notes
            self._generate_analysis_notes(asteroid)
    
    def _generate_analysis_notes(self, asteroid: NEO) -> None:
        """
        Generate analysis notes for the selected asteroid.
        
        Args:
            asteroid: NEO object
        """
        notes = []
        
        if asteroid.hazardous:
            notes.append("ALERT: This asteroid is classified as potentially hazardous.")
        
        if asteroid.close_approaches and len(asteroid.close_approaches) > 0:
            approach = asteroid.close_approaches[0]
            
            # Add information about the closest approach
            notes.append(f"Closest approach occurs on {approach.approach_time.strftime('%Y-%m-%d')}.")
            
            # Risk assessment
            if hasattr(approach, 'distance_au'):
                if approach.distance_au < 0.05:
                    notes.append("This object will pass within 0.05 AU of Earth.")
                    if approach.distance_au < 0.01:
                        notes.append("CAUTION: Extremely close approach detected.")
            
            # Velocity assessment
            if hasattr(approach, 'velocity_kps'):
                if approach.velocity_kps > 30:
                    notes.append(f"High velocity approach: {approach.velocity_kps:.2f} km/s")
        
        # Add information about physical properties
        if hasattr(asteroid, 'diameter_min') and hasattr(asteroid, 'diameter_max'):
            avg_diameter = (asteroid.diameter_min + asteroid.diameter_max) / 2
            if avg_diameter > 1.0:
                notes.append(f"Large object: approximately {avg_diameter:.2f} km in diameter.")
                notes.append("Objects larger than 1 km can cause global effects on impact.")
            elif avg_diameter > 0.14:
                notes.append(f"Medium object: approximately {avg_diameter:.2f} km in diameter.")
                notes.append("Objects larger than 140m can cause regional devastation.")
        
        # Set notes in the text area
        if notes:
            self.notes_text.setText("\n".join(notes))
        else:
            self.notes_text.setText("No specific analysis notes available for this object.")
    
    def clear_data(self) -> None:
        """Clear all displayed data in the panel."""
        self.name_label.setText("Name: N/A")
        self.id_label.setText("ID: N/A")
        self.magnitude_label.setText("Absolute Magnitude: N/A")
        self.diameter_label.setText("Estimated Diameter: N/A")
        self.hazardous_label.setText("Potentially Hazardous: N/A")
        self.hazardous_label.setStyleSheet("")
        
        self.date_label.setText("Date: N/A")
        self.distance_label.setText("Miss Distance: N/A")
        self.velocity_label.setText("Relative Velocity: N/A")
        self.orbiting_body_label.setText("Orbiting Body: N/A")
        
        self.risk_label.setText("Risk Level: N/A")
        self.impact_probability_label.setText("Impact Probability: N/A")
        self.energy_label.setText("Impact Energy: N/A")
        self.torino_label.setText("Torino Scale: N/A")
        self.risk_progress.setValue(0)
        
        self.notes_text.setText("")


class AsteroidView(QWidget):
    """
    Main asteroid monitoring view that integrates visualization and threat analysis.
    
    This widget combines an orbital visualization with threat analysis and
    control panels to provide a comprehensive asteroid monitoring interface.
    """
    
    # Signal emitted when a new asteroid is selected
    asteroid_selected = pyqtSignal(str)
    
    # Signal emitted when a potentially hazardous asteroid is detected
    hazard_detected = pyqtSignal(NEO)
    
    def __init__(self, parent=None):
        """Initialize the asteroid view with all components."""
        super().__init__(parent)
        
        # Initialize NEO tracker (with your API key)
        # For now we use a placeholder - in production, get this from config
        self.api_key = "DEMO_KEY"  # Replace with your NASA API key
        self.neo_tracker = NEOTracker(self.api_key)
        
        # Data storage
        self.asteroids = {}  # Dictionary of NEO objects keyed by ID
        self.filtered_asteroids = {}  # Filtered subset of asteroids
        self.selected_asteroid_id = None
        
        # Set up the UI components
        self.setup_ui()
        
        # Set up the update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_asteroid_data)
        
        # Initial data load
        self.load_asteroid_data()
    
    def setup_ui(self):
        """Set up the UI components for the asteroid view."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Add title and info bar
        title_layout = QHBoxLayout()
        title_label = QLabel("Asteroid Monitoring System")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.status_label = QLabel("Status: Initializing...")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.status_label)
        main_layout.addLayout(title_layout)
        
        # Create main splitter for visualization and analysis
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Visualization and control panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Add orbit visualization
        self.orbit_viz = OrbitVisualization()
        left_layout.addWidget(self.orbit_viz)
        
        # Add control panel for visualization
        control_panel = self.create_control_panel()
        left_layout.addWidget(control_panel)
        
        # Right side: Threat analysis and asteroid list
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Add threat analysis panel
        self.threat_panel = ThreatAnalysisPanel()
        right_layout.addWidget(self.threat_panel)
        
        # Add asteroid list
        list_group = QGroupBox("Asteroid List")
        list_layout = QVBoxLayout(list_group)
        
        # Search and filter controls
        filter_layout = QHBoxLayout()
        self.search_input = QComboBox()
        self.search_input.setEditable(True)
        self.search_input.setPlaceholderText("Search asteroids...")
        self.search_input.currentTextChanged.connect(self.on_search_changed)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Asteroids", "Potentially Hazardous", "Close Approaches", "Recent Discoveries"])
        self.filter_combo.currentTextChanged.connect(self.apply_filters)
        
        filter_layout.addWidget(QLabel("Search:"))
        filter_layout.addWidget(self.search_input, 3)
        filter_layout.addWidget(QLabel("Filter:"))
        filter_layout.addWidget(self.filter_combo, 2)
        
        list_layout.addLayout(filter_layout)
        
        # Asteroid table
        self.asteroid_table = QTableWidget()
        self.asteroid_table.setColumnCount(5)
        self.asteroid_table.setHorizontalHeaderLabels(["Name", "Diameter (km)", "Closest Approach", "Distance (AU)", "Hazardous"])
        self.asteroid_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.asteroid_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.asteroid_table.itemSelectionChanged.connect(self.on_asteroid_selected)
        
        list_layout.addWidget(self.asteroid_table)
        
        right_layout.addWidget(list_group)
        
        # Add widgets to splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([500, 500])  # Initial sizes
        
        main_layout.addWidget(main_splitter)
        
        # Add status bar
        status_layout = QHBoxLayout()
        self.update_status = QLabel("Last Update: Never")
        self.asteroid_count = QLabel("Asteroids: 0")
        self.hazardous_count = QLabel("Hazardous: 0")
        
        status_layout.addWidget(self.update_status)
        status_layout.addStretch()
        status_layout.addWidget(self.asteroid_count)
        status_layout.addWidget(self.hazardous_count)
        
        main_layout.addLayout(status_layout)
    
    def create_control_panel(self):
        """Create the control panel for visualization options and data updates."""
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout(control_group)
        
        # Visualization controls
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Scale slider
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale:"))
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(10, 200)
        self.scale_slider.setValue(50)
        self.scale_slider.valueChanged.connect(self.on_scale_changed)
        scale_layout.addWidget(self.scale_slider)
        viz_layout.addLayout(scale_layout)
        
        # Display options
        self.show_orbits = QCheckBox("Show Orbits")
        self.show_orbits.setChecked(True)
        self.show_orbits.toggled.connect(self.update_display_options)
        
        self.show_approach_vectors = QCheckBox("Show Approach Vectors")
        self.show_approach_vectors.setChecked(True)
        self.show_approach_vectors.toggled.connect(self.update_display_options)
        
        self.show_labels = QCheckBox("Show Labels")
        self.show_labels.setChecked(True)
        self.show_labels.toggled.connect(self.update_display_options)
        
        viz_layout.addWidget(self.show_orbits)
        viz_layout.addWidget(self.show_approach_vectors)
        viz_layout.addWidget(self.show_labels)
        
        control_layout.addWidget(viz_group)
        
        # Data update controls
        update_group = QGroupBox("Data Updates")
        update_layout = QVBoxLayout(update_group)
        
        self.auto_update = QCheckBox("Real-time Updates")
        self.auto_update.setChecked(False)
        self.auto_update.toggled.connect(self.toggle_auto_update)
        
        update_layout.addWidget(self.auto_update)
        
        # Update frequency
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Update Frequency:"))
        self.update_freq = QComboBox()
        self.update_freq.addItems(["1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour"])
        self.update_freq.setCurrentIndex(1)  # Default to 5 minutes
        freq_layout.addWidget(self.update_freq)
        update_layout.addLayout(freq_layout)
        
        # Manual update button
        self.update_button = QPushButton("Update Now")
        self.update_button.clicked.connect(self.update_asteroid_data)
        update_layout.addWidget(self.update_button)
        
        control_layout.addWidget(update_group)
        
        # Data source controls
        source_group = QGroupBox("Data Source")
        source_layout = QVBoxLayout(source_group)
        
        # Time range
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time Range:"))
        self.time_range = QComboBox()
        self.time_range.addItems(["Next 7 days", "Next 30 days", "Next 60 days", "Next 90 days"])
        self.time_range.currentTextChanged.connect(self.on_time_range_changed)
        time_layout.addWidget(self.time_range)
        source_layout.addLayout(time_layout)
        
        # Minimum diameter filter
        diameter_layout = QHBoxLayout()
        diameter_layout.addWidget(QLabel("Min Diameter:"))
        self.min_diameter = QComboBox()
        self.min_diameter.addItems(["All", ">50m", ">100m", ">500m", ">1km", ">10km"])
        self.min_diameter.currentTextChanged.connect(self.apply_filters)
        diameter_layout.addWidget(self.min_diameter)
        source_layout.addLayout(diameter_layout)
        
        # Hazardous filter
        self.hazardous_only = QCheckBox("Potentially Hazardous Only")
        self.hazardous_only.toggled.connect(self.apply_filters)
        source_layout.addWidget(self.hazardous_only)
        
        control_layout.addWidget(source_group)
        
        return control_group
    
    def load_asteroid_data(self) -> None:
        """
        Initial load of asteroid data from the NASA API.
        
        This loads asteroid data for the default time period and applies
        initial filtering.
        """
        self.status_label.setText("Status: Loading data...")
        
        try:
            # Get time range from UI selection
            time_range_text = self.time_range.currentText()
            days = int(time_range_text.split()[1])
            
            # Calculate start and end dates
            start_date = datetime.now()
            end_date = start_date + timedelta(days=days)
            
            # Disable update button during load
            self.update_button.setEnabled(False)
            
            # Cancel any existing loading thread
            if hasattr(self, 'loading_thread') and self.loading_thread.isRunning():
                self.loading_thread.cancel()
                self.loading_thread.wait()
            
            # Create progress bar for loading indication
            if not hasattr(self, 'progress_bar'):
                self.progress_bar = QProgressBar()
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(0)
                status_layout = self.layout().itemAt(self.layout().count() - 1).layout()
                status_layout.insertWidget(1, self.progress_bar)
            else:
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(True)
            
            # Create and start loading thread
            self.loading_thread = NEOLoadingThread(
                self.neo_tracker, start_date, end_date
            )
            self.loading_thread.data_loaded.connect(self.on_data_loaded)
            self.loading_thread.error_occurred.connect(self.on_loading_error)
            self.loading_thread.progress_updated.connect(self.on_loading_progress)
            self.loading_thread.start()
            
            self.status_label.setText("Status: Fetching data from NASA API...")
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            self.status_label.setText(f"Status: Error! {error_msg}")
            logging.error(error_msg, exc_info=True)
    
    def update_asteroid_data(self) -> None:
        """
        Update asteroid data with latest information from NASA API.
        """
        self.status_label.setText("Status: Updating data...")
        self.load_asteroid_data()
    
    def on_data_loaded(self, asteroids: Dict[str, NEO]) -> None:
        """
        Handle loaded asteroid data from the worker thread.
        
        Args:
            asteroids: Dictionary of NEO objects keyed by ID
        self.update_button.setEnabled(True)
        
        # Hide progress bar
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setVisible(False)
        
        # Apply filters to the loaded data
        self.apply_filters()
        
        # Update UI components
        self.update_status_counts()
        self.populate_search_dropdown()
        
        self.status_label.setText(f"Status: Loaded {len(asteroids)} asteroids")
        self.update_status.setText(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Check for potentially hazardous asteroids
        self.check_for_hazards()
    
        self.check_for_hazards()
    
    def on_loading_error(self, error_msg: str) -> None:
        """
        Handle errors during data loading.
        
        Args:
            error_msg: Error message to display
        """
        self.update_button.setEnabled(True)
        self.status_label.setText(f"Status: Error! {error_msg}")
        logging.error(f"Data loading error: {error_msg}")
        
        # Hide progress bar
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setVisible(False)
    
    def on_loading_progress(self, progress: int, message: str) -> None:
        """
        Handle progress updates from the loading thread.
        
        Args:
            progress: Progress percentage (0-100)
            message: Progress status message
        """
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(progress)
        
        # Update status message
        self.status_label.setText(f"Status: {message}")
    
    def apply_filters(self) -> None:
        """
        Apply all active filters to the asteroid data and update the display.
        """
        # Start with all asteroids
        filtered = self.asteroids.copy()
        
        # Apply filter by type
        filter_type = self.filter_combo.currentText()
        if filter_type == "Potentially Hazardous":
            filtered = {id: ast for id, ast in filtered.items() if ast.hazardous}
        elif filter_type == "Close Approaches":
            filtered = {id: ast for id, ast in filtered.items() 
                       if ast.close_approaches and len(ast.close_approaches) > 0}
        elif filter_type == "Recent Discoveries":
            # Filter to asteroids discovered in the last 30 days
            # This would require discovery date info from the API
            # For now, we'll use a placeholder based on ID (newer IDs often have higher numbers)
            sorted_ids = sorted(filtered.keys(), reverse=True)
            recent_ids = sorted_ids[:min(50, len(sorted_ids))]
            filtered = {id: filtered[id] for id in recent_ids}
        
        # Apply minimum diameter filter
        min_diameter_text = self.min_diameter.currentText()
        if min_diameter_text != "All":
            # Extract numeric value from text (e.g., ">100m" -> 0.1)
            size_value = float(min_diameter_text[1:-1])
            size_unit = min_diameter_text[-1]
            
            # Convert to kilometers
            if size_unit == 'm':
                size_km = size_value / 1000
            else:  # Assume km
                size_km = size_value
                
            # Apply filter
            filtered = {id: ast for id, ast in filtered.items() 
                       if hasattr(ast, 'diameter_min') and ast.diameter_min >= size_km}
        
        # Apply hazardous only filter
        if self.hazardous_only.isChecked():
            filtered = {id: ast for id, ast in filtered.items() if ast.hazardous}
        
        # Apply search filter if present
        search_text = self.search_input.currentText().strip().lower()
        if search_text:
            filtered = {id: ast for id, ast in filtered.items() 
                       if search_text in ast.name.lower() or search_text in ast.id.lower()}
        
        # Update filtered data and refresh display
        self.filtered_asteroids = filtered
        self.update_asteroid_table()
        self.update_visualization()
        
        # Update status counts
        self.update_status_counts()
    
    def update_status_counts(self) -> None:
        """Update status bar with current asteroid counts."""
        total_count = len(self.asteroids)
        hazardous_count = sum(1 for ast in self.asteroids.values() if ast.hazardous)
        filtered_count = len(self.filtered_asteroids)
        
        self.asteroid_count.setText(f"Asteroids: {filtered_count}/{total_count}")
        self.hazardous_count.setText(f"Hazardous: {hazardous_count}")
    
    def populate_search_dropdown(self) -> None:
        """Populate the search dropdown with asteroid names."""
        current_text = self.search_input.currentText()
        
        self.search_input.clear()
        self.search_input.addItem("")  # Empty option for no filter
        
        # Add asteroid names (limit to avoid overwhelming the dropdown)
        names = sorted([ast.name for ast in self.asteroids.values()])[:100]
        self.search_input.addItems(names)
        
        # Restore current text
        if current_text:
            self.search_input.setCurrentText(current_text)
    
    def update_asteroid_table(self) -> None:
        """Update the asteroid table with current filtered data."""
        # Clear table
        self.asteroid_table.setRowCount(0)
        
        if not self.filtered_asteroids:
            return
            
        # Populate table with data
        for row, (asteroid_id, asteroid) in enumerate(self.filtered_asteroids.items()):
            self.asteroid_table.insertRow(row)
            
            # Name
            name_item = QTableWidgetItem(asteroid.name)
            self.asteroid_table.setItem(row, 0, name_item)
            
            # Diameter
            if hasattr(asteroid, 'diameter_min') and hasattr(asteroid, 'diameter_max'):
                diameter_text = f"{asteroid.diameter_min:.2f} - {asteroid.diameter_max:.2f}"
            else:
                # Approximate from absolute magnitude
                diameter = 1329 * 10**(-asteroid.absolute_magnitude/5) / 2
                diameter_text = f"~{diameter:.2f}"
            diameter_item = QTableWidgetItem(diameter_text)
            self.asteroid_table.setItem(row, 1, diameter_item)
            
            # Closest approach
            if asteroid.close_approaches and len(asteroid.close_approaches) > 0:
                approach = asteroid.close_approaches[0]
                approach_date = approach.approach_time.strftime("%Y-%m-%d")
                approach_item = QTableWidgetItem(approach_date)
                
                # Distance
                distance_text = f"{approach.distance_au:.6f}"
                distance_item = QTableWidgetItem(distance_text)
                
                self.asteroid_table.setItem(row, 2, approach_item)
                self.asteroid_table.setItem(row, 3, distance_item)
            else:
                # No close approach data
                self.asteroid_table.setItem(row, 2, QTableWidgetItem("N/A"))
                self.asteroid_table.setItem(row, 3, QTableWidgetItem("N/A"))
            
            # Hazardous
            hazardous_text = "YES" if asteroid.hazardous else "NO"
            hazardous_item = QTableWidgetItem(hazardous_text)
            
            # Set text color based on hazardous status
            if asteroid.hazardous:
                hazardous_item.setForeground(QBrush(QColor(255, 30, 30)))
            else:
                hazardous_item.setForeground(QBrush(QColor(30, 180, 30)))
                
            self.asteroid_table.setItem(row, 4, hazardous_item)
            
            # Store asteroid ID as item data
            name_item.setData(Qt.ItemDataRole.UserRole, asteroid_id)
        
        # Resize columns to contents
        self.asteroid_table.resizeColumnsToContents()
        
        # Select the previously selected asteroid if it's still in the filtered list
        if self.selected_asteroid_id in self.filtered_asteroids:
            for row in range(self.asteroid_table.rowCount()):
                item = self.asteroid_table.item(row, 0)
                if item.data(Qt.ItemDataRole.UserRole) == self.selected_asteroid_id:
                    self.asteroid_table.selectRow(row)
                    break
    
    def update_visualization(self) -> None:
        """Update the orbit visualization with the filtered asteroid data."""
        # Update with current filtered data
        self.orbit_viz.set_asteroids(self.filtered_asteroids)
        
        # Refresh selected asteroid if set
        if self.selected_asteroid_id:
            self.orbit_viz.select_asteroid(self.selected_asteroid_id)
    
    def check_for_hazards(self) -> None:
        """Check for potentially hazardous asteroids and emit warnings."""
        hazardous_asteroids = []
        
        for asteroid in self.asteroids.values():
            if asteroid.hazardous:
                if asteroid.close_approaches and asteroid.close_approaches[0].distance_au < 0.05:
                    hazardous_asteroids.append(asteroid)
        
        # Sort by closest approach
        hazardous_asteroids.sort(key=lambda a: a.close_approaches[0].distance_au if a.close_approaches else float('inf'))
        
        # Emit warnings for the most concerning asteroids
        for asteroid in hazardous_asteroids[:3]:  # Limit to top 3 to avoid overwhelming
            self.hazard_detected.emit(asteroid)
    
    def on_asteroid_selected(self) -> None:
        """Handle asteroid selection from the table."""
        selected_rows = self.asteroid_table.selectedItems()
        if not selected_rows:
            self.selected_asteroid_id = None
            self.orbit_viz.select_asteroid(None)
            self.threat_panel.set_asteroid(None)
            return
            
        # Get first selected row
        row = selected_rows[0].row()
        name_item = self.asteroid_table.item(row, 0)
        
        # Get asteroid ID from item data
        asteroid_id = name_item.data(Qt.ItemDataRole.UserRole)
        self.selected_asteroid_id = asteroid_id
        
        # Update visualization and threat panel
        self.orbit_viz.select_asteroid(asteroid_id)
        self.threat_panel.set_asteroid(self.filtered_asteroids.get(asteroid_id))
        
        # Emit selection signal
        self.asteroid_selected.emit(asteroid_id)
    
    def on_search_changed(self, text: str) -> None:
        """Handle changes to the search input."""
        self.apply_filters()
    
    def update_display_options(self) -> None:
        """Update display options for the visualization."""
        # This would pass display preferences to the orbit visualization
        # For now, this is a placeholder as the actual implementation
        # would depend on OrbitVisualization capabilities
        pass
    
    def on_scale_changed(self, value: int) -> None:
        """Handle changes to the scale slider."""
        # Update the scale in the orbit visualization
        self.orbit_viz.scale = value
        self.orbit_viz.update()
    
    def on_time_range_changed(self, text: str) -> None:
        """Handle changes to the time range selection."""
        # This will trigger a new data load with the updated time range
        self.load_asteroid_data()
    
    def toggle_auto_update(self, enabled: bool) -> None:
        """Toggle automatic updates."""
        if enabled:
            # Get update frequency in minutes
            frequency_text = self.update_freq.currentText()
            minutes = int(frequency_text.split()[0])
            
            # Set timer interval in milliseconds
            self.update_timer.setInterval(minutes * 60 * 1000)
            self.update_timer.start()
        else:
            self.update_timer.stop()


class NEOLoadingThread(QThread):
    """
    Thread for asynchronous loading of asteroid data from NASA's API.
    
    This thread handles the loading of NEO data in the background to keep
    the UI responsive during potentially lengthy API calls.
    """
    # Signal emitted when data is successfully loaded
    data_loaded = pyqtSignal(dict)
    
    # Signal emitted when an error occurs during loading
    error_occurred = pyqtSignal(str)
    
    # Signal for progress updates during loading
    progress_updated = pyqtSignal(int, str)
    
    def __init__(self, neo_tracker, start_date, end_date):
        """
        Initialize the loading thread.
        
        Args:
            neo_tracker: NEOTracker instance to use for data fetching
            start_date: Start date for asteroid data
            end_date: End date for asteroid data
        """
        super().__init__()
        self.neo_tracker = neo_tracker
        self.start_date = start_date
        self.end_date = end_date
        self.canceled = False
    
    def run(self):
        """
        Execute the data loading operation in the background thread.
        
        This method fetches asteroid data from the NASA API, processes it,
        and emits signals with the results or error messages.
        """
        try:
            # Emit initial progress update
            self.progress_updated.emit(0, "Initializing data fetch...")
            
            # Format dates for the API
            start_str = self.start_date.strftime("%Y-%m-%d")
            end_str = self.end_date.strftime("%Y-%m-%d")
            
            # Check if operation is canceled
            if self.canceled:
                return
                
            # Emit progress update before API call
            self.progress_updated.emit(10, f"Fetching NEO data from {start_str} to {end_str}...")
            
            # Get asteroid data from the API
            asteroids = self.neo_tracker.get_neo_data(start_str, end_str)
            
            # Check if operation is canceled
            if self.canceled:
                return
                
            # Emit progress update before processing
            self.progress_updated.emit(50, f"Processing data for {len(asteroids)} asteroids...")
            
            # Process close approaches for each asteroid
            for i, (asteroid_id, asteroid) in enumerate(asteroids.items()):
                # Update progress every 10 asteroids
                if i % 10 == 0:
                    progress = 50 + min(40, int((i / len(asteroids)) * 40))
                    self.progress_updated.emit(progress, f"Processing asteroid {i+1}/{len(asteroids)}...")
                
                # Get close approaches if not already loaded
                if not asteroid.close_approaches:
                    try:
                        approaches = self.neo_tracker.get_close_approaches(asteroid_id)
                        asteroid.close_approaches = approaches
                    except Exception as approach_error:
                        # Log error but continue with other asteroids
                        logging.warning(f"Failed to get close approaches for {asteroid_id}: {approach_error}")
                
                # Check if operation is canceled
                if self.canceled:
                    return
            
            # Analyze close approaches for hazard assessment
            self.progress_updated.emit(90, "Analyzing close approaches and hazard levels...")
            self.neo_tracker.analyze_close_approaches(asteroids)
            
            # Final progress update
            self.progress_updated.emit(100, "Data loading complete!")
            
            # Emit the loaded data
            self.data_loaded.emit(asteroids)
            
        except Exception as e:
            # Log the error
            error_msg = f"Error loading asteroid data: {str(e)}"
            logging.error(error_msg, exc_info=True)
            
            # Emit the error signal
            self.error_occurred.emit(error_msg)
    
    def cancel(self):
        """
        Cancel the currently running operation.
        
        This method sets a flag that causes the run method to exit early.
        """
        self.canceled = True
        

# Add the AsteroidView.on_time_range_changed method that was referenced but missing
    def on_time_range_changed(self, text: str) -> None:
        """
        Handle changes to the time range selection.
        
        This method triggers a new data load with the updated time range.
        
        Args:
            text: Selected time range text
        """
        # This will trigger a new data load with the updated time range
        self.load_asteroid_data()
