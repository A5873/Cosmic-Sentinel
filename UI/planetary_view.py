#!/usr/bin/env python3
"""
Planetary View Module

This module implements the visualization and interactive components for
displaying planetary positions, orbits, and related astronomical data.
It provides both 2D and 3D views for planetary tracking along with
controls for adjusting time and simulation parameters.
"""

import math
import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

from PyQt6.QtCore import (
    Qt, QTimer, QRectF, QPointF, QDateTime, QDate, QTime, 
    pyqtSignal, pyqtSlot, QObject
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, 
    QLinearGradient, QPainterPath, QRadialGradient
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QGroupBox, QCheckBox, QSplitter,
    QDateTimeEdit, QFrame, QTabWidget, QToolButton, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsPathItem,
    QGraphicsItem, QGraphicsItemGroup, QTableWidget, QTableWidgetItem,
    QHeaderView, QScrollArea, QToolBar, QStyle, QStyleOptionButton
)

from API.planetary_tracking import PlanetaryTracker
from models.planet_data import PlanetData, OrbitData, CoordinateSystem


class ViewMode(Enum):
    """Enum representing the different view modes for the planetary visualization."""
    VIEW_2D = 0
    VIEW_3D = 1


class OrbitalVisualization(QGraphicsView):
    """
    A widget that provides a graphical visualization of planets and their orbits.
    
    This component renders a dynamic view of the solar system with accurate
    positioning of planets based on astronomical calculations. It supports both
    2D (top-down) and 3D perspectives with interactive controls.
    """
    
    planet_selected = pyqtSignal(str)  # Signal emitted when a planet is selected
    time_changed = pyqtSignal(QDateTime)  # Signal emitted when the visualization time changes
    
    # Planet colors with realistic approximations
    PLANET_COLORS = {
        "Mercury": QColor(169, 169, 169),  # Gray
        "Venus": QColor(255, 198, 73),     # Yellow-orange
        "Earth": QColor(78, 127, 216),     # Blue
        "Mars": QColor(193, 68, 14),       # Red
        "Jupiter": QColor(240, 198, 116),  # Light brown
        "Saturn": QColor(207, 181, 59),    # Gold
        "Uranus": QColor(209, 231, 231),   # Pale cyan
        "Neptune": QColor(43, 98, 194),    # Deep blue
        "Pluto": QColor(181, 144, 95),     # Brown (included for completeness)
    }
    
    # Relative sizes for visualization (not to scale but proportional)
    PLANET_SIZES = {
        "Sun": 20.0,
        "Mercury": 2.5,
        "Venus": 4.0,
        "Earth": 4.2,
        "Mars": 3.0,
        "Jupiter": 12.0,
        "Saturn": 10.0,
        "Uranus": 7.0,
        "Neptune": 6.8,
        "Pluto": 1.5,
    }
    
    def __init__(self, parent=None):
        """Initialize the orbital visualization component."""
        super().__init__(parent)
        
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        # Setting up the scene
        self.scene().setSceneRect(-500, -500, 1000, 1000)
        
        # Initialization parameters
        self.view_mode = ViewMode.VIEW_2D
        self.scale_factor = 5.0  # AU to pixel scale
        self.show_labels = True
        self.show_orbits = True
        self.selected_planet = None
        self.current_time = QDateTime.currentDateTime()
        self.planet_items = {}  # Dictionary to store planet graphics items
        self.orbit_items = {}   # Dictionary to store orbit graphics items
        self.label_items = {}   # Dictionary to store planet label items
        
        # Planet and orbit data
        self.planet_data = {}
        self.orbit_data = {}
        
        # Add the sun
        self._add_sun()
        
        # Setup initial view
        self.resetTransform()
        self.scale(1, 1)
        self.centerOn(0, 0)
        
    def _add_sun(self):
        """Add the sun to the center of the scene."""
        sun_size = self.PLANET_SIZES["Sun"]
        sun = QGraphicsEllipseItem(-sun_size/2, -sun_size/2, sun_size, sun_size)
        
        # Create a radial gradient for the sun
        gradient = QRadialGradient(0, 0, sun_size/2)
        gradient.setColorAt(0, QColor(255, 255, 200))
        gradient.setColorAt(0.7, QColor(255, 165, 0))
        gradient.setColorAt(1, QColor(255, 69, 0))
        
        sun.setBrush(QBrush(gradient))
        sun.setPen(QPen(Qt.PenStyle.NoPen))
        sun.setZValue(100)  # Ensure sun is on top
        
        # Add a custom property for identification
        sun.setData(0, "Sun")
        
        self.scene().addItem(sun)
        self.planet_items["Sun"] = sun
        
        # Add sun label
        if self.show_labels:
            sun_label = self.scene().addText("Sun")
            sun_label.setDefaultTextColor(QColor(255, 255, 200))
            sun_label.setPos(-sun_label.boundingRect().width()/2, sun_size/2 + 5)
            self.label_items["Sun"] = sun_label
    
    def update_planet_positions(self, planet_data_dict: Dict[str, PlanetData]):
        """
        Update the positions of planets based on the provided planet data.
        
        Args:
            planet_data_dict: Dictionary mapping planet names to PlanetData objects
        """
        self.planet_data = planet_data_dict
        
        # Clear existing planets (except the sun)
        for name, item in list(self.planet_items.items()):
            if name != "Sun":
                self.scene().removeItem(item)
                self.planet_items.pop(name)
        
        # Clear existing labels (except the sun's)
        for name, item in list(self.label_items.items()):
            if name != "Sun":
                self.scene().removeItem(item)
                self.label_items.pop(name)
                
        # Clear existing orbits
        for item in self.orbit_items.values():
            self.scene().removeItem(item)
        self.orbit_items.clear()
        
        # Add planets at their current positions
        for planet_name, planet_data in planet_data_dict.items():
            if planet_name == "Sun":
                continue  # Skip the sun as it's already added
                
            # Get position data
            if self.view_mode == ViewMode.VIEW_2D:
                # Use x, y coordinates for 2D view (ecliptic plane)
                x = planet_data.position.x * self.scale_factor
                y = planet_data.position.y * self.scale_factor
            else:
                # TODO: Implement 3D perspective when in 3D mode
                x = planet_data.position.x * self.scale_factor
                y = planet_data.position.y * self.scale_factor
            
            # Create planet item
            size = self.PLANET_SIZES.get(planet_name, 3.0)
            planet = QGraphicsEllipseItem(-size/2, -size/2, size, size)
            
            # Set the planet's color
            color = self.PLANET_COLORS.get(planet_name, QColor(200, 200, 200))
            planet.setBrush(QBrush(color))
            planet.setPen(QPen(Qt.PenStyle.NoPen))
            
            # Position the planet
            planet.setPos(x, y)
            planet.setData(0, planet_name)  # Store planet name for identification
            planet.setZValue(10)  # Ensure planets are above orbits but below sun
            
            # Make the planet interactive
            planet.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            
            self.scene().addItem(planet)
            self.planet_items[planet_name] = planet
            
            # Add label
            if self.show_labels:
                label = self.scene().addText(planet_name)
                label.setDefaultTextColor(color.lighter(150))
                label.setPos(x - label.boundingRect().width()/2, y + size/2 + 5)
                self.label_items[planet_name] = label
            
            # Draw orbit if we have orbit data and orbits are enabled
            if self.show_orbits and planet_name in self.orbit_data:
                orbit = self._create_orbit_path(planet_name)
                self.scene().addItem(orbit)
                self.orbit_items[planet_name] = orbit
    
    def update_orbit_data(self, orbit_data_dict: Dict[str, OrbitData]):
        """
        Update the orbit data for planets.
        
        Args:
            orbit_data_dict: Dictionary mapping planet names to OrbitData objects
        """
        self.orbit_data = orbit_data_dict
        
        # If we should show orbits, update them
        if self.show_orbits:
            self._update_orbits()
    
    def _update_orbits(self):
        """Update orbit visualizations based on current orbit data."""
        # Clear existing orbits
        for item in self.orbit_items.values():
            self.scene().removeItem(item)
        self.orbit_items.clear()
        
        # Create new orbit paths
        for planet_name in self.orbit_data:
            if planet_name == "Sun":
                continue  # Sun doesn't have an orbit
                
            orbit = self._create_orbit_path(planet_name)
            self.scene().addItem(orbit)
            self.orbit_items[planet_name] = orbit
    
    def _create_orbit_path(self, planet_name: str) -> QGraphicsPathItem:
        """
        Create a path item representing a planet's orbit.
        
        Args:
            planet_name: Name of the planet whose orbit to create
            
        Returns:
            A QGraphicsPathItem representing the orbit
        """
        orbit_data = self.orbit_data.get(planet_name)
        if not orbit_data:
            # Create an empty path if we don't have orbit data
            return QGraphicsPathItem()
        
        # Create a path for the elliptical orbit
        path = QPainterPath()
        
        # Calculate ellipse parameters
        a = orbit_data.semi_major_axis * self.scale_factor
        b = a * math.sqrt(1 - orbit_data.eccentricity**2)
        c = orbit_data.eccentricity * a  # Distance from center to focus
        
        # Start drawing from the right side of the ellipse
        path.moveTo(a, 0)
        
        # Add the elliptical arc - approximate with cubic bezier curves
        # For simplicity, we'll use a simple approximation here
        # In a complete implementation, bezier approximation of ellipse would be used
        cx = -c  # sun at focus (offset from center)
        rect = QRectF(-a + cx, -b, 2*a, 2*b)
        path.addEllipse(rect)
        
        # Create the path item
        orbit_item = QGraphicsPathItem(path)
        
        # Set cosmetic properties
        pen = QPen(self.PLANET_COLORS.get(planet_name, QColor(200, 200, 200)))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(1)
        orbit_item.setPen(pen)
        orbit_item.setZValue(5)  # Below planets but above background
        
        return orbit_item
    
    def set_time(self, date_time: QDateTime):
        """
        Set the time for the orbital visualization.
        
        Args:
            date_time: The date and time to visualize
        """
        if self.current_time != date_time:
            self.current_time = date_time
            self.time_changed.emit(date_time)
            # This would trigger a recalculation of planet positions
            # via the connected tracker
    
    def set_view_mode(self, mode: ViewMode):
        """
        Set the visualization view mode (2D or 3D).
        
        Args:
            mode: The view mode to set
        """
        if self.view_mode != mode:
            self.view_mode = mode
            # Update the visualization based on the new mode
            if self.planet_data:
                self.update_planet_positions(self.planet_data)
    
    def set_show_labels(self, show: bool):
        """
        Set whether planet labels should be displayed.
        
        Args:
            show: True to show labels, False to hide them
        """
        if self.show_labels != show:
            self.show_labels = show
            
            # Update label visibility
            for name, label in self.label_items.items():
                label.setVisible(show)
    
    def set_show_orbits(self, show: bool):
        """
        Set whether planetary orbits should be displayed.
        
        Args:
            show: True to show orbits, False to hide them
        """
        if self.show_orbits != show:
            self.show_orbits = show
            
            # Update orbit visibility
            for orbit in self.orbit_items.values():
                orbit.setVisible(show)
                
            # If orbits are now visible but we don't have them drawn,
            # and we have orbit data, create them
            if show and not self:

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Planetary View Module

This module provides UI components for visualizing planetary positions and tracking data
in both 2D and 3D representations using PyQt6 and pyqtgraph.
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDateTimeEdit, QDockWidget, 
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QSplitter, QTabWidget, QVBoxLayout, QWidget
)
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# Import our planetary tracking API
from API.planetary_tracking import PlanetaryTracker, PlanetData


class PlanetaryView(QWidget):
    """
    A PyQt6 widget for displaying planetary positions and tracking data.
    
    This widget provides both 2D and 3D visualizations of planetary positions,
    with controls for time selection and various view options.
    """
    
    # Signal emitted when the view is updated with new data
    view_updated = pyqtSignal(datetime)
    
    # Signal emitted when a planet is selected/clicked in the view
    planet_selected = pyqtSignal(str)  # Planet name
    
    # Signal emitted when a notable event is detected
    event_detected = pyqtSignal(str, str)  # Event type, description
    
    def __init__(self, parent=None):
        """Initialize the planetary view widget."""
        super().__init__(parent)
        
        # Initialize the planetary tracker
        self.tracker = PlanetaryTracker()
        
        # Current time for planetary positions
        self.current_time = datetime.now()
        
        # Planet data cache
        self.planet_data: Dict[str, PlanetData] = {}
        
        # Planet colors for visualization
        self.planet_colors = {
            'mercury': (173, 167, 186),
            'venus': (255, 198, 73),
            'earth': (16, 130, 205),
            'mars': (226, 123, 88),
            'jupiter': (253, 167, 88),
            'saturn': (194, 194, 194),
            'uranus': (175, 238, 238),
            'neptune': (93, 118, 203),
            'pluto': (153, 153, 153),
            'sun': (255, 215, 0)
        }
        
        # Setup UI components
        self._setup_ui()
        
        # Initialize update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_real_time)
        
        # Initial data update
        self.update_data()
    
    def _setup_ui(self):
        """Set up all UI components for the planetary view."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tab widget for 2D/3D views
        self.view_tabs = QTabWidget()
        
        # Create 2D and 3D view widgets
        self.view_2d = self._create_2d_view()
        self.view_3d = self._create_3d_view()
        
        # Add views to tabs
        self.view_tabs.addTab(self.view_2d, "2D View")
        self.view_tabs.addTab(self.view_3d, "3D View")
        
        # Create control panel
        control_panel = self._create_control_panel()
        
        # Create status panel
        status_panel = self._create_status_panel()
        
        # Add main components to layout
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(self.view_tabs)
        
        # Right side panels (controls and status)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(control_panel)
        right_layout.addWidget(status_panel)
        
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([700, 300])  # Default size distribution
        
        main_layout.addWidget(main_splitter)
    
    def _create_2d_view(self) -> QWidget:
        """Create the 2D view widget using pyqtgraph."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Configure pyqtgraph
        pg.setConfigOptions(antialias=True)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Add axis labels
        self.plot_widget.setLabel('left', 'Y (AU)')
        self.plot_widget.setLabel('bottom', 'X (AU)')
        self.plot_widget.setTitle('Solar System - Top View')
        
        # Create plot items
        self.orbit_plots = {}
        self.planet_plots = {}
        
        # Add sun at the center
        sun_symbol = pg.ScatterPlotItem()
        sun_symbol.addPoints(
            [0], [0], 
            size=15, 
            brush=pg.mkBrush(*self.planet_colors['sun']), 
            symbol='o'
        )
        self.plot_widget.addItem(sun_symbol)
        
        layout.addWidget(self.plot_widget)
        return widget
    
    def _create_3d_view(self) -> QWidget:
        """Create the 3D view widget using pyqtgraph's OpenGL capabilities."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create 3D view widget
        self.view_3d_widget = gl.GLViewWidget()
        self.view_3d_widget.setCameraPosition(distance=20, elevation=30, azimuth=45)
        
        # Add coordinate grid
        grid = gl.GLGridItem()
        grid.setSize(20, 20, 1)
        grid.setSpacing(1, 1, 1)
        self.view_3d_widget.addItem(grid)
        
        # Add axes
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [10, 0, 0]]), color=(1, 0, 0, 1), width=2)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 10, 0]]), color=(0, 1, 0, 1), width=2)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 10]]), color=(0, 0, 1, 1), width=2)
        self.view_3d_widget.addItem(x_axis)
        self.view_3d_widget.addItem(y_axis)
        self.view_3d_widget.addItem(z_axis)
        
        # Create containers for 3D objects
        self.orbit_paths_3d = {}
        self.planet_items_3d = {}
        
        # Add sun at the center
        sun_mesh = gl.MeshData.sphere(rows=10, cols=20)
        sun_item = gl.GLMeshItem(
            meshdata=sun_mesh, 
            smooth=True, 
            color=QColor(*self.planet_colors['sun']),
            glOptions='additive',
            shader='shaded'
        )
        sun_item.scale(0.3, 0.3, 0.3)
        sun_item.translate(0, 0, 0)
        self.view_3d_widget.addItem(sun_item)
        
        layout.addWidget(self.view_3d_widget)
        return widget
    
    def _create_control_panel(self) -> QWidget:
        """Create the control panel with time selection and view options."""
        control_panel = QGroupBox("Controls")
        layout = QVBoxLayout(control_panel)
        
        # Time control section
        time_group = QGroupBox("Time Selection")
        time_layout = QGridLayout(time_group)
        
        # Date and time selector
        self.date_time_edit = QDateTimeEdit(self.current_time)
        self.date_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.date_time_edit.setCalendarPopup(True)
        self.date_time_edit.dateTimeChanged.connect(self.on_time_changed)
        
        # Time navigation buttons
        self.btn_now = QPushButton("Now")
        self.btn_now.clicked.connect(self.set_time_to_now)
        
        self.btn_prev_day = QPushButton("◀ Day")
        self.btn_prev_day.clicked.connect(lambda: self.adjust_time(days=-1))
        
        self.btn_next_day = QPushButton("Day ▶")
        self.btn_next_day.clicked.connect(lambda: self.adjust_time(days=1))
        
        # Real-time update toggle
        self.chk_real_time = QCheckBox("Real-time Updates")
        self.chk_real_time.toggled.connect(self.toggle_real_time_updates)
        
        # Speed slider for time simulation
        self.time_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_speed_slider.setMinimum(1)
        self.time_speed_slider.setMaximum(100)
        self.time_speed_slider.setValue(10)
        self.time_speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.time_speed_slider.setTickInterval(10)
        self.time_speed_slider.valueChanged.connect(self.set_simulation_speed)
        
        speed_label = QLabel("Simulation Speed:")
        
        # Arrange time controls
        time_layout.addWidget(QLabel("Date & Time:"), 0, 0)
        time_layout.addWidget(self.date_time_edit, 0, 1, 1, 3)
        time_layout.addWidget(self.btn_prev_day, 1, 0)
        time_layout.addWidget(self.btn_now, 1, 1)
        time_layout.addWidget(self.btn_next_day, 1, 2)
        time_layout.addWidget(self.chk_real_time, 2, 0, 1, 3)
        time_layout.addWidget(speed_label, 3, 0)
        time_layout.addWidget(self.time_speed_slider, 3, 1, 1, 3)
        
        # View options section
        view_group = QGroupBox("View Options")
        view_layout = QGridLayout(view_group)
        
        # Planet selection
        self.planet_selector = QComboBox()
        planets = ["All Planets", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
        self.planet_selector.addItems(planets)
        self.planet_selector.currentTextChanged.connect(self.on_planet_selected)
        
        # View type options
        self.view_type = QComboBox()
        self.view_type.addItems(["Top View", "Side View", "Front View", "Free View"])
        self.view_type.currentTextChanged.connect(self.change_view_perspective)
        
        # Display options
        self.chk_show_orbits = QCheckBox("Show Orbits")
        self.chk_show_orbits.setChecked(True)
        self.chk_show_orbits.toggled.connect(self.toggle_orbit_display)
        
        self.chk_show_labels = QCheckBox("Show Labels")
        self.chk_show_labels.setChecked(True)
        self.chk_show_labels.toggled.connect(self.toggle_labels)
        
        self.chk_show_distances = QCheckBox("Show Distances")
        self.chk_show_distances.setChecked(False)
        self.chk_show_distances.toggled.connect(self.toggle_distances)
        
        # Arrange view controls
        view_layout.addWidget(QLabel("Focus Planet:"), 0, 0)
        view_layout.addWidget(self.planet_selector, 0, 1)
        view_layout.addWidget(QLabel("View Perspective:"), 1, 0)
        view_layout.addWidget(self.view_type, 1, 1)
        view_layout.addWidget(self.chk_show_orbits, 2, 0)
        view_layout.addWidget(self.chk_show_labels, 2, 1)
        view_layout.addWidget(self.chk_show_distances, 3, 0)
        
        # Update button
        self.btn_update = QPushButton("Update View")
        self.btn_update.clicked.connect(self.update_data)
        
        # Add all sections to main control panel
        layout.addWidget(time_group)
        layout.addWidget(view_group)
        layout.addWidget(self.btn_update)
        layout.addStretch()
        
        return control_panel
    
    def _create_status_panel(self) -> QWidget:
        """Create the status panel for displaying information about alignments and events."""
        status_panel = QGroupBox("Status & Events")
        layout = QVBoxLayout(status_panel)
        
        # Current planet info
        info_group = QGroupBox("Selected Planet")
        info_layout = QGridLayout(info_group)
        
        self.selected_planet_label = QLabel("No planet selected")
        self.selected_planet_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        
        self.planet_distance_label = QLabel("Distance from Earth: N/A")
        self.planet_position_label = QLabel("Position (x, y, z): N/A")
        self.planet_velocity_label = QLabel("Velocity: N/A")
        
        info_layout.addWidget(self.selected_planet_label, 0, 0, 1, 2)
        info_layout.addWidget(self.planet_distance_label, 1, 0, 1, 2)
        info_layout.addWidget(self.planet_position_label, 2, 0, 1, 2)
        info_layout.addWidget(self.planet_velocity_label, 3, 0, 1, 2)
        
        # Upcoming events and alignments
        events_group = QGroupBox("Astronomical Events")
        events_layout = QVBoxLayout(events_group)
        
        self.events_list = QLabel("No upcoming events")
        events_layout.addWidget(self.events_list)
        
        # Planetary alignments
        alignments_group = QGroupBox("Planetary Alignments")
        alignments_layout = QVBoxLayout(alignments_group)
        
        self.alignments_list = QLabel("No current alignments")
        alignments_layout.addWidget(self.alignments_list)
        
        # Add groups to main layout
        layout.addWidget(info_group)
        layout.addWidget(events_group)
        layout.addWidget(alignments_group)
        layout.addStretch()
        
        return status_panel
    
    def update_data(self):
        """Update planet data and refresh the visualization."""
        try:
            # Get planet data for the current time
            self.planet_data = self.tracker.get_planet_positions(self.current_time)
            
            # Update 2D view
            self._update_2d_view()
            
            # Update 3D view
            self._update_3d_view()
            
            # Check for events and alignments
            self._check_for_events()
            
            # Emit signal that view has been updated
            self.view_updated.emit(self.current_time)
            
        except Exception as e:
            print(f"Error updating planetary data: {e}")
    
    def _update_2d_view(self):
        """Update the 2D visualization with current planet data."""
        # Clear existing planet plots (except sun)
        for planet, plot in list(self.planet_plots.items()):
            if planet != 'sun':
                self.plot_widget.removeItem(plot)
                self.planet_plots.pop(planet)
        
        # Clear orbit plots
        for orbit in self.orbit_plots.values():
            self.plot_widget.removeItem(orbit)
        self.orbit_plots.clear()
        
        # Add planet positions
        for planet_name, data in self.planet_data.items():
            if planet_name.lower() == 'sun':
                continue  # Sun is already at the center
            
            # Get planet position (in AU)
            x = data.position.x
            y = data.position.y
            
            # Create scatter plot for planet
            color = self.planet_colors.get(planet_name.lower(), (200, 200, 200))
            size = 10  # Base size, can be adjusted based on planet
            if planet_name.lower() in ['jupiter', 'saturn']:
                size = 15
            elif planet_name.lower() in ['uranus', 'neptune']:
                size = 12
            elif planet_name.lower() in ['mercury', 'venus', 'earth', 'mars']:
                size = 8
            
            # Add planet to plot
            planet_plot = pg.ScatterPlotItem()
            planet_plot.addPoints([x], [y], size=size, brush=pg.mkBrush(*color), symbol='o')
            self.plot_widget.addItem(planet_plot)
            self.planet_plots[planet_name.lower()] = planet_plot
            
            # Add planet label if enabled
            if self.chk_show_labels.isChecked():
                label = pg.TextItem(planet_name, color=(255, 255, 255))
                label.setPos(x, y + 0.2)  # Position label above planet
                self.plot_widget.addItem(label)
                # Store label with planet for later removal
                if f"{planet_name.lower()}_label" not in self.planet_plots:
                    self.planet_plots[f"{planet_name.lower()}_label"] = label
            
            # Draw orbit if enabled
            if self.chk_show_orbits.isChecked() and hasattr(data, 'orbit'):
                self._draw_orbit_2d(planet_name, data.orbit)
        
        # Update plot range to ensure all planets are visible
        self.plot_widget.setRange(xRange=(-10, 10), yRange=(-10, 10), padding=0.1)
    
    def _draw_orbit_2d(self, planet_name, orbit_data):
        """Draw the orbit for a planet in the 2D view."""
        if not orbit_data:
            return
            
        # Get orbit parameters
        a = orbit_data.semi_major_axis
        e = orbit_data.eccentricity
        
        # Calculate points for elliptical orbit
        theta = np.linspace(0, 2*np.pi, 100)
        r = a * (1 - e**2) / (1 + e * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Create orbit plot
        color = self.planet_colors.get(planet_name.lower(), (200, 200, 200))
        orbit_plot = pg.PlotCurveItem(x, y, pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DotLine))
        self.plot_widget.addItem(orbit_plot)
        self.orbit_plots[planet_name.lower()] = orbit_plot
    
    def _update_3d_view(self):
        """Update the 3D visualization with current planet data."""
        # Clear existing planets (except sun)
        for planet, item in list(self.planet_items_3d.items()):
            if planet != 'sun':
                self.view_3d_widget.removeItem(item)
                self.planet_items_3d.pop(planet)
        
        # Clear orbit paths
        for path in self.orbit_paths_3d.values():
            self.view_3d_widget.removeItem(path)
        self.orbit_paths_3d.clear()
        
        # Add planets to 3D view
        for planet_name, data in self.planet_data.items():
            if planet_name.lower() == 'sun':
                continue  # Sun is already at the center
            
            # Get 3D position
            x = data.position.x
            y = data.position.y
            z = data.position.z if hasattr(data.position, 'z') else 0
            
            # Create planet mesh
            color = self.planet_colors.get(planet_name.lower(), (200, 200, 200))
            size = 0.1  # Base size, adjust based on planet
            if planet_name.lower() in ['jupiter', 'saturn']:
                size = 0.25
            elif planet_name.lower() in ['uranus', 'neptune']:
                size = 0.2
            
            # Create sphere for planet
            planet_mesh = gl.MeshData.sphere(rows=10, cols=20)
            planet_item = gl.GLMeshItem(
                meshdata=planet_mesh,
                smooth=True,
                color=QColor(*color),
                shader='shaded'
            )
            planet_item.scale(size, size, size)
            planet_item.translate(x, y, z)
            self.view_3d_widget.addItem(planet_item)
            self.planet_items_3d[planet_name.lower()] = planet_item
            
            # Draw orbit if enabled
            if self.chk_show_orbits.isChecked() and hasattr(data, 'orbit'):
                self._draw_orbit_3d(planet_name, data.orbit)
    
    def _draw_orbit_3d(self, planet_name, orbit_data):
        """Draw the orbit for a planet in the 3D view."""
        if not orbit_data:
            return
            
        # Get orbit parameters
        a = orbit_data.semi_major_axis
        e = orbit_data.eccentricity
        
        # Calculate points for elliptical orbit
        theta = np.linspace(0, 2*np.pi, 100)
        r = a * (1 - e**2) / (1 + e * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros_like(theta)  # Flat orbit for simplicity
        
        # Apply inclination and other orbital elements if available
        if hasattr(orbit_data, 'inclination'):
            # Apply inclination rotation (simplified)
            inclination = np.radians(orbit_data.inclination)
            z_inclined = -y * np.sin(inclination)
            y_inclined = y * np.cos(inclination)
            y = y_inclined
            z = z_inclined
        
        # Create orbit path
        color = self.planet_colors.get(planet_name.lower(), (200, 200, 200))
        orbit_pos = np.column_stack((x, y, z))
        orbit_path = gl.GLLinePlotItem(
            pos=orbit_pos,
            color=QColor(*color),
            width=1,
            mode='line_strip'
        )
        self.view_3d_widget.addItem(orbit_path)
        self.orbit_paths_3d[planet_name.lower()] = orbit_path
    
    def _check_for_events(self):
        """Check for astronomical events and alignments."""
        # Get events from tracker
        events = self.tracker.get_planetary_events(self.current_time)
        
        # Update events display
        if events:
            events_text = "<br>".join([f"• {event}" for event in events])
            self.events_list.setText(events_text)
            
            # Emit signal for new events
            for event in events:
                self.event_detected.emit("Planetary Event", event)
        else:
            self.events_list.setText("No upcoming events")
        
        # Check for alignments
        alignments = self.tracker.get_planetary_alignments(self.current_time)
        
        # Update alignments display
        if alignments:
            alignments_text = "<br>".join([f"• {alignment}" for alignment in alignments])
            self.alignments_list.setText(alignments_text)
        else:
            self.alignments_list.setText("No current alignments")
    
    def set_time_to_now(self):
        """Set the visualization time to the current time."""
        self.current_time = datetime.now()
        self.date_time_edit.setDateTime(self.current_time)
        self.update_data()
    
    def on_time_

