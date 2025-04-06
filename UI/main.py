#!/usr/bin/env python3
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QStatusBar, QMenuBar,
                            QMenu, QLabel, QFrame)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QFont


class CosmicSentinelApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set window properties
        self.setWindowTitle("Cosmic Sentinel")
        self.setMinimumSize(900, 600)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Add title label
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

        main_layout.addLayout(button_layout)

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System ready")

        # Create menu bar
        self.create_menus()

    def create_menus(self):
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

        # Tools menu
        tools_menu = self.menuBar().addMenu("&Tools")

        settings_action = QAction("&Settings", self)
        tools_menu.addAction(settings_action)

        # Help menu
        help_menu = self.menuBar().addMenu("&Help")

        about_action = QAction("&About", self)
        help_menu.addAction(about_action)

        help_action = QAction("&Help Contents", self)
        help_action.setShortcut("F1")
        help_menu.addAction(help_action)


def main():
    app = QApplication(sys.argv)
    window = CosmicSentinelApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
