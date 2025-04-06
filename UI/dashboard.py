from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel, QTextEdit, QPushButton 
import matplotlib.pyplot as plt
from matlpotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
import sqlite3
from services.forecast_deep import forecast_event

class Dashboard(QMainWindow):
    def__init__(self)
        super().__init__()
        self.setWindowTitle("Cosmic Sentinel - Dashboard")
        self.seyGeometry(100, 100, 900, 600)

        layout = QVBoxLayout()

        # Prediction chart
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # AI prediction
        self.prediction_label = QLabel("sentinel AI Prediction: ")

layout.addWidget(self.prediction_label)

        self.predict_button = QPushButton(" Analyze Risk Level")

self.predict_button.clicked.connect(self.run_prediction)

layout.addWidget(self.predict_button)
        central_widget = QWidget()
        central_widget.setLayout(layout)

self.setCentralWidget(central_widget)
        
        self.update_chart()
    
        def update_chart(self):
            conn = sqlite3.connect("data/cosmic_sentinel.db")
            cursor = conn.cursor()
            cursor.execute("SELECT name, impact_prob FROM celestial_objects ORDER BY discovery_date DESC LIMIT 10")
            data = cursor.fetchall()
            conn.close()

            names, risks = zip(*data)

            self.ax.clear()
            self.ax.bar(names, risks, color=["red" if x > 0.1 else "green" for x in risks])
            self.ax.set_title("Sentinel AI Risk Predictions")
            self.ax.set_ylabel("Impact Probability")
            self.ax.set_xticklabels(names, rotation=45)
            self.canvas.draw()

        def run_predictions(self):
            prediction = forecast_event()

    self.prediction_label.serText(f"Sentinel AI Prediction: {prediction}")

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    dashboard = Dashboard()
    dashboard.show()
    sys.exit(app.exec_())

# Displays AI Risk Prediction visually
# Shows colot-coded risk levels red = high, green = low
