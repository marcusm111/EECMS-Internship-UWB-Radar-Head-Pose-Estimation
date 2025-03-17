"""
Continuous Inference Module for Head Movement Classification with Visual Display

This module provides functionality for continuously monitoring a directory for new sensor data files,
processing them as they arrive, and running inference using the trained head movement classification model.
It displays results in a large, sophisticated GUI popup.

Example usage:
    model, device, classes, stats = load_model()
    start_continuous_inference(
        model=model,
        device=device, 
        class_names=classes, 
        norm_stats=stats,
        input_dir="sensor_data",
        archive_dir="processed_data",
        results_file="inference_results.csv"
    )
"""

import os
import time
import csv
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
import sys
import threading

import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
                             QWidget, QProgressBar, QFrame, QSizePolicy, QDesktopWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QFont, QPalette, QColor, QLinearGradient, QGradient, QPainter, QPen, QBrush

# Import from your existing modules
from inference import load_model, load_spectrogram, run_inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("continuous_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class InferenceSignals(QObject):
    """Signal class for thread-safe communication with GUI"""
    inference_result = pyqtSignal(str, float, dict)
    processing_started = pyqtSignal()
    processing_completed = pyqtSignal()

class HeadMovementDisplay(QMainWindow):
    """
    GUI window to display head movement detection results in real-time.
    """
    
    def __init__(self, class_names):
        super().__init__()
        self.class_names = class_names
        self.initUI()
        
    def initUI(self):
        """Initialize the UI components"""
        # Configure window
        self.setWindowTitle("Head Movement Detection")
        self.setStyleSheet("background-color: #2E3440;")  # Dark background
        
        # Set default window size instead of percentage of screen
        default_width, default_height = 1000, 800
        self.resize(default_width, default_height)
        self.setMinimumSize(800, 600)  # Smaller minimum size to allow more flexibility
        
        # Center the window
        self.center()
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Add title
        title_label = QLabel("Head Movement Detection")
        title_label.setFont(QFont("Arial", 22, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #ECEFF4; margin-bottom: 10px;")
        title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        title_label.setMinimumHeight(50)  # Increased minimum height
        main_layout.addWidget(title_label)
        
        # Add status indicator
        self.status_label = QLabel("Waiting for movement data...")
        self.status_label.setFont(QFont("Arial", 14))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #D8DEE9; margin-bottom: 15px;")
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.status_label.setMinimumHeight(40)  # Increased minimum height
        self.status_label.setWordWrap(True)  # Enable word wrapping
        main_layout.addWidget(self.status_label)
        
        # Create detection result frame
        detection_frame = QFrame()
        detection_frame.setFrameShape(QFrame.StyledPanel)
        detection_frame.setStyleSheet("""
            QFrame {
                background-color: #3B4252;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        detection_layout = QVBoxLayout(detection_frame)
        detection_layout.setContentsMargins(15, 15, 15, 15)
        
        # Movement type display
        self.movement_label = QLabel("No Movement Detected")
        self.movement_label.setFont(QFont("Arial", 28, QFont.Bold))  # Reduced default font size
        self.movement_label.setAlignment(Qt.AlignCenter)
        self.movement_label.setStyleSheet("color: #88C0D0; margin: 10px;")
        self.movement_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.movement_label.setMinimumHeight(100)  # Significantly increased minimum height
        self.movement_label.setWordWrap(True)  # Enable word wrapping
        detection_layout.addWidget(self.movement_label)
        
        # Confidence display
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setFont(QFont("Arial", 18))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("color: #A3BE8C; margin-bottom: 10px;")
        self.confidence_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.confidence_label.setMinimumHeight(50)  # Increased minimum height
        detection_layout.addWidget(self.confidence_label)
        
        # Add detection frame to main layout
        main_layout.addWidget(detection_frame, 2)  # Give it more space with stretch factor
        
        # Create probability bars container
        prob_container = QFrame()
        prob_container.setFrameShape(QFrame.StyledPanel)
        prob_container.setStyleSheet("""
            QFrame {
                background-color: #3B4252;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        prob_layout = QVBoxLayout(prob_container)
        prob_layout.setSpacing(10)
        prob_layout.setContentsMargins(15, 15, 15, 15)
        
        # Probability title
        prob_title = QLabel("Probability Distribution")
        prob_title.setFont(QFont("Arial", 16, QFont.Bold))
        prob_title.setAlignment(Qt.AlignCenter)
        prob_title.setStyleSheet("color: #ECEFF4; margin-bottom: 10px;")
        prob_title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        prob_title.setMinimumHeight(40)  # Increased minimum height
        prob_layout.addWidget(prob_title)
        
        # Container widget for probability bars
        bars_container = QWidget()
        bars_layout = QVBoxLayout(bars_container)
        bars_layout.setSpacing(8)  # Reduced spacing slightly for more compact layout
        
        # Create progress bars for each class
        self.prob_bars = {}
        for class_name in self.class_names:
            # Create frame for each bar
            bar_frame = QFrame()
            bar_frame.setFrameShape(QFrame.NoFrame)
            bar_frame.setMinimumHeight(40)  # Increased minimum height for better visibility
            bar_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            
            class_layout = QHBoxLayout(bar_frame)
            class_layout.setContentsMargins(5, 0, 5, 0)
            class_layout.setSpacing(10)
            
            # Simple fixed-width label for class name - wide enough for any class name
            name_label = QLabel(class_name)
            name_label.setFixedWidth(150)  # Fixed width that should fit all class names
            name_label.setFont(QFont("Arial", 12))
            name_label.setStyleSheet("color: #D8DEE9;")
            name_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            class_layout.addWidget(name_label)
            class_layout.addWidget(name_label)
            
            # Progress bar for probability
            bar = QProgressBar()
            bar.setMinimum(0)
            bar.setMaximum(100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setMinimumHeight(25)  # Slightly reduced height
            bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #434C5E;
                    border-radius: 5px;
                    background-color: #2E3440;
                }
                QProgressBar::chunk {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5E81AC, stop:1 #88C0D0);
                    border-radius: 3px;
                }
            """)
            class_layout.addWidget(bar)
            
            # Value label
            value_label = QLabel("0.00%")
            value_label.setMinimumWidth(80)
            value_label.setFixedWidth(80)  # Slightly reduced width
            value_label.setFont(QFont("Arial", 12))
            value_label.setStyleSheet("color: #D8DEE9;")
            value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            value_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            class_layout.addWidget(value_label)
            
            # Add to layout and store references
            bars_layout.addWidget(bar_frame)
            self.prob_bars[class_name] = (bar, value_label)
        
        # Add the bars container to the probability layout
        prob_layout.addWidget(bars_container)
        
        # Add probability container to main layout
        main_layout.addWidget(prob_container, 3)  # Give probability section more space
        
        # Processing indicator
        processing_layout = QHBoxLayout()
        self.processing_label = QLabel("Processing Status: Idle")
        self.processing_label.setFont(QFont("Arial", 12))
        self.processing_label.setStyleSheet("color: #ECEFF4;")
        self.processing_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.processing_label.setMinimumHeight(30)  # Increased minimum height
        processing_layout.addWidget(self.processing_label)
        main_layout.addLayout(processing_layout)
        
        # Create signals object
        self.signals = InferenceSignals()
        self.signals.inference_result.connect(self.update_display)
        self.signals.processing_started.connect(self.on_processing_started)
        self.signals.processing_completed.connect(self.on_processing_completed)
        
    def center(self):
        """Center the window on the screen"""
        frame_geometry = self.frameGeometry()
        screen_center = QDesktopWidget().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())
    
    @pyqtSlot(str, float, dict)
    def update_display(self, prediction: str, confidence: float, probabilities: Dict[str, float]):
        """Update the display with new inference results"""
        # Update movement label with appropriate font size for visibility
        font_size = 28  # Default font size
        if len(prediction) > 12:
            font_size = 24
        if len(prediction) > 18:
            font_size = 20
            
        self.movement_label.setFont(QFont("Arial", font_size, QFont.Bold))
        self.movement_label.setText(prediction)
        
        # Update confidence
        conf_percentage = confidence * 100
        conf_color = "#A3BE8C" if conf_percentage > 70 else "#EBCB8B" if conf_percentage > 40 else "#BF616A"
        self.confidence_label.setText(f"Confidence: {conf_percentage:.2f}%")
        self.confidence_label.setStyleSheet(f"color: {conf_color}; margin-bottom: 10px;")
        
        # Update all probability bars
        for class_name, (bar, value_label) in self.prob_bars.items():
            prob_value = probabilities.get(class_name, 0) * 100
            bar.setValue(int(prob_value))
            
            # Format percentage with fixed precision
            value_text = f"{prob_value:.2f}%"
            value_label.setText(value_text)
            
            # Highlight the detected class
            if class_name == prediction:
                bar.setStyleSheet("""
                    QProgressBar {
                        border: 2px solid #434C5E;
                        border-radius: 5px;
                        background-color: #2E3440;
                    }
                    QProgressBar::chunk {
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #A3BE8C, stop:1 #8FBCBB);
                        border-radius: 3px;
                    }
                """)
                value_label.setStyleSheet("color: #A3BE8C; font-weight: bold;")
            else:
                bar.setStyleSheet("""
                    QProgressBar {
                        border: 2px solid #434C5E;
                        border-radius: 5px;
                        background-color: #2E3440;
                    }
                    QProgressBar::chunk {
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5E81AC, stop:1 #88C0D0);
                        border-radius: 3px;
                    }
                """)
                value_label.setStyleSheet("color: #D8DEE9;")
        
        # Update status - add timestamp for better feedback
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(f"Movement Detected: {prediction} [{timestamp}]")
    
    @pyqtSlot()
    def on_processing_started(self):
        """Called when processing of data begins"""
        self.processing_label.setText("Processing Status: Active")
        self.processing_label.setStyleSheet("color: #A3BE8C;")
    
    @pyqtSlot()
    def on_processing_completed(self):
        """Called when processing of data completes"""
        self.processing_label.setText("Processing Status: Idle")
        self.processing_label.setStyleSheet("color: #ECEFF4;")
    
    # Override the resize event to ensure font sizes are updated
    def resizeEvent(self, event):
        """Handle window resize events to adjust font sizes if needed"""
        # Let the parent class handle the basic resize behavior
        super().resizeEvent(event)
        
        # Get the current window size
        width = self.width()
        height = self.height()
        
        # Adjust font sizes based on window dimensions
        if width < 900:  # Small window
            self.movement_label.setFont(QFont("Arial", 20, QFont.Bold))
            self.confidence_label.setFont(QFont("Arial", 14))
        else:  # Larger window - use default sizes with text length adjustment
            # Adjust movement label based on content length
            text = self.movement_label.text()
            if len(text) > 12:
                self.movement_label.setFont(QFont("Arial", 24, QFont.Bold))
            elif len(text) > 18:
                self.movement_label.setFont(QFont("Arial", 20, QFont.Bold))
            else:
                self.movement_label.setFont(QFont("Arial", 28, QFont.Bold))
            
            self.confidence_label.setFont(QFont("Arial", 18))

def setup_directories(input_dir: str, archive_dir: str) -> Tuple[str, str]:
    """
    Set up the necessary directories for file monitoring.
    
    Args:
        input_dir: Directory to monitor for new sensor files
        archive_dir: Directory to move processed files to
        
    Returns:
        input_dir: Absolute path to input directory
        archive_dir: Absolute path to archive directory
    """
    # Create absolute paths
    input_dir = os.path.abspath(input_dir)
    archive_dir = os.path.abspath(archive_dir)
    
    # Create directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    
    logger.info(f"Monitoring directory: {input_dir}")
    logger.info(f"Archive directory: {archive_dir}")
    
    return input_dir, archive_dir


def setup_results_file(results_file: str) -> str:
    """
    Set up the CSV file for storing inference results.
    
    Args:
        results_file: Path to the results file
        
    Returns:
        results_file: Absolute path to results file
    """
    results_file = os.path.abspath(results_file)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Initialize the results file with headers if it doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 
                'Front File', 
                'Left File', 
                'Right File', 
                'Prediction', 
                'Confidence',
                'Probabilities'
            ])
    
    logger.info(f"Results will be saved to: {results_file}")
    return results_file


def find_matching_sensor_files(
    front_file: str, 
    input_dir: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find matching left and right sensor files for a given front sensor file.
    
    Args:
        front_file: Front sensor file name (contains 'sF')
        input_dir: Directory containing sensor files
        
    Returns:
        left_file: Matching left sensor file path or None
        right_file: Matching right sensor file path or None
    """
    # Create the expected filenames for left and right sensors
    base_name = front_file.replace('sF', 's')
    left_file = os.path.join(input_dir, base_name.replace('s', 'sL'))
    right_file = os.path.join(input_dir, base_name.replace('s', 'sR'))
    
    # Check if both files exist
    if os.path.exists(left_file) and os.path.exists(right_file):
        return left_file, right_file
    
    return None, None


def archive_files(
    front_file: str, 
    left_file: str, 
    right_file: str, 
    archive_dir: str
) -> bool:
    """
    Move processed sensor files to the archive directory.
    
    Args:
        front_file: Path to front sensor file
        left_file: Path to left sensor file
        right_file: Path to right sensor file
        archive_dir: Directory to move processed files to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create a subdirectory with timestamp to group related files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(front_file).replace('sF', '')
        group_dir = os.path.join(archive_dir, f"{timestamp}_{base_name.replace('.txt', '')}")
        os.makedirs(group_dir, exist_ok=True)
        
        # Move files to archive
        shutil.move(front_file, os.path.join(group_dir, os.path.basename(front_file)))
        shutil.move(left_file, os.path.join(group_dir, os.path.basename(left_file)))
        shutil.move(right_file, os.path.join(group_dir, os.path.basename(right_file)))
        
        logger.debug(f"Archived files to {group_dir}")
        return True
    except Exception as e:
        logger.error(f"Error archiving files: {e}")
        return False


def save_results(
    results_file: str,
    front_file: str,
    left_file: str,
    right_file: str,
    prediction: str,
    confidence: float,
    probabilities: Dict[str, float]
) -> None:
    """
    Save inference results to the CSV file.
    
    Args:
        results_file: Path to the results CSV file
        front_file: Path to front sensor file
        left_file: Path to left sensor file
        right_file: Path to right sensor file
        prediction: Predicted class
        confidence: Confidence score
        probabilities: Dictionary of class probabilities
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format probabilities as a string
        probs_str = ";".join([f"{k}:{v:.4f}" for k, v in probabilities.items()])
        
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                os.path.basename(front_file),
                os.path.basename(left_file),
                os.path.basename(right_file),
                prediction,
                f"{confidence:.4f}",
                probs_str
            ])
        
        logger.debug(f"Results saved to {results_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def process_file_set(
    front_file: str,
    left_file: str,
    right_file: str,
    model: torch.nn.Module,
    device: torch.device,
    class_names: List[str],
    norm_stats: Dict[str, float],
    results_file: str,
    archive_dir: str,
    signals: Optional[InferenceSignals] = None
) -> bool:
    """
    Process a complete set of sensor files (front, left, right).
    
    Args:
        front_file: Path to front sensor file
        left_file: Path to left sensor file
        right_file: Path to right sensor file
        model: Trained PyTorch model
        device: PyTorch device
        class_names: List of class names
        norm_stats: Dictionary with mean and std values for normalization
        results_file: Path to the results CSV file
        archive_dir: Directory to move processed files to
        signals: Optional signals object for GUI updates
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Signal processing started if GUI is active
        if signals:
            signals.processing_started.emit()
        
        # Load the spectrogram
        tensor = load_spectrogram(front_file)
        
        # Run inference
        prediction, confidence, probabilities = run_inference(
            model, tensor, device, class_names,
            mean=norm_stats['mean'], std=norm_stats['std']
        )
        
        # Log the results
        logger.info(f"Prediction: {prediction} with {confidence:.2%} confidence")
        logger.info(f"Top probabilities: {sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        # Save results to CSV
        save_results(
            results_file,
            front_file,
            left_file,
            right_file,
            prediction,
            confidence,
            probabilities
        )
        
        # Update GUI if available
        if signals:
            signals.inference_result.emit(prediction, confidence, probabilities)
        
        # Archive the files
        archive_files(front_file, left_file, right_file, archive_dir)
        
        # Signal processing completed if GUI is active
        if signals:
            signals.processing_completed.emit()
            
        return True
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        if signals:
            signals.processing_completed.emit()
        return False


def continuous_inference_loop(
    model: torch.nn.Module,
    device: torch.device,
    class_names: List[str],
    norm_stats: Dict[str, float],
    input_dir: str,
    archive_dir: str,
    results_file: str,
    poll_interval: float = 0.5,
    signals: Optional[InferenceSignals] = None
) -> None:
    """
    Main loop for continuous inference.
    
    Args:
        model: Trained PyTorch model
        device: PyTorch device
        class_names: List of class names
        norm_stats: Dictionary with mean and std values for normalization
        input_dir: Directory to monitor for new sensor files
        archive_dir: Directory to move processed files to
        results_file: Path to the results CSV file
        poll_interval: Time in seconds between directory checks
        signals: Optional signals object for GUI updates
    """
    processed_files = set()  # Track processed front files
    
    logger.info("Starting continuous inference loop...")
    logger.info(f"Monitoring directory: {input_dir}")
    logger.info(f"Poll interval: {poll_interval}s")
    
    try:
        while True:
            # Find all front sensor files in the directory
            front_files = [
                os.path.join(input_dir, f) for f in os.listdir(input_dir)
                if f.endswith(".txt") and "sF" in f and os.path.join(input_dir, f) not in processed_files
            ]
            
            if front_files:
                logger.debug(f"Found {len(front_files)} new front sensor files")
            
            # Process each front file if matching left and right files exist
            for front_file in front_files:
                left_file, right_file = find_matching_sensor_files(
                    os.path.basename(front_file), 
                    input_dir
                )
                
                if left_file and right_file:
                    logger.info(f"Processing complete set for {os.path.basename(front_file)}")
                    
                    # Process the file set
                    success = process_file_set(
                        front_file,
                        left_file,
                        right_file,
                        model,
                        device,
                        class_names,
                        norm_stats,
                        results_file,
                        archive_dir,
                        signals
                    )
                    
                    if success:
                        # Mark as processed
                        processed_files.add(front_file)
                        
                        # Keep the processed set at a reasonable size
                        if len(processed_files) > 1000:
                            # Remove oldest entries (assuming chronological processing)
                            processed_files = set(list(processed_files)[-500:])
            
            # Sleep to prevent high CPU usage
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        logger.info("Continuous inference stopped by user")
    except Exception as e:
        logger.error(f"Continuous inference loop error: {e}")
        raise


def start_continuous_inference(
    model: torch.nn.Module = None,
    device: torch.device = None,
    class_names: List[str] = None,
    norm_stats: Dict[str, float] = None,
    input_dir: str = "sensor_data",
    archive_dir: str = "processed_data",
    results_file: str = "inference_results.csv",
    model_path: str = "final_head_movement_model.pth",
    config_path: str = "final_config.yaml",
    poll_interval: float = 0.5
) -> None:
    """
    Start the continuous inference system with GUI display.
    
    Args:
        model: Trained PyTorch model (optional, will be loaded if None)
        device: PyTorch device (optional, will be set if None)
        class_names: List of class names (optional, will be loaded if None)
        norm_stats: Dictionary with mean and std values (optional, will be loaded if None)
        input_dir: Directory to monitor for new sensor files
        archive_dir: Directory to move processed files to
        results_file: Path to the results CSV file
        model_path: Path to the saved model weights file (used if model is None)
        config_path: Path to the configuration file (used if model is None)
        poll_interval: Time in seconds between directory checks
    """
    # Set up directories
    input_dir, archive_dir = setup_directories(input_dir, archive_dir)
    
    # Set up results file
    results_file = setup_results_file(results_file)
    
    # Load model if not provided
    if model is None or device is None or class_names is None or norm_stats is None:
        logger.info(f"Loading model from {model_path}")
        model, device, class_names, norm_stats = load_model(model_path, config_path)
    
    # Initialize QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create display window
    display = HeadMovementDisplay(class_names)
    display.show()
    
    # Run inference loop in a separate thread
    inference_thread = threading.Thread(
        target=continuous_inference_loop,
        args=(
            model, device, class_names, norm_stats,
            input_dir, archive_dir, results_file, poll_interval,
            display.signals
        ),
        daemon=True
    )
    inference_thread.start()
    
    # Start the application event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    # Example usage when run as a script
    start_continuous_inference(
        input_dir="sensor_data",
        archive_dir="processed_data",
        results_file="results/inference_results.csv",
        poll_interval=0.5
    )