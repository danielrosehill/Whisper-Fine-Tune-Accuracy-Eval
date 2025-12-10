#!/usr/bin/env python3
"""
STT Evaluation Recording GUI
Records audio samples for comparing Whisper fine-tunes against original models.
Data is saved to the HuggingFace dataset repository.
"""

import sys
import json
import wave
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QProgressBar, QComboBox,
    QMessageBox, QFrame, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont


# HuggingFace dataset path
HF_DATASET_PATH = Path.home() / "repos/hugging-face/datasets/private/Small-STT-Eval-Audio-Dataset/dataset"


class AudioSignals(QObject):
    """Signals for audio recording thread communication."""
    recording_finished = pyqtSignal()
    level_update = pyqtSignal(float)


class AudioRecorder:
    """Handles audio recording with pause/resume support."""

    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.paused = False
        self.audio_data = []
        self.stream = None
        self.signals = AudioSignals()

    def start(self):
        """Start recording."""
        self.audio_data = []
        self.recording = True
        self.paused = False

        def callback(indata, frames, time, status):
            if self.recording and not self.paused:
                self.audio_data.append(indata.copy())
                rms = np.sqrt(np.mean(indata**2))
                self.signals.level_update.emit(rms)

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            callback=callback
        )
        self.stream.start()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.signals.recording_finished.emit()

    def save(self, filepath):
        """Save recorded audio to WAV file."""
        if not self.audio_data:
            return False

        audio = np.concatenate(self.audio_data, axis=0)
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return True

    def get_duration(self):
        if not self.audio_data:
            return 0.0
        total_frames = sum(len(chunk) for chunk in self.audio_data)
        return total_frames / self.sample_rate


class RecorderWindow(QMainWindow):
    """Main recording GUI window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("STT Evaluation Recorder")
        self.setMinimumSize(800, 600)

        # Paths - data goes to HF dataset
        self.dataset_path = HF_DATASET_PATH
        self.source_truth_path = self.dataset_path / "source-truth"
        self.recordings_path = self.dataset_path / "recordings"
        self.progress_file = self.dataset_path / "progress.json"

        # Ensure directories exist
        self.recordings_path.mkdir(parents=True, exist_ok=True)

        # Load progress tracking
        self.completed = self.load_progress()

        # Load passages (all of them)
        self.all_passages = self.load_all_passages()

        # Filter to only unrecorded passages
        self.show_all = False
        self.passages = self.get_pending_passages()
        self.current_index = 0

        # Audio recorder
        self.recorder = AudioRecorder()
        self.recorder.signals.level_update.connect(self.update_level_meter)
        self.recorder.signals.recording_finished.connect(self.on_recording_finished)

        # Timer for duration display
        self.duration_timer = QTimer()
        self.duration_timer.timeout.connect(self.update_duration)

        # State
        self.is_recording = False
        self.is_paused = False

        self.setup_ui()
        self.load_current_passage()
        self.update_progress()

    def load_progress(self):
        """Load progress from JSON file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    data = json.load(f)
                    return set(data.get('completed', []))
            except:
                pass
        return set()

    def save_progress(self):
        """Save progress to JSON file."""
        data = {
            'completed': sorted(list(self.completed)),
            'last_updated': datetime.now().isoformat()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_all_passages(self):
        """Load all passage files."""
        passages = []
        if self.source_truth_path.exists():
            files = sorted(self.source_truth_path.glob("*.txt"))
            for f in files:
                passages.append({
                    'name': f.stem,
                    'path': f,
                    'text': f.read_text().strip()
                })
        return passages

    def get_pending_passages(self):
        """Get only unrecorded passages."""
        if self.show_all:
            return self.all_passages
        return [p for p in self.all_passages if p['name'] not in self.completed]

    def setup_ui(self):
        """Setup the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Progress section
        progress_frame = QFrame()
        progress_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        progress_layout = QVBoxLayout(progress_frame)

        progress_header = QHBoxLayout()
        self.progress_label = QLabel("Progress: 0 / 0")
        self.progress_label.setFont(QFont("", 12))
        progress_header.addWidget(self.progress_label)

        self.remaining_label = QLabel("Remaining: 0")
        self.remaining_label.setFont(QFont("", 12))
        progress_header.addWidget(self.remaining_label)
        progress_header.addStretch()

        self.show_all_checkbox = QCheckBox("Show completed")
        self.show_all_checkbox.stateChanged.connect(self.toggle_show_all)
        progress_header.addWidget(self.show_all_checkbox)

        progress_layout.addLayout(progress_header)

        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)

        layout.addWidget(progress_frame)

        # Passage info
        info_layout = QHBoxLayout()
        self.passage_name_label = QLabel("Passage: ")
        self.passage_name_label.setFont(QFont("", 11, QFont.Weight.Bold))
        info_layout.addWidget(self.passage_name_label)
        info_layout.addStretch()

        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("", 11))
        info_layout.addWidget(self.status_label)
        layout.addLayout(info_layout)

        # Text display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFont(QFont("", 16))
        self.text_display.setMinimumHeight(150)
        layout.addWidget(self.text_display)

        # Level meter
        meter_layout = QHBoxLayout()
        meter_layout.addWidget(QLabel("Level:"))
        self.level_meter = QProgressBar()
        self.level_meter.setMaximum(100)
        self.level_meter.setTextVisible(False)
        self.level_meter.setFixedHeight(20)
        meter_layout.addWidget(self.level_meter)

        self.duration_label = QLabel("0:00")
        self.duration_label.setFont(QFont("", 12, QFont.Weight.Bold))
        self.duration_label.setMinimumWidth(60)
        meter_layout.addWidget(self.duration_label)
        layout.addLayout(meter_layout)

        # Recording controls
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setSpacing(10)

        self.record_btn = QPushButton("⏺ Record")
        self.record_btn.setFont(QFont("", 14))
        self.record_btn.setMinimumHeight(50)
        self.record_btn.clicked.connect(self.on_record)
        self.record_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        controls_layout.addWidget(self.record_btn)

        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setFont(QFont("", 14))
        self.pause_btn.setMinimumHeight(50)
        self.pause_btn.clicked.connect(self.on_pause)
        self.pause_btn.setEnabled(False)
        controls_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setFont(QFont("", 14))
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white;")
        controls_layout.addWidget(self.stop_btn)

        self.save_btn = QPushButton("💾 Save")
        self.save_btn.setFont(QFont("", 14))
        self.save_btn.setMinimumHeight(50)
        self.save_btn.clicked.connect(self.on_save)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("background-color: #2196F3; color: white;")
        controls_layout.addWidget(self.save_btn)

        layout.addWidget(controls_frame)

        # Navigation
        nav_layout = QHBoxLayout()

        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.setFont(QFont("", 12))
        self.prev_btn.clicked.connect(self.prev_passage)
        nav_layout.addWidget(self.prev_btn)

        nav_layout.addStretch()

        self.skip_btn = QPushButton("Skip →")
        self.skip_btn.setFont(QFont("", 12))
        self.skip_btn.clicked.connect(self.skip_passage)
        nav_layout.addWidget(self.skip_btn)

        self.next_btn = QPushButton("Next →")
        self.next_btn.setFont(QFont("", 12))
        self.next_btn.clicked.connect(self.next_passage)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)

        layout.addLayout(nav_layout)

        # Microphone selector
        mic_layout = QHBoxLayout()
        mic_layout.addWidget(QLabel("Microphone:"))
        self.mic_combo = QComboBox()
        self.populate_microphones()
        mic_layout.addWidget(self.mic_combo)
        mic_layout.addStretch()
        layout.addLayout(mic_layout)

    def populate_microphones(self):
        """Populate microphone dropdown and default to Q2U if available."""
        devices = sd.query_devices()
        q2u_index = -1
        combo_index = 0
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                self.mic_combo.addItem(f"{dev['name']}", i)
                if 'q2u' in dev['name'].lower():
                    q2u_index = combo_index
                combo_index += 1
        if q2u_index >= 0:
            self.mic_combo.setCurrentIndex(q2u_index)

    def toggle_show_all(self, state):
        """Toggle showing all passages vs only pending."""
        self.show_all = bool(state)
        self.passages = self.get_pending_passages()
        self.current_index = 0
        self.load_current_passage()
        self.update_progress()

    def load_current_passage(self):
        """Load and display current passage."""
        if not self.passages:
            self.text_display.setText("🎉 All passages recorded! Check 'Show completed' to review.")
            self.passage_name_label.setText("Passage: None")
            self.record_btn.setEnabled(False)
            return

        self.record_btn.setEnabled(True)
        passage = self.passages[self.current_index]
        self.passage_name_label.setText(f"Passage: {passage['name']} ({self.current_index + 1}/{len(self.passages)})")
        self.text_display.setText(passage['text'])

        # Check if already recorded
        if passage['name'] in self.completed:
            self.status_label.setText("✓ Already recorded")
            self.status_label.setStyleSheet("color: green;")
        else:
            self.status_label.setText("Not recorded")
            self.status_label.setStyleSheet("color: orange;")

    def update_progress(self):
        """Update progress indicators."""
        total = len(self.all_passages)
        recorded = len(self.completed)
        remaining = total - recorded

        self.progress_label.setText(f"Progress: {recorded} / {total} recorded")
        self.remaining_label.setText(f"Remaining: {remaining}")
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(recorded)

    def on_record(self):
        """Start recording."""
        self.is_recording = True
        self.is_paused = False

        device_idx = self.mic_combo.currentData()
        if device_idx is not None:
            sd.default.device[0] = device_idx

        self.recorder.start()
        self.duration_timer.start(100)

        self.record_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)

        self.status_label.setText("🔴 Recording...")
        self.status_label.setStyleSheet("color: red;")

    def on_pause(self):
        """Pause/resume recording."""
        if self.is_paused:
            self.recorder.resume()
            self.is_paused = False
            self.pause_btn.setText("⏸ Pause")
            self.status_label.setText("🔴 Recording...")
            self.status_label.setStyleSheet("color: red;")
        else:
            self.recorder.pause()
            self.is_paused = True
            self.pause_btn.setText("▶ Resume")
            self.status_label.setText("⏸ Paused")
            self.status_label.setStyleSheet("color: orange;")

    def on_stop(self):
        """Stop recording."""
        self.recorder.stop()
        self.duration_timer.stop()
        self.is_recording = False
        self.is_paused = False

        self.record_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ Pause")
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.skip_btn.setEnabled(True)

        self.status_label.setText("Stopped - Ready to save")
        self.status_label.setStyleSheet("color: blue;")

    def on_recording_finished(self):
        pass

    def on_save(self):
        """Save the recording and update progress."""
        if not self.passages:
            return

        passage = self.passages[self.current_index]
        wav_path = self.recordings_path / f"{passage['name']}.wav"

        if self.recorder.save(wav_path):
            # Mark as completed
            self.completed.add(passage['name'])
            self.save_progress()

            self.status_label.setText("✓ Saved!")
            self.status_label.setStyleSheet("color: green;")
            self.save_btn.setEnabled(False)
            self.update_progress()

            # Auto-advance to next unrecorded passage
            if not self.show_all:
                self.passages = self.get_pending_passages()
                if self.passages:
                    self.current_index = 0
                    self.load_current_passage()
                    self.reset_controls()
                else:
                    self.load_current_passage()  # Show completion message
            else:
                self.next_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "Error", "No audio to save!")

    def update_level_meter(self, rms):
        level = min(100, int(rms * 1000))
        self.level_meter.setValue(level)

    def update_duration(self):
        duration = self.recorder.get_duration()
        mins = int(duration // 60)
        secs = int(duration % 60)
        self.duration_label.setText(f"{mins}:{secs:02d}")

    def prev_passage(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_passage()
            self.reset_controls()

    def next_passage(self):
        if self.current_index < len(self.passages) - 1:
            self.current_index += 1
            self.load_current_passage()
            self.reset_controls()

    def skip_passage(self):
        self.next_passage()

    def reset_controls(self):
        self.record_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.level_meter.setValue(0)
        self.duration_label.setText("0:00")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = RecorderWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
