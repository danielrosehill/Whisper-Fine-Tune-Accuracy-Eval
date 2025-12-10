#!/usr/bin/env python3
"""
STT Fine-Tune Evaluation GUI

PyQt6 GUI for running and monitoring STT fine-tune evaluation.
Features:
- Visual progress monitoring
- Incremental result saving (resume on crash)
- Real-time WER display
- Final analysis comparison
"""

import subprocess
import json
import csv
import time
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import tempfile
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QTextEdit, QTableWidget,
    QTableWidgetItem, QGroupBox, QCheckBox, QScrollArea, QMessageBox,
    QHeaderView, QSplitter, QFrame, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor

import werpy
from whisper_normalizer.english import EnglishTextNormalizer

# Paths
SCRIPT_DIR = Path(__file__).parent
MODELS_CONFIG = SCRIPT_DIR / "models.json"
DATASET_PATH = Path.home() / "repos/hugging-face/datasets/private/Small-STT-Eval-Audio-Dataset/data"
METADATA_CSV = DATASET_PATH / "metadata.csv"
OUTPUT_PATH = SCRIPT_DIR / "results"
INDIVIDUAL_RESULTS_PATH = OUTPUT_PATH / "individual"
PROGRESS_FILE = OUTPUT_PATH / "evaluation_progress.json"


@dataclass
class TranscriptionResult:
    """Single transcription result."""
    audio_file: str
    model: str
    reference: str
    hypothesis: str
    reference_normalized: str
    hypothesis_normalized: str
    wer: float
    duration_seconds: float


@dataclass
class ModelSummary:
    """Summary statistics for a model."""
    model: str
    total_files: int
    average_wer: float
    median_wer: float
    min_wer: float
    max_wer: float
    total_time_seconds: float
    avg_time_per_file: float


def load_models_from_config() -> dict[str, Path]:
    """Load model paths from models.json config file."""
    with open(MODELS_CONFIG) as f:
        config = json.load(f)

    models = {}

    # Load fine-tuned GGML models
    ft_ggml = config["fine_tuned"]["ggml"]
    ft_base = Path(ft_ggml["base_path"])
    for size, filename in ft_ggml["models"].items():
        models[f"finetune-{size}"] = ft_base / filename

    # Load original GGML models
    orig_ggml = config["original"]["ggml"]
    orig_base = Path(orig_ggml["base_path"])
    for size, filename in orig_ggml["models"].items():
        models[f"original-{size}"] = orig_base / filename

    return models


def get_available_audio_files():
    """Load audio files and transcriptions from metadata.csv."""
    audio_files = []

    with open(METADATA_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = DATASET_PATH / row['file_name']
            if audio_path.exists():
                audio_files.append({
                    "audio": audio_path,
                    "transcription": row['transcription'],
                    "name": audio_path.stem,
                    "category": row.get('category', '')
                })

    return audio_files


def transcribe_with_whisper_cli(audio_path: Path, model_path: Path) -> tuple[str, float]:
    """Transcribe audio using whisper-cli (whisper.cpp)."""
    start_time = time.time()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        output_base = f.name[:-4]

    cmd = [
        "whisper-cli-vulkan",
        "-m", str(model_path),
        "-f", str(audio_path),
        "-l", "en",
        "-np",
        "-nt",
        "-of", output_base,
        "-otxt",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        duration = time.time() - start_time

        output_file = Path(f"{output_base}.txt")
        if output_file.exists():
            transcription = output_file.read_text().strip()
            output_file.unlink()
            return transcription, duration
        else:
            return result.stdout.strip(), duration

    except subprocess.TimeoutExpired:
        return "[TIMEOUT]", time.time() - start_time
    except Exception as e:
        return f"[ERROR: {e}]", time.time() - start_time


def get_individual_result_path(model_name: str, audio_file: str) -> Path:
    """Get path for individual result JSON file."""
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    return INDIVIDUAL_RESULTS_PATH / safe_model / f"{audio_file}.json"


def save_individual_result(result: TranscriptionResult):
    """Save a single transcription result to its own JSON file."""
    path = get_individual_result_path(result.model, result.audio_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(asdict(result), f, indent=2)


def load_completed_results() -> dict[str, list[dict]]:
    """Load all previously completed individual results."""
    results = {}
    if not INDIVIDUAL_RESULTS_PATH.exists():
        return results

    for model_dir in INDIVIDUAL_RESULTS_PATH.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            model_results = []
            for json_file in sorted(model_dir.glob("*.json")):
                try:
                    with open(json_file) as f:
                        model_results.append(json.load(f))
                except Exception:
                    pass  # Skip corrupted files
            if model_results:
                results[model_name] = model_results
    return results


def is_result_completed(model_name: str, audio_file: str) -> bool:
    """Check if a specific result already exists."""
    return get_individual_result_path(model_name, audio_file).exists()


def calculate_summary(model_name: str, results: list[TranscriptionResult]) -> ModelSummary:
    """Calculate summary statistics for a model's results."""
    if not results:
        return ModelSummary(
            model=model_name,
            total_files=0,
            average_wer=0,
            median_wer=0,
            min_wer=0,
            max_wer=0,
            total_time_seconds=0,
            avg_time_per_file=0
        )

    wers = [r.wer for r in results]
    times = [r.duration_seconds for r in results]
    sorted_wers = sorted(wers)

    return ModelSummary(
        model=model_name,
        total_files=len(results),
        average_wer=sum(wers) / len(wers),
        median_wer=sorted_wers[len(sorted_wers) // 2],
        min_wer=min(wers),
        max_wer=max(wers),
        total_time_seconds=sum(times),
        avg_time_per_file=sum(times) / len(times)
    )


class EvaluationWorker(QThread):
    """Background worker for running evaluations."""
    progress = pyqtSignal(str, int, int, str)  # model, current, total, message
    result_ready = pyqtSignal(str, dict)  # model, result dict
    model_complete = pyqtSignal(str, dict)  # model, summary dict
    all_complete = pyqtSignal(dict)  # all summaries
    log_message = pyqtSignal(str)  # log text
    error = pyqtSignal(str)  # error message

    def __init__(self, models: dict[str, Path], audio_files: list,
                 completed_results: Optional[dict] = None):
        super().__init__()
        self.models = models
        self.audio_files = audio_files
        self.completed_results = completed_results or {}
        self.normalizer = EnglishTextNormalizer()
        self.running = True
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def stop(self):
        self.running = False

    def run(self):
        all_results = dict(self.completed_results)
        summaries = {}

        # Calculate summaries for already-completed models
        for model_name, results in all_results.items():
            result_objects = [TranscriptionResult(**r) for r in results]
            summaries[model_name] = asdict(calculate_summary(model_name, result_objects))

        total_models = len(self.models)
        models_done = len([m for m in self.models if m in all_results])

        for model_idx, (model_name, model_path) in enumerate(self.models.items()):
            if not self.running:
                break

            # Skip already completed models
            if model_name in all_results:
                self.log_message.emit(f"Skipping {model_name} (already completed)")
                continue

            if not model_path.exists():
                self.log_message.emit(f"WARNING: Model not found: {model_name}")
                continue

            self.log_message.emit(f"\n{'='*50}")
            self.log_message.emit(f"Evaluating: {model_name}")
            self.log_message.emit(f"Model path: {model_path}")
            self.log_message.emit(f"{'='*50}")

            model_results = []
            total_files = len(self.audio_files)

            for i, file_info in enumerate(self.audio_files):
                if not self.running:
                    break

                audio_name = file_info['name']

                # Skip if already completed (individual file exists)
                if is_result_completed(model_name, audio_name):
                    self.log_message.emit(
                        f"  [{i+1}/{total_files}] {audio_name}: SKIPPED (already done)"
                    )
                    # Load existing result
                    existing_path = get_individual_result_path(model_name, audio_name)
                    with open(existing_path) as f:
                        existing = json.load(f)
                    model_results.append(TranscriptionResult(**existing))
                    self.progress.emit(model_name, i + 1, total_files,
                                       f"Skipped {audio_name} (cached)")
                    continue

                self.progress.emit(model_name, i + 1, total_files,
                                   f"Processing {audio_name}")

                reference = file_info['transcription']
                hypothesis, duration = transcribe_with_whisper_cli(
                    file_info['audio'], model_path
                )

                ref_normalized = self.normalizer(reference)
                hyp_normalized = self.normalizer(hypothesis)

                wer = werpy.wer([ref_normalized], [hyp_normalized])

                result = TranscriptionResult(
                    audio_file=audio_name,
                    model=model_name,
                    reference=reference,
                    hypothesis=hypothesis,
                    reference_normalized=ref_normalized,
                    hypothesis_normalized=hyp_normalized,
                    wer=wer,
                    duration_seconds=duration
                )
                model_results.append(result)

                # Save individual result immediately
                save_individual_result(result)

                # Emit individual result
                self.result_ready.emit(model_name, asdict(result))

                self.log_message.emit(
                    f"  [{i+1}/{total_files}] {audio_name}: "
                    f"WER={wer:.2%} ({duration:.1f}s)"
                )

                # Also update aggregate progress file
                all_results[model_name] = [asdict(r) for r in model_results]
                self.save_progress(all_results, summaries)

            if model_results:
                all_results[model_name] = [asdict(r) for r in model_results]
                summary = calculate_summary(model_name, model_results)
                summaries[model_name] = asdict(summary)

                # Save progress incrementally
                self.save_progress(all_results, summaries)
                self.model_complete.emit(model_name, asdict(summary))

                self.log_message.emit(f"\n{model_name} complete:")
                self.log_message.emit(f"  Average WER: {summary.average_wer:.2%}")
                self.log_message.emit(f"  Total time: {summary.total_time_seconds:.1f}s")

        if self.running:
            self.save_final_results(all_results, summaries)
            self.all_complete.emit(summaries)

    def save_progress(self, all_results: dict, summaries: dict):
        """Save progress for resume capability."""
        OUTPUT_PATH.mkdir(exist_ok=True)
        progress_data = {
            "timestamp": self.run_timestamp,
            "last_updated": datetime.now().isoformat(),
            "completed_models": list(all_results.keys()),
            "summaries": summaries,
            "results": all_results
        }
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f, indent=2)

    def save_final_results(self, all_results: dict, summaries: dict):
        """Save final results to timestamped files."""
        OUTPUT_PATH.mkdir(exist_ok=True)
        timestamp = self.run_timestamp

        # Detailed results JSON
        json_path = OUTPUT_PATH / f"detailed_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "summaries": summaries,
                "results": all_results
            }, f, indent=2)

        # Summary CSV
        csv_path = OUTPUT_PATH / f"summary_{timestamp}.csv"
        if summaries:
            first_summary = list(summaries.values())[0]
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(first_summary.keys()))
                writer.writeheader()
                for s in summaries.values():
                    writer.writerow(s)

        self.log_message.emit(f"\nResults saved to: {OUTPUT_PATH}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STT Fine-Tune Evaluation")
        self.setMinimumSize(1200, 800)

        self.worker: Optional[EvaluationWorker] = None
        self.all_results: dict = {}
        self.summaries: dict = {}

        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Header
        header = QLabel("STT Fine-Tune Evaluation")
        header.setFont(QFont("", 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Models and Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Models group
        models_group = QGroupBox("Available Models")
        models_layout = QVBoxLayout(models_group)
        self.model_checkboxes: dict[str, QCheckBox] = {}

        # We'll populate this after loading
        self.models_container = QWidget()
        self.models_container_layout = QVBoxLayout(self.models_container)
        models_layout.addWidget(self.models_container)

        # Select all/none buttons
        btn_row = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_models)
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self.select_no_models)
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(select_none_btn)
        models_layout.addLayout(btn_row)

        left_layout.addWidget(models_group)

        # Info group
        info_group = QGroupBox("Dataset Info")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("Loading...")
        info_layout.addWidget(self.info_label)
        left_layout.addWidget(info_group)

        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)

        self.start_btn = QPushButton("Start Evaluation")
        self.start_btn.clicked.connect(self.start_evaluation)
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_evaluation)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")

        self.resume_btn = QPushButton("Resume Previous Run")
        self.resume_btn.clicked.connect(self.resume_evaluation)
        self.resume_btn.setEnabled(False)

        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.resume_btn)

        left_layout.addWidget(controls_group)
        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # Right panel - Tabs
        right_panel = QTabWidget()

        # Progress tab
        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)

        # Overall progress
        overall_group = QGroupBox("Overall Progress")
        overall_layout = QVBoxLayout(overall_group)
        self.overall_label = QLabel("Ready")
        self.overall_progress = QProgressBar()
        overall_layout.addWidget(self.overall_label)
        overall_layout.addWidget(self.overall_progress)
        progress_layout.addWidget(overall_group)

        # Current model progress
        current_group = QGroupBox("Current Model")
        current_layout = QVBoxLayout(current_group)
        self.current_model_label = QLabel("Not started")
        self.current_progress = QProgressBar()
        self.current_file_label = QLabel("")
        current_layout.addWidget(self.current_model_label)
        current_layout.addWidget(self.current_progress)
        current_layout.addWidget(self.current_file_label)
        progress_layout.addWidget(current_group)

        # Log output
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("monospace", 9))
        log_layout.addWidget(self.log_text)
        progress_layout.addWidget(log_group)

        right_panel.addTab(progress_widget, "Progress")

        # Results tab
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Model", "Avg WER", "Median WER", "Min WER", "Max WER"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        results_layout.addWidget(self.results_table)

        right_panel.addTab(results_widget, "Results Summary")

        # Comparison tab
        comparison_widget = QWidget()
        comparison_layout = QVBoxLayout(comparison_widget)

        self.comparison_text = QTextEdit()
        self.comparison_text.setReadOnly(True)
        self.comparison_text.setFont(QFont("monospace", 10))
        comparison_layout.addWidget(self.comparison_text)

        right_panel.addTab(comparison_widget, "Fine-tune vs Original")

        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])

    def load_data(self):
        """Load models and audio files."""
        try:
            self.models = load_models_from_config()
            self.audio_files = get_available_audio_files()

            # Populate model checkboxes
            for name, path in sorted(self.models.items()):
                cb = QCheckBox(name)
                if path.exists():
                    cb.setChecked(True)
                    cb.setToolTip(str(path))
                else:
                    cb.setEnabled(False)
                    cb.setText(f"{name} (not found)")
                self.model_checkboxes[name] = cb
                self.models_container_layout.addWidget(cb)

            # Update info
            available = sum(1 for p in self.models.values() if p.exists())
            self.info_label.setText(
                f"Audio files: {len(self.audio_files)}\n"
                f"Models available: {available}/{len(self.models)}\n"
                f"Dataset: {DATASET_PATH.name}"
            )

            # Check for resume data
            if PROGRESS_FILE.exists():
                self.resume_btn.setEnabled(True)
                with open(PROGRESS_FILE) as f:
                    progress = json.load(f)
                completed = len(progress.get("completed_models", []))
                self.resume_btn.setText(
                    f"Resume Previous Run ({completed} models done)"
                )

            self.log("Data loaded successfully")
            self.log(f"Found {len(self.audio_files)} audio files")
            self.log(f"Found {available} available models")

        except Exception as e:
            self.log(f"ERROR loading data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load data: {e}")

    def log(self, message: str):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def select_all_models(self):
        for cb in self.model_checkboxes.values():
            if cb.isEnabled():
                cb.setChecked(True)

    def select_no_models(self):
        for cb in self.model_checkboxes.values():
            cb.setChecked(False)

    def get_selected_models(self) -> dict[str, Path]:
        """Get dict of selected models."""
        return {
            name: self.models[name]
            for name, cb in self.model_checkboxes.items()
            if cb.isChecked() and self.models[name].exists()
        }

    def start_evaluation(self, completed_results: Optional[dict] = None):
        """Start the evaluation."""
        selected = self.get_selected_models()
        if not selected:
            QMessageBox.warning(self, "No Models", "Please select at least one model")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)

        self.all_results = completed_results or {}
        self.summaries = {}
        self.results_table.setRowCount(0)

        # Calculate overall progress
        total_ops = len(selected) * len(self.audio_files)
        completed_ops = sum(
            len(self.all_results.get(m, []))
            for m in selected
        )
        self.overall_progress.setMaximum(total_ops)
        self.overall_progress.setValue(completed_ops)

        self.worker = EvaluationWorker(selected, self.audio_files, completed_results)
        self.worker.progress.connect(self.on_progress)
        self.worker.result_ready.connect(self.on_result)
        self.worker.model_complete.connect(self.on_model_complete)
        self.worker.all_complete.connect(self.on_all_complete)
        self.worker.log_message.connect(self.log)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_worker_finished)

        self.log("\n" + "="*50)
        self.log("Starting evaluation...")
        self.log(f"Selected models: {', '.join(selected.keys())}")
        self.log("="*50)

        self.worker.start()

    def resume_evaluation(self):
        """Resume from saved progress."""
        try:
            with open(PROGRESS_FILE) as f:
                progress = json.load(f)

            completed = progress.get("results", {})
            self.log(f"Resuming with {len(completed)} completed models")

            # Pre-populate summaries table
            for model_name, summary in progress.get("summaries", {}).items():
                self.add_summary_row(summary)

            self.start_evaluation(completed)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to resume: {e}")

    def stop_evaluation(self):
        """Stop the running evaluation."""
        if self.worker:
            self.log("Stopping evaluation...")
            self.worker.stop()

    def on_progress(self, model: str, current: int, total: int, message: str):
        """Handle progress update."""
        self.current_model_label.setText(f"Model: {model}")
        self.current_progress.setMaximum(total)
        self.current_progress.setValue(current)
        self.current_file_label.setText(message)

        # Update overall progress
        selected = self.get_selected_models()
        model_list = list(selected.keys())
        model_idx = model_list.index(model) if model in model_list else 0
        overall = model_idx * len(self.audio_files) + current
        self.overall_progress.setValue(overall)

        pct = (overall / self.overall_progress.maximum() * 100) if self.overall_progress.maximum() > 0 else 0
        self.overall_label.setText(f"Overall: {pct:.1f}%")

    def on_result(self, model: str, result: dict):
        """Handle individual transcription result."""
        if model not in self.all_results:
            self.all_results[model] = []
        self.all_results[model].append(result)

    def on_model_complete(self, model: str, summary: dict):
        """Handle model completion."""
        self.summaries[model] = summary
        self.add_summary_row(summary)

    def add_summary_row(self, summary: dict):
        """Add a row to the results table."""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)

        self.results_table.setItem(row, 0, QTableWidgetItem(summary["model"]))
        self.results_table.setItem(row, 1, QTableWidgetItem(f"{summary['average_wer']:.2%}"))
        self.results_table.setItem(row, 2, QTableWidgetItem(f"{summary['median_wer']:.2%}"))
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{summary['min_wer']:.2%}"))
        self.results_table.setItem(row, 4, QTableWidgetItem(f"{summary['max_wer']:.2%}"))

        # Color code based on WER
        wer = summary['average_wer']
        if wer < 0.1:
            color = QColor(200, 255, 200)  # Green
        elif wer < 0.2:
            color = QColor(255, 255, 200)  # Yellow
        else:
            color = QColor(255, 200, 200)  # Red

        for col in range(5):
            item = self.results_table.item(row, col)
            if item:
                item.setBackground(color)

    def on_all_complete(self, summaries: dict):
        """Handle evaluation completion."""
        self.log("\n" + "="*50)
        self.log("EVALUATION COMPLETE")
        self.log("="*50)

        # Generate comparison text
        self.update_comparison(summaries)

    def update_comparison(self, summaries: dict):
        """Update the comparison tab."""
        lines = ["FINE-TUNE VS ORIGINAL COMPARISON", "="*50, ""]

        sizes = ['tiny', 'base', 'small', 'medium', 'large']

        for size in sizes:
            ft_key = f"finetune-{size}"
            orig_key = f"original-{size}"

            if ft_key in summaries and orig_key in summaries:
                ft_wer = summaries[ft_key]['average_wer']
                orig_wer = summaries[orig_key]['average_wer']
                improvement = orig_wer - ft_wer
                pct = (improvement / orig_wer * 100) if orig_wer > 0 else 0

                if improvement > 0:
                    symbol = "[+]"
                    result = f"{abs(pct):.1f}% improvement"
                else:
                    symbol = "[-]"
                    result = f"{abs(pct):.1f}% regression"

                lines.append(f"{size.upper():8}: Fine-tune {ft_wer:.2%} vs Original {orig_wer:.2%}")
                lines.append(f"         {symbol} {result}")
                lines.append("")

        lines.append("="*50)
        lines.append("")
        lines.append("RANKING BY AVERAGE WER")
        lines.append("-"*50)

        sorted_summaries = sorted(summaries.items(), key=lambda x: x[1]['average_wer'])
        for rank, (name, s) in enumerate(sorted_summaries, 1):
            lines.append(f"{rank}. {name}: {s['average_wer']:.2%}")

        self.comparison_text.setText("\n".join(lines))

    def on_error(self, message: str):
        """Handle error."""
        self.log(f"ERROR: {message}")
        QMessageBox.critical(self, "Error", message)

    def on_worker_finished(self):
        """Handle worker thread completion."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.resume_btn.setEnabled(PROGRESS_FILE.exists())
        self.current_model_label.setText("Complete")
        self.overall_label.setText("Evaluation finished")
        self.log("Worker thread finished")

    def closeEvent(self, event):
        """Handle window close."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Evaluation is running. Stop and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                self.worker.wait(5000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
