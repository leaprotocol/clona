import os
from nicegui import ui
import logging
from datetime import datetime
import shutil
import exifread
import asyncio
import gphoto2 as gp

from logging_handler import UILogger
from camera_controls import CameraControls
from dataset_controls import DatasetControls
from photo_capture import PhotoCapture
from batch_capture import BatchCapture

class LensAnalysisUI:
    def __init__(self, camera_manager, dataset_manager):
        self.camera_manager = camera_manager
        self.dataset_manager = dataset_manager
        self.current_dataset = None
        
        # Initialize components without UI elements
        self.dataset_controls = DatasetControls(dataset_manager, on_dataset_change=self.handle_dataset_selection)
        self.photo_capture = PhotoCapture(camera_manager, dataset_manager)
        self.batch_capture = BatchCapture(camera_manager, self.photo_capture)
        self.camera_controls = CameraControls(camera_manager)

    def run(self):
        """Run the UI event loop."""
        ui.run()

    async def setup_ui(self):
        """Setup main UI layout"""
        # Create main container for photo capture
        photo_capture_container = ui.column().classes('w-full')
        
        # Setup all components
        await self.camera_controls.setup_controls()
        await self.photo_capture.setup_controls(photo_capture_container)
        
        # Create main layout sections
        with ui.row().classes('w-full gap-4'):
            # Left panel - Dataset management
            with ui.column().classes('w-1/3'):
                await self.dataset_controls.setup_controls()
                
            # Right panel - Analysis area
            with ui.column().classes('w-2/3'):
                self.create_analysis_area()
                
        # Add logging area at bottom
        with ui.row().classes('w-full mt-4'):
            self.setup_ui_logging()
        
        # Initial refresh of dataset list
        await self.refresh_dataset_list()

    def create_analysis_area(self):
        """Create analysis area"""
        with ui.card().classes('w-full'):
            ui.label('Analysis').classes('text-xl font-bold')
            self.scenario_details = ui.column().classes('w-full')
            
            # Analysis buttons
            with ui.row().classes('gap-2'):
                ui.button(
                    'Capture Photo',
                    on_click=self.dataset_controls.handle_capture_photo
                ).classes('bg-blue-500 text-white')
                ui.button(
                    'Import RAW',
                    on_click=self.dataset_controls.handle_import_raw
                ).classes('bg-blue-500 text-white')

    async def handle_dataset_selection(self, dataset):
        """Handle dataset selection change"""
        try:
            logging.debug(f"Raw dataset input: {dataset}")
            
            # Handle empty selection
            if not dataset:
                self.current_dataset = None
                return True
            
            # Get dataset ID
            dataset_id = dataset['id'] if isinstance(dataset, dict) else str(dataset)
            
            logging.debug(f"Processing dataset selection with ID: {dataset_id}")
            
            if not dataset_id:
                logging.error("No valid dataset ID found in selection")
                ui.notify('Invalid dataset selection', type='negative')
                return False
            
            # Load the dataset
            self.current_dataset = await self.dataset_manager.load_dataset(dataset_id)
            
            if self.current_dataset:
                logging.info(f"Successfully loaded dataset: {dataset_id}")
                await self.refresh_dataset_list()
                ui.notify('Dataset loaded successfully', type='positive')
                return True
            else:
                logging.error(f"Failed to load dataset with ID: {dataset_id}")
                ui.notify('Failed to load dataset', type='negative')
                return False
            
        except Exception as e:
            logging.error(f"Error in handle_dataset_selection: {str(e)}")
            ui.notify('Error selecting dataset', type='negative')
            self.current_dataset = None
            return False

    async def refresh_dataset_list(self):
        """Refresh the list of available datasets"""
        try:
            datasets = await self.dataset_manager.list_datasets()
            self.dataset_controls.dataset_list.clear()
            
            if not datasets:
                with self.dataset_controls.dataset_list:
                    ui.label('No datasets available').classes('text-gray-500 italic')
                return False

            for dataset in datasets:
                with self.dataset_controls.dataset_list:
                    with ui.card().classes('w-full p-2'):
                        with ui.row().classes('justify-between items-center'):
                            ui.label(dataset['name']).classes(
                                'font-bold text-blue-500'
                                if dataset == self.current_dataset
                                else ''
                            )
                            with ui.row().classes('gap-1'):
                                ui.button(
                                    'Select',
                                    on_click=lambda d=dataset: self.dataset_controls.select_dataset(d)
                                ).classes('bg-blue-500 text-white p-1 text-sm')
                                ui.button(
                                    'Delete',
                                    on_click=lambda d=dataset: self.dataset_controls.delete_dataset(d)
                                ).classes('bg-red-500 text-white p-1 text-sm')
            return True
        except Exception as e:
            with self.dataset_controls.dataset_list:
                ui.label(f'Error loading datasets: {str(e)}').classes('text-red-500')
            return False

    def setup_ui_logging(self):
        """Set up logging display in the UI"""
        self.log_display = ui.textarea(
            label='Application Logs',
            value=''
        ).classes('w-full h-40')

        class UILogHandler(logging.Handler):
            def __init__(self, callback):
                super().__init__()
                self.callback = callback

            def emit(self, record):
                log_entry = self.format(record)
                self.callback(log_entry + "\n")

        def update_log(msg):
            current = self.log_display.value
            self.log_display.value = current + msg

        handler = UILogHandler(update_log)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        ))
        logging.getLogger().addHandler(handler)
