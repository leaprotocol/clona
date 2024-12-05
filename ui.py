import os
from nicegui import ui
import logging
import time
from typing import Callable
from datetime import datetime
import gphoto2 as gp  # Add this import

class LensAnalysisUI:
    def __init__(self, camera_manager, dataset_manager):
        self.camera_manager = camera_manager
        self.dataset_manager = dataset_manager
        self.current_dataset = None
        self.current_scenario = None
        self.log_display = None
        self.status_label = None
        self.capture_button = None
        self.connect_button = None
        self.disconnect_button = None
        self.settings_button = None
        self.dataset_list = None
        self.scenario_details = None

    def setup_ui_logging(self):
        """Set up logging display in the UI"""
        self.log_display = ui.textarea(
            label='Application Logs',
            value=''
        ).classes('w-full h-40')

        class UILogHandler(logging.Handler):
            def __init__(self, callback: Callable[[str], None]):
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


    def update_camera_status(self):
        """Update camera status display and button states"""
        connected = self.camera_manager.connected
        if connected:
            self.status_label.text = '● Camera Connected'
            self.status_label.classes('text-green-500', remove='text-red-500')  # Add green, remove red
            self.connect_button.disable()
            self.disconnect_button.enable()
            self.capture_button.enable()
            self.settings_button.enable()
        else:
            self.status_label.text = '○ Camera Disconnected'
            self.status_label.classes('text-red-500', remove='text-green-500')  # Add red, remove green
            self.connect_button.enable()
            self.disconnect_button.disable()
            self.capture_button.disable()
            self.settings_button.disable()

    def handle_connect_camera(self):
        """Handle camera connection"""
        success = self.camera_manager.initialize_camera()
        ui.notify('Camera connected successfully' if success else 'Failed to connect camera',
                  type='positive' if success else 'negative')
        self.update_camera_status()

    def handle_disconnect_camera(self):
        """Handle camera disconnection"""
        self.camera_manager.release_camera()
        ui.notify('Camera disconnected')
        self.update_camera_status()

    def handle_capture_photo(self):
        """Handle photo capture"""
        if not self.camera_manager.connected:
            ui.notify('Camera not connected', type='negative')
            return

        if not self.current_scenario:
            ui.notify('Please select a scenario first', type='warning')
            return

        # For vignette scenario, ask for aperture setting first
        if self.current_scenario['type'] == 'vignette':
            self.show_aperture_selection_dialog()
        else:
            self.do_capture()  # Regular capture for other scenarios

    def show_aperture_selection_dialog(self):
        """Show dialog for selecting multiple apertures for batch capture"""
        # Get available apertures from camera
        available_apertures = self.print_available_apertures()
        if not available_apertures:
            ui.notify('Could not get aperture settings from camera', type='negative')
            return

        dialog = ui.dialog()

        def close_dialog():
            dialog.close()
            ui.notify('Capture cancelled', type='warning')

        with dialog, ui.card().classes('p-4 min-w-[300px]'):
            # Add close button in top right
            with ui.row().classes('w-full justify-between items-center'):
                ui.label('Select Aperture Values').classes('text-xl')
                ui.button(text='✕', on_click=close_dialog).classes('text-gray-500')

            ui.separator().classes('my-4')

            aperture_switches = []
            for aperture in available_apertures:
                switch = ui.switch(f'{aperture}')
                aperture_switches.append((aperture, switch))

            with ui.row().classes('gap-2 justify-end mt-4'):
                ui.button(
                    'Cancel',
                    on_click=close_dialog
                ).classes('bg-gray-500 text-white')

                ui.button(
                    'Capture Series',
                    on_click=lambda: self.do_batch_capture(
                        dialog,
                        [ap for ap, sw in aperture_switches if sw.value]
                    )
                ).classes('bg-green-500 text-white')

        # Allow dialog to be closed by clicking outside or pressing Escape
        dialog.on_escape = close_dialog
        dialog.on_click_outside = close_dialog

        dialog.open()
    def do_capture_with_aperture(self, dialog, aperture):
        """Capture photo with the selected aperture value"""
        dialog.close()

        # Ensure captures directory exists
        os.makedirs("captures", exist_ok=True)

        ui.notify('Capturing photo...', type='ongoing')

        # Try capture with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                path = self.camera_manager.capture_image("captures")
                if path:
                    # Add photo to scenario with metadata
                    photo_info = {
                        'filename': os.path.basename(path),
                        'path': path,
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                        'metadata': {
                            'aperture': aperture
                        }
                    }

                    if 'photos' not in self.current_scenario:
                        self.current_scenario['photos'] = []
                    self.current_scenario['photos'].append(photo_info)

                    # Update UI
                    self.select_scenario(self.current_scenario)
                    ui.notify(f'Photo captured at {aperture}', type='positive')
                    return
                else:
                    if attempt < max_retries - 1:
                        logging.warning(f"Capture attempt {attempt + 1} failed, retrying...")
                        time.sleep(2)  # Wait before retry
                    else:
                        ui.notify('Failed to capture photo after multiple attempts', type='negative')

            except Exception as e:
                if attempt < max_retries - 1:
                    logging.error(f"Error on capture attempt {attempt + 1}: {e}")
                    time.sleep(2)  # Wait before retry
                else:
                    ui.notify(f'Error capturing photo: {str(e)}', type='negative')
                    logging.error(f"Error capturing photo: {e}")


    def do_capture(self):
        """Regular capture without aperture metadata"""
        # Ensure captures directory exists
        os.makedirs("captures", exist_ok=True)

        ui.notify('Capturing photo...', type='ongoing')

        try:
            path = self.camera_manager.capture_image("captures")
            if path:
                # Add photo to scenario with basic metadata
                photo_info = {
                    'filename': os.path.basename(path),
                    'path': path,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'metadata': {}
                }

                if 'photos' not in self.current_scenario:
                    self.current_scenario['photos'] = []
                self.current_scenario['photos'].append(photo_info)

                # Update UI
                self.select_scenario(self.current_scenario)
                ui.notify('Photo captured successfully', type='positive')
            else:
                ui.notify('Failed to capture photo', type='negative')
        except Exception as e:
            ui.notify(f'Error capturing photo: {str(e)}', type='negative')
            logging.error(f"Error capturing photo: {e}")

    def create_camera_controls(self):
        """Create camera control section"""
        with ui.card().classes('w-full mb-4'):
            with ui.row().classes('w-full items-center'):
                ui.label('Camera Control').classes('text-xl mr-4')
                self.status_label = ui.label('○ Camera Disconnected').classes('text-red-500')

            with ui.row().classes('gap-2 mt-2'):
                self.connect_button = ui.button(
                    'Connect Camera',
                    on_click=self.handle_connect_camera
                ).classes('bg-blue-500 text-white')

                self.disconnect_button = ui.button(
                    'Disconnect Camera',
                    on_click=self.handle_disconnect_camera
                ).classes('bg-red-500 text-white')
                self.disconnect_button.disable()

            with ui.row().classes('gap-2 mt-2'):
                self.capture_button = ui.button(
                    '📸 Capture Photo',
                    on_click=self.handle_capture_photo
                ).classes('bg-green-500 text-white')
                self.capture_button.disable()

                self.settings_button = ui.button(
                    '⚙️ Camera Settings',
                    on_click=self.show_camera_settings
                ).classes('bg-purple-500 text-white')
                self.settings_button.disable()

    def create_dataset_dialog(self):
        """Show dialog for creating a new dataset"""
        dialog = ui.dialog()
        with dialog, ui.card():
            ui.label('Create New Dataset').classes('text-xl mb-4')

            dataset_name = ui.input(
                label='Dataset Name',
                placeholder='Enter dataset name'
            ).classes('w-full mb-4')

            with ui.row().classes('gap-2 justify-end'):
                ui.button(
                    'Cancel',
                    on_click=dialog.close
                ).classes('bg-gray-500 text-white')

                ui.button(
                    'Create',
                    on_click=lambda: self.handle_create_dataset(dataset_name.value, dialog)
                ).classes('bg-green-500 text-white')

        dialog.open()

    def handle_create_dataset(self, name: str, dialog):
        """Handle dataset creation"""
        if not name:
            ui.notify('Please enter a dataset name', type='warning')
            return

        try:
            dataset = self.dataset_manager.create_dataset(name)
            if dataset:
                dialog.close()
                ui.notify(f'Dataset "{name}" created successfully', type='positive')
                self.refresh_dataset_list()
            else:
                ui.notify('Failed to create dataset', type='negative')
        except Exception as e:
            ui.notify(f'Error creating dataset: {str(e)}', type='negative')

    def create_scenario_dialog(self):
        """Show dialog for creating a new scenario"""
        if not self.current_dataset:
            ui.notify('Please select a dataset first', type='warning')
            return

        dialog = ui.dialog()
        with dialog, ui.card():
            ui.label('Create New Scenario').classes('text-xl mb-4')

            scenario_type = ui.select(
                label='Scenario Type',
                options=[
                    ('vignette', 'Vignetting Test'),
                    ('bokeh', 'Bokeh Analysis'),
                    ('sharpness', 'Sharpness Test'),
                    ('distortion', 'Distortion Analysis')
                ]
            ).classes('w-full mb-4')

            focal_length = ui.number('Focal Length (mm)', value=50, min=1).classes('w-full mb-4')

            notes = ui.textarea(
                label='Notes (optional)',
                placeholder='Enter any additional notes about this test...'
            ).classes('w-full mb-4')

            with ui.row().classes('gap-2 justify-end'):
                ui.button(
                    'Cancel',
                    on_click=dialog.close
                ).classes('bg-gray-500 text-white')

                ui.button(
                    'Create',
                    on_click=lambda: self.handle_create_scenario(
                        scenario_type.value,
                        focal_length.value,
                        notes.value,
                        dialog
                    )
                ).classes('bg-green-500 text-white')

        dialog.open()

    def handle_create_scenario(self, scenario_type: str, focal_length: float, notes: str, dialog):
        """Handle scenario creation"""
        if not scenario_type:
            ui.notify('Please select a scenario type', type='warning')
            return

        try:
            # Get just the first part of the tuple if it is one
            actual_type = scenario_type[0] if isinstance(scenario_type, tuple) else scenario_type

            metadata = {
                'focal_length': focal_length,
                'notes': notes if notes else '',
                'created': datetime.now().strftime("%Y%m%d_%H%M%S")
            }

            scenario = self.dataset_manager.create_scenario(
                self.current_dataset['id'],
                actual_type,  # Use the extracted type
                metadata
            )

            if scenario:
                dialog.close()
                ui.notify(f'Scenario created successfully', type='positive')
                self.select_scenario(scenario)
                self.refresh_dataset_list()
            else:
                ui.notify('Failed to create scenario', type='negative')
        except Exception as e:
            ui.notify(f'Error creating scenario: {str(e)}', type='negative')

    def show_camera_settings(self):
        """Show camera settings dialog"""
        with ui.dialog() as dialog, ui.card():
            ui.label('Camera Settings').classes('text-xl mb-4')

            with ui.column().classes('gap-4'):
                ui.label('Camera settings dialog placeholder')
                ui.button('Close', on_click=dialog.close)

    def select_dataset(self, dataset):
        """Select a dataset and show its details"""
        self.current_dataset = dataset
        self.current_scenario = None
        self.scenario_details.clear()
        with self.scenario_details:
            ui.label(f'Dataset: {dataset["name"]}').classes('text-xl')
            if 'metadata' in dataset and 'lens_details' in dataset['metadata']:
                ui.label(f'Lens: {dataset["metadata"]["lens_details"]}')

            if 'scenarios' in dataset and dataset['scenarios']:
                with ui.column().classes('mt-4 gap-2'):
                    ui.label('Scenarios:').classes('font-bold')
                    for scenario in dataset['scenarios']:
                        with ui.card().classes('w-full p-2'):
                            with ui.row().classes('justify-between items-center'):
                                # Handle both string and tuple types
                                scenario_type = scenario['type']
                                display_type = scenario_type[1] if isinstance(scenario_type,
                                                                              tuple) else scenario_type.title()
                                ui.label(display_type).classes('font-bold')
                                ui.button('Select',
                                          on_click=lambda s=scenario: self.select_scenario(s)
                                          ).classes('bg-blue-500 text-white p-1 text-sm')
            else:
                ui.label('No scenarios yet - create one to begin testing').classes('text-gray-500 italic mt-4')

        self.refresh_dataset_list()

    def select_scenario(self, scenario):
        """Select and display a scenario"""
        self.current_scenario = scenario
        self.scenario_details.clear()

        with self.scenario_details:
            ui.label(f"Scenario: {scenario['type'].title()}").classes('text-xl mb-2')
            ui.label(f"Focal Length: {scenario['metadata']['focal_length']}mm").classes('mb-1')

            if scenario['metadata'].get('notes'):
                with ui.card().classes('w-full mb-4 bg-gray-50'):
                    ui.label('Notes:').classes('font-bold')
                    ui.label(scenario['metadata']['notes'])

            # Photos and Analysis section
            photos = scenario.get('photos', [])
            with ui.card().classes('w-full mb-4'):
                ui.label('Photos').classes('font-bold mb-2')

                if not photos:
                    ui.label('No photos taken yet').classes('text-gray-500 italic')
                else:
                    with ui.grid(columns=2).classes('gap-4'):
                        for photo in photos:
                            with ui.card().classes('p-4'):
                                ui.label(f"Filename: {photo['filename']}").classes('font-bold')
                                ui.label(f"Taken: {photo['timestamp']}")
                                if 'metadata' in photo and 'aperture' in photo['metadata']:
                                    ui.label(f"Aperture: f/{photo['metadata']['aperture']}")

                                with ui.row().classes('gap-2 mt-2'):
                                    if scenario['type'] == 'vignette':
                                        ui.button(
                                            'View Analysis',
                                            on_click=lambda p=photo: self.show_photo_analysis(scenario, p)
                                        ).classes('bg-blue-500 text-white p-2')

                                    ui.button(
                                        'Delete Photo',
                                        on_click=lambda p=photo: self.delete_photo(scenario, p)
                                    ).classes('bg-red-500 text-white p-2')

            # Capture controls
            with ui.card().classes('w-full p-4'):
                ui.label('Capture').classes('font-bold mb-2')
                with ui.row().classes('gap-2'):
                    if scenario['type'] == 'vignette':
                        ui.button(
                            '📸 Capture with Aperture',
                            on_click=self.show_aperture_selection_dialog
                        ).classes('bg-green-500 text-white')
                    else:
                        ui.button(
                            '📸 Capture Photo',
                            on_click=self.handle_capture_photo
                        ).classes('bg-green-500 text-white')

        def delete_photo(self, scenario, photo):
            """Delete a photo from the scenario"""
            try:
                # Remove the file
                if os.path.exists(photo['path']):
                    os.remove(photo['path'])

                # Remove from photos list
                scenario['photos'].remove(photo)

                # Remove analysis visualization if it exists
                if 'analysis' in photo and 'visualization_path' in photo['analysis']:
                    if os.path.exists(photo['analysis']['visualization_path']):
                        os.remove(photo['analysis']['visualization_path'])

                # Refresh display
                self.select_scenario(scenario)
                ui.notify('Photo deleted', type='warning')
            except Exception as e:
                ui.notify(f'Error deleting photo: {str(e)}', type='negative')
                logging.error(f"Error deleting photo: {e}")


    def refresh_dataset_list(self):
        """Refresh the dataset list display"""
        self.dataset_list.clear()

        try:
            datasets = self.dataset_manager.list_datasets()
            if not datasets:
                with self.dataset_list:
                    ui.label('No datasets available').classes('text-gray-500 italic')
            else:
                for dataset in datasets:
                    with self.dataset_list:
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
                                        on_click=lambda d=dataset: self.select_dataset(d)
                                    ).classes('bg-blue-500 text-white p-1 text-sm')
                                    ui.button(
                                        'Delete',
                                        on_click=lambda d=dataset: self.delete_dataset(d)
                                    ).classes('bg-red-500 text-white p-1 text-sm')
        except Exception as e:
            with self.dataset_list:
                ui.label(f'Error loading datasets: {str(e)}').classes('text-red-500')

    def delete_dataset(self, dataset):
        """Delete a dataset"""
        try:
            if self.dataset_manager.delete_dataset(dataset['id']):
                ui.notify(f'Dataset deleted successfully', type='positive')
                if self.current_dataset and self.current_dataset['id'] == dataset['id']:
                    self.current_dataset = None
                    self.scenario_details.clear()
                    with self.scenario_details:
                        ui.label('No dataset selected').classes('text-gray-500 italic')
                self.refresh_dataset_list()
            else:
                ui.notify('Failed to delete dataset', type='negative')
        except Exception as e:
            ui.notify(f'Error deleting dataset: {str(e)}', type='negative')

    def create_main_page(self):
        """Create the main application page"""

        @ui.page('/')
        def main_page():
            with ui.column().classes('w-full max-w-5xl mx-auto p-4 gap-4'):
                # Title and header
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Lens Analysis Application').classes('text-2xl')
                    with ui.row().classes('gap-2'):
                        ui.button('Help',
                                  on_click=lambda: ui.notify('Help functionality coming soon', type='info')
                                  ).classes('bg-gray-500 text-white')

                # Camera Controls
                with ui.card().classes('w-full p-4'):
                    self.create_camera_controls()

                # Dataset and Scenario Management
                with ui.row().classes('w-full gap-4'):
                    # Left panel - Dataset List
                    with ui.card().classes('w-1/3 p-4'):
                        with ui.row().classes('justify-between items-center mb-4'):
                            ui.label('Datasets').classes('text-xl')
                            ui.button(
                                'New Dataset',
                                on_click=self.create_dataset_dialog
                            ).classes('bg-green-500 text-white')

                        # Dataset list container
                        self.dataset_list = ui.column().classes('gap-2 max-h-96 overflow-y-auto')
                        self.refresh_dataset_list()

                    # Right panel - Scenario Management
                    with ui.card().classes('w-2/3 p-4'):
                        with ui.row().classes('justify-between items-center mb-4'):
                            ui.label('Scenario').classes('text-xl')
                            ui.button(
                                'New Scenario',
                                on_click=self.create_scenario_dialog
                            ).classes('bg-blue-500 text-white')

                        # Scenario details container
                        self.scenario_details = ui.card().classes('w-full p-4')
                        with self.scenario_details:
                            ui.label('No dataset selected').classes('text-gray-500 italic')

                # Log Display
                with ui.card().classes('w-full p-4'):
                    with ui.row().classes('justify-between items-center mb-2'):
                        ui.label('System Logs').classes('text-xl')
                        ui.button(
                            'Clear',
                            on_click=lambda: setattr(self.log_display, 'value', '')
                        ).classes('bg-gray-500 text-white text-sm')
                    self.setup_ui_logging()

                # Start periodic status updates
                ui.timer(2.0, self.update_camera_status)

    def do_batch_capture(self, dialog, apertures):
        """Capture multiple photos with different aperture values"""
        if not apertures:
            ui.notify('Please select at least one aperture value', type='warning')
            return

        logging.info(f"Starting batch capture for apertures: {apertures}")
        dialog.close()
        os.makedirs("captures", exist_ok=True)

        total = len(apertures)
        for idx, aperture in enumerate(apertures, 1):
            logging.info(f"Processing aperture {aperture} ({idx}/{total})")
            ui.notify(f'Setting aperture to {aperture} ({idx}/{total})...', type='info')

            try:
                # Get camera config
                logging.info("Getting camera config...")
                camera_config = self.camera_manager.camera.get_config()

                # Find and set aperture
                logging.info("Looking for aperture setting...")
                OK, aperture_widget = gp.gp_widget_get_child_by_name(camera_config, 'aperture')
                if OK >= gp.GP_OK:
                    logging.info(f"Setting aperture value to {aperture}")
                    # Set the aperture value
                    aperture_widget.set_value(aperture)
                    logging.info("Applying camera config...")
                    self.camera_manager.camera.set_config(camera_config)
                    time.sleep(1)  # Short delay to let camera apply setting
                else:
                    logging.error("Could not find aperture setting")
                    ui.notify('Could not find aperture setting', type='negative')
                    continue

                # Capture photo
                logging.info("Initiating capture...")
                ui.notify(f'Capturing photo at {aperture}...', type='ongoing')
                path = self.camera_manager.capture_image("captures")

                if path:
                    logging.info(f"Photo saved to {path}")
                    photo_info = {
                        'filename': os.path.basename(path),
                        'path': path,
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                        'metadata': {
                            'aperture': aperture
                        }
                    }

                    if 'photos' not in self.current_scenario:
                        self.current_scenario['photos'] = []
                    self.current_scenario['photos'].append(photo_info)

                    # Update UI
                    self.select_scenario(self.current_scenario)
                    ui.notify(f'Photo captured at {aperture}', type='positive')
                else:
                    logging.error(f"Failed to capture photo at {aperture}")
                    ui.notify(f'Failed to capture photo at {aperture}', type='negative')

            except Exception as e:
                logging.error(f"Error during capture at {aperture}: {e}")
                ui.notify(f'Error during capture at {aperture}: {str(e)}', type='negative')
                continue

        logging.info("Batch capture complete")
        ui.notify('Batch capture complete', type='positive')



    def do_capture_confirmed(self, dialog, aperture):
        """Execute single capture after user confirms aperture is set"""
        dialog.close()

        ui.notify(f'Capturing photo at {aperture}...', type='ongoing')

        try:
            path = self.camera_manager.capture_image("captures")
            if path:
                # Add photo to scenario with metadata
                photo_info = {
                    'filename': os.path.basename(path),
                    'path': path,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'metadata': {
                        'aperture': aperture
                    }
                }

                if 'photos' not in self.current_scenario:
                    self.current_scenario['photos'] = []
                self.current_scenario['photos'].append(photo_info)

                # Update UI
                self.select_scenario(self.current_scenario)
                ui.notify(f'Photo captured at {aperture}', type='positive')
            else:
                ui.notify('Failed to capture photo', type='negative')
        except Exception as e:
            ui.notify(f'Error capturing photo: {str(e)}', type='negative')
            logging.error(f"Error capturing photo: {e}")
    def print_available_apertures(self):
        """Print available aperture settings for the camera"""
        try:
            camera_config = self.camera_manager.camera.get_config()
            OK, aperture_widget = gp.gp_widget_get_child_by_name(camera_config, 'aperture')
            if OK >= gp.GP_OK:
                choices = [aperture_widget.get_choice(i) for i in range(aperture_widget.count_choices())]
                logging.info(f"Available aperture settings: {choices}")
                return choices
        except Exception as e:
            logging.error(f"Error getting aperture settings: {e}")
            return None

    def show_photo_analysis(self, scenario, photo_info):
        """Show analysis dialog for a photo"""
        dialog = ui.dialog()
        with dialog, ui.card().classes('p-4 min-w-[400px]'):
            ui.label('Photo Analysis').classes('text-xl mb-4')

            if 'analysis' not in photo_info:
                ui.button(
                    'Analyze Vignetting',
                    on_click=lambda: self.run_photo_analysis(scenario, photo_info, dialog)
                ).classes('bg-blue-500 text-white')
            else:
                # Show analysis results
                results = photo_info['analysis']['vignetting_results']
                ui.label(f"Overall vignetting score: {results['vignetting_score']:.1f}/100").classes('font-bold')

                # Show corner ratios in a grid
                with ui.grid(columns=2).classes('gap-4 mt-4'):
                    for corner, ratio in results['corner_ratios'].items():
                        ui.label(f"{corner}: {ratio:.2f}")

                # Show visualization
                if 'visualization_path' in photo_info['analysis']:
                    ui.image(photo_info['analysis']['visualization_path']).classes('mt-4')

            ui.button('Close', on_click=dialog.close).classes('bg-gray-500 text-white mt-4')

    def run_photo_analysis(self, scenario, photo_info, dialog=None):
        """Run analysis on a photo"""
        try:
            from analysis import analyze_scenario_photo  # Import at use time

            if analyze_scenario_photo(scenario, photo_info):
                ui.notify('Analysis complete', type='positive')
                # Refresh the UI
                if dialog:
                    dialog.close()
                self.show_photo_analysis(scenario, photo_info)
            else:
                ui.notify('Analysis failed', type='negative')
        except Exception as e:
            ui.notify(f'Error during analysis: {str(e)}', type='negative')
            logging.error(f"Analysis error: {e}")


    def run(self):
        """Start the UI"""
        self.create_main_page()
        ui.run()

