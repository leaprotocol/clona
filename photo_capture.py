from nicegui import ui

class PhotoCapture:
    def __init__(self, camera_manager, dataset_manager):
        self.camera_manager = camera_manager
        self.dataset_manager = dataset_manager
        self.log_display = None  # Initialize as None, create during setup

    async def setup_controls(self, container):
        """Setup photo capture controls within the given container"""
        with container:
            self.log_display = ui.textarea().classes('log-display')

    async def do_capture(self, dataset_path, aperture):
        """Regular capture with full metadata"""
        try:
            os.makedirs(dataset_path, exist_ok=True)
            capture_result = await self.camera_manager.capture_image("captures")
            if not capture_result:
                ui.notify('Failed to capture photo', type='negative')
                return

            await self.process_capture(capture_result, dataset_path, aperture)
        except Exception as e:
            ui.notify(f'Error capturing photo: {str(e)}', type='negative')
            logging.error(f"Error capturing photo: {e}")

    async def process_capture(self, capture_result, dataset_path, aperture):
        """Process captured image and save metadata"""
        try:
            temp_path = capture_result['path']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_filename = f"{self.current_scenario['type']}_{aperture}_{timestamp}_{os.path.basename(temp_path)}"
            final_path = os.path.join(dataset_path, final_filename)

            # Move the file asynchronously
            await asyncio.to_thread(shutil.move, temp_path, final_path)

            # Get camera settings from capture result
            camera_settings = capture_result['metadata']['camera_settings']

            # Create and save photo info
            photo_info = {
                'filename': final_filename,
                'path': final_path,
                'timestamp': timestamp,
                'metadata': {
                    'scenario_type': self.current_scenario['type'],
                    'scenario_id': self.current_scenario['id'],
                    'camera_settings': camera_settings,
                    'aperture': aperture,
                    'shutter_speed': camera_settings.get('shutter_speed'),
                    'iso': camera_settings.get('iso'),
                    'lens_name': camera_settings.get('lens_name'),
                    'camera_model': camera_settings.get('camera_model'),
                    'focal_length': camera_settings.get('focal_length')
                }
            }

            if 'photos' not in self.current_scenario:
                self.current_scenario['photos'] = []
            self.current_scenario['photos'].append(photo_info)

            if await self.dataset_manager.update_scenario(
                    self.current_dataset['id'],
                    self.current_scenario
            ):
                ui.notify('Photo captured and saved with full metadata', type='positive')
                await self.select_scenario(self.current_scenario)
            else:
                ui.notify('Photo captured but failed to update dataset', type='warning')

        except Exception as e:
            ui.notify(f'Error processing capture: {str(e)}', type='negative')
            logging.error(f"Error processing capture: {e}") 