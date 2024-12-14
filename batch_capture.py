class BatchCapture:
    def __init__(self, camera_manager, photo_capture):
        self.camera_manager = camera_manager
        self.photo_capture = photo_capture
        self.log_display = None

    async def do_batch_capture(self, dialog, apertures):
        if not apertures:
            ui.notify("No apertures selected for batch capture.", type='negative')
            return

        try:
            dialog.close()
            await asyncio.sleep(0.5)
            
            camera_config, widget_ref = await self.get_aperture_widget()
            if not widget_ref:
                return

            dataset_path = os.path.join("datasets", self.current_dataset['id'], "photos")
            await self.capture_series(camera_config, widget_ref, apertures, dataset_path)

        except Exception as e:
            logging.error(f"Error in batch capture: {e}")
            ui.notify('Error during batch capture', type='negative') 