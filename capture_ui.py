class CaptureUI:
    def __init__(self, camera_manager, dataset_manager):
        self.camera_manager = camera_manager
        self.dataset_manager = dataset_manager
        self.current_scenario = None
        self.log_display = None

    async def do_capture(self, dataset_path, aperture):
        """Regular capture with full metadata"""
        # Implementation from original file
        # Lines referenced from ui.py:
        # startLine: 461
        # endLine: 515

    async def do_batch_capture(self, dialog, apertures):
        """Perform batch capture for selected apertures"""
        # Implementation from original file
        # Lines referenced from ui.py:
        # startLine: 1127
        # endLine: 1180 