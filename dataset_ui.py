class DatasetUI:
    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager
        self.dataset_list = None
        self.current_dataset = None

    def refresh_dataset_list(self):
        """Refresh the list of available datasets"""
        # Implementation from original file
        # Lines referenced from ui.py:
        # startLine: 323
        # endLine: 355 