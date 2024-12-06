# dataset_manager.py
import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any


# dataset_manager.py
import os
import json
import logging
import shutil
import zipfile
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

class DatasetManager:
    def __init__(self, base_path: str = "datasets"):
        """Initialize dataset manager"""
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def export_dataset(self, dataset_id: str, export_path: str) -> bool:
        """Export a dataset to a zip file"""
        try:
            dataset = self.load_dataset(dataset_id)
            if not dataset:
                logging.error(f"Dataset {dataset_id} not found for export")
                return False

            dataset_dir = os.path.join(self.base_path, dataset_id)
            if not os.path.exists(dataset_dir):
                logging.error(f"Dataset directory {dataset_dir} not found")
                return False

            # Create zip file name with dataset name and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_name = f"{dataset['name']}_{timestamp}.zip"
            zip_path = os.path.join(export_path, zip_name)

            # Create the export directory if it doesn't exist
            os.makedirs(export_path, exist_ok=True)

            # Create zip file
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through all files in the dataset directory
                for root, _, files in os.walk(dataset_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Calculate arc name (path within zip file)
                        arc_name = os.path.relpath(file_path, self.base_path)
                        zipf.write(file_path, arc_name)

            logging.info(f"Dataset exported to {zip_path}")
            return True

        except Exception as e:
            logging.error(f"Error exporting dataset: {e}")
            return False

    def import_dataset(self, zip_path: str) -> Optional[Dict]:
        """Import a dataset from a zip file"""
        try:
            if not os.path.exists(zip_path):
                logging.error(f"Import file not found: {zip_path}")
                return None

            # Create temporary directory for extraction
            temp_dir = os.path.join(self.base_path, "_temp_import")
            os.makedirs(temp_dir, exist_ok=True)

            try:
                # Extract zip file
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    zipf.extractall(temp_dir)

                # Find the dataset directory (should be first subdirectory)
                subdirs = [d for d in os.listdir(temp_dir)
                          if os.path.isdir(os.path.join(temp_dir, d))]
                if not subdirs:
                    raise ValueError("No dataset found in zip file")

                dataset_id = subdirs[0]
                dataset_temp_path = os.path.join(temp_dir, dataset_id)

                # Load and validate dataset info
                info_path = os.path.join(dataset_temp_path, "info.json")
                if not os.path.exists(info_path):
                    raise ValueError("Dataset info.json not found")

                with open(info_path, 'r') as f:
                    dataset_info = json.load(f)

                # Create new dataset ID to avoid conflicts
                new_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_dataset_id = f"{dataset_info['name']}_{new_timestamp}"
                new_dataset_path = os.path.join(self.base_path, new_dataset_id)

                # Move the dataset to its final location
                shutil.move(dataset_temp_path, new_dataset_path)

                # Update dataset info with new ID
                dataset_info['id'] = new_dataset_id
                dataset_info['imported'] = new_timestamp
                with open(os.path.join(new_dataset_path, "info.json"), 'w') as f:
                    json.dump(dataset_info, f, indent=2)

                logging.info(f"Dataset imported: {new_dataset_id}")
                return dataset_info

            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

        except Exception as e:
            logging.error(f"Error importing dataset: {e}")
            return None

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets"""
        try:
            datasets = []
            for dataset_id in os.listdir(self.base_path):
                info_path = os.path.join(self.base_path, dataset_id, "info.json")
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                        datasets.append(info)
            return sorted(datasets, key=lambda x: x.get('created', ''), reverse=True)
        except Exception as e:
            logging.error(f"Error listing datasets: {e}")
            return []

    def create_dataset(self, name: str, metadata: Dict = None) -> Optional[Dict]:
        """Create a new dataset"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_id = f"{name}_{timestamp}"
            dataset_path = os.path.join(self.base_path, dataset_id)

            if os.path.exists(dataset_path):
                logging.error(f"Dataset {dataset_id} already exists")
                return None

            os.makedirs(dataset_path)
            os.makedirs(os.path.join(dataset_path, "scenarios"))
            os.makedirs(os.path.join(dataset_path, "images"))

            dataset_info = {
                "id": dataset_id,
                "name": name,
                "created": timestamp,
                "metadata": metadata or {},
                "scenarios": []
            }

            with open(os.path.join(dataset_path, "info.json"), 'w') as f:
                json.dump(dataset_info, f, indent=2)

            logging.info(f"Created dataset: {dataset_id}")
            return dataset_info

        except Exception as e:
            logging.error(f"Error creating dataset: {e}")
            return None

    def load_dataset(self, dataset_id: str) -> Optional[Dict]:
        """Load a dataset by ID"""
        try:
            info_path = os.path.join(self.base_path, dataset_id, "info.json")
            if not os.path.exists(info_path):
                logging.error(f"Dataset {dataset_id} not found")
                return None

            with open(info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            return None

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset"""
        try:
            dataset_path = os.path.join(self.base_path, dataset_id)
            if not os.path.exists(dataset_path):
                logging.error(f"Dataset {dataset_id} not found")
                return False

            # Delete all files and directories
            for root, dirs, files in os.walk(dataset_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(dataset_path)

            logging.info(f"Deleted dataset: {dataset_id}")
            return True
        except Exception as e:
            logging.error(f"Error deleting dataset: {e}")
            return False

    def create_scenario(self, dataset_id: str, scenario_type: str, metadata: Dict) -> Optional[Dict]:
        """Create a new scenario in a dataset"""
        try:
            dataset = self.load_dataset(dataset_id)
            if not dataset:
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scenario_id = f"{scenario_type}_{timestamp}"
            scenario_path = os.path.join(self.base_path, dataset_id, "scenarios", scenario_id)
            os.makedirs(scenario_path)

            scenario_info = {
                "id": scenario_id,
                "type": scenario_type,
                "created": timestamp,
                "metadata": metadata,
                "photos": []
            }

            # Save scenario info
            with open(os.path.join(scenario_path, "info.json"), 'w') as f:
                json.dump(scenario_info, f, indent=2)

            # Update dataset info
            dataset["scenarios"].append(scenario_info)
            with open(os.path.join(self.base_path, dataset_id, "info.json"), 'w') as f:
                json.dump(dataset, f, indent=2)

            logging.info(f"Created scenario: {scenario_id}")
            return scenario_info

        except Exception as e:
            logging.error(f"Error creating scenario: {e}")
            return None

    def add_photo_to_scenario(self, dataset_id: str, scenario_id: str, photo_path: str) -> bool:
        """Add a photo to a scenario"""
        try:
            dataset = self.load_dataset(dataset_id)
            if not dataset:
                return False

            # Find scenario
            scenario = next(
                (s for s in dataset["scenarios"] if s["id"] == scenario_id),
                None
            )
            if not scenario:
                logging.error(f"Scenario {scenario_id} not found")
                return False

            # Copy photo to dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(photo_path)
            new_filename = f"{timestamp}_{filename}"
            new_path = os.path.join(self.base_path, dataset_id, "images", new_filename)

            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            # Copy the file
            with open(photo_path, 'rb') as src, open(new_path, 'wb') as dst:
                dst.write(src.read())

            # Update scenario info
            photo_info = {
                "filename": new_filename,
                "original_path": photo_path,
                "timestamp": timestamp
            }
            scenario["photos"].append(photo_info)

            # Save updated dataset info
            with open(os.path.join(self.base_path, dataset_id, "info.json"), 'w') as f:
                json.dump(dataset, f, indent=2)

            logging.info(f"Added photo to scenario: {new_filename}")
            return True

        except Exception as e:
            logging.error(f"Error adding photo to scenario: {e}")
            return False

    def update_scenario(self, dataset_id: str, scenario):
        """Update scenario information in dataset"""
        try:
            dataset = self.load_dataset(dataset_id)
            if not dataset:
                return False

            # Find and update the scenario
            for i, s in enumerate(dataset['scenarios']):
                if s['id'] == scenario['id']:
                    dataset['scenarios'][i] = scenario
                    break

            # Save updated dataset info
            info_path = os.path.join(self.base_path, dataset_id, "info.json")
            with open(info_path, 'w') as f:
                json.dump(dataset, f, indent=2)

            logging.info(f"Updated scenario in dataset {dataset_id}")
            return True
        except Exception as e:
            logging.error(f"Error updating scenario: {e}")
            return False