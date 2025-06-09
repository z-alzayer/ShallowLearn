"""
Simple label inspection tool with training/validation tagging and CSV export.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from typing import Dict, List, Optional
from IPython.display import display, clear_output
import ipywidgets as widgets
from pathlib import Path


class LabelInspector:
    """Simple label inspector with training/validation tagging."""
    
    def __init__(self, dataloader, rgb_bands: List[int] = [3, 2, 1]):
        """
        Initialize the label inspector.
        
        Parameters:
        -----------
        dataloader : DataLoader
            PyTorch DataLoader containing images and labels
        rgb_bands : List[int], default=[3, 2, 1]
            Band indices for RGB visualization (0-indexed)
        """
        self.dataloader = dataloader
        self.rgb_bands = rgb_bands
        self.current_idx = 0
        self.dataset_size = len(dataloader.dataset)
        
        # Track decisions for each file (lazy loading)
        self.decisions = {}  # {file_path: {'status': 'train/val/skip', 'notes': ''}}
        self.file_paths = []  # Cache file paths for navigation
        
        # Initialize file paths for decision tracking (lightweight)
        print("Initializing file paths...")
        for i in range(self.dataset_size):
            # Get file path from dataset's satellite_files list (no loading)
            sat_type, file_path = dataloader.dataset.satellite_files[i]
            self.file_paths.append(file_path)
            self.decisions[file_path] = {'status': 'undecided', 'notes': ''}
        print(f"Ready to inspect {len(self.file_paths)} samples")
        
        self.setup_widgets()
    
    def setup_widgets(self):
        """Setup interactive widgets."""
        # Navigation
        self.idx_slider = widgets.IntSlider(
            value=0, min=0, max=self.dataset_size - 1,
            description='Sample:', layout=widgets.Layout(width='400px')
        )
        
        self.prev_btn = widgets.Button(description='◀ Previous', button_style='info')
        self.next_btn = widgets.Button(description='Next ▶', button_style='info')
        
        # Decision buttons
        self.train_btn = widgets.Button(description='✓ Use for Training', button_style='success')
        self.val_btn = widgets.Button(description='✓ Use for Validation', button_style='warning')
        self.skip_btn = widgets.Button(description='✗ Skip/Bad Quality', button_style='danger')
        
        # Notes
        self.notes_text = widgets.Textarea(
            placeholder='Optional notes about this sample...',
            description='Notes:',
            layout=widgets.Layout(width='500px', height='60px')
        )
        
        # Status display
        self.status_output = widgets.Output()
        
        # Export
        self.export_btn = widgets.Button(description='💾 Export Decisions to CSV', button_style='primary')
        self.csv_filename = widgets.Text(
            value='label_decisions.csv',
            description='CSV filename:',
            layout=widgets.Layout(width='300px')
        )
        
        # Event handlers
        self.idx_slider.observe(self.on_index_change, names='value')
        self.prev_btn.on_click(lambda b: self.change_index(-1))
        self.next_btn.on_click(lambda b: self.change_index(1))
        self.train_btn.on_click(lambda b: self.mark_sample('train'))
        self.val_btn.on_click(lambda b: self.mark_sample('validation'))
        self.skip_btn.on_click(lambda b: self.mark_sample('skip'))
        self.notes_text.observe(self.on_notes_change, names='value')
        self.export_btn.on_click(self.export_decisions)
    
    def display_interface(self):
        """Display the inspection interface."""
        # Navigation controls
        nav_box = widgets.HBox([self.prev_btn, self.idx_slider, self.next_btn])
        
        # Decision controls
        decision_box = widgets.HBox([self.train_btn, self.val_btn, self.skip_btn])
        
        # Export controls
        export_box = widgets.HBox([self.csv_filename, self.export_btn])
        
        # Layout
        control_panel = widgets.VBox([
            widgets.HTML("<h3>🔍 Label Inspector</h3>"),
            nav_box,
            widgets.HTML("<br><b>Mark this sample as:</b>"),
            decision_box,
            self.notes_text,
            widgets.HTML("<br><b>Export decisions:</b>"),
            export_box,
            self.status_output
        ])
        
        display(control_panel)
        self.show_sample()
    
    def on_index_change(self, change):
        """Handle index change."""
        self.save_current_notes()  # Save notes before switching
        self.current_idx = change['new']
        self.show_sample()
    
    def change_index(self, delta):
        """Change index by delta."""
        self.save_current_notes()
        new_idx = max(0, min(self.dataset_size - 1, self.current_idx + delta))
        self.idx_slider.value = new_idx
    
    def mark_sample(self, status):
        """Mark current sample with given status."""
        file_path = self.file_paths[self.current_idx]
        
        # Update decision
        self.decisions[file_path]['status'] = status
        self.decisions[file_path]['notes'] = self.notes_text.value
        
        # Update button styles to show selection
        self.update_button_styles(status)
        
        # Auto-advance to next sample
        if self.current_idx < self.dataset_size - 1:
            self.change_index(1)
        
        self.update_status_display()
    
    def update_button_styles(self, current_status):
        """Update button styles based on current status."""
        # Reset all buttons
        self.train_btn.button_style = 'success'
        self.val_btn.button_style = 'warning'
        self.skip_btn.button_style = 'danger'
        
        # Highlight selected button
        if current_status == 'train':
            self.train_btn.button_style = 'info'
        elif current_status == 'validation':
            self.val_btn.button_style = 'info'
        elif current_status == 'skip':
            self.skip_btn.button_style = 'info'
    
    def on_notes_change(self, change):
        """Handle notes text change."""
        # Notes are saved when switching samples or marking
        pass
    
    def save_current_notes(self):
        """Save current notes for the current sample."""
        if self.file_paths and self.current_idx < len(self.file_paths):
            file_path = self.file_paths[self.current_idx]
            self.decisions[file_path]['notes'] = self.notes_text.value
    
    def show_sample(self):
        """Display current sample."""
        if not self.file_paths:
            return
        
        # Load sample on-demand (lazy loading)
        try:
            print(f"Loading sample {self.current_idx + 1}/{self.dataset_size}...")
            sample = self.dataloader.dataset[self.current_idx]
            image = sample['image']  # Shape: (C, H, W)
            file_path = sample['file_path']
        except Exception as e:
            print(f"Error loading sample {self.current_idx}: {e}")
            # Show error and allow user to skip
            plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5, f'Error loading sample {self.current_idx}\n{str(e)}', 
                    ha='center', va='center', fontsize=12, color='red')
            plt.title(f'Sample {self.current_idx + 1}/{self.dataset_size} - ERROR')
            plt.axis('off')
            plt.show()
            return
        
        # Clear previous plot
        plt.close('all')
        
        # Create RGB image for display
        rgb_image = self.create_rgb_image(image)
        
        # Setup plot
        has_labels = 'labels' in sample and sample['labels'] is not None
        
        if has_labels:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Show image
            ax1.imshow(rgb_image)
            ax1.set_title(f'Image: {Path(file_path).name}')
            ax1.axis('off')
            
            # Show labels
            labels = sample['labels'].cpu().numpy()
            im = ax2.imshow(labels, cmap='tab10')
            ax2.set_title('Labels')
            ax2.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
            ax1.imshow(rgb_image)
            ax1.set_title(f'Image: {Path(file_path).name}')
            ax1.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Load current decision and notes
        current_decision = self.decisions[file_path]
        self.notes_text.value = current_decision['notes']
        self.update_button_styles(current_decision['status'])
        self.update_status_display()
    
    def create_rgb_image(self, image_tensor):
        """Create RGB image from tensor."""
        # Convert to numpy and normalize
        img = image_tensor.cpu().numpy()
        img = np.nan_to_num(img, nan=0.0)
        
        # Normalize each band to 0-1
        normalized = np.zeros_like(img)
        for i in range(img.shape[0]):
            band = img[i]
            band_min, band_max = band.min(), band.max()
            if band_max > band_min:
                normalized[i] = (band - band_min) / (band_max - band_min)
            else:
                normalized[i] = band
        
        # Create RGB
        try:
            rgb = np.stack([
                normalized[self.rgb_bands[0]],
                normalized[self.rgb_bands[1]],
                normalized[self.rgb_bands[2]]
            ], axis=2)
        except IndexError:
            # Fallback to first 3 bands
            rgb = np.stack([
                normalized[0],
                normalized[1] if normalized.shape[0] > 1 else normalized[0],
                normalized[2] if normalized.shape[0] > 2 else normalized[0]
            ], axis=2)
        
        return np.clip(rgb, 0, 1)
    
    def update_status_display(self):
        """Update status display."""
        with self.status_output:
            clear_output(wait=True)
            
            # Count decisions
            counts = {'train': 0, 'validation': 0, 'skip': 0, 'undecided': 0}
            for decision in self.decisions.values():
                counts[decision['status']] += 1
            
            # Current sample info
            file_path = self.file_paths[self.current_idx]
            current_status = self.decisions[file_path]['status']
            
            print(f"📊 Progress: {self.current_idx + 1}/{self.dataset_size}")
            print(f"📁 Current: {Path(file_path).name}")
            print(f"🏷️  Status: {current_status}")
            print(f"🔢 Summary - Train: {counts['train']}, Val: {counts['validation']}, Skip: {counts['skip']}, Undecided: {counts['undecided']}")
    
    def export_decisions(self, button):
        """Export decisions to CSV."""
        # Prepare data
        data = []
        for i, file_path in enumerate(self.file_paths):
            decision = self.decisions[file_path]
            
            # Determine satellite type from filename
            filename = Path(file_path).name.upper()
            if any(sat in filename for sat in ["S2A", "S2B"]):
                sat_type = 'sentinel2'
            else:
                sat_type = 'landsat'
            
            data.append({
                'file_path': file_path,
                'filename': Path(file_path).name,
                'satellite_type': sat_type,
                'status': decision['status'],
                'notes': decision['notes'],
                'bands': str(self.dataloader.dataset.bands)
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        filename = self.csv_filename.value
        df.to_csv(filename, index=False)
        
        # Show summary
        with self.status_output:
            clear_output(wait=True)
            print(f"✅ Exported {len(data)} decisions to {filename}")
            print("\nSummary:")
            print(df['status'].value_counts())
            print(f"\nColumns: {list(df.columns)}")


def quick_inspect(dataloader, index: int):
    """Quick look at a single sample."""
    sample = dataloader.dataset[index]
    image = sample['image']
    
    # Create RGB
    img = image.cpu().numpy()
    img = np.nan_to_num(img, nan=0.0)
    
    # Normalize
    normalized = np.zeros_like(img)
    for i in range(img.shape[0]):
        band = img[i]
        band_min, band_max = band.min(), band.max()
        if band_max > band_min:
            normalized[i] = (band - band_min) / (band_max - band_min)
    
    # RGB (assuming bands 3,2,1 are R,G,B)
    try:
        rgb = np.stack([normalized[3], normalized[2], normalized[1]], axis=2)
    except IndexError:
        rgb = np.stack([normalized[0], normalized[0], normalized[0]], axis=2)
    
    rgb = np.clip(rgb, 0, 1)
    
    # Plot
    has_labels = 'labels' in sample
    if has_labels:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(rgb)
        ax1.set_title(f'Sample {index}')
        ax1.axis('off')
        
        ax2.imshow(sample['labels'].cpu().numpy(), cmap='tab10')
        ax2.set_title('Labels')
        ax2.axis('off')
    else:
        plt.figure(figsize=(6, 4))
        plt.imshow(rgb)
        plt.title(f'Sample {index} (No labels)')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"File: {Path(sample['file_path']).name}")
    print(f"Satellite: {sample['satellite_type']}")
    print(f"Shape: {image.shape}")