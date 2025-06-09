"""
Simple fallback label inspector for when ipywidgets has issues.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path


class SimpleLabelInspector:
    """Simple command-line label inspector."""
    
    def __init__(self, dataloader, rgb_bands: List[int] = [3, 2, 1]):
        self.dataloader = dataloader
        self.rgb_bands = rgb_bands
        self.current_idx = 0
        self.dataset_size = len(dataloader.dataset)
        
        # Track decisions
        self.decisions = {}
        self.file_paths = []
        
        # Initialize file paths
        print("Initializing file paths...")
        for i in range(self.dataset_size):
            sat_type, file_path = dataloader.dataset.satellite_files[i]
            self.file_paths.append(file_path)
            self.decisions[file_path] = {'status': 'undecided', 'notes': ''}
        print(f"Ready to inspect {len(self.file_paths)} samples")
    
    def show_current_sample(self):
        """Show current sample with matplotlib."""
        try:
            print(f"\nLoading sample {self.current_idx + 1}/{self.dataset_size}...")
            sample = self.dataloader.dataset[self.current_idx]
            
            image = sample['image']
            file_path = sample['file_path']
            
            # Create RGB image
            rgb_image = self.create_rgb_image(image)
            
            # Clear previous plots
            plt.close('all')
            
            # Setup plot
            has_labels = 'labels' in sample and sample['labels'] is not None
            
            if has_labels:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Show image
                ax1.imshow(rgb_image)
                ax1.set_title(f'Sample {self.current_idx + 1}: {Path(file_path).name}')
                ax1.axis('off')
                
                # Show labels
                labels = sample['labels'].cpu().numpy()
                im = ax2.imshow(labels, cmap='tab10')
                ax2.set_title('Labels')
                ax2.axis('off')
                plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
                ax1.imshow(rgb_image)
                ax1.set_title(f'Sample {self.current_idx + 1}: {Path(file_path).name}')
                ax1.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Show current status
            current_decision = self.decisions[file_path]
            print(f"📁 File: {Path(file_path).name}")
            print(f"🏷️  Current status: {current_decision['status']}")
            print(f"📝 Notes: {current_decision['notes']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading sample {self.current_idx}: {e}")
            return False
    
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
    
    def run_interactive(self):
        """Run interactive inspection loop."""
        print("\n" + "="*60)
        print("🔍 SIMPLE LABEL INSPECTOR")
        print("="*60)
        print("Commands:")
        print("  'n' or 'next'     - Next sample")
        print("  'p' or 'prev'     - Previous sample")
        print("  'g <num>'         - Go to sample number")
        print("  't' or 'train'    - Mark as training")
        print("  'v' or 'val'      - Mark as validation")
        print("  's' or 'skip'     - Mark as skip")
        print("  'note <text>'     - Add note")
        print("  'status'          - Show summary")
        print("  'export'          - Export to CSV")
        print("  'q' or 'quit'     - Quit")
        print("="*60)
        
        # Show first sample
        self.show_current_sample()
        
        while True:
            try:
                command = input(f"\n[{self.current_idx + 1}/{self.dataset_size}] Command: ").strip().lower()
                
                if command in ['q', 'quit']:
                    break
                    
                elif command in ['n', 'next']:
                    if self.current_idx < self.dataset_size - 1:
                        self.current_idx += 1
                        self.show_current_sample()
                    else:
                        print("Already at last sample")
                        
                elif command in ['p', 'prev']:
                    if self.current_idx > 0:
                        self.current_idx -= 1
                        self.show_current_sample()
                    else:
                        print("Already at first sample")
                        
                elif command.startswith('g '):
                    try:
                        target = int(command.split()[1]) - 1  # Convert to 0-indexed
                        if 0 <= target < self.dataset_size:
                            self.current_idx = target
                            self.show_current_sample()
                        else:
                            print(f"Invalid sample number. Range: 1-{self.dataset_size}")
                    except (ValueError, IndexError):
                        print("Usage: g <number>")
                        
                elif command in ['t', 'train']:
                    self.mark_current('train')
                    
                elif command in ['v', 'val']:
                    self.mark_current('validation')
                    
                elif command in ['s', 'skip']:
                    self.mark_current('skip')
                    
                elif command.startswith('note '):
                    note_text = command[5:]  # Remove 'note '
                    file_path = self.file_paths[self.current_idx]
                    self.decisions[file_path]['notes'] = note_text
                    print(f"✅ Note added: {note_text}")
                    
                elif command == 'status':
                    self.show_status()
                    
                elif command == 'export':
                    self.export_csv()
                    
                else:
                    print("Unknown command. Type 'q' to quit.")
                    
            except KeyboardInterrupt:
                print("\nQuitting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def mark_current(self, status):
        """Mark current sample with status."""
        file_path = self.file_paths[self.current_idx]
        self.decisions[file_path]['status'] = status
        print(f"✅ Marked as: {status}")
        
        # Auto-advance
        if self.current_idx < self.dataset_size - 1:
            self.current_idx += 1
            self.show_current_sample()
        else:
            print("Last sample reached!")
    
    def show_status(self):
        """Show summary of decisions."""
        counts = {'train': 0, 'validation': 0, 'skip': 0, 'undecided': 0}
        for decision in self.decisions.values():
            counts[decision['status']] += 1
        
        print(f"\n📊 SUMMARY:")
        print(f"   Training: {counts['train']}")
        print(f"   Validation: {counts['validation']}")
        print(f"   Skip: {counts['skip']}")
        print(f"   Undecided: {counts['undecided']}")
        print(f"   Total: {sum(counts.values())}")
    
    def export_csv(self, filename="label_decisions.csv"):
        """Export decisions to CSV."""
        data = []
        for file_path in self.file_paths:
            decision = self.decisions[file_path]
            
            # Determine satellite type
            fname = Path(file_path).name.upper()
            sat_type = 'sentinel2' if any(s in fname for s in ["S2A", "S2B"]) else 'landsat'
            
            data.append({
                'file_path': file_path,
                'filename': Path(file_path).name,
                'satellite_type': sat_type,
                'status': decision['status'],
                'notes': decision['notes'],
                'bands': str(self.dataloader.dataset.bands)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        
        print(f"✅ Exported {len(data)} decisions to {filename}")
        print(f"   Training: {(df['status'] == 'train').sum()}")
        print(f"   Validation: {(df['status'] == 'validation').sum()}")
        print(f"   Skip: {(df['status'] == 'skip').sum()}")


def test_widgets():
    """Test if ipywidgets is working properly."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
        
        print("Testing widget functionality...")
        
        # Test basic widget
        test_btn = widgets.Button(description="Test Button")
        test_output = widgets.Output()
        
        def on_click(b):
            with test_output:
                print("✅ Button clicked! Widgets are working.")
        
        test_btn.on_click(on_click)
        
        display(widgets.VBox([
            widgets.HTML("<h4>Widget Test</h4>"),
            test_btn,
            test_output
        ]))
        
        return True
        
    except Exception as e:
        print(f"❌ Widget test failed: {e}")
        return False


if __name__ == "__main__":
    print("Usage:")
    print("""
    # Try the widget version first
    if test_widgets():
        from ShallowLearn.label_inspector import LabelInspector
        inspector = LabelInspector(dataloader)
        inspector.display_interface()
    else:
        # Fallback to simple version
        from simple_inspector import SimpleLabelInspector
        inspector = SimpleLabelInspector(dataloader)
        inspector.run_interactive()
    """)