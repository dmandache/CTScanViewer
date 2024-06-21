import warnings
warnings.filterwarnings('ignore')  # Ignore warnings

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive_output
from IPython.display import display, clear_output
import time

# List of DICOM tags to display
DICOM_TAGS_TO_DISPLAY = ['patient_id', 'date']

def load_nifti_ras(file_path):
    """Load a NIfTI file and return the image data oriented in RAS+."""
    img = nib.load(file_path)
    
    # Get the image data and affine matrix
    data = img.get_fdata()
    affine = img.affine
    
    # Determine the current orientation
    current_ornt = nib.orientations.io_orientation(affine)
    
    # Define the target orientation (RAS+)
    ras_ornt = np.array([[0, 1], [1, 1], [2, 1]])
    
    # Calculate the transformation to the target orientation
    transform = nib.orientations.ornt_transform(current_ornt, ras_ornt)
    
    # Apply the transformation to the data
    return nib.orientations.apply_orientation(data, transform)

def clip_hu_values(ct_scan, min_hu, max_hu):
    """Clip the Hounsfield Unit (HU) values of the CT scan."""
    ct_scan = np.clip(ct_scan, min_hu, max_hu)
    return ct_scan

class CTScanViewer:
    def __init__(self, df, ct_scan_col, segmentation_col, HU_min=-100, HU_max=400):
        self.df = df  # DataFrame containing scan data
        self.ct_scan_col = ct_scan_col  # Column name for CT scan file paths
        self.segmentation_col = segmentation_col  # Column name for segmentation file paths
        self.HU_min = HU_min  # Minimum HU value for clipping
        self.HU_max = HU_max  # Maximum HU value for clipping
        self.current_index = 0  # Index of the current scan
        self.view_plane = 'axial'  # Initial view plane
        self.slice_idx = 0  # Index of displayed slice
        self.ct_scan = np.zeros([2, 2, 2]) # Initialize 3D array for CT scan
        self.segmentation = np.zeros([2, 2, 2]) # Initialize 3D array for segmentation
        
        self.init_widgets()  # Initialize widgets
        self.load_data()  # Load the initial scan data
                           
    def init_widgets(self):
        """Initialize interactive widgets."""
        self.slice_slider = widgets.IntSlider(
            min=0, max=100, step=1, value=0, description='Slice ', layout=widgets.Layout(width='600px'))
        self.slice_slider.observe(self.on_slice_change, names='value')  # Update slice on slider change
        
        self.alpha_slider = widgets.FloatSlider(value=0.3, min=0, max=1, step=0.1, description='α', orientation='vertical')
               
        self.plane_selector = widgets.ToggleButtons(
            options=['axial', 'sagittal', 'coronal'], description='Plane ')
        self.plane_selector.observe(self.on_plane_change, names='value')  # Update plane on selection change
        
        self.next_button = widgets.Button(description="Next")
        self.next_button.layout.object_position = 'right'
        self.next_button.on_click(self.on_next)  # Load next scan on button click
        
        self.progress_bar = widgets.FloatProgress(
            value=0, min=0, max=1, description='Loading:', bar_style='info')
                
        self.info_display = widgets.HTML(value="")  # HTML widget to display scan info

        ui_top = widgets.VBox([self.plane_selector, self.slice_slider])  # Top UI elements
        out = widgets.interactive_output(self.update_display, {'slice_idx': self.slice_slider, 'view_plane': self.plane_selector, 'alpha': self.alpha_slider})
        ui_bot = widgets.HBox([out, self.alpha_slider, self.info_display, self.next_button, self.progress_bar])  # Bottom UI elements
        
        display(ui_top, ui_bot)  # Display the widgets
        
    def load_data(self):
        """Load CT scan and segmentation data."""
        self.progress_bar.layout.visibility = 'visible'
        self.progress_bar.value = 0
        self.progress_bar.bar_style = 'info'
        self.progress_bar.description = 'Loading...'
        
        row = self.df.iloc[self.current_index]  # Get the current scan data
        
        self.progress_bar.value = 0.1
        self.ct_scan = load_nifti_ras(row[self.ct_scan_col])  # Load CT scan
        self.progress_bar.value = 0.4
        self.ct_scan = clip_hu_values(self.ct_scan, self.HU_min, self.HU_max)  # Clip HU values
        self.progress_bar.value = 0.6
        self.segmentation = load_nifti_ras(row[self.segmentation_col])  # Load segmentation
        self.progress_bar.value = 0.8
        
        self.update_info_display()  # Update scan info display
        self.update_slice_slider()  # Update the slice slider
        
        self.progress_bar.value = 1
        self.progress_bar.bar_style = 'success'
        self.progress_bar.description = 'Loaded'
        time.sleep(0.5)
        self.progress_bar.layout.visibility = 'hidden'

    def update_slice_slider(self):
        """Update the slice slider based on the selected view plane."""
        if self.view_plane == 'axial':
            self.num_slices = self.ct_scan.shape[2]
            self.slice_idx = np.argmax(np.sum(self.segmentation, axis=(0, 1)))
        elif self.view_plane == 'sagittal':
            self.num_slices = self.ct_scan.shape[0]
            self.slice_idx = np.argmax(np.sum(self.segmentation, axis=(1, 2)))
        elif self.view_plane == 'coronal':
            self.num_slices = self.ct_scan.shape[1]
            self.slice_idx = np.argmax(np.sum(self.segmentation, axis=(0, 2)))
            
        self.slice_slider.max = self.num_slices - 1
        self.slice_slider.value = self.slice_idx

    def update_display(self, slice_idx, view_plane, alpha=0.5):
        """Update the CT scan display based on the selected slice and view plane."""
        self.view_plane = view_plane
        
        if view_plane == 'axial':
            ct_slice = self.ct_scan[:, :, slice_idx]
            seg_slice = self.segmentation[:, :, slice_idx]
        elif view_plane == 'sagittal':
            ct_slice = self.ct_scan[slice_idx, :, :]
            seg_slice = self.segmentation[slice_idx, :, :]
        elif view_plane == 'coronal':
            ct_slice = self.ct_scan[:, slice_idx, :]
            seg_slice = self.segmentation[:, slice_idx, :]
        
        plt.figure(figsize=(8, 8))
        plt.imshow(ct_slice.T, cmap='gray', origin='lower')
        plt.imshow(np.ma.masked_where(seg_slice == 0, seg_slice).T, cmap='jet', alpha=alpha, origin='lower')
        plt.contour(seg_slice.T, colors='red', linewidths=0.5, alpha=alpha, origin='lower')
        plt.show()
        
    def update_info_display(self):
        """Update the scan info display."""
        row = self.df.iloc[self.current_index]
        
        info = f"<b>Scan Info:</b><br>"
        for column in row.index:
            if column in DICOM_TAGS_TO_DISPLAY:
                info += f"<b>{column}:</b> {row[column]}<br>"
        self.info_display.value = info
        
    def on_slice_change(self, change):
        """Handle slice slider change event."""
        self.slice_ix = self.slice_slider.value

    def on_plane_change(self, change):
        """Handle view plane change event."""
        self.view_plane = self.plane_selector.value  # Update view plane
        self.update_slice_slider()  # Update the slice slider
        
    def on_next(self, button):
        """Handle next button click event."""
        self.current_index = (self.current_index + 1) % len(self.df)  # Increment scan index
        self.load_data()  # Load the next scan data
