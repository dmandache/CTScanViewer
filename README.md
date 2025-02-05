# CTScanViewer

## Overview

CTScanViewer is a Jupyter Notebook widget for visualizing 3D CT scans along with their organ segmentations. The widget allows users to navigate through different slices, adjust the transparency of the segmentation overlay, and switch between different viewing planes (axial, sagittal, coronal). It also provides a progress bar to indicate loading status and displays metadata for each scan.

![preview ux]((https://github.com/dmandache/CTScanViewer/blob/main/preview.png)

## Features

- **Slice Navigation**: Navigate through CT scan slices using a slider.
- **Transparency Adjustment**: Adjust the transparency of the segmentation overlay with a slider.
- **Plane Selection**: Choose the viewing plane (axial, sagittal, coronal) using a dropdown menu.
- **Progress Bar**: Visual indication of the loading process.
- **Metadata Display**: View additional information about the CT scan and segmentation.
- **Next Button**: Easily switch to the next scan in the dataframe.

## Requirements

- Python 3.x
- Jupyter Notebook or JupyterLab
- nibabel
- numpy
- matplotlib
- ipywidgets
- pandas

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/CTScanViewer.git
cd CTScanViewer
```

2. Install the required Python packages:

```bash
pip install nibabel numpy matplotlib ipywidgets pandas
```

## Usage

1. Prepare a dataframe containing paths to the CT scans and segmentations, along with any additional metadata:

```python
import pandas as pd

df = pd.DataFrame({
    'ct_scan_path': ['path_to_ct_scan1.nii', 'path_to_ct_scan2.nii'],
    'segmentation_path': ['path_to_segmentation1.nii', 'path_to_segmentation2.nii'],
    'patient_id': [1, 2],
    'age': [65, 70],
    'sex': ['M', 'F']
})
```

2. Create an instance of the `CTScanViewer` with your dataframe:

```python
viewer = CTScanViewer(df)
```

3. Run the code in a Jupyter Notebook cell to display the interactive widget.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## Acknowledgments

- [nibabel](https://nipy.org/nibabel/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [ipywidgets](https://ipywidgets.readthedocs.io/)
- [pandas](https://pandas.pydata.org/)
