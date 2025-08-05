# SVG to PNG Converter

## Overview
The SVG to PNG Converter is a Python-based desktop application with a graphical user interface (GUI) that allows users to convert SVG (Scalable Vector Graphics) files to PNG (Portable Network Graphics) format. The application provides options to adjust the scale, DPI, and background colour of the output PNG, making it suitable for both screen and print purposes. Built using `customtkinter` for the GUI and `cairosvg` for conversion, it offers a user-friendly experience with error handling and dynamic feedback.

## Features
- **File Selection**: Select an SVG file using a file dialog.
- **Scale Adjustment**: Adjust the output size with a slider (0.1x to 5.0x).
- **DPI Settings**: Choose from predefined DPI options (72, 150, 300, 600) for screen or print quality.
- **Background colour**: Select transparent, white, or black background for the PNG.
- **Dimension Parsing**: Automatically reads SVG dimensions, supporting units like px, pt, mm, cm, and in.
- **Reset and Continue**: After conversion, resets the interface and prompts the user to either convert another file or exit.
- **Error Handling**: Displays informative messages for invalid files, missing selections, or conversion errors.

## Requirements
- Python 3.6 or higher
- Required Python packages:
  - `customtkinter` (for the GUI)
  - `cairosvg` (for SVG to PNG conversion)
- The `xml.dom.minidom` module is included in Python's standard library.

## Installation
1. **Clone or Download the Project**:
   - Clone the repository or download the `main.py` file to your local machine.
   ```bash
   git clone <https://github.com/gappeah/SVG2PNG.git>
   ```

2. **Install Dependencies**:
   - Install the required Python packages using pip:
   ```bash
   pip install customtkinter cairosvg
   ```

3. **Ensure Python is Installed**:
   - Verify that Python 3.6+ is installed:
   ```bash
   python --version
   ```

## Usage
1. **Run the Application**:
   - Navigate to the project directory and run the script:
   ```bash
   python main.py
   ```

2. **Interface Overview**:
   - The application opens a window with a dark theme and a 600x400 resolution.
   - **Select SVG File**: Click the "Select SVG File" button to choose an SVG file.
   - **Adjust Settings**:
     - Use the scale slider to adjust the output size (default: 1.0x).
     - Select DPI from the dropdown (default: 300 for print).
     - Choose a background colour (default: transparent).
   - **Export PNG**: Click the "Export PNG" button, choose an output location, and save the PNG file.

3. **Post-Conversion**:
   - After a successful conversion, a success message displays the output path.
   - The interface resets, and a prompt asks if you want to convert another file:
     - **Yes**: Opens the file selection dialog for another SVG.
     - **No**: Closes the application.

## Code Structure
- **main.py**:
  - Contains the `SVGConverterApp` class, which inherits from `customtkinter.CTk`.
  - Implements the GUI with buttons, sliders, dropdowns, and labels.
  - Handles file selection, SVG dimension parsing, and PNG conversion.
  - Includes a `parse_length` function to convert SVG units (px, pt, mm, cm, in) to pixels.
  - Manages state reset and user prompts after conversion.

## Limitations
- The application assumes SVG files have valid `width` and `height` attributes.
- Unsupported SVG units (e.g., percentages) may cause parsing errors.
- Requires a working installation of `cairosvg`, which may have system-specific dependencies (e.g., Cairo libraries).

## Troubleshooting
- **Error: "No file selected"**: Ensure an SVG file is selected before clicking "Export PNG".
- **Error: "Failed to parse SVG dimensions"**: Verify the SVG file has valid `width` and `height` attributes in supported units.
- **Error: "Conversion failed"**: Check if `cairosvg` is properly installed and the SVG file is not corrupted.
- **Missing Dependencies**: Run `pip install customtkinter cairosvg` to install required packages.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with [customtkinter](https://github.com/TomSchimansky/CustomTkinter) for the modern GUI.
- Uses [cairosvg](https://cairosvg.org/) for SVG to PNG conversion.