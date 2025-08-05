import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import cairosvg
import os
from xml.dom import minidom


def parse_length(length_str):
    import re
    match = re.fullmatch(r"([0-9.]+)([a-z%]*)", length_str.strip())
    if not match:
        raise ValueError(f"Invalid length: {length_str}")
    value, unit = match.groups()
    value = float(value)
    unit = unit.lower()

    # Convert units to pixels
    if unit == "":  # assume pixels
        return value
    elif unit == "px":
        return value
    elif unit == "pt":
        return value * 1.3333  # 1 pt = 1.3333 px
    elif unit == "mm":
        return value * 3.7795
    elif unit == "cm":
        return value * 37.795
    elif unit == "in":
        return value * 96
    else:
        raise ValueError(f"Unsupported unit: {unit}")


class SVGConverterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SVG to PNG Converter")
        self.geometry("600x400")
        ctk.set_appearance_mode("Dark")

        # State
        self.svg_path = None
        self.native_width = None
        self.native_height = None

        # File selection
        self.file_btn = ctk.CTkButton(self, text="Select SVG File", command=self.select_file)
        self.file_btn.pack(pady=10)

        # Info label
        self.info_label = ctk.CTkLabel(self, text="No file selected")
        self.info_label.pack(pady=5)

        # Scale slider
        self.scale_label = ctk.CTkLabel(self, text="Scale: 1.0x")
        self.scale_label.pack(pady=5)
        self.scale_slider = ctk.CTkSlider(self, from_=0.1, to=5.0, number_of_steps=49, command=self.update_scale_label)
        self.scale_slider.set(1.0)
        self.scale_slider.pack(pady=5)

        # DPI options
        self.dpi_label = ctk.CTkLabel(self, text="DPI")
        self.dpi_label.pack(pady=5)
        self.dpi_dropdown = ctk.CTkOptionMenu(self, values=["72 (Screen)", "150", "300 (Print)", "600"], command=None)
        self.dpi_dropdown.set("300 (Print)")
        self.dpi_dropdown.pack(pady=5)

        # Background color
        self.bg_label = ctk.CTkLabel(self, text="Background")
        self.bg_label.pack(pady=5)
        self.bg_dropdown = ctk.CTkOptionMenu(self, values=["transparent", "white", "black"])
        self.bg_dropdown.set("transparent")
        self.bg_dropdown.pack(pady=5)

        # Export button
        self.export_btn = ctk.CTkButton(self, text="Export PNG", command=self.export_png)
        self.export_btn.pack(pady=20)

    def update_scale_label(self, value):
        self.scale_label.configure(text=f"Scale: {float(value):.1f}x")

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("SVG files", "*.svg")])
        if path:
            self.svg_path = path
            self.info_label.configure(text=os.path.basename(path))
            self.get_svg_dimensions(path)

    def get_svg_dimensions(self, path):
        try:
            doc = minidom.parse(path)
            svg = doc.getElementsByTagName("svg")[0]
            width_str = svg.getAttribute("width")
            height_str = svg.getAttribute("height")

            self.native_width = parse_length(width_str)
            self.native_height = parse_length(height_str)
            doc.unlink()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse SVG dimensions: {e}")

    def export_png(self):
        if not self.svg_path:
            messagebox.showwarning("No file", "Please select an SVG file first.")
            return

        scale = self.scale_slider.get()
        dpi_str = self.dpi_dropdown.get().split(" ")[0]
        dpi = int(dpi_str)
        bg = self.bg_dropdown.get()
        output_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if not output_path:
            return

        try:
            width = int(self.native_width * scale)
            height = int(self.native_height * scale)

            cairosvg.svg2png(
                url=self.svg_path,
                write_to=output_path,
                output_width=width,
                output_height=height,
                dpi=dpi,
                background_color=None if bg == "transparent" else bg
            )
            messagebox.showinfo("Success", f"PNG exported successfully to:\n{output_path}")

            # Reset application state
            self.svg_path = None
            self.native_width = None
            self.native_height = None
            self.info_label.configure(text="No file selected")
            self.scale_slider.set(1.0)
            self.scale_label.configure(text="Scale: 1.0x")
            self.dpi_dropdown.set("300 (Print)")
            self.bg_dropdown.set("transparent")

            # Prompt user to upload another file or close
            response = messagebox.askyesno("Continue?", "Would you like to convert another SVG file?")
            if response:
                self.select_file()
            else:
                self.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Conversion failed: {e}")


if __name__ == "__main__":
    app = SVGConverterApp()
    app.mainloop()