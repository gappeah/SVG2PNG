"""
Raster and SVG Converter + Vectorizer
- Supports opening SVG, PNG, JPG/JPEG
- Preview raster images
- Vectorize raster images via color quantization + contour tracing or grayscale thresholding
- Preview the vectorized result before saving
- Export final SVG and (for SVG inputs) render to PNG

Dependencies:
pip install pillow numpy opencv-python svgwrite cairosvg customtkinter

"""

import os
import io
import math
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
import svgwrite
import cairosvg


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


class RasterVectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Raster & SVG Vectorizer")
        self.geometry("1000x700")
        ctk.set_appearance_mode("Dark")

        # State
        self.input_path = None
        self.pil_image = None  # original raster preview (PIL Image)
        self.preview_photo = None
        self.svg_width = None
        self.svg_height = None
        self.last_svg = None  # svgwrite.Drawing instance

        # Layout: left controls, right preview
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        controls = ctk.CTkFrame(self, width=320)
        controls.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)

        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)

        # Controls
        self.select_btn = ctk.CTkButton(controls, text="Select Image or SVG", command=self.select_file)
        self.select_btn.pack(pady=(12, 6), padx=12, fill="x")

        self.info_label = ctk.CTkLabel(controls, text="No file selected")
        self.info_label.pack(pady=6, padx=12)

        # Vectorization mode
        ctk.CTkLabel(controls, text="Vectorization Mode:").pack(pady=(8, 2), padx=12, anchor="w")
        self.mode_option = ctk.CTkOptionMenu(controls, values=["Color Quantization + Contours", "Grayscale Threshold + Contours"], command=None)
        self.mode_option.set("Color Quantization + Contours")
        self.mode_option.pack(pady=6, padx=12, fill="x")

        # Number of colors
        ctk.CTkLabel(controls, text="Number of colors (k)").pack(pady=(8, 2), padx=12, anchor="w")
        self.k_slider = ctk.CTkSlider(controls, from_=2, to=16, number_of_steps=14, command=self._update_k_label)
        self.k_slider.set(6)
        self.k_slider.pack(pady=6, padx=12, fill="x")
        self.k_label = ctk.CTkLabel(controls, text="k = 6")
        self.k_label.pack(pady=(0,8), padx=12)

        # Threshold
        ctk.CTkLabel(controls, text="Threshold (for grayscale mode)").pack(pady=(8, 2), padx=12, anchor="w")
        self.threshold_slider = ctk.CTkSlider(controls, from_=0, to=255, number_of_steps=255, command=self._update_threshold_label)
        self.threshold_slider.set(128)
        self.threshold_slider.pack(pady=6, padx=12, fill="x")
        self.threshold_label = ctk.CTkLabel(controls, text="threshold = 128")
        self.threshold_label.pack(pady=(0,8), padx=12)

        # Fill color mode
        ctk.CTkLabel(controls, text="Fill mode for vector shapes").pack(pady=(8, 2), padx=12, anchor="w")
        self.fill_option = ctk.CTkOptionMenu(controls, values=["Original colors", "Single chosen color"], command=None)
        self.fill_option.set("Original colors")
        self.fill_option.pack(pady=6, padx=12, fill="x")

        self.color_btn = ctk.CTkButton(controls, text="Choose Fill Color", command=self.choose_color)
        self.color_btn.pack(pady=6, padx=12, fill="x")
        self.chosen_color = "#000000"
        self.color_preview = ctk.CTkLabel(controls, text=f"Chosen: {self.chosen_color}")
        self.color_preview.pack(pady=(0,8), padx=12)

        # Smoothing and min area
        ctk.CTkLabel(controls, text="Contour smoothing (approx poly epsilon factor)").pack(pady=(8,2), padx=12, anchor="w")
        self.epsilon_slider = ctk.CTkSlider(controls, from_=0.0, to=5.0, number_of_steps=50, command=self._update_epsilon_label)
        self.epsilon_slider.set(1.0)
        self.epsilon_slider.pack(pady=6, padx=12, fill="x")
        self.epsilon_label = ctk.CTkLabel(controls, text="epsilon factor = 1.0")
        self.epsilon_label.pack(pady=(0,8), padx=12)

        ctk.CTkLabel(controls, text="Minimum contour area (pixels)").pack(pady=(8,2), padx=12, anchor="w")
        self.min_area_entry = ctk.CTkEntry(controls)
        self.min_area_entry.insert(0, "100")
        self.min_area_entry.pack(pady=6, padx=12, fill="x")

        # Buttons: Preview vectorize, Confirm save, Export PNG (for SVG input), Reset
        btn_frame = ctk.CTkFrame(controls)
        btn_frame.pack(pady=12, padx=12, fill="x")

        self.preview_vec_btn = ctk.CTkButton(btn_frame, text="Preview Vectorize", command=self.preview_vectorize)
        self.preview_vec_btn.pack(side="left", expand=True, padx=6, pady=6)

        self.save_vec_btn = ctk.CTkButton(btn_frame, text="Confirm & Save SVG", command=self.save_svg)
        self.save_vec_btn.pack(side="left", expand=True, padx=6, pady=6)

        self.export_png_btn = ctk.CTkButton(controls, text="Export SVG to PNG (if SVG loaded)", command=self.export_png)
        self.export_png_btn.pack(pady=6, padx=12, fill="x")

        self.reset_btn = ctk.CTkButton(controls, text="Reset", command=self.reset_all)
        self.reset_btn.pack(pady=6, padx=12, fill="x")

        # Preview canvas
        self.canvas = tk.Canvas(self.preview_frame, bg="#222222")
        self.canvas.grid(row=0, column=0, sticky="nswe")

        # Status bar
        self.status = ctk.CTkLabel(self, text="Ready", anchor="w")
        self.status.grid(row=1, column=0, columnspan=2, sticky="we", padx=10, pady=(0,8))

    def _update_k_label(self, v):
        self.k_label.configure(text=f"k = {int(float(v))}")

    def _update_threshold_label(self, v):
        self.threshold_label.configure(text=f"threshold = {int(float(v))}")

    def _update_epsilon_label(self, v):
        self.epsilon_label.configure(text=f"epsilon factor = {float(v):.2f}")

    def choose_color(self):
        c = colorchooser.askcolor(title="Choose fill color")
        if c and c[1]:
            self.chosen_color = c[1]
            self.color_preview.configure(text=f"Chosen: {self.chosen_color}")

    def set_status(self, text):
        self.status.configure(text=text)
        self.update()

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Images and SVG", "*.png *.jpg *.jpeg *.svg")])
        if not path:
            return
        self.input_path = path
        self.info_label.configure(text=os.path.basename(path))
        self.set_status(f"Loaded {os.path.basename(path)}")

        try:
            self.load_image(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def load_image(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".svg":
            # Render SVG to PNG bytes via cairosvg then open with PIL
            png_bytes = cairosvg.svg2png(url=path)
            pil = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            # try to read svg width/height attributes
            try:
                from xml.dom import minidom
                doc = minidom.parse(path)
                svg = doc.getElementsByTagName('svg')[0]
                w = svg.getAttribute('width')
                h = svg.getAttribute('height')
                if w and h:
                    self.svg_width = int(parse_length(w))
                    self.svg_height = int(parse_length(h))
                doc.unlink()
            except Exception:
                self.svg_width = pil.width
                self.svg_height = pil.height

            self.pil_image = pil
        else:
            pil = Image.open(path).convert("RGBA")
            self.pil_image = pil
            self.svg_width = pil.width
            self.svg_height = pil.height

        self.show_preview(self.pil_image)

    def show_preview(self, pil_img):
        # Fit into canvas
        cw = self.preview_frame.winfo_width() or 600
        ch = self.preview_frame.winfo_height() or 600
        max_w = max(200, cw - 20)
        max_h = max(200, ch - 20)
        img = pil_img.copy()
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(max_w//2+10, max_h//2+10, image=self.preview_photo)

    def preview_vectorize(self):
        if not self.pil_image:
            messagebox.showwarning("No image", "Please select an image first.")
            return

        mode = self.mode_option.get()
        k = int(self.k_slider.get())
        threshold = int(self.threshold_slider.get())
        epsilon_factor = float(self.epsilon_slider.get())
        min_area = int(self.min_area_entry.get() or 100)
        fill_single = (self.fill_option.get() == "Single chosen color")
        chosen_color = self.chosen_color

        self.set_status("Vectorizing (preview)...")
        try:
            svg = self._vectorize_image(self.pil_image, mode, k, threshold, epsilon_factor, min_area, fill_single, chosen_color)
            self.last_svg = svg
            # Render svg to PNG for preview
            svg_str = svg.tostring()
            png_bytes = cairosvg.svg2png(bytestring=svg_str)
            preview_pil = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            self.show_preview(preview_pil)
            self.set_status("Preview ready — review and click 'Confirm & Save SVG' to export")
        except Exception as e:
            messagebox.showerror("Vectorization error", f"Failed to vectorize: {e}")
            self.set_status("Error during vectorization")

    def _vectorize_image(self, pil_img, mode, k, threshold, epsilon_factor, min_area, fill_single, chosen_color):
        # Convert to numpy array (BGRA) for OpenCV
        img = np.array(pil_img)
        h, w = img.shape[:2]
        # Work on a smaller version if extremely large to speed up processing
        max_dim = 1024
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        # Convert from RGBA to RGB for processing
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        contours_by_color = []  # list of tuples (fill_color_hex, list_of_contours)

        if mode == "Color Quantization + Contours":
            # k-means color quantization
            Z = rgb.reshape((-1, 3)).astype(np.float32)
            # criteria and apply kmeans
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
            K = max(1, min(32, k))
            _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = centers.astype('uint8')
            labels = labels.flatten()
            quant = centers[labels].reshape((h, w, 3))

            # for each center color create a mask and find contours
            for i, c in enumerate(centers):
                mask = (labels.reshape((h, w)) == i).astype('uint8') * 255
                # find contours
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                good_cnts = []
                for cnt in cnts:
                    area = cv2.contourArea(cnt)
                    if area < min_area:
                        continue
                    # approximate to reduce points
                    peri = cv2.arcLength(cnt, True)
                    eps = epsilon_factor * 0.01 * peri
                    approx = cv2.approxPolyDP(cnt, eps, True)
                    # scale back to original image size if we downscaled
                    if scale != 1.0:
                        approx = (approx.astype('float32') / scale).astype('int')
                    good_cnts.append(approx)
                if good_cnts:
                    hexc = '#%02x%02x%02x' % (int(c[0]), int(c[1]), int(c[2]))
                    contours_by_color.append((hexc, good_cnts))

        else:
            # Grayscale thresholding
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            good_cnts = []
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                peri = cv2.arcLength(cnt, True)
                eps = epsilon_factor * 0.01 * peri
                approx = cv2.approxPolyDP(cnt, eps, True)
                if scale != 1.0:
                    approx = (approx.astype('float32') / scale).astype('int')
                good_cnts.append(approx)
            sel_color = chosen_color if fill_single else '#000000'
            if good_cnts:
                contours_by_color.append((sel_color, good_cnts))

        # Create SVG
        dwg = svgwrite.Drawing(size=(f"{self.svg_width or w}px", f"{self.svg_height or h}px"))
        # Set viewBox to real pixel dims
        dwg.viewbox(0, 0, self.svg_width or w, self.svg_height or h)

        # Background: transparent by default
        for fill_color, cnts in contours_by_color:
            for cnt in cnts:
                # cnt is Nx1x2
                pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                if len(pts) < 3:
                    continue
                # svgwrite uses lists of tuples
                if self.fill_option.get() == "Single chosen color":
                    fillc = chosen_color
                else:
                    fillc = fill_color
                # Create polygon
                try:
                    dwg.add(dwg.polygon(points=pts, fill=fillc, stroke='none'))
                except Exception:
                    # fallback: create polyline + close
                    dwg.add(dwg.polyline(points=pts, fill=fillc, stroke='none'))

        return dwg

    def save_svg(self):
        if not self.last_svg:
            messagebox.showwarning("No SVG", "There is no generated SVG to save. Please press 'Preview Vectorize' first.")
            return
        path = filedialog.asksaveasfilename(defaultextension='.svg', filetypes=[('SVG files', '*.svg')])
        if not path:
            return
        try:
            self.last_svg.saveas(path)
            messagebox.showinfo("Saved", f"SVG saved to: {path}")
            self.set_status(f"Saved SVG: {path}")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save SVG: {e}")

    def export_png(self):
        # If an SVG file is loaded or last_svg exists, allow exporting to PNG
        if self.input_path and self.input_path.lower().endswith('.svg'):
            svg_source = None
            # export the loaded file directly
            svg_source = self.input_path
        elif self.last_svg:
            svg_source = self.last_svg.tostring()
        else:
            messagebox.showwarning("No SVG available", "No SVG to export. Load an SVG or create a vector preview first.")
            return

        out = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG files', '*.png')])
        if not out:
            return
        try:
            if isinstance(svg_source, str) and os.path.exists(svg_source):
                cairosvg.svg2png(url=svg_source, write_to=out)
            else:
                cairosvg.svg2png(bytestring=svg_source, write_to=out)
            messagebox.showinfo('Exported', f'PNG exported to: {out}')
            self.set_status(f'Exported PNG: {out}')
        except Exception as e:
            messagebox.showerror('Export error', f'Failed to export PNG: {e}')

    def reset_all(self):
        self.input_path = None
        self.pil_image = None
        self.preview_photo = None
        self.last_svg = None
        self.info_label.configure(text='No file selected')
        self.canvas.delete('all')
        self.set_status('Ready')


if __name__ == '__main__':
    app = RasterVectorApp()
    app.mainloop()
