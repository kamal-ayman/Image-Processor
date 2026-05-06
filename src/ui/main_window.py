import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2
import numpy as np
from typing import Dict, List, Type

from ..app.image_service import ImageService
from ..domain.image_operation import ImageOperation
from ..infrastructure.filters.spatial.spatial_filters import *
from ..infrastructure.filters.frequency.frequency_filters import *
from ..infrastructure.noise.noise_operations import *
from ..infrastructure.morphology.morphological_operations import *
from ..infrastructure.enhancement.enhancement_operations import *
from ..infrastructure.transform.transform_operations import *
from ..infrastructure.detection.detection_operations import *

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.service = ImageService()
        self.title("Image Processor Project")
        self.geometry("1300x800")
        
        # UI State
        self.categories = {
            "Transform": ["Grayscale", "Gray to Binary", "RGB to Binary"],
            "Spatial Filters": ["Mean", "Median", "Max", "Min", "Midpoint"],
            "Frequency Filters": ["Ideal LPF", "Ideal HPF", "Butterworth LPF", "Butterworth HPF", "Gaussian LPF", "Gaussian HPF"],
            "Noise": ["Gaussian Noise", "Salt & Pepper", "Rayleigh Noise"],
            "Morphology": ["Erosion", "Dilation", "Opening", "Closing", "Boundary"],
            "Enhancement": ["Hist Equalization", "Contrast Stretch", "Brightness", "Negative", "Log", "Gamma"],
            "Detection/Sharp": ["Point Detection", "Point Sharpening", "Line Detection", "Line Sharpening"]
        }
        
        # Operation Mapping
        self.ops: Dict[str, ImageOperation] = {
            "Grayscale": RgbToGrayscale(),
            "Gray to Binary": GrayToBinary(),
            "RGB to Binary": RgbToBinary(),
            "Mean": MeanFilter(),
            "Median": MedianFilter(),
            "Max": MaxFilter(),
            "Min": MinFilter(),
            "Midpoint": MidpointFilter(),
            "Ideal LPF": IdealLowPassFilter(),
            "Ideal HPF": IdealHighPassFilter(),
            "Butterworth LPF": ButterworthLowPassFilter(),
            "Butterworth HPF": ButterworthHighPassFilter(),
            "Gaussian LPF": GaussianLowPassFilter(),
            "Gaussian HPF": GaussianHighPassFilter(),
            "Gaussian Noise": GaussianNoise(),
            "Salt & Pepper": SaltAndPepperNoise(),
            "Rayleigh Noise": RayleighNoise(),
            "Erosion": Erosion(),
            "Dilation": Dilation(),
            "Opening": Opening(),
            "Closing": Closing(),
            "Boundary": BoundaryExtraction(),
            "Hist Equalization": HistogramEqualization(),
            "Contrast Stretch": ContrastStretching(),
            "Brightness": BrightnessAdjustment(),
            "Negative": ImageNegative(),
            "Log": LogTransformation(),
            "Gamma": GammaCorrection(),
            "Point Detection": PointDetection(),
            "Point Sharpening": PointSharpening(),
            "Line Detection": LineDetection(),
            "Line Sharpening": LineSharpening()
        }

        self._setup_ui()

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Image Processor", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=20, padx=20)

        self.btn_load = ctk.CTkButton(self.sidebar, text="Load Image", command=self._load_image)
        self.btn_load.pack(pady=10, padx=20)

        self.btn_reset = ctk.CTkButton(self.sidebar, text="Reset Image", fg_color="gray", command=self._reset_image)
        self.btn_reset.pack(pady=5, padx=20)

        ctk.CTkLabel(self.sidebar, text="Category:").pack(pady=(20, 0))
        self.cat_menu = ctk.CTkOptionMenu(self.sidebar, values=list(self.categories.keys()), command=self._update_ops)
        self.cat_menu.pack(pady=5, padx=20)

        ctk.CTkLabel(self.sidebar, text="Operation:").pack(pady=(10, 0))
        self.op_menu = ctk.CTkOptionMenu(self.sidebar, values=self.categories["Transform"])
        self.op_menu.pack(pady=5, padx=20)

        ctk.CTkLabel(self.sidebar, text="Parameter:").pack(pady=(20, 0))
        self.slider = ctk.CTkSlider(self.sidebar, from_=0, to=1, command=self._on_slider_move)
        self.slider.pack(pady=5, padx=20)
        self.slider_label = ctk.CTkLabel(self.sidebar, text="0.5")
        self.slider_label.pack()

        self.btn_apply = ctk.CTkButton(self.sidebar, text="Apply & Update", fg_color="#2ecc71", hover_color="#27ae60", command=self._apply_operation)
        self.btn_apply.pack(pady=30, padx=20)

        # --- Canvas ---
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.img_container = ctk.CTkFrame(self.canvas_frame, fg_color="transparent")
        self.img_container.pack(expand=True, fill="both")

        self.label_orig = ctk.CTkLabel(self.img_container, text="No Image Loaded")
        self.label_orig.pack(side="left", expand=True, padx=10)

        self.label_proc = ctk.CTkLabel(self.img_container, text="")
        self.label_proc.pack(side="right", expand=True, padx=10)

    def _load_image(self):
        path = filedialog.askopenfilename()
        if path and self.service.load_image(path):
            self._update_display()

    def _reset_image(self):
        self.service.reset()
        self._update_display()

    def _update_ops(self, cat):
        ops = self.categories.get(cat, [])
        self.op_menu.configure(values=ops)
        self.op_menu.set(ops[0])

    def _on_slider_move(self, val):
        self.slider_label.configure(text=f"{val:.2f}")

    def _apply_operation(self):
        op_name = self.op_menu.get()
        p = self.slider.get()
        operation = self.ops.get(op_name)
        
        if not operation: return

        # Simple parameter mapping logic
        kwargs = {}
        if op_name in ["Ideal LPF", "Ideal HPF", "Butterworth LPF", "Butterworth HPF", "Gaussian LPF", "Gaussian HPF"]:
            kwargs['d0'] = p * 200
        elif op_name in ["Gray to Binary", "RGB to Binary"]:
            kwargs['threshold'] = int(p * 255)
        elif "Noise" in op_name:
            if op_name == "Gaussian Noise":
                kwargs['sigma'] = p * 100
            elif op_name == "Salt & Pepper Noise":
                kwargs['salt_prob'] = p/10
                kwargs['pepper_prob'] = p/10
            elif op_name == "Rayleigh Noise":
                kwargs['b'] = p * 100
        elif op_name in ["Mean", "Median", "Max", "Min", "Midpoint"]:
            kwargs['kernel_size'] = int(p * 20) | 1
        elif op_name == "Gamma":
            kwargs['gamma'] = max(0.01, p * 4)
        elif op_name == "Brightness":
            kwargs['offset'] = (p - 0.5) * 200
        elif "Sharpening" in op_name:
            kwargs['alpha'] = p * 5

        self.service.apply_operation(operation, **kwargs)
        self._update_display()

    def _update_display(self):
        orig = self.service.get_original_image()
        proc = self.service.get_current_image()

        if orig is not None:
            self._show_image(orig, self.label_orig)
        if proc is not None:
            self._show_image(proc, self.label_proc)

    def _show_image(self, img_np, label):
        # Convert BGR (OpenCV) to RGB (PIL)
        if len(img_np.shape) == 3:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_np
        
        img_pil = Image.fromarray(img_rgb)
        
        # Fit to size
        img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(450, 450))
        label.configure(image=img_ctk, text="")
        label.image = img_ctk
