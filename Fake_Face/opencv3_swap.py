import cv2
import numpy as np
import dlib
import tkinter as tk
from tkinter import Tk, Button, filedialog, Label, Frame, messagebox, ttk, Scale
from PIL import Image, ImageTk
import os
import time


class FaceSwapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Face Swap Application")
        self.root.geometry("1000x800")
        self.root.minsize(900, 700)

        # Configuration
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(self.predictor_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Dlib model: {str(e)}")
            self.root.destroy()
            return

        # State variables
        self.source_img = None
        self.target_img = None
        self.output_img = None
        self.processing = False
        self.face_enhance = True
        self.blend_amount = 0.5
        self.resize_factor = 1.0

        # UI Setup
        self.create_widgets()

        # Performance tracking
        self.last_operation_time = 0

    def create_widgets(self):
        # Main frames
        self.main_frame = Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image display frame
        self.image_frame = Frame(self.main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        # Source image panel
        self.source_frame = Frame(self.image_frame)
        self.source_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.source_label = Label(self.source_frame, text="Source Image", font=('Helvetica', 12, 'bold'))
        self.source_label.pack()

        self.source_image_panel = Label(self.source_frame, bg='#f0f0f0', relief=tk.SUNKEN)
        self.source_image_panel.pack(fill=tk.BOTH, expand=True)

        # Target image panel
        self.target_frame = Frame(self.image_frame)
        self.target_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.target_label = Label(self.target_frame, text="Target Image", font=('Helvetica', 12, 'bold'))
        self.target_label.pack()

        self.target_image_panel = Label(self.target_frame, bg='#f0f0f0', relief=tk.SUNKEN)
        self.target_image_panel.pack(fill=tk.BOTH, expand=True)

        # Output image panel (only shown after processing)
        self.output_frame = Frame(self.image_frame)
        self.output_label = Label(self.output_frame, text="Output Image", font=('Helvetica', 12, 'bold'))
        self.output_label.pack()

        self.output_image_panel = Label(self.output_frame, bg='#f0f0f0', relief=tk.SUNKEN)
        self.output_image_panel.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.columnconfigure(1, weight=1)
        self.image_frame.rowconfigure(0, weight=1)

        # Control frame
        self.control_frame = Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=(10, 0))

        # Buttons
        self.btn_frame = Frame(self.control_frame)
        self.btn_frame.pack(side=tk.LEFT, padx=5)

        self.load_source_btn = Button(self.btn_frame, text="Load Source Image", width=20,
                                      command=lambda: self.load_image(True))
        self.load_source_btn.grid(row=0, column=0, padx=5, pady=5)

        self.load_target_btn = Button(self.btn_frame, text="Load Target Image", width=20,
                                      command=lambda: self.load_image(False))
        self.load_target_btn.grid(row=0, column=1, padx=5, pady=5)

        self.swap_btn = Button(self.btn_frame, text="Swap Faces", width=20,
                               command=self.swap_faces, state=tk.DISABLED)
        self.swap_btn.grid(row=1, column=0, padx=5, pady=5)

        self.save_btn = Button(self.btn_frame, text="Save Result", width=20,
                               command=self.save_result, state=tk.DISABLED)
        self.save_btn.grid(row=1, column=1, padx=5, pady=5)

        # Settings frame
        self.settings_frame = Frame(self.control_frame)
        self.settings_frame.pack(side=tk.RIGHT, padx=5)

        # Blend amount slider
        self.blend_label = Label(self.settings_frame, text="Blend Amount:")
        self.blend_label.grid(row=0, column=0, sticky='w')

        self.blend_slider = Scale(self.settings_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                  command=self.update_blend_amount)
        self.blend_slider.set(50)
        self.blend_slider.grid(row=0, column=1, sticky='ew')

        # Resize factor slider
        self.resize_label = Label(self.settings_frame, text="Resize Factor:")
        self.resize_label.grid(row=1, column=0, sticky='w')

        self.resize_slider = Scale(self.settings_frame, from_=10, to=100, orient=tk.HORIZONTAL,
                                   command=self.update_resize_factor)
        self.resize_slider.set(100)
        self.resize_slider.grid(row=1, column=1, sticky='ew')

        # Face enhance checkbox
        self.enhance_var = tk.BooleanVar(value=True)
        self.enhance_check = tk.Checkbutton(self.settings_frame, text="Enhance Face",
                                            variable=self.enhance_var, command=self.toggle_enhance)
        self.enhance_check.grid(row=2, column=0, columnspan=2, sticky='w')

        # Status bar
        self.status_frame = Frame(self.root, bd=1, relief=tk.SUNKEN)
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)

        self.status_label = Label(self.status_frame, text="Ready", anchor=tk.W)
        self.status_label.pack(fill=tk.X)

        # Configure weights for main frame
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def update_blend_amount(self, value):
        self.blend_amount = float(value) / 100.0
        if self.output_img is not None:
            self.show_output_image()

    def update_resize_factor(self, value):
        self.resize_factor = float(value) / 100.0
        if self.source_img is not None:
            self.show_image_in_label(self.source_img, self.source_image_panel)
        if self.target_img is not None:
            self.show_image_in_label(self.target_img, self.target_image_panel)
        if self.output_img is not None:
            self.show_output_image()

    def toggle_enhance(self):
        self.face_enhance = self.enhance_var.get()

    def load_image(self, is_source=True):
        if self.processing:
            return

        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select Source Image" if is_source else "Select Target Image",
                                          filetypes=filetypes)
        if not path:
            return

        try:
            # Read with OpenCV
            img = cv2.imread(path)
            if img is None:
                raise ValueError("Unsupported image format or corrupted file")

            # Convert to RGB for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if is_source:
                self.source_img = img
                self.show_image_in_label(self.source_img, self.source_image_panel)
            else:
                self.target_img = img
                self.show_image_in_label(self.target_img, self.target_image_panel)

            # Enable swap button if both images are loaded
            if self.source_img is not None and self.target_img is not None:
                self.swap_btn.config(state=tk.NORMAL)

            self.update_status(f"Loaded {'source' if is_source else 'target'} image: {os.path.basename(path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.update_status("Error loading image")

    def show_image_in_label(self, image_cv, label_widget):
        if image_cv is None:
            return

        try:
            # Resize based on current factor
            height, width = image_cv.shape[:2]
            new_width = int(width * self.resize_factor)
            new_height = int(height * self.resize_factor)

            # Convert to RGB and resize
            img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (new_width, new_height))

            # Convert to PIL Image and then to PhotoImage
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Update the label
            label_widget.config(image=img_tk)
            label_widget.image = img_tk

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")

    def show_output_image(self):
        if self.output_img is None:
            return

        try:
            # Apply blending if needed
            if self.blend_amount < 1.0 and self.target_img is not None:
                # Ensure images are the same size
                if self.output_img.shape != self.target_img.shape:
                    self.target_img = cv2.resize(self.target_img, (self.output_img.shape[1], self.output_img.shape[0]))

                blended = cv2.addWeighted(self.output_img, self.blend_amount,
                                          self.target_img, 1.0 - self.blend_amount, 0)
            else:
                blended = self.output_img

            # Show in the output panel
            self.show_image_in_label(blended, self.output_image_panel)

            # Show the output frame if not already shown
            if not self.output_frame.winfo_ismapped():
                self.output_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
                self.image_frame.columnconfigure(2, weight=1)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display output image: {str(e)}")

    def get_landmarks(self, img):
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Enhance contrast for better detection
            gray = cv2.equalizeHist(gray)

            # Detect faces
            faces = self.detector(gray, 1)
            if len(faces) == 0:
                return None

            # Get landmarks for the first face
            shape = self.predictor(gray, faces[0])
            return [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect landmarks: {str(e)}")
            return None

    def apply_affine_transform(self, src, src_tri, dst_tri, size):
        warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
        dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return dst

    def calculate_delaunay_triangles(self, rect, points):
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            subdiv.insert(p)

        triangle_list = subdiv.getTriangleList()
        triangles = []

        for t in triangle_list:
            pts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]

            # Check if all points are within the image
            if all(0 <= pt[0] < rect[2] and 0 <= pt[1] < rect[3] for pt in pts):
                idx = []
                for pt in pts:
                    # Find the index of the closest landmark point
                    distances = [np.linalg.norm(np.array(pt) - np.array(p)) for p in points]
                    min_idx = np.argmin(distances)
                    if distances[min_idx] < 5:  # Threshold for matching
                        idx.append(min_idx)

                if len(idx) == 3:
                    triangles.append((idx[0], idx[1], idx[2]))

        return triangles

    def warp_triangle(self, img1, img2, t1, t2):
        # Calculate bounding rectangles for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        # Offset points by left top corner of the respective rectangles
        t1_rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
        t2_rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]

        # Create mask for the triangle
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)

        # Warp the source image to match the destination triangle
        img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        warped = self.apply_affine_transform(img1_rect, t1_rect, t2_rect, (r2[2], r2[3]))
        warped = warped * mask

        # Copy the warped triangle to the destination image
        img2_area = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]].astype(np.float32)
        img2_area *= (1.0 - mask)
        img2_area += warped
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = np.clip(img2_area, 0, 255).astype(np.uint8)

    def enhance_face(self, face_img):
        # Apply some basic enhancement to make the swapped face look better
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)

            # Split channels
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge channels and convert back to BGR
            enhanced_lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # Apply slight sharpening
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

            return enhanced

        except Exception as e:
            print(f"Enhancement failed: {str(e)}")
            return face_img

    def swap_faces(self):
        if self.processing:
            return

        if self.source_img is None or self.target_img is None:
            messagebox.showerror("Error", "Please load both source and target images.")
            return

        self.processing = True
        self.update_status("Processing... Please wait")
        self.root.config(cursor="watch")
        self.root.update()

        try:
            start_time = time.time()

            # Get landmarks for both images
            points1 = self.get_landmarks(self.source_img)
            points2 = self.get_landmarks(self.target_img)

            if points1 is None or points2 is None:
                messagebox.showerror("Error", "Face not detected in one of the images.")
                return

            # Create a copy of the target image for warping
            img1_warped = np.copy(self.target_img)

            # Calculate Delaunay triangulation
            rect = (0, 0, self.target_img.shape[1], self.target_img.shape[0])
            dt = self.calculate_delaunay_triangles(rect, points2)

            if not dt:
                messagebox.showerror("Error", "Failed to calculate Delaunay triangulation.")
                return

            # Warp each triangle from source to target
            for tri in dt:
                t1 = [points1[tri[0]], points1[tri[1]], points1[tri[2]]]
                t2 = [points2[tri[0]], points2[tri[1]], points2[tri[2]]]
                self.warp_triangle(self.source_img, img1_warped, t1, t2)

            # Prepare mask for seamless cloning
            hull2 = [points2[i] for i in cv2.convexHull(np.array(points2), returnPoints=False).flatten()]
            mask = np.zeros(self.target_img.shape, dtype=self.target_img.dtype)
            cv2.fillConvexPoly(mask, np.int32(hull2), (255, 255, 255))

            # Find center of the face for seamless cloning
            r = cv2.boundingRect(np.float32([hull2]))
            center = (r[0] + r[2] // 2, r[1] + r[3] // 2)

            # Apply face enhancement if enabled
            if self.face_enhance:
                img1_warped = self.enhance_face(img1_warped)

            # Perform seamless cloning
            self.output_img = cv2.seamlessClone(img1_warped, self.target_img, mask, center, cv2.NORMAL_CLONE)

            # Show the result
            self.show_output_image()
            self.save_btn.config(state=tk.NORMAL)

            # Update status with performance info
            self.last_operation_time = time.time() - start_time
            self.update_status(f"Done! Processing time: {self.last_operation_time:.2f} seconds")

        except Exception as e:
            messagebox.showerror("Error", f"Face swap failed: {str(e)}")
            self.update_status("Face swap failed")

        finally:
            self.processing = False
            self.root.config(cursor="")

    def save_result(self):
        if self.output_img is None:
            messagebox.showerror("Error", "No result to save.")
            return

        filetypes = [("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp"), ("All files", "*.*")]
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=filetypes)

        if path:
            try:
                cv2.imwrite(path, self.output_img)
                messagebox.showinfo("Saved", f"Image successfully saved to:\n{path}")
                self.update_status(f"Result saved to {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
                self.update_status("Save failed")


if __name__ == "__main__":
    root = Tk()
    app = FaceSwapApp(root)
    root.mainloop()