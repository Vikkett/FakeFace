import cv2
import numpy as np
import dlib
from tkinter import Tk, Button, filedialog, Label, Frame, messagebox
from PIL import Image, ImageTk

# Load Dlib model
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Globals
source_img = None
target_img = None
output_img = None

# GUI Setup
root = Tk()
root.title("Face Swap Application")
root.geometry("850x650")

# Image preview labels
frame_top = Frame(root)
frame_top.pack()

source_label = Label(frame_top, text="Source Image")
source_label.grid(row=0, column=0, padx=10)

target_label = Label(frame_top, text="Target Image")
target_label.grid(row=0, column=1, padx=10)

source_image_panel = Label(frame_top)
source_image_panel.grid(row=1, column=0, padx=10)

target_image_panel = Label(frame_top)
target_image_panel.grid(row=1, column=1, padx=10)

# Function to display image in GUI
def show_image_in_label(image_cv, label):
    img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((400, 300))
    img_tk = ImageTk.PhotoImage(image=img_pil)
    label.config(image=img_tk)
    label.image = img_tk

# Load image
def load_image(is_source=True):
    global source_img, target_img
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not path:
        return
    img = cv2.imread(path)
    if img is None:
        messagebox.showerror("Error", "Image could not be read.")
        return
    if is_source:
        source_img = img
        show_image_in_label(source_img, source_image_panel)
    else:
        target_img = img
        show_image_in_label(target_img, target_image_panel)

# Landmark extraction
def get_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    return [(shape.part(i).x, shape.part(i).y) for i in range(68)]

# Affine warp
def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, size, None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

# Triangle helper
def rect_contains(rect, point):
    return rect[0] <= point[0] <= rect[0]+rect[2] and rect[1] <= point[1] <= rect[1]+rect[3]

def calculate_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)
    triangle_list = subdiv.getTriangleList()
    triangles = []
    for t in triangle_list:
        pts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
        if all(rect_contains(rect, pt) for pt in pts):
            idx = []
            for pt in pts:
                for i, p in enumerate(points):
                    if abs(pt[0] - p[0]) < 1 and abs(pt[1] - p[1]) < 1:
                        idx.append(i)
                        break
            if len(idx) == 3:
                triangles.append((idx[0], idx[1], idx[2]))
    return triangles

# Warp triangle
def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2_rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    size = (r2[2], r2[3])
    warped = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    warped = warped * mask

    img2_area = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]].astype(np.float32)
    img2_area *= (1.0 - mask)
    img2_area += warped
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = np.clip(img2_area, 0, 255).astype(np.uint8)

# Face swap logic
def swap_faces():
    global output_img
    if source_img is None or target_img is None:
        messagebox.showerror("Error", "Please load both source and target images.")
        return

    points1 = get_landmarks(source_img)
    points2 = get_landmarks(target_img)
    if points1 is None or points2 is None:
        messagebox.showerror("Error", "Face not detected in one of the images.")
        return

    img1_warped = np.copy(target_img)
    rect = (0, 0, target_img.shape[1], target_img.shape[0])
    dt = calculate_delaunay_triangles(rect, points2)

    for tri in dt:
        t1 = [points1[tri[0]], points1[tri[1]], points1[tri[2]]]
        t2 = [points2[tri[0]], points2[tri[1]], points2[tri[2]]]
        warp_triangle(source_img, img1_warped, t1, t2)

    hull2 = [points2[i] for i in cv2.convexHull(np.array(points2), returnPoints=False).flatten()]
    mask = np.zeros(target_img.shape, dtype=target_img.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull2), (255, 255, 255))
    r = cv2.boundingRect(np.float32([hull2]))
    center = (r[0] + r[2]//2, r[1] + r[3]//2)

    output_img = cv2.seamlessClone(img1_warped, target_img, mask, center, cv2.NORMAL_CLONE)
    cv2.imshow("Swapped Face", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Save result
def save_result():
    global output_img
    if output_img is None:
        messagebox.showerror("Error", "No result to save.")
        return
    path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
    if path:
        cv2.imwrite(path, output_img)
        messagebox.showinfo("Saved", f"Image saved to:\n{path}")

# Buttons
frame_buttons = Frame(root)
frame_buttons.pack(pady=15)

Button(frame_buttons, text="Load Source Image", width=20, command=lambda: load_image(True)).grid(row=0, column=0, padx=10)
Button(frame_buttons, text="Load Target Image", width=20, command=lambda: load_image(False)).grid(row=0, column=1, padx=10)
Button(frame_buttons, text="Swap Faces", width=20, command=swap_faces).grid(row=1, column=0, padx=10, pady=10)
Button(frame_buttons, text="Save Result", width=20, command=save_result).grid(row=1, column=1, padx=10, pady=10)

root.mainloop()
