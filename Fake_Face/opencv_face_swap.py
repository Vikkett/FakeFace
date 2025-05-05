import cv2
import numpy as np
import face_recognition
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk

root = Tk()
root.title("Advanced Face Swap (Triangulation)")
root.geometry("900x700")

image_label = Label(root)
image_label.pack()

source_img = None
target_img = None
result_img = None

def load_image(is_source=True):
    global source_img, target_img
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks = face_recognition.face_landmarks(img_rgb)

    if not landmarks:
        print("No face detected.")
        return

    if is_source:
        source_img = img
    else:
        target_img = img

    show_image(img_rgb)

def show_image(img):
    img_pil = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    image_label.configure(image=img_tk)
    image_label.image = img_tk

def get_landmarks(img):
    points = []
    face_landmarks_list = face_recognition.face_landmarks(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not face_landmarks_list:
        return []

    landmark = face_landmarks_list[0]
    for key in landmark:
        for point in landmark[key]:
            points.append(point)

    # Add chin + outer face contour if not included
    face_locations = face_recognition.face_locations(img)
    if face_locations:
        top, right, bottom, left = face_locations[0]
        points += [
            (left, top), (right, top), (right, bottom), (left, bottom)
        ]
    return points

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def rect_contains(rect, point):
    x, y, w, h = rect
    if point[0] < x or point[0] > x + w:
        return False
    if point[1] < y or point[1] > y + h:
        return False
    return True

def get_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList()
    triangle_indices = []

    for t in triangleList:
        pts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
        index = []
        for pt in pts:
            for i, p in enumerate(points):
                if abs(pt[0] - p[0]) < 2 and abs(pt[1] - p[1]) < 2:
                    index.append(i)
        if len(index) == 3:
            triangle_indices.append(index)
    return triangle_indices

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((int(t2[i][0] - r2[0])), int(t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    size = (r2[2], r2[3])
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    img2_rect = img2_rect * mask
    img2_area = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    img2_area = img2_area * (1 - mask)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_area + img2_rect

def swap_faces_advanced():
    global source_img, target_img, result_img
    if source_img is None or target_img is None:
        print("Load both images first.")
        return

    src_pts = get_landmarks(source_img)
    tgt_pts = get_landmarks(target_img)

    if not src_pts or not tgt_pts or len(src_pts) != len(tgt_pts):
        print("Could not extract landmarks properly.")
        return

    tgt_img_warped = np.copy(target_img)
    rect = (0, 0, target_img.shape[1], target_img.shape[0])
    triangles = get_triangles(rect, tgt_pts)

    for tri in triangles:
        t1 = [src_pts[i] for i in tri]
        t2 = [tgt_pts[i] for i in tri]
        warp_triangle(source_img, tgt_img_warped, t1, t2)

    # Create mask and seamless clone
    hull8U = cv2.convexHull(np.array(tgt_pts), returnPoints=True)
    mask = np.zeros(target_img.shape, dtype=target_img.dtype)
    cv2.fillConvexPoly(mask, hull8U, (255, 255, 255))

    center = (target_img.shape[1] // 2, target_img.shape[0] // 2)
    output = cv2.seamlessClone(tgt_img_warped, target_img, mask, center, cv2.NORMAL_CLONE)

    result_img = output
    result_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    show_image(result_rgb)

def save_result():
    global result_img
    if result_img is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if file_path:
            cv2.imwrite(file_path, result_img)
            print("Saved.")
    else:
        print("No result to save.")

# Buttons
Button(root, text="Load Source Face", command=lambda: load_image(True)).pack(pady=5)
Button(root, text="Load Target Face", command=lambda: load_image(False)).pack(pady=5)
Button(root, text="Advanced Face Swap", command=swap_faces_advanced).pack(pady=5)
Button(root, text="Save Result", command=save_result).pack(pady=5)

root.mainloop()
