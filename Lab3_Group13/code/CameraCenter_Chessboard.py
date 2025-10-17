import cv2
import numpy as np
import os


def find_and_calibrate(image_dir, pattern_size, square_size, display=False):
    """Detect chessboard in images under image_dir, run calibrateCamera, return calibration and obj/img points."""
    cols, rows = pattern_size  # pattern_size provided as (cols, rows)

    # prepare object points for one view
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    obj_points = []
    img_points = []

    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist.")
        return None

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png'))])
    if len(image_files) == 0:
        print(f"No images found in '{image_dir}'")
        return None

    last_gray = None
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: cannot read {img_path}, skipping")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        last_gray = gray

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp.copy())
            img_points.append(corners_sub)
            if display:
                disp = cv2.drawChessboardCorners(img.copy(), (cols, rows), corners_sub, found)
                cv2.imshow('corners', disp)
                cv2.waitKey(200)
        else:
            print(f"No corners in {img_name}")

    if display:
        cv2.destroyAllWindows()

    if len(obj_points) == 0 or last_gray is None:
        print("No valid corners detected. Calibration cannot be performed.")
        return None

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, last_gray.shape[::-1], None, None)

    print(f"Calibration done. {len(obj_points)} views used. Reproj err approx: ")
    # compute mean reproj
    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += err
    mean_error = mean_error / len(obj_points)
    print(f"  mean reprojection error: {mean_error:.4f} pixels")

    return {
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'objp': objp,
        'img_points': img_points,
        'obj_points': obj_points,
        'pattern': (cols, rows)
    }


def create_chessboard_ply_from_objpoints(obj_points, rows, cols, filename, duplicate_vertices=True, camera_rvecs=None, camera_tvecs=None):
    """Create a black/white chessboard mesh PLY from obj_points list (use first view as base).
    obj_points: list of (rows*cols,3) arrays (same format as returned by find_and_calibrate)
    rows, cols: inner corner counts (rows, cols)
    """
    if not obj_points or len(obj_points) == 0:
        print('No obj_points provided, cannot create chessboard mesh')
        return

    base = np.asarray(obj_points[0])
    try:
        grid = base.reshape(rows, cols, 3)
    except Exception:
        # fallback if order differs
        grid = base.reshape(cols, rows, 3).transpose(1, 0, 2)

    if duplicate_vertices:
        # Create flat mesh: each triangle has its own 3 vertices (no sharing) so color isn't interpolated
        flat_vertices = []
        flat_colors = []
        faces = []
        vid = 0
        for i in range(rows - 1):
            for j in range(cols - 1):
                p00 = np.array(grid[i, j], dtype=float)
                p10 = np.array(grid[i, j + 1], dtype=float)
                p01 = np.array(grid[i + 1, j], dtype=float)
                p11 = np.array(grid[i + 1, j + 1], dtype=float)
                # face color by square parity
                if (i + j) % 2 == 0:
                    face_color = (255, 255, 255)
                else:
                    face_color = (0, 0, 0)

                # triangle 1: p00, p10, p11
                flat_vertices.append((float(p00[0]), float(p00[1]), float(p00[2])))
                flat_colors.append(face_color)
                flat_vertices.append((float(p10[0]), float(p10[1]), float(p10[2])))
                flat_colors.append(face_color)
                flat_vertices.append((float(p11[0]), float(p11[1]), float(p11[2])))
                flat_colors.append(face_color)
                faces.append((vid, vid + 1, vid + 2, face_color))
                vid += 3

                # triangle 2: p00, p11, p01
                flat_vertices.append((float(p00[0]), float(p00[1]), float(p00[2])))
                flat_colors.append(face_color)
                flat_vertices.append((float(p11[0]), float(p11[1]), float(p11[2])))
                flat_colors.append(face_color)
                flat_vertices.append((float(p01[0]), float(p01[1]), float(p01[2])))
                flat_colors.append(face_color)
                faces.append((vid, vid + 1, vid + 2, face_color))
                vid += 3

        vertices = flat_vertices
        vertex_colors = flat_colors
    else:
        vertices = []
        vertex_colors = []
        vertex_map = {}
        idx = 0
        for i in range(rows):
            for j in range(cols):
                x, y, z = grid[i, j]
                key = (round(float(x), 6), round(float(y), 6), round(float(z), 6))
                if key not in vertex_map:
                    vertex_map[key] = idx
                    vertices.append((float(x), float(y), float(z)))
                    # vertex color by parity
                    if (i + j) % 2 == 0:
                        vertex_colors.append((255, 255, 255))
                    else:
                        vertex_colors.append((0, 0, 0))
                    idx += 1

        faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                p0 = vertex_map[(round(float(grid[i, j, 0]), 6), round(float(grid[i, j, 1]), 6), round(float(grid[i, j, 2]), 6))]
                p1 = vertex_map[(round(float(grid[i, j + 1, 0]), 6), round(float(grid[i, j + 1, 1]), 6), round(float(grid[i, j + 1, 2]), 6))]
                p2 = vertex_map[(round(float(grid[i + 1, j + 1, 0]), 6), round(float(grid[i + 1, j + 1, 1]), 6), round(float(grid[i + 1, j + 1, 2]), 6))]
                p3 = vertex_map[(round(float(grid[i + 1, j, 0]), 6), round(float(grid[i + 1, j, 1]), 6), round(float(grid[i + 1, j, 2]), 6))]
                # face color by square parity (i+j)
                if (i + j) % 2 == 0:
                    c = (255, 255, 255)
                else:
                    c = (0, 0, 0)
                faces.append((p0, p1, p2, c))
                faces.append((p0, p2, p3, c))

    # append camera markers as small red pyramids (so they are visible in mesh viewers)
    if camera_rvecs is not None and camera_tvecs is not None:
        # estimate a reasonable marker scale from the chessboard spacing
        try:
            # grid spacing: distance between neighboring inner corners
            d1 = np.linalg.norm(grid[0, 1] - grid[0, 0])
            d2 = np.linalg.norm(grid[1, 0] - grid[0, 0])
            marker_scale = 0.2 * (d1 + d2) / 2.0
        except Exception:
            marker_scale = 0.2

        for rvec, tvec in zip(camera_rvecs, camera_tvecs):
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1) if tvec is not None else np.zeros((3, 1))
            cam_pos = (-R.T @ t).flatten()
            apex = np.array(cam_pos, dtype=float)

            # create three small base vertices around the apex (axis-aligned offsets)
            b1 = apex + np.array([marker_scale, 0.0, 0.0])
            b2 = apex + np.array([0.0, marker_scale, 0.0])
            b3 = apex + np.array([0.0, 0.0, marker_scale])

            # add base vertices
            idx_b1 = len(vertices)
            vertices.append((float(b1[0]), float(b1[1]), float(b1[2])))
            vertex_colors.append((255, 0, 0))
            idx_b2 = len(vertices)
            vertices.append((float(b2[0]), float(b2[1]), float(b2[2])))
            vertex_colors.append((255, 0, 0))
            idx_b3 = len(vertices)
            vertices.append((float(b3[0]), float(b3[1]), float(b3[2])))
            vertex_colors.append((255, 0, 0))

            # add apex vertex
            idx_apex = len(vertices)
            vertices.append((float(apex[0]), float(apex[1]), float(apex[2])))
            vertex_colors.append((255, 0, 0))

            # create triangular faces (apex->b1->b2, apex->b2->b3, apex->b3->b1) and base face
            faces.append((idx_apex, idx_b1, idx_b2, (255, 0, 0)))
            faces.append((idx_apex, idx_b2, idx_b3, (255, 0, 0)))
            faces.append((idx_apex, idx_b3, idx_b1, (255, 0, 0)))
            faces.append((idx_b1, idx_b2, idx_b3, (255, 0, 0)))

    # write mesh PLY with vertex colors and face colors
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(vertices)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write(f'element face {len(faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')

        for (v, col) in zip(vertices, vertex_colors):
            f.write(f"{v[0]} {v[1]} {v[2]} {col[0]} {col[1]} {col[2]}\n")

        for (a, b, c, col) in faces:
            f.write(f"3 {a} {b} {c} {col[0]} {col[1]} {col[2]}\n")

    print(f"Wrote chessboard mesh to {filename} (vertices: {len(vertices)}, faces: {len(faces)})")


def main():
    # --- defaults ---
    images_dir = 'tmp'
    pattern = '11x8'  # cols x rows
    square_size = 25.0
    out_mesh = 'chessboard_with_cameras_mesh_flat.ply'


    cols, rows = map(int, pattern.split('x'))

    print(f"Using defaults: images='{images_dir}', pattern={cols}x{rows}, square={square_size}")

    res = find_and_calibrate(images_dir, (cols, rows), square_size, display=False)
    if res is None:
        print('Calibration failed or no valid images found.')
        return

    # Save calibration for reference
    np.savez('camera_params_generated.npz', mtx=res['mtx'], dist=res['dist'], rvecs=res['rvecs'], tvecs=res['tvecs'])
    print("Saved calibration to camera_params_generated.npz")

    # write only the flat mesh (duplicated vertices) including camera markers
    create_chessboard_ply_from_objpoints(res['obj_points'], rows, cols, out_mesh, duplicate_vertices=True, camera_rvecs=res['rvecs'], camera_tvecs=res['tvecs'])
    print(f'Wrote flat chessboard mesh to {out_mesh}')


if __name__ == '__main__':
    main()