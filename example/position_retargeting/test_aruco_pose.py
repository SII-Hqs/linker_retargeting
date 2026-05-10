import cv2
import cv2.aruco as aruco
import numpy as np
import json

# ========== 1. 读取内参 ==========
with open("camera_params/color_intrinsics.json", "r") as f:
    data = json.load(f)

fx = data["fx"]
fy = data["fy"]
cx = data["cx"]
cy = data["cy"]

dist_dict = data["distortion"]

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

dist = np.array([
    dist_dict["k1"],
    dist_dict["k2"],
    dist_dict["p1"],
    dist_dict["p2"],
    dist_dict["k3"]
], dtype=np.float32)

# ========== 2. marker 实际边长 ==========
marker_length = 0.04   # 4 cm = 0.04 m

# ========== 3. ArUco 字典 ==========
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

detector = aruco.ArucoDetector(aruco_dict, parameters)

# ========== 4. 打开摄像头 ==========
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取图像失败")
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(frame, corners, ids)

        # 单 marker pose
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, K, dist
        )

        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, K, dist, rvecs[i], tvecs[i], 0.03)

            marker_id = int(ids[i][0])
            tx, ty, tz = tvecs[i][0]
            text = f"id={marker_id} t=({tx:.3f},{ty:.3f},{tz:.3f})m"
            cv2.putText(frame, text, (20, 40 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("aruco pose test", frame)
    key = cv2.waitKey(1)
    if key == 27:   # ESC 退出
        break

cap.release()
cv2.destroyAllWindows()