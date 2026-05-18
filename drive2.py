import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
# CONFIG

VIDEO_PATH          = r"C:\Users\Admin\Desktop\Self-DrivingCarSimulation\Highway.mp4"
YOLO_MODEL          = "yolov8n.pt"
YOLO_EVERY_N        = 5
RESIZE_WIDTH        = 480
DISPLAY_WIDTH       = 1000
TARGET_FPS          = 20
DECISION_BUFFER     = 12
CANNY_LOW           = 30
CANNY_HIGH          = 120
HOUGH_THRESHOLD     = 30
HOUGH_MIN_LEN       = 40
HOUGH_MAX_GAP       = 80
LANE_WIDTH_METERS   = 3.7
CONFIDENCE_THRESH   = 0.40
PERSON_CONF_THRESH  = 0.70
VEHICLE_CONF_THRESH = 0.40
PERSON_MIN_HEIGHT   = 0.15
VEHICLE_DANGER_ZONE = 0.35
VEHICLE_SIZE_DANGER = 0.18
ALLOWED_CLASSES     = {
    "person", "car", "truck", "bus",
    "bicycle", "motorcycle"
}
# LANE DETECTION

def preprocess_frame(frame):
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred  = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges    = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    return edges

def apply_roi(edges):
    h, w    = edges.shape
    roi_pts = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.42), int(h * 0.60)),
        (int(w * 0.58), int(h * 0.60)),
        (int(w * 0.95), h),
    ]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_pts, 255)
    return cv2.bitwise_and(edges, mask), roi_pts

def detect_lane_lines(masked_edges):
    lines = cv2.HoughLinesP(
        masked_edges, 1, np.pi / 180,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_MAX_GAP,
    )
    left_lines, right_lines = [], []
    if lines is None:
        return left_lines, right_lines

    h, w = masked_edges.shape
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        if dx == 0:
            continue
        slope = (y2 - y1) / dx
        if slope > 0.4 and x1 < w // 2:
            left_lines.append(line)
        elif slope < -0.4 and x1 > w // 2:
            right_lines.append(line)

    return left_lines, right_lines

def average_lane_line(lines, image_shape):
    if not lines:
        return None
    h = image_shape[0]
    pts = np.array([[x, y] for line in lines for x, y in [line[0][:2], line[0][2:]]])
    [vx, vy, cx, cy] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    t_bottom = (h - cy) / vy
    t_top    = (int(h * 0.60) - cy) / vy
    x_bot    = int(cx + t_bottom * vx)
    x_top    = int(cx + t_top    * vx)
    return (x_bot, h, x_top, int(h * 0.60))
# Lane memory

_last_left_avg  = None
_last_right_avg = None
_lane_memory_frames = 0
LANE_MEMORY_MAX = 20

def average_lane_line_with_memory(lines, image_shape, last_avg, alpha=0.85):
    new_avg = average_lane_line(lines, image_shape)
    if new_avg is None:
        return last_avg
    if last_avg is None:
        return new_avg
    blended = tuple(int(alpha * n + (1 - alpha) * l) for n, l in zip(new_avg, last_avg))
    return blended

def draw_lanes(frame, left_avg, right_avg, roi_pts):
    overlay = frame.copy()

    points = []
    if left_avg:
        x1, y1, x2, y2 = left_avg
        cv2.line(frame, (x1, y1), (x2, y2), (0, 220, 0), 3)
        points += [(x1, y1), (x2, y2)]
    if right_avg:
        x1, y1, x2, y2 = right_avg
        cv2.line(frame, (x1, y1), (x2, y2), (0, 220, 0), 3)
        points += [(x1, y1), (x2, y2)]

    if left_avg and right_avg:
        lx1, ly1, lx2, ly2 = left_avg
        rx1, ry1, rx2, ry2 = right_avg
        poly = np.array([[lx1, ly1], [lx2, ly2], [rx2, ry2], [rx1, ry1]], np.int32)
        cv2.fillPoly(overlay, [poly], (0, 180, 0))
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    return frame
# MEASUREMENTS

def calculate_offset(left_avg, right_avg, image_width):
    if left_avg is None or right_avg is None:
        return 0.0
    lane_center   = (left_avg[0] + right_avg[0]) / 2
    image_center  = image_width / 2
    offset_pixels = lane_center - image_center
    return (offset_pixels / image_width) * LANE_WIDTH_METERS

def calculate_curvature(left_avg, right_avg, image_height):
    avg = left_avg or right_avg
    if avg is None:
        return float("inf")
    x1, y1, x2, y2 = avg
    dx, dy = x2 - x1, y2 - y1
    if dy == 0:
        return float("inf")
    slope = dx / dy
    curvature = abs(1 / slope) * 1000 if slope != 0 else float("inf")
    return min(curvature, 9999.0)

def estimate_speed(prev_gray, curr_gray, fps, speed_buffer):
    diff      = cv2.absdiff(prev_gray, curr_gray)
    mean_diff = float(np.mean(diff))
    speed_kmh = mean_diff * fps * 0.30
    speed_buffer.append(speed_kmh)
    return round(float(np.mean(speed_buffer)), 1)
# OBJECT DETECTION

def run_object_detection(model, frame):
    return model.predict(frame, verbose=False)

def is_valid_detection(label, confidence, x1, y1, x2, y2, frame_shape):
    if label not in ALLOWED_CLASSES:
        return False

    if label == "person":
        if confidence < PERSON_CONF_THRESH:
            return False
        box_height_ratio = (y2 - y1) / frame_shape[0]
        if box_height_ratio < PERSON_MIN_HEIGHT:
            return False
    elif label in ("car", "truck", "bus", "bicycle", "motorcycle"):
        if confidence < VEHICLE_CONF_THRESH:
            return False
    else:
        if confidence < CONFIDENCE_THRESH:
            return False

    return True

def get_lane_x_at_y(lane_avg, target_y):
    if lane_avg is None:
        return None
    x_bot, y_bot, x_top, y_top = lane_avg
    if y_bot == y_top:
        return x_bot
    t = (target_y - y_bot) / (y_top - y_bot)
    return int(x_bot + t * (x_top - x_bot))

def get_lane_boundaries_at_y(left_avg, right_avg, frame_width, target_y, frame_height):
    left_x  = get_lane_x_at_y(left_avg,  target_y)
    right_x = get_lane_x_at_y(right_avg, target_y)

    height_ratio  = 1.0 - (target_y / frame_height)
    fallback_half = int(frame_width * max(0.08, 0.15 - height_ratio * 0.08))

    if left_x is not None and right_x is not None:
        margin     = int((right_x - left_x) * 0.10)
        lane_left  = left_x  + margin
        lane_right = right_x - margin
    elif left_x is not None:
        lane_left  = left_x
        lane_right = left_x + fallback_half * 2
    elif right_x is not None:
        lane_right = right_x
        lane_left  = right_x - fallback_half * 2
    else:
        cx         = int(frame_width * 0.55)
        lane_left  = cx - fallback_half
        lane_right = cx + fallback_half

    return lane_left, lane_right

def is_in_ego_lane(x1, y1, x2, y2, left_avg, right_avg, frame_shape):
    fh, fw  = frame_shape[:2]
    box_cx  = (x1 + x2) // 2

    if box_cx < fw * 0.45:
        return False

    check_y = int((y1 + y2) / 2)
    lane_left, lane_right = get_lane_boundaries_at_y(left_avg, right_avg, fw, check_y, fh)
    return lane_left <= box_cx <= lane_right

def estimate_proximity(x1, y1, x2, y2, frame_shape):
    fh, fw = frame_shape[:2]
    bottom_ratio = y2 / fh
    width_ratio  = (x2 - x1) / fw

    if bottom_ratio > (1 - VEHICLE_DANGER_ZONE) or width_ratio > VEHICLE_SIZE_DANGER:
        return "CLOSE"
    elif bottom_ratio > 0.55 or width_ratio > 0.10:
        return "NEAR"
    return "FAR"

def draw_detections(frame, results, model, left_avg, right_avg):
    if results is None:
        return
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence  = float(box.conf)
            label       = model.names[int(box.cls)]

            if not is_valid_detection(label, confidence, x1, y1, x2, y2, frame.shape):
                continue

            is_vehicle  = label in ("car", "truck", "bus", "bicycle", "motorcycle")
            in_ego_lane = is_in_ego_lane(x1, y1, x2, y2, left_avg, right_avg, frame.shape) if is_vehicle else False
            proximity   = estimate_proximity(x1, y1, x2, y2, frame.shape) if is_vehicle else None

            if label == "person":
                color = (0, 60, 220)
            elif is_vehicle and in_ego_lane and proximity == "CLOSE":
                color = (0, 0, 255)
            elif is_vehicle and in_ego_lane and proximity == "NEAR":
                color = (0, 140, 255)
            else:
                color = (200, 200, 200)

            thickness = 3 if (is_vehicle and in_ego_lane and proximity == "CLOSE") else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if is_vehicle:
                tag  = f"{label}, [{proximity}] , {confidence:.0%}"
            else:
                tag = f"{label}  {confidence:.0%}"

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
# DECISION ENGINE

def compute_decision(left_lines, right_lines, results, model, offset, frame_shape, left_avg, right_avg):
    if results:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf)
                label      = model.names[int(box.cls)]

                if not is_valid_detection(label, confidence, x1, y1, x2, y2, frame_shape):
                    continue

                if label == "person":
                    return "STOP — Pedestrian"
                if label in ("car", "truck", "bus", "bicycle", "motorcycle"):
                    if not is_in_ego_lane(x1, y1, x2, y2, left_avg, right_avg, frame_shape):
                        continue
                    proximity = estimate_proximity(x1, y1, x2, y2, frame_shape)
                    if proximity == "CLOSE":
                        return "STOP — Car Ahead"
                    elif proximity == "NEAR":
                        return "Slow Down"

    if not left_lines and not right_lines:
        return "Go Straight"

    left_count  = len(left_lines)
    right_count = len(right_lines)

    if abs(offset) > 0.6:
        return "Turn Right" if offset < 0 else "Turn Left"

    if left_count > right_count * 1.5:
        return "Turn Right"
    elif right_count > left_count * 1.5:
        return "Turn Left"

    return "Go Straight"

def smooth_decision(raw, buffer):
    buffer.append(raw)
    return max(set(buffer), key=buffer.count)
# HUD OVERLAY

DECISION_COLORS = {
    "Go Straight"        : (200, 200, 200),
    "Turn Left"          : (255, 200,   0),
    "Turn Right"         : (255, 200,   0),
    "Slow Down"          : (  0, 165, 255),
    "STOP — Pedestrian"  : (  0,   0, 220),
    "STOP — Sign"        : (  0,   0, 220),
    "STOP — Car Ahead"   : (  0,   0, 255),
}

def draw_hud(frame, decision):
    h, w = frame.shape[:2]
    color = DECISION_COLORS.get(decision, (255, 255, 255))

    font_d = cv2.FONT_HERSHEY_SIMPLEX
    scale  = 0.65
    thick  = 2
    (tw, th), _ = cv2.getTextSize(decision, font_d, scale, thick)
    pad  = 7
    tx   = w - tw - pad * 2 - 12
    ty   = 12 + th + pad

    cv2.putText(frame, decision, (tx + 1, ty + 1),
                font_d, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(frame, decision, (tx, ty),
                font_d, scale, color, thick, cv2.LINE_AA)

    ax, ay = w - 12 - (tw // 2) - pad, ty + 35
    if "Left" in decision:
        cv2.arrowedLine(frame, (ax + 22, ay), (ax - 22, ay), color, 2, tipLength=0.4)
    elif "Right" in decision:
        cv2.arrowedLine(frame, (ax - 22, ay), (ax + 22, ay), color, 2, tipLength=0.4)
    elif "STOP" in decision:
        cv2.circle(frame, (ax, ay), 12, color, -1)
    else:
        cv2.arrowedLine(frame, (ax, ay + 22), (ax, ay - 22), color, 2, tipLength=0.4)

    return frame
# MAIN LOOP

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    model = YOLO(YOLO_MODEL)

    decision_buf = deque(maxlen=DECISION_BUFFER)

    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    def maybe_resize(f):
        if RESIZE_WIDTH > 0 and f.shape[1] > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / f.shape[1]
            return cv2.resize(f, (RESIZE_WIDTH, int(f.shape[0] * scale)))
        return f

    first = maybe_resize(first)

    frame_count  = 0
    last_results = None
    last_left_avg  = None
    last_right_avg = None

    print("[ Driving Car Simulation ] — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame       = maybe_resize(frame)
        frame_count += 1
        frame_start = time.time()

        edges                   = preprocess_frame(frame)
        masked_edges, roi_pts   = apply_roi(edges)
        left_lines, right_lines = detect_lane_lines(masked_edges)
        left_avg                = average_lane_line_with_memory(left_lines,  frame.shape, last_left_avg)
        right_avg               = average_lane_line_with_memory(right_lines, frame.shape, last_right_avg)
        last_left_avg           = left_avg
        last_right_avg          = right_avg

        if frame_count % YOLO_EVERY_N == 0:
            last_results = run_object_detection(model, frame)
        results = last_results

        offset = calculate_offset(left_avg, right_avg, frame.shape[1])

        raw_decision    = compute_decision(left_lines, right_lines, results, model, offset, frame.shape, left_avg, right_avg)
        stable_decision = smooth_decision(raw_decision, decision_buf)

        output = draw_lanes(frame.copy(), left_avg, right_avg, roi_pts)
        draw_detections(output, results, model, left_avg, right_avg)
        output = draw_hud(output, stable_decision)

        display = cv2.resize(output, (DISPLAY_WIDTH, int(DISPLAY_WIDTH * output.shape[0] / output.shape[1])))
        cv2.imshow("Driving Car Simulation", display)

        elapsed = time.time() - frame_start
        sleep_time = (1.0 / TARGET_FPS) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[ Simulation ended ]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ Simulation stopped by user ]")
        cv2.destroyAllWindows()