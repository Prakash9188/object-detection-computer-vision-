
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

TEMPLATE_PATH = "template.png"
SCENE_PATH = "scene.png"

# SIFT parameters
SIFT_FEATURES = 2000

# Lowe's ratio test
RATIO_THRESHOLD = 0.87

# Hough Transform parameters
POSITION_BIN_SIZE = 20
SCALE_BIN_SIZE = 0.20
ANGLE_BIN_SIZE = 15

# RANSAC parameters
RANSAC_ITERATIONS = 100
RANSAC_ERROR_THRESHOLD = 5.0

# Multiple object detection
MAX_OBJECTS = 5
MIN_HOUGH_VOTES = 5

# Non-Maximum Suppression
NMS_IOU_THRESHOLD = 0.5


# ============================================================
# STEP 1: LOAD IMAGES
# ============================================================

def load_images():

    template = cv2.imread(TEMPLATE_PATH)
    scene = cv2.imread(SCENE_PATH)

    if template is None:
        raise FileNotFoundError(
            f"Could not load {TEMPLATE_PATH}"
        )

    if scene is None:
        raise FileNotFoundError(
            f"Could not load {SCENE_PATH}"
        )

    print("Images loaded successfully.")

    return template, scene


# ============================================================
# STEP 2: SIFT FEATURE EXTRACTION
# ============================================================

def extract_sift_features(template, scene):

    # Convert both images to grayscale.
    template_gray = cv2.cvtColor(
        template,
        cv2.COLOR_BGR2GRAY
    )

    scene_gray = cv2.cvtColor(
        scene,
        cv2.COLOR_BGR2GRAY
    )

    # Create SIFT detector.
    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES
    )

    # Extract keypoints and descriptors.
    template_keypoints, template_descriptors = \
        sift.detectAndCompute(
            template_gray,
            None
        )

    scene_keypoints, scene_descriptors = \
        sift.detectAndCompute(
            scene_gray,
            None
        )

    if template_descriptors is None:
        raise RuntimeError(
            "No SIFT descriptors found in template."
        )

    if scene_descriptors is None:
        raise RuntimeError(
            "No SIFT descriptors found in scene."
        )

    print(
        "Template keypoints:",
        len(template_keypoints)
    )

    print(
        "Scene keypoints:",
        len(scene_keypoints)
    )

    print(
        "Template descriptor shape:",
        template_descriptors.shape
    )

    print(
        "Scene descriptor shape:",
        scene_descriptors.shape
    )

    return (
        template_gray,
        scene_gray,
        template_keypoints,
        scene_keypoints,
        template_descriptors,
        scene_descriptors
    )


# ============================================================
# STEP 3: FAST SIFT FEATURE MATCHING
# ============================================================

def match_features(
        scene_descriptors,
        template_descriptors,
        ratio_threshold=RATIO_THRESHOLD):

    print("\nMatching SIFT features...")

    # SIFT descriptors use Euclidean distance.
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    # Find the two nearest template descriptors
    # for every scene descriptor.
    knn_matches = matcher.knnMatch(
        scene_descriptors,
        template_descriptors,
        k=2
    )

    good_matches = []

    # Apply Lowe's ratio test.
    for pair in knn_matches:

        if len(pair) < 2:
            continue

        best_match = pair[0]
        second_best_match = pair[1]

        ratio = (
            best_match.distance /
            (second_best_match.distance + 1e-10)
        )

        if ratio < ratio_threshold:

            good_matches.append({
                "scene_index": best_match.queryIdx,
                "template_index": best_match.trainIdx,
                "distance": best_match.distance,
                "ratio": ratio
            })

    print(
        "Number of good matches:",
        len(good_matches)
    )

    return good_matches


# ============================================================
# STEP 4: GENERALIZED HOUGH TRANSFORM
# ============================================================

def predict_object_parameters(
        template_keypoint,
        scene_keypoint,
        template_center_x,
        template_center_y):

    # Template keypoint coordinates.
    xt, yt = template_keypoint.pt

    # Scene keypoint coordinates.
    xs, ys = scene_keypoint.pt

    # Estimate scale.
    scale = (
        scene_keypoint.size /
        (template_keypoint.size + 1e-10)
    )

    # Estimate rotation.
    angle = (
        scene_keypoint.angle -
        template_keypoint.angle
    )

    angle = angle % 360

    # Vector from template keypoint
    # to template center.
    dx = template_center_x - xt
    dy = template_center_y - yt

    # Convert angle to radians.
    theta = np.radians(angle)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rotate the vector.
    rotated_dx = (
        dx * cos_theta -
        dy * sin_theta
    )

    rotated_dy = (
        dx * sin_theta +
        dy * cos_theta
    )

    # Apply scale.
    rotated_dx *= scale
    rotated_dy *= scale

    # Predict object center.
    center_x = xs + rotated_dx
    center_y = ys + rotated_dy

    return (
        center_x,
        center_y,
        scale,
        angle
    )


# ============================================================
# CREATE HOUGH BIN
# ============================================================

def create_hough_bin(
        center_x,
        center_y,
        scale,
        angle):

    x_bin = int(
        center_x / POSITION_BIN_SIZE
    )

    y_bin = int(
        center_y / POSITION_BIN_SIZE
    )

    scale_bin = int(
        scale / SCALE_BIN_SIZE
    )

    angle_bin = int(
        angle / ANGLE_BIN_SIZE
    )

    return (
        x_bin,
        y_bin,
        scale_bin,
        angle_bin
    )


# ============================================================
# BUILD HOUGH SPACE
# ============================================================

def build_hough_space(
        matches,
        scene_keypoints,
        template_keypoints,
        template_center_x,
        template_center_y):

    hough_space = {}

    for match in matches:

        scene_index = match["scene_index"]
        template_index = match["template_index"]

        scene_kp = scene_keypoints[scene_index]
        template_kp = template_keypoints[template_index]

        center_x, center_y, scale, angle = \
            predict_object_parameters(
                template_kp,
                scene_kp,
                template_center_x,
                template_center_y
            )

        bin_id = create_hough_bin(
            center_x,
            center_y,
            scale,
            angle
        )

        # Store only the match.
        # We do not need to store center_x,
        # center_y, scale and angle again.
        if bin_id not in hough_space:
            hough_space[bin_id] = []

        hough_space[bin_id].append(match)

    return hough_space


# ============================================================
# STEP 5: AFFINE RANSAC
# ============================================================

def ransac_affine(
        matches,
        scene_points,
        template_points,
        iterations=RANSAC_ITERATIONS,
        error_threshold=RANSAC_ERROR_THRESHOLD):

    # At least 3 point correspondences are
    # required for an affine transformation.
    if len(matches) < 3:
        return None, []

    # --------------------------------------------------------
    # Convert matches into point arrays.
    # --------------------------------------------------------

    src_points = np.array(
        [
            template_points[
                match["template_index"]
            ]
            for match in matches
        ],
        dtype=np.float32
    )

    dst_points = np.array(
        [
            scene_points[
                match["scene_index"]
            ]
            for match in matches
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # OpenCV performs RANSAC internally.
    #
    # This is considerably faster than manually
    # implementing 500 Python iterations.
    # --------------------------------------------------------

    matrix, inlier_mask = cv2.estimateAffine2D(
        src_points,
        dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=error_threshold,
        maxIters=iterations,
        confidence=0.99,
        refineIters=10
    )

    if matrix is None:
        return None, []

    # Convert mask into Boolean array.
    inlier_mask = (
        inlier_mask.ravel().astype(bool)
    )

    # Get the actual inlier matches.
    inliers = [
        matches[i]
        for i in range(len(matches))
        if inlier_mask[i]
    ]

    # Need at least 3 inliers for a useful
    # affine detection.
    if len(inliers) < 3:
        return None, []

    return matrix, inliers


# ============================================================
# STEP 6: TRANSFORM TEMPLATE CORNERS
# ============================================================

def get_transformed_corners(
        template_gray,
        affine_matrix):

    template_h, template_w = \
        template_gray.shape

    # Four corners of template.
    corners = np.array(
        [
            [0, 0],
            [template_w, 0],
            [template_w, template_h],
            [0, template_h]
        ],
        dtype=np.float32
    )

    # OpenCV can transform all four points
    # at once.
    transformed_corners = cv2.transform(
        corners.reshape(-1, 1, 2),
        affine_matrix
    )

    return transformed_corners.reshape(-1, 2)


# ============================================================
# STEP 7: MULTIPLE OBJECT DETECTION
# ============================================================

def detect_multiple_objects(
        good_matches,
        scene_keypoints,
        template_keypoints,
        template_center_x,
        template_center_y,
        template_gray,
        max_objects=MAX_OBJECTS,
        min_hough_votes=MIN_HOUGH_VOTES):

    # Work on a copy so that the original
    # matches remain unchanged.
    remaining_matches = good_matches.copy()

    detections = []

    # --------------------------------------------------------
    # Precompute keypoint coordinates.
    # This avoids repeatedly accessing .pt.
    # --------------------------------------------------------

    template_points = np.array(
        [
            kp.pt
            for kp in template_keypoints
        ],
        dtype=np.float32
    )

    scene_points = np.array(
        [
            kp.pt
            for kp in scene_keypoints
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Search for multiple objects.
    # --------------------------------------------------------

    for object_number in range(max_objects):

        print(
            f"\nSearching for object "
            f"{object_number + 1}..."
        )

        if len(remaining_matches) < 3:
            print(
                "Not enough remaining matches."
            )
            break

        # Build Hough space.
        hough_space = build_hough_space(
            remaining_matches,
            scene_keypoints,
            template_keypoints,
            template_center_x,
            template_center_y
        )

        if not hough_space:
            print("No Hough candidates.")
            break

        # Sort bins according to number of votes.
        sorted_bins = sorted(
            hough_space.items(),
            key=lambda item: len(item[1]),
            reverse=True
        )

        found_object = False

        # ----------------------------------------------------
        # Try strongest Hough clusters first.
        # ----------------------------------------------------

        for bin_id, cluster_matches in sorted_bins:

            # Ignore weak clusters.
            if len(cluster_matches) < min_hough_votes:
                break

            print(
                "Trying Hough cluster with",
                len(cluster_matches),
                "matches"
            )

            # ------------------------------------------------
            # Run affine RANSAC.
            # ------------------------------------------------

            matrix, inliers = ransac_affine(
                cluster_matches,
                scene_points,
                template_points
            )

            if matrix is None:
                continue

            if len(inliers) < 3:
                continue

            # ------------------------------------------------
            # Calculate transformed template corners.
            # ------------------------------------------------

            corners = get_transformed_corners(
                template_gray,
                matrix
            )

            # ------------------------------------------------
            # Store detection.
            # ------------------------------------------------

            detection = {
                "matrix": matrix,
                "inliers": inliers,
                "hough_bin": bin_id,
                "corners": corners,
                "score": len(inliers)
            }

            detections.append(detection)

            print(
                "Object detected!",
                "Inliers:",
                len(inliers)
            )

            # ------------------------------------------------
            # Remove matches used by this object.
            # ------------------------------------------------

            inlier_ids = {
                (
                    match["scene_index"],
                    match["template_index"]
                )
                for match in inliers
            }

            remaining_matches = [
                match
                for match in remaining_matches
                if (
                    match["scene_index"],
                    match["template_index"]
                ) not in inlier_ids
            ]

            found_object = True

            break

        if not found_object:

            print(
                "No more valid objects found."
            )

            break

    return detections


# ============================================================
# STEP 8: BOUNDING BOX
# ============================================================

def corners_to_bbox(corners):

    x_min = np.min(corners[:, 0])
    y_min = np.min(corners[:, 1])

    x_max = np.max(corners[:, 0])
    y_max = np.max(corners[:, 1])

    return [
        x_min,
        y_min,
        x_max,
        y_max
    ]


# ============================================================
# STEP 9: IoU interscetion / union
# ============================================================

def calculate_iou(box1, box2):

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection coordinates.
    intersection_x_min = max(
        x1_min,
        x2_min
    )

    intersection_y_min = max(
        y1_min,
        y2_min
    )

    intersection_x_max = min(
        x1_max,
        x2_max
    )

    intersection_y_max = min(
        y1_max,
        y2_max
    )

    # Intersection dimensions.
    width = max(
        0,
        intersection_x_max -
        intersection_x_min
    )

    height = max(
        0,
        intersection_y_max -
        intersection_y_min
    )

    intersection_area = (
        width * height
    )

    # Individual box areas.
    area1 = (
        (x1_max - x1_min) *
        (y1_max - y1_min)
    )

    area2 = (
        (x2_max - x2_min) *
        (y2_max - y2_min)
    )

    # Union area.
    union_area = (
        area1 +
        area2 -
        intersection_area
    )

    if union_area <= 0:
        return 0.0

    return (
        intersection_area /
        union_area
    )


# ============================================================
# STEP 10: NON-MAXIMUM SUPPRESSION
# ============================================================

def non_maximum_suppression(
        detections,
        iou_threshold=NMS_IOU_THRESHOLD):

    if not detections:
        return []

    # Make sure every detection has a bbox.
    for detection in detections:

        detection["bbox"] = \
            corners_to_bbox(
                detection["corners"]
            )

        detection["score"] = \
            len(detection["inliers"])

    # Strongest detection first.
    detections = sorted(
        detections,
        key=lambda d: d["score"],
        reverse=True
    )

    kept_detections = []

    while detections:

        # Select strongest detection.
        best = detections.pop(0)

        kept_detections.append(best)

        remaining = []

        for detection in detections:

            iou = calculate_iou(
                best["bbox"],
                detection["bbox"]
            )

            if iou < iou_threshold:

                remaining.append(
                    detection
                )

            else:

                print(
                    "Duplicate detection removed.",
                    "IoU:",
                    round(iou, 3)
                )

        detections = remaining

    return kept_detections


# ============================================================
# STEP 11: DRAW FINAL DETECTIONS
# ============================================================

def draw_detections(
        scene,
        detections):

    result = scene.copy()

    for i, detection in enumerate(
        detections
    ):

        # We already calculated the corners
        # during detection.
        corners = np.int32(
            detection["corners"]
        )

        # Draw four sides of the
        # transformed template.
        for j in range(4):

            start_point = tuple(
                corners[j]
            )

            end_point = tuple(
                corners[
                    (j + 1) % 4
                ]
            )

            cv2.line(
                result,
                start_point,
                end_point,
                (0, 255, 0),
                3
            )

        # Label position.
        label_x = int(
            corners[0][0]
        )

        label_y = int(
            corners[0][1] - 10
        )

        cv2.putText(
            result,
            "Object " + str(i + 1),
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return result


# ============================================================
# STEP 12: DISPLAY SIFT KEYPOINTS
# ============================================================

def show_sift_keypoints(
        template_gray,
        scene_gray,
        template_keypoints,
        scene_keypoints):

    template_image = cv2.drawKeypoints(
        template_gray,
        template_keypoints,
        None
    )

    scene_image = cv2.drawKeypoints(
        scene_gray,
        scene_keypoints,
        None
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.subplot(1, 2, 1)

    plt.imshow(
        template_image,
        cmap="gray"
    )

    plt.title(
        "Template SIFT Keypoints"
    )

    plt.axis("off")

    plt.subplot(1, 2, 2)

    plt.imshow(
        scene_image,
        cmap="gray"
    )

    plt.title(
        "Scene SIFT Keypoints"
    )

    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("SIFT + HOUGH + AFFINE RANSAC OBJECT DETECTION")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load images.
    # --------------------------------------------------------

    template, scene = load_images()

    # --------------------------------------------------------
    # 2. Extract SIFT features.
    # --------------------------------------------------------

    (
        template_gray,
        scene_gray,
        template_keypoints,
        scene_keypoints,
        template_descriptors,
        scene_descriptors
    ) = extract_sift_features(
        template,
        scene
    )

    # --------------------------------------------------------
    # 3. Match SIFT descriptors.
    # --------------------------------------------------------

    good_matches = match_features(
        scene_descriptors,
        template_descriptors,
        RATIO_THRESHOLD
    )

    # --------------------------------------------------------
    # Check if enough matches exist.
    # --------------------------------------------------------

    if len(good_matches) < 3:

        print(
            "\nNot enough good matches "
            "for affine transformation."
        )

        return

    # --------------------------------------------------------
    # 4. Template center.
    # --------------------------------------------------------

    template_h, template_w = \
        template_gray.shape

    template_center_x = \
        template_w / 2.0

    template_center_y = \
        template_h / 2.0

    # --------------------------------------------------------
    # 5. Detect multiple objects.
    # --------------------------------------------------------

    detections = detect_multiple_objects(
        good_matches,
        scene_keypoints,
        template_keypoints,
        template_center_x,
        template_center_y,
        template_gray,
        max_objects=MAX_OBJECTS,
        min_hough_votes=MIN_HOUGH_VOTES
    )

    print(
        "\nCandidate detections:",
        len(detections)
    )

    # --------------------------------------------------------
    # 6. Apply NMS.
    # --------------------------------------------------------

    detections = non_maximum_suppression(
        detections,
        NMS_IOU_THRESHOLD
    )

    print(
        "Objects after NMS:",
        len(detections)
    )

    # --------------------------------------------------------
    # 7. Print detection information.
    # --------------------------------------------------------

    for i, detection in enumerate(
        detections
    ):

        print(
            "\nObject",
            i + 1
        )

        print(
            "Number of inliers:",
            len(
                detection["inliers"]
            )
        )

        print(
            "Bounding box:",
            detection["bbox"]
        )

        print(
            "Affine matrix:"
        )

        print(
            detection["matrix"]
        )

    # --------------------------------------------------------
    # 8. Draw final detections.
    # --------------------------------------------------------

    result = draw_detections(
        scene,
        detections
    )

    # --------------------------------------------------------
    # 9. Display result.
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 8)
    )

    plt.imshow(
        cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        "Final Detected Objects"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()

    print("\nDetection completed.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
