
import cv2
import numpy as np

def generate_all_forensic_stages(image_path):
    print("Initializing comprehensive forensic analysis protocol")
    
    # ---------------------------------------------------------
    # Read the original image
    # ---------------------------------------------------------
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("Error: Image not found. Please verify the file path.")
        return

    # ==========================================
    # Stage 1: Grayscale Conversion
    # ==========================================
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Stage_1_Grayscale.jpg", gray_img)
    print("- Stage 1 generated: Grayscale image")

    # ==========================================
    # Stage 2: Keypoint Extraction and Drawing
    # ==========================================
    orb = cv2.ORB_create(nfeatures=5000)
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)
    
    # Draw all detected keypoints on the original image (in green)
    keypoints_img = cv2.drawKeypoints(original_img, keypoints, None, color=(0, 255, 0), flags=0)
    cv2.imwrite("Stage_2_Keypoints.jpg", keypoints_img)
    print("- Stage 2 generated: Detected keypoints")

    # ---------------------------------------------------------
    # Initial Matching and Filtering
    # ---------------------------------------------------------
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors, descriptors, k=3)
    
    good_matches = []
    for match in matches:
        if len(match) >= 3:
            m1, m2, m3 = match[0], match[1], match[2]
            pt1 = np.array(keypoints[m1.queryIdx].pt)
            pt2 = np.array(keypoints[m2.trainIdx].pt)
            spatial_distance = np.linalg.norm(pt1 - pt2)
            
            if spatial_distance >= 80 and m2.distance < 0.70 * m3.distance:
                good_matches.append(m2)

    # ---------------------------------------------------------
    # Geometric Filtering (RANSAC) and Preparation
    # ---------------------------------------------------------
    if len(good_matches) > 15:
        src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        
        # Prepare empty copies for drawing
        lines_img = original_img.copy()            # For Stage 3
        black_mask = np.zeros_like(gray_img)       # For Stage 4 (Mask)
        patch_radius = 18                          # Patch radius for edges
        
        for i, match in enumerate(good_matches):
            if matches_mask[i] == 1: 
                pt1 = tuple(np.round(keypoints[match.queryIdx].pt).astype(int))
                pt2 = tuple(np.round(keypoints[match.trainIdx].pt).astype(int))
                
                # ==========================================
                # Stage 3: Draw Matching Lines
                # ==========================================
                cv2.line(lines_img, pt1, pt2, (0, 255, 0), thickness=2)
                cv2.circle(lines_img, pt1, radius=4, color=(0, 0, 255), thickness=-1)
                cv2.circle(lines_img, pt2, radius=4, color=(255, 0, 0), thickness=-1)
                
                # ==========================================
                # Stage 4: Prepare the Black Forensic Mask
                # ==========================================
                cv2.circle(black_mask, pt1, patch_radius, 255, -1)
                cv2.circle(black_mask, pt2, patch_radius, 255, -1)

        # Save Stage 3 image
        cv2.imwrite("Stage_3_Matched_Lines.jpg", lines_img)
        print("- Stage 3 generated: Geometric matching lines")
        
        # Merge mask with original image to extract only the forged region
        extracted_forgery = cv2.bitwise_and(original_img, original_img, mask=black_mask)
        # Saved as PNG to maintain maximum quality and sharpness of extracted edges
        cv2.imwrite("Stage_4_Extracted_Forgery.png", extracted_forgery)
        print("- Stage 4 generated: Extracted forgery with black background")
        
        print("\nProcess completed successfully. The four images are ready in the working directory.")
        
    else:
        print("\nNo significant forgery detected for extraction.")

if __name__ == "__main__":
    # Path of the image to be analyzed - update this path to point to your image file
    IMAGE_PATH = "D:\\2\\Forgery Image.png" 
    generate_all_forensic_stages(IMAGE_PATH)
