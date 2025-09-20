import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import time
from collections import deque
from datetime import datetime
from sklearn.cluster import KMeans

def calculate_fps(fps_history):
    current_time = time.time()
    fps_history.append(current_time)
    
    # Calculate FPS over the last second
    while fps_history and current_time - fps_history[0] > 1.0:
        fps_history.popleft()
    
    return len(fps_history)

def draw_status_bar(frame, fps, shirt_name, color_rec=None, style_rec=None):
    # Extract filename without extension for display
    shirt_display = os.path.splitext(os.path.basename(shirt_name))[0]
    
    # Create semi-transparent background for status bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)  # Increased height for recommendations
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Add FPS and shirt name
    cv2.putText(frame, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Active Shirt: {shirt_display}", (frame.shape[1]//3, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add recommendations if available
    if color_rec:
        cv2.putText(frame, f"Color: {color_rec}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if style_rec:
        cv2.putText(frame, f"Style: {style_rec}", (frame.shape[1]//3, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame

def draw_thumbnail(frame, shirt_image, mask):
    # Calculate thumbnail size (15% of frame width)
    thumb_width = int(frame.shape[1] * 0.15)
    aspect_ratio = shirt_image.shape[0] / shirt_image.shape[1]
    thumb_height = int(thumb_width * aspect_ratio)
    
    # Resize shirt and mask for thumbnail
    thumb_shirt = cv2.resize(shirt_image, (thumb_width, thumb_height))
    thumb_mask = cv2.resize(mask, (thumb_width, thumb_height))
    
    # Convert mask to 3 channels if needed
    if len(thumb_mask.shape) == 2:
        thumb_mask = cv2.cvtColor(thumb_mask, cv2.COLOR_GRAY2BGR)
    
    # Position thumbnail in top-right corner with padding
    padding = 10
    y_start = padding
    y_end = y_start + thumb_height
    x_start = frame.shape[1] - thumb_width - padding
    x_end = x_start + thumb_width
    
    # Create semi-transparent overlay for thumbnail
    alpha = thumb_mask / 255.0
    roi = frame[y_start:y_end, x_start:x_end].astype(float)
    overlay = (thumb_shirt.astype(float) * alpha + roi * (1 - alpha)).astype(np.uint8)
    frame[y_start:y_end, x_start:x_end] = overlay
    
    # Draw border around thumbnail
    cv2.rectangle(frame, (x_start-1, y_start-1), (x_end+1, y_end+1), (255, 255, 255), 1)
    
    return frame

def save_screenshot(frame):
    # Create screenshots directory if it doesn't exist
    screenshots_dir = "screenshots"
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)
        
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(screenshots_dir, f"virtual_tryon_{timestamp}.png")
    
    # Save the image
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved as {filename}")

# --- Define your UNet model architecture here 
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout_prob=0.1):
        super(UNet, self).__init__()
        self.dropout_prob = dropout_prob

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.middle = nn.Sequential(
            CBR(512, 1024),
            nn.Dropout(dropout_prob)
        )
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        m = self.middle(self.pool(e4))
        d4 = self.up4(m)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return torch.sigmoid(out)

# Initialize device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize your UNet model with dropout
model = UNet(in_channels=3, out_channels=1, dropout_prob=0.1)

# Load the saved weights from local path
weights_path = "new_segmentation_unet.pth"
try:
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Loading from training checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print(f"Successfully loaded weights from: {weights_path}")
    
    model = model.to(DEVICE)
    model.eval()    # Quick test to ensure model works
    dummy_input = torch.randn(1, 3, 256, 192).to(DEVICE)
    with torch.no_grad():
        try:
            _ = model(dummy_input)
            print("Model successfully verified with test input")
        except Exception as e:
            print(f"Error during model verification: {e}")
            exit()
            
except FileNotFoundError:
    print(f"Error: Weights file not found at: {weights_path}")
    print("Make sure the weights file is in the same directory as this script")
    exit()
except Exception as e:
    print(f"Error loading weights: {e}")
    exit()

# Define the preprocessing steps for the shirt image
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 192)),  # Common aspect ratio for fashion images (4:3)
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Add robustness to lighting
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Initialize MediaPipe Pose and Camera ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --- Function to open file dialog and select an image ---
def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Shirt Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    return file_path

# --- Get the shirt image using file dialog ---
shirt_image_path = select_image()

if not shirt_image_path:
    print("No image selected. Exiting.")
    exit()

# --- Load shirt image ---
shirt_image = cv2.imread(shirt_image_path, cv2.IMREAD_UNCHANGED)
if shirt_image is None:
    print("Error loading image. Please check the file path.")
    exit()

# --- Replace the extract_shirt function with UNet inference ---
def segment_shirt(img_np):
    try:
        # Convert to RGB for preprocessing
        img_pil = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            mask_prob = output.squeeze().cpu().numpy()

        # Apply threshold with higher confidence
        binary_mask = (mask_prob > 0.3).astype(np.uint8) * 255
        
        if binary_mask.sum() == 0:
            print("Warning: Model produced empty segmentation mask")
            return np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        
        # Resize the mask to original image size
        mask_resized = cv2.resize(binary_mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Post-processing to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)  # Fill small holes
        mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)   # Remove small noise
        
        # Find contours and keep only the largest one (the shirt)
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_resized = np.zeros_like(mask_resized)
            cv2.drawContours(mask_resized, [largest_contour], -1, 255, -1)
        
        return mask_resized
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

# Initialize FPS calculation
fps_history = deque(maxlen=150)

# Get the segmentation mask from the UNet model
shirt_mask = segment_shirt(shirt_image)

# Apply the mask to extract the shirt (assuming white background)
shirt_no_bg = cv2.bitwise_and(shirt_image, shirt_image, mask=shirt_mask)

# Show the extracted shirt and its mask (for debugging)
cv2.imshow("Extracted Shirt (UNet)", shirt_no_bg)
cv2.imshow("Shirt Mask (UNet)", shirt_mask)

# --- Set up webcam and window ---
cap = cv2.VideoCapture(0)
cv2.namedWindow('Virtual Try-On', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Virtual Try-On', 800, 600)

shirt_name = shirt_image_path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        
        # Get body measurements
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Get face landmarks for skin tone detection
        face_landmarks = [
            landmarks[mp_pose.PoseLandmark.NOSE],
            landmarks[mp_pose.PoseLandmark.LEFT_EYE],
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE],
            landmarks[mp_pose.PoseLandmark.LEFT_EAR],
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR],
        ]
        
        # Analyze body type and skin tone
        body_type = analyze_body_type(landmarks)
        skin_color, skin_tone = detect_skin_tone(frame, face_landmarks)
        
        # Get color and style recommendations
        if skin_color is not None:
            _, color_recommendation = get_color_recommendation(skin_tone, cv2.mean(shirt_no_bg)[:3])
        else:
            color_recommendation = "Could not analyze skin tone"
            
        if body_type:
            _, style_recommendation = get_style_recommendation(body_type, shirt_no_bg)
        else:
            style_recommendation = "Could not analyze body type"
            
        # Get frame dimensions
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        
        # Calculate shirt width based on shoulder width with increased scaling
        shoulder_distance = abs(left_shoulder.x - right_shoulder.x)
        shirt_width = int(shoulder_distance * frame_width * 1.8)  # Increased from 1.4
        
        # Calculate torso height and use it for shirt height
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        torso_height = abs(hip_y - shoulder_y) * frame_height
          # Set shirt height based on torso height with increased scaling
        aspect_ratio = shirt_no_bg.shape[0] / shirt_no_bg.shape[1]
        shirt_height = int(torso_height * 1.7)  # Increased from 1.3
        
        # Resize shirt and mask
        shirt_resized = cv2.resize(shirt_no_bg, (shirt_width, shirt_height))
        mask_resized = cv2.resize(shirt_mask, (shirt_width, shirt_height))

        # Calculate shirt position
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shirt_x = int(shoulder_center_x * frame_width - shirt_width / 2)
        shirt_y = int(shoulder_y * frame_height - shirt_height * 0.2)  # Place slightly below shoulders

        # Ensure shirt stays within frame bounds
        shirt_x = max(0, min(shirt_x, frame_width - shirt_width))
        shirt_y = max(0, min(shirt_y, frame_height - shirt_height))
        shirt_end_x = shirt_x + shirt_width
        shirt_end_y = shirt_y + shirt_height

        try:
            shirt_region = frame[shirt_y:shirt_end_y, shirt_x:shirt_end_x]
            resized_mask = cv2.resize(mask_resized, (shirt_region.shape[1], shirt_region.shape[0])).astype(np.uint8)
            resized_shirt = cv2.resize(shirt_resized, (shirt_region.shape[1], shirt_region.shape[0]))

            # Ensure the mask has the same number of channels as the shirt region (grayscale to BGR)
            if len(resized_mask.shape) == 2:
                resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)

            # Use the mask to blend the shirt with the background
            alpha = resized_mask / 255.0
            frame_roi = frame[shirt_y:shirt_end_y, shirt_x:shirt_end_x].astype(float)
            shirt_overlay = (resized_shirt.astype(float) * alpha + frame_roi * (1 - alpha)).astype(np.uint8)
            frame[shirt_y:shirt_end_y, shirt_x:shirt_end_x] = shirt_overlay

        except ValueError:
            print("Error: Shirt region out of bounds (still).")        # Calculate FPS
    fps = calculate_fps(fps_history)

    # Draw status bar with recommendations if available
    frame = draw_status_bar(frame, fps, shirt_name, 
                          color_rec=color_recommendation if 'color_recommendation' in locals() else None,
                          style_rec=style_recommendation if 'style_recommendation' in locals() else None)

    # Draw thumbnail
    frame = draw_thumbnail(frame, shirt_no_bg, shirt_mask)

    # Show frame and handle key events
    cv2.imshow('Virtual Try-On', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  
        break
    elif key == ord('s'): 
        save_screenshot(frame)

cap.release()
cv2.destroyAllWindows()

def detect_skin_tone(frame, face_landmarks):
    # Extract face region
    face_points = []
    for landmark in face_landmarks:
        face_points.append([int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])])
    
    if not face_points:
        return None, None
    
    # Convert to numpy array
    face_points = np.array(face_points)
    
    # Get bounding box of face
    x, y, w, h = cv2.boundingRect(face_points)
    face_region = frame[y:y+h, x:x+w]
    
    if face_region.size == 0:
        return None, None
    
    # Convert to RGB and reshape for KMeans
    face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
    pixels = face_region_rgb.reshape((-1, 3))
    
    # Use KMeans to find dominant skin color
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    
    # Classify skin tone
    # Using a simplified classification system
    l = 0.2126 * dominant_color[0] + 0.7152 * dominant_color[1] + 0.0722 * dominant_color[2]
    
    if l > 200:
        return dominant_color, "fair"
    elif l > 170:
        return dominant_color, "medium"
    elif l > 140:
        return dominant_color, "olive"
    else:
        return dominant_color, "dark"

def analyze_body_type(landmarks):
    # Get key measurements
    shoulder_width = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x - 
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
    hip_width = abs(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x - 
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x)
    waist_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + 
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
    shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
    torso_length = waist_y - shoulder_y
    
    # Determine body type
    shoulder_hip_ratio = shoulder_width / hip_width
    
    if shoulder_hip_ratio > 1.1:
        return "inverted triangle"
    elif shoulder_hip_ratio < 0.9:
        return "pear"
    elif torso_length > 0.3:  # Relative to height
        return "rectangle"
    else:
        return "hourglass"

def get_color_recommendation(skin_tone, shirt_color):
    # Define complementary colors for each skin tone
    skin_tone_matches = {
        "fair": ["navy", "burgundy", "forest green", "purple", "blue"],
        "medium": ["red", "orange", "olive", "teal", "white"],
        "olive": ["coral", "turquoise", "purple", "cream", "brown"],
        "dark": ["white", "yellow", "bright blue", "red", "pink"]
    }
    
    # Convert BGR shirt color to HSV for better color analysis
    shirt_hsv = cv2.cvtColor(np.uint8([[shirt_color]]), cv2.COLOR_BGR2HSV)[0][0]
    
    if skin_tone in skin_tone_matches:
        recommended_colors = skin_tone_matches[skin_tone]
        # Basic color matching logic
        if shirt_hsv[1] < 50:  # Low saturation (close to white/black/gray)
            return True, "This neutral color works well with your skin tone."
        elif skin_tone == "fair" and shirt_hsv[2] > 200:  # Too bright
            return False, "This color might be too bright for your skin tone. Try darker shades."
        elif skin_tone == "dark" and shirt_hsv[2] < 50:  # Too dark
            return False, "This color might be too dark. Try brighter or more vibrant colors."
        else:
            return True, "This color complements your skin tone well."
    
    return None, "Could not analyze color compatibility."

def get_style_recommendation(body_type, shirt_image):
    # Calculate shirt proportions
    height, width = shirt_image.shape[:2]
    aspect_ratio = height / width
    
    recommendations = {
        "inverted triangle": {
            "good": aspect_ratio < 1.5,
            "message": "Fitted shirts that taper at the waist work best for your inverted triangle shape."
        },
        "pear": {
            "good": aspect_ratio > 1.2,
            "message": "Longer shirts with a slight flare can balance your proportions."
        },
        "rectangle": {
            "good": 1.2 <= aspect_ratio <= 1.8,
            "message": "This classic fit works well for your straight body type."
        },
        "hourglass": {
            "good": 1.3 <= aspect_ratio <= 1.7,
            "message": "Fitted shirts that accentuate your waist suit your hourglass shape."
        }
    }
    
    if body_type in recommendations:
        return recommendations[body_type]["good"], recommendations[body_type]["message"]
    
    return None, "Could not analyze style compatibility."