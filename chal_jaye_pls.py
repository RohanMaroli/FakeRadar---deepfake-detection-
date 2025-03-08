import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import train_loader, val_loader
from deepfake_model import model
import time
import numpy as np

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Feature Extraction Functions
def extract_deepfake_features(frame, landmarks):
    """Extracts key deepfake-related features from a video frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Inconsistent Lighting
    inconsistent_lighting = np.var(gray)
    
    # Misaligned Facial Features
    eye_distance = abs(landmarks.part(36).x - landmarks.part(45).x)
    
    # Blurry / Jagged Edges
    edge_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Color Mismatch & Resolution Differences
    color_mismatch = np.mean(cv2.absdiff(frame[:, :, 0], frame[:, :, 1]))
    resolution_difference = np.mean(cv2.GaussianBlur(gray, (5,5), 0))
    
    # Jittery Motion & Abnormal Blink Rates
    jittery_motion = edge_var  # Approximate representation
    abnormal_blink_rate = eye_distance / inconsistent_lighting  # Estimation
    
    # Irregular Lip Movement & Lack of Microexpressions
    lip_movement = abs(landmarks.part(48).y - landmarks.part(54).y)
    micro_expression_lack = np.std(gray)
    
    # Eye Gaze Anomalies & Unnatural Head Movements
    eye_gaze_anomalies = abs(landmarks.part(42).x - landmarks.part(39).x)
    unnatural_head_movement = abs(landmarks.part(27).x - landmarks.part(30).x)
    
    return [
        inconsistent_lighting, eye_distance, edge_var, color_mismatch, resolution_difference, 
        jittery_motion, abnormal_blink_rate, lip_movement, micro_expression_lack,
        eye_gaze_anomalies, unnatural_head_movement
    ]

# Training Loop
num_epochs = 10
print("Training started...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    start_time = time.time()
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    val_loss, val_accuracy = 0, 0

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    end_time = time.time()

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {total_loss/len(train_loader):.4f}, "
          f"Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss/len(val_loader):.4f}, "
          f"Val Acc: {val_accuracy:.2f}%, "
          f"Time: {end_time - start_time:.2f}s")

# Save the trained model
torch.save(model.state_dict(), "deepfake_model_2.0.pth")
print("Model saved successfully!")
