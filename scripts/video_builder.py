import cv2
import os
import shutil
from configs import VIDEO_DIR

class VideoBuilder:
    def __init__(self, filename, fps, width, height):
        self.filename = filename
        # Initialize with a temporary or default path
        self.output_path = os.path.join(str(VIDEO_DIR), self.filename)
        self.fps = fps
        self.width = width
        self.height = height
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID'
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        if not self.writer.isOpened():
            raise RuntimeError(f"❌ Cannot open video writer for: {self.output_path}")

        # Ensure subdirectories exist
        os.makedirs(os.path.join(str(VIDEO_DIR), "fakes"), exist_ok=True)
        os.makedirs(os.path.join(str(VIDEO_DIR), "reals"), exist_ok=True)

    def update(self, frame, bbox, label, conf):
        """
        Draws bounding box, prediction text, and confidence bar on the frame and writes it to the video.
        """
        # Determine label and color
        label = str(label).upper()
        is_fake = label == "FAKE"
        color = (0, 0, 255) if is_fake else (0, 255, 0) # Red for FAKE, Green for REAL
        
        # Draw bounding box
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Draw large text label
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

        # Draw confidence bar
        if is_fake:
            prob_fake = conf
            # Update target path for later moving
            self.target_path = os.path.join(str(VIDEO_DIR), "fakes", self.filename)
        else:
            prob_fake = 1.0 - conf
            # Update target path for later moving
            self.target_path = os.path.join(str(VIDEO_DIR), "reals", self.filename)
        
        bar_x = 10
        bar_y = 100
        bar_w = 400
        bar_h = 30
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        
        # Filled bar (proportional to prob_fake)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * prob_fake), bar_y + bar_h), color, -1)
        
        # Bar Label
        bar_text = "FAKE %" if prob_fake > 0.5 else "REAL %"
        cv2.putText(frame, bar_text, (bar_x + 10, bar_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self.writer.write(frame)

    def release(self):
        if self.writer:
            self.writer.release()
            
        # Move file to target path if set and different
        if hasattr(self, 'target_path') and self.target_path != self.output_path:
            try:
                print(f"Moving video to: {self.target_path}")
                shutil.move(self.output_path, self.target_path)
            except Exception as e:
                print(f"❌ Failed to move video: {e}")
