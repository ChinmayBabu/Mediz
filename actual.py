import os
import cv2
import supervision as sv
from inference import InferencePipeline

# Setup
MODEL_ID = "mediz/6"
INPUT_VIDEO = r"D:\Mediz\input.mp4"
OUTPUT_VIDEO = r"D:\Mediz\output_annotated.mp4"

# Annotators
mask_annotator = sv.MaskAnnotator(opacity=0.35)
box_annotator = sv.BoxAnnotator(thickness=1)
label_annotator = sv.LabelAnnotator()

# Initialize OpenCV VideoWriter
cap = cv2.VideoCapture(INPUT_VIDEO)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
cap.release()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

def open_cv_sink(predictions: dict, video_frame):
    detections = sv.Detections.from_inference(predictions)
    
    # Annotate
    img = mask_annotator.annotate(scene=video_frame.image.copy(), detections=detections)
    img = box_annotator.annotate(scene=img, detections=detections)
    
    labels = [p["class"] for p in predictions["predictions"]]
    img = label_annotator.annotate(scene=img, detections=detections, labels=labels)

    # Save frame
    out.write(img)

# Start Pipeline
pipeline = InferencePipeline.init(
    model_id=MODEL_ID,
    video_reference=INPUT_VIDEO,
    on_prediction=open_cv_sink,
)

print(f"Saving to {OUTPUT_VIDEO} using OpenCV...")
pipeline.start()
pipeline.join()

out.release()
print("Export complete.")