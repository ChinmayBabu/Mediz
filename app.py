import os
import cv2
import supervision as sv
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

# 1. INITIALIZE ANNOTATORS 
# Setup: Lighter colors and thin lines
# 'opacity' controls the surface mask (0.2 is very light)
mask_annotator = sv.MaskAnnotator(opacity=0.35) 
# 'thickness' controls the box line (1 is a fine line)
box_annotator = sv.BoxAnnotator(thickness=1) 
label_annotator = sv.LabelAnnotator()

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # Convert predictions to Supervision format
    detections = sv.Detections.from_inference(predictions)
    
    # --- ANNOTATION CHAIN ---
    
    # A. Draw the light surface overlay (The mask)
    annotated_image = mask_annotator.annotate(
        scene=video_frame.image.copy(), 
        detections=detections
    )
    
    # B. Draw matching thin bounding boxes
    annotated_image = box_annotator.annotate(
        scene=annotated_image, 
        detections=detections
    )
    
    # C. Add text labels
    labels = [p["class"] for p in predictions["predictions"]]
    annotated_image = label_annotator.annotate(
        scene=annotated_image, 
        detections=detections, 
        labels=labels
    )

    # Display result
    cv2.imshow("Mediz Inference", annotated_image)
    
    # EXIT ON 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        os._exit(0)

# 4. START PIPELINE
# Ensure your API Key is set in your environment variables 
# OR add: os.environ["ROBOFLOW_API_KEY"] = "YOUR_KEY" above.
pipeline = InferencePipeline.init(
    model_id="mediz/6",
    video_reference=r"D:\Mediz\input.mp4",
    on_prediction=my_custom_sink,
)

pipeline.start()
pipeline.join()