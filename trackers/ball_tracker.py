from ultralytics import YOLO
import cv2
import pickle # pickle is used to serialize and deserialize a Python object structure. In this case, we will use it to store the player detections

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None: # If read_from_stub is True and the stub path is provided, then we will read the player detections from the stub path
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)

        if stub_path is not None: # If the stub path is provided, then we will store the player detections in the stub path
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f) 
        
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        return ball_detections
            
    def detect_frame(self, frames):
        results = self.model.predict(frames, conf=0.15)[0]    # This will return the results of the model. The model will track the players in the frames and return the results. The results will contain the bounding boxes of the players, the class id of the players, and the class names of the players
        id_name_dict = results.names                                        # This will return the names of the classes that the model was trained on. In this case, it will return the names of the players

        ball_dict = {} # This dictionary will store the player id and the player name
        
        # This loop will iterate over the boxes that the model detected
        for box in results.boxes: 
            result = box.xyxy.tolist()[0]                   # This will return the coordinates of the player
            ball_dict[1] = result                    # This will store the player id and the player name in the dictionary
    
        return ball_dict
        
    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw the bounding boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_video_frames.append(frame)
        return output_video_frames