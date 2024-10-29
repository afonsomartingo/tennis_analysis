from ultralytics import YOLO

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect_frames(self, frames):
        player_detections = []

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        return player_detections
            
    def detect_frame(self, frames):
        results = self.model.track(frames, persist=True)        # persist=True means that the tracker will remember the players from the previous frame
        id_name_dict = results.names                            # This will return the names of the classes that the model was trained on. In this case, it will return the names of the players

        player_dict = {} # This dictionary will store the player id and the player name
        
        # This loop will iterate over the boxes that the model detected
        for box in results.boxes: 
            track_id = int(box.id.tolist()[0])              # This will return the id of the player
            result = box.xyxy.tolist()[0]                   # This will return the coordinates of the player
            object_cls_id = box.cls.tolist()[0]             # This will return the class id of the player
            object_cls_name = id_name_dict[object_cls_id]   # This will return the class name of the player
            if object_cls_name == "person":                 # If the class name is person, then we will store the player id and the player name in the dictionary
                player_dict[track_id] = result              # This will store the player id and the player name in the dictionary
    
        return player_dict
        