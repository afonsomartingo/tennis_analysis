from ultralytics import YOLO
import cv2
import pickle # pickle is used to serialize and deserialize a Python object structure. In this case, we will use it to store the player detections
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self, ball_positions):

        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        
        df_ball_positions['ball_hit'] = 0
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff() # calculate the difference between the current and previous value

        minimum_change_for_hit = 25
        for i in range(1,len(df_ball_positions) - int(minimum_change_for_hit*1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                # check the next few frames to see if the ball is hit or not  
                # iloc is used to access the index of the dataframe
                for change_frame in range(i+1, i+int(minimum_change_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0          
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1
                # if the change count is greater than the minimum change for hit, then the ball is hit
                if change_count>minimum_change_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:                # If read_from_stub is True and the stub path is provided, then we will read the player detections from the stub path
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:                                   # If the stub path is provided, then we will store the player detections in the stub path
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f) 

        return ball_detections
            
    def detect_frame(self, frames):
        results = self.model.predict(frames, conf=0.15)[0]          # This will return the results of the model. The model will track the players in the frames and return the results. The results will contain the bounding boxes of the players, the class id of the players, and the class names of the players
        id_name_dict = results.names                                # This will return the names of the classes that the model was trained on. In this case, it will return the names of the players

        ball_dict = {}                                              # This dictionary will store the player id and the player name
        
        # This loop will iterate over the boxes that the model detected
        for box in results.boxes: 
            result = box.xyxy.tolist()[0]                   # This will return the coordinates of the player
            ball_dict[1] = result                    # This will store the player id and the player name in the dictionary
    
        return ball_dict
        
    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames