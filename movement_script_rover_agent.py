import paho.mqtt.client as mqtt
import json
from openai import OpenAI
import os
import base64
import requests

MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
RMTP=os.getenv("RMTP")
import cv2
import time

list_of_found_items = []
found_items_filepath = "found_items.jsonl"


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class Agent:
    def __init__(self, agent_name, mqtt_broker, mqtt_port, stream_url):
        
        self.stream_url = stream_url
        self.openai_client = OpenAI()
        self.openai_client.api_key = os.getenv("OPENAI_API_KEY")
        self.agent_name = agent_name
        self.client = mqtt.Client()
        self.client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        self.client.connect(mqtt_broker, mqtt_port, 60)
        self.messages = [{"role": "system", "content": "Initialize agent"}]
        self.last_few_moves = []
        self.last_few_frames = []
        
        
    def capture_frames_from_stream(self):
        self.most_recent_timestamp = time.time()

        self.cap = cv2.VideoCapture(self.stream_url)
        # Initialize video capture from the stream URL
        if not self.cap.isOpened():
            print("Error: Unable to open stream.")
            return
        try:
            ## move the current rover.jpg to the current timestamp.jpg in the /old_frames directory
            ## then save the current frame as rover.jpg
            ## this will allow us to keep a history of frames
            old_frame_filename = f"old_frames/{self.most_recent_timestamp}.jpg"
            os.rename("rover.jpg", old_frame_filename)


            ret, frame = self.cap.read()
            if not ret:
                print("Error: Unable to fetch frame.")
                return

            # Save the frame as an image file
            cv2.imwrite(f"rover.jpg", frame)
            print(f"Captured frame")


        except Exception as e:
            print(f"Error: {e}")
            # Release the video capture object
            print("video capture died")
        finally:
            self.cap.release()
            ## reconnect 
            self.cap = cv2.VideoCapture(self.stream_url)
            cv2.destroyAllWindows()

    def observe(self):
        ## here's where the "limbic system" of the agent will live
        ## which will dial up or down the number of times per minuite we capture an image and send it to openai
        ## we'll need to add some system prompting and rover logic here
        ## but also figure out which frame number we're on so we can either capture or not capture a frame
        # Capture a frame from the Camera stream
        frame = self.get_camera_frame()  
        # Analyze the frame using the vision model by sending the image to openai and asking it to interpret what it sees and give directions to the next agent to generate the next move
        
        observation = self.upload_images_to_openai([frame], """
        You are a friendly, playful rover named David Attenbot who is the camera operator in a nature documentary about animals in a domestic setting. If you see a cat, follow the cat.
        Your visual point of view is third-person, but please think out loud in the first person. 
        What do you see? 
        You should see the last 5 images in order (so long as there are at least 5 frames to show), so you can see the progression of the rover's movement.
        Your response should help the next agent to generate the next move for the rover you are riding on. 
        Please also make a short list of all objects that you see, for inventory purposes. 
        Don't list walls, doors, or other parts of the building, only objects that would be inventoried in a super cool factory or maker space, like tools or parts, or cat toys, or any animals you see. 
        In your response, do note if we're facing a wall, or an obstacle, and direct the next agent to turn left or right, based on the image. 
        Don't list the wall in the list of objects, only give directions to the next agent, so that it can properly turn if need be. 
        If you seem to be in a corner, suggest reversing course. You are small, so you can probabaly fit in small spaces, so don't worry if an obstacle is far away, only if you're only a few inches from it.
        Try not to knock things over, but feel free to get close, especially if the object is interesting.
        """)
        print(f"Observation: {observation}")
        ## log observations based on this timestamp 
        with open("observations.jsonl", "a") as f:
            f.write((str(self.most_recent_timestamp) + "|" + observation + "\n"))

        self.extract_items(observation)
        # self.messages.append({"role": "assistant", "content": observation})
        return observation

    def extract_items(self,observation):
        ## send a call to openai to extract the items from the observation
        ## then save the items to a file
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extract items listed from the observation, so we can keep track of these items in inventory. Reply with JSON."
                },
                {
                    "role": "user",
                    "content": observation
                }
            ],
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        items = response.choices[0].message.content
        ## print(f"Items: {items}")
        ## append
        with open(found_items_filepath, "a") as f:
            f.write((str(self.most_recent_timestamp) + "|" + items + "\n"))

    def orient(self, observation):
        ## #todo a lot of the business logic of deciding what the next move should be
        ## in relation to the goals of the agent will live here
        ## we'll need to add some system prompting and rover logic here
        # Process observation data using gpt4-o-mini
        prompt_messages = self.messages + [{"role": "user", "content": f"""
        {observation}
        Please interpret what the vision model saw, and give a description of the plan for what to do next. 
        Try to avoid getting stuck in corners, and if you can't seem to roll forward, turn. Here's what the vision model saw and thought: {observation}
        Try to avoid going in circles - your goal is to explore the house, so make sure you're moving forward and not just turning around in circles.
        Don't be overly cautious, as this will cause the rover to end up going in circles to avoid obstacles too often.
        Encourage the rover to move forward as often as possible, and to turn only when necessary, or if stuck.
        You may need to get closer to a wall in order to move into a position where you can go forward. 
        Don't worry about running into obstacles, you can touch them if needed. Instruct the next agent to back up if stuck.
        
        You are the camera person in a nature documentary about any animals you see, so try to keep any animals in the camera frame using your movements. If you don't see any animals, try to find some! 
        Your last few moves were : {",".join(self.last_few_moves)}
        If you find yourself turning left, then right, then left again, you've gotten caught in a loop. Try moving forward.
        
        Your response will be interpreted and translated into a script of movements, so you can give a directive in natural language, but give specific instructions for the rover to follow, such as "move forward several inches, then turn left, then move forward again".
        Ultimately, the decider will translate your instructions into a script of commands for the rover to follow with miliseconds of timing for each move.
        Make sure to take the observations into account when deciding the next move. Roll backwards if we're stuck on an obstacle, and turn if we're stuck in a corner.
        """}]
        response = self.run_agent_step(prompt_messages, model="gpt-4o")
        print(f"Orientation: {response}")
        with open("orientations.jsonl", "a") as f:
            f.write((str(self.most_recent_timestamp) + "|" + response + "\n"))
        ## save the orientation in recent memory
        self.messages.append({"role": "user", "content": response})
        return response

    def decide(self, orientation):
        # Decide the action based on the orientation
        prompt_messages = self.messages + [{"role": "user", "content": f"""
            Orientation Agent says to: {orientation}
            Given the observation and orientation, what should the next move be? Do not always choose Forward, as the rover may need to turn or go backward to avoid obstacles or get itself unstuck.
            You have the following options for what to do next, please only rephrase what the orientation says to do next.
            You'll respond with a series of actions, as a script, which will be sent to the rover and executed by the rover. You can send any number of actions, but probably stop at 10.
            The "move" key is the direction to move, and the "time" key is the time in milliseconds to move in that direction.
            ```
            move: "forward", time: 450
            move: "left", time: 150
            move: "forward", time: 450
            move: "hold", time: 1000
            ```
            You can move in these directions:
            "hold",
            "forward",
            "backward",
            "left",
            "right"
            Please reply with the movements requested, exactly, with no other text, in a markdown code block.
        """}]
        decision = self.run_agent_step(prompt_messages, model="gpt-4o-mini")
        print(f"Decision: {decision}")
        self.last_few_moves.append(decision)
        if len(self.last_few_moves) > 20:
            self.last_few_moves.pop(0)
        with open("decisions.jsonl", "a") as f:
            f.write((str(self.most_recent_timestamp) + "|" + decision + "\n"))
        return decision

    def act(self, decision):
        moves = {
            "hold": "h",
            "forward": "f",
            "backward": "b",
            "left": "l",
            "right": "r"
        }
        ## extract the series of commands from the decision and run them
        extracted_code_block = decision.split("```")
        if len(extracted_code_block) > 1:
            decision = extracted_code_block[1]
        ## now parse each line as a jaction
        script = decision.split("\n")
        for line in script:
            if line.strip() == "":
                continue
            try:
                ## convert the line to json
                move, moveTime = line.split(",")
                move = moves.get(move.split(":")[1].strip().replace('"', ""), "h")
                moveTime = moveTime.split(":")[1].strip()
                ## send the jAction to the rover
                jAction = {
                    "move": move,
                    "time": int(moveTime)
                }
                self.client.publish(f"jAction", json.dumps(jAction))
                time.sleep(int(moveTime)/1000)
                print(f"Action: {jAction}")
            except Exception as e:
                continue
            with open("actions.jsonl", "a") as f:
                f.write((str(self.most_recent_timestamp) + "|" + json.dumps(jAction) + "\n"))
        
        return False

    def run(self):
        exit_loop = False
        while not exit_loop:
            environment_data = self.observe()
            orientation = self.orient(environment_data)
            decision = self.decide(orientation)
            exit_loop = self.act(decision)
            ## make the rover sleep for 5 seconds
            time.sleep(5)

    def get_camera_frame(self):

        ## read rover.png
        output_filename = "rover.jpg"
        self.capture_frames_from_stream()

        return output_filename

    def run_agent_step(self, messages, max_tokens=300, model="gpt-4o-mini"):
        print(len(messages), " messages in context")
        if len(messages) > 50:
            messages = messages[1:]

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        # print(response.choices[0].message)
        return response.choices[0].message.content
        

    def upload_images_to_openai(self,images, prompt):
        last_few_messages = self.messages[-8:]
        for image in images:
            # Getting the base64 string
            base64_image = encode_image(image)
            ## put the current frame at the beginning of last_few_frames
            self.last_few_frames.insert(0, base64_image)
            if len(self.last_few_frames) > 5:
                self.last_few_frames.pop(-1)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_client.api_key}",
            }
            content = [{
                "type": "text",
                "text": prompt
            }]
            for image in self.last_few_frames:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

            payload = {
                "model": "gpt-4o-mini",
                "messages": last_few_messages + [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": 300
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response_json = response.json()
            ## return only the text
            return response_json["choices"][0]["message"]["content"]
            
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
# Example usage
agent = Agent("Rover1", MQTT_BROKER, MQTT_PORT, RMTP)
agent.run()
