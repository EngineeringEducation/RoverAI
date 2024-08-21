# File: observe/capture_and_analyze.py

from self_modifying_rover_agent import PrioritizedStep

def step_wrapper():
    def capture_and_analyze(agent, environment, prior_steps):
        frame = agent.get_camera_frame()
        observation = agent.upload_images_to_openai([frame], "You are a friendly, playful rover. Your visual point of view is third-person, but please think out loud in the first person. What do you see? Your response should help the next agent to generate the next move for the rover you are riding on. Please also make a short list of all objects that you see, for inventory purposes. Don't list walls, doors, or other parts of the building, only objects that would be inventoried in a super cool factory or maker space, like tools or parts, or cat toys, or any animals you see.")
        print(f"Observation: {observation}")
        
        with open("observations.jsonl", "a") as f:
            f.write(f"{agent.most_recent_timestamp}|{observation}\n")
        
        # Extract items from observation
        response = agent.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract items listed from the observation, so we can keep track of these items in inventory. Reply with JSON."},
                {"role": "user", "content": observation}
            ],
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        items = response.choices[0].message.content
        print(f"Items: {items}")
        
        with open("found_items.jsonl", "a") as f:
            f.write(f"{agent.most_recent_timestamp}|{items}\n")
        
        environment['observation'] = observation
        environment['items'] = items

    def utility_function(context):
        return 10 if 'obstacle' in context.get('last_observation', '') else 5

    return PrioritizedStep(utility_function, capture_and_analyze)

