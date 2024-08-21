# File: orient/interpret_observation.py

from self_modifying_rover_agent import PrioritizedStep

def step_wrapper():
    def interpret_observation(agent, environment, prior_steps):
        observation = environment.get('observation', '')
        agent.messages.append({"role": "user", "content": f"Please interpret what the vision model saw, and give a description of the plan for what to do next, try to avoid getting stuck in corners, and if you can't seem to roll forward, turn: {observation}"})
        response = agent.run_agent_step(agent.messages)
        print(f"Orientation: {response}")
        
        with open("orientations.jsonl", "a") as f:
            f.write(f"{agent.most_recent_timestamp}|{response}\n")
        
        environment['orientation'] = response

    def utility_function(context):
        base_priority = 8
        num_items = len(context.get('items', []))
        return base_priority + min(num_items, 5)  # Cap the item bonus at 5

    return PrioritizedStep(utility_function, interpret_observation)