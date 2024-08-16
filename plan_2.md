## Rover Improvements

## LLM Usage
- LM Router so we can route simple requests to cheaper models
- logging properly so we can associate images with observations & inventory

## Vision System
- Just captures an image and sends it to openai
    - get the image capture to send to cheaper, vision-classifier models that can run on small consumer hardware
    - get the image capture to identify objects and save those somehow
- seems to lose the feed, perhaps needs to ... do something else with the feed

## mapping and telemetry
- indoor navigation through beacons?
- https://stella-cv.readthedocs.io/en/latest/example.html for mapping
- https://github.com/AprilRobotics/apriltag
- https://github.com/uupks/openvslam
- https://github.com/raulmur/ORB_SLAM2
- https://www.ros.org/blog/getting-started/ probably should know all about this


## navigation 
- probably some goals will help
- having direct prompting about the plan would help
- if beacons are available, can tag things locally to beacons
- do more deterministic navigation 
