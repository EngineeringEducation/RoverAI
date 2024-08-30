## Next Steps

## Vision
- [x] Send multiple snapshots for comparison
- Detect image similarity prior to sending snapshots

## Better Actions
- [x] write function to allow rover to send multiple commands at once, along with delays between them
- [x] Better capture intent of action

## Better Cognitive Architecture to support Better Actions
- The observe phase could compare images for major changes and only involve the orienter if needed
- The orient phase could check to see if we're "still executing" the last thing and then optionally skip decide/act if no new information (no major visual changes) is available and we're still executing
- The decide phase should respond to being skipped by skipping
- The Act phase could also wait for the script to finish executing.

## Deduplication of Inventory
- currently we amass a lot of different descriptions of objects that are materially the same object. We could dedupe them by giving the list of detected objects each time, which will prompt the vision model to likely describe objects in the same way.

## Location-independent time-axis item-cluster tagging
- We tend to see the same things near each other. While we may not have spatial data, this gives us landmark data.
- We could try to tag items as in the same "shot" or as clustered together, and just note that those items are nearby each other whenever we have a positive sight on them having been grouped, along with the timestamp as an axis, so that we can say when the toy mouse was last near the scratching post. This would allow an LLM to answer the question "Where is my yellow drill" with the response "it is near a red tool chest and a blue car"

## Self-Modification
Allowing the rover to self-modify can have good or disasterous results, we'll have to think hard about the architecture.