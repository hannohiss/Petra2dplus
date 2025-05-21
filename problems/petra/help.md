# Petra
Petrol station replenishment task

## Actions
- Choose a station and vehicle to refuel

## Action masking
- Mask all visited stations
- Mask all vehicles that are empty and only allow go back to depot, if they are not already there
- Mask all vehicles that are timed up and only allow go back to depot, if they are not already there

- batch processing hack, with self.finished()
- if vehicle is not at depot and has no time left, it can violate the time constraint
- exclude depot for empty vehicles

## Terminate Episode Conditions