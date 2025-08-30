# 🐳 DuckieBot Container Commands Reference

## Basic Container Operations

### Enter Container
```bash
# SSH to DuckieBot first
ssh duckie@blueduckie.local

# Enter the main container
docker exec -it dt-core /bin/bash
```

### Build Workspace
```bash
# Inside container
cd /code/catkin_ws
catkin_make
source devel/setup.bash
```

### Run Lane Following
```bash
# Inside container (after build)
roslaunch lane_follower advanced_lane_following.launch
```

## One-Liner Commands (from your laptop)

### Build Only
```bash
ssh duckie@blueduckie.local "docker exec dt-core bash -c 'cd /code/catkin_ws && catkin_make'"
```

### Build and Source
```bash
ssh duckie@blueduckie.local "docker exec dt-core bash -c 'cd /code/catkin_ws && catkin_make && source devel/setup.bash'"
```

### Full Build and Run
```bash
ssh duckie@blueduckie.local "docker exec dt-core bash -c 'cd /code/catkin_ws && catkin_make && source devel/setup.bash && roslaunch lane_follower advanced_lane_following.launch'"
```

## Debugging Commands

### Check ROS Master
```bash
ssh duckie@blueduckie.local "docker exec dt-core bash -c 'rostopic list'"
```

### Monitor Topics
```bash
ssh duckie@blueduckie.local "docker exec dt-core bash -c 'rostopic echo /lane_pose'"
```

### Check Logs
```bash
ssh duckie@blueduckie.local "docker logs dt-core"
```

## Quick Scripts

### Use the helper script:
```bash
./build_and_run.sh blueduckie
```

### Or create your own alias:
```bash
alias duckiebuild="ssh duckie@blueduckie.local 'docker exec dt-core bash -c \"cd /code/catkin_ws && catkin_make\"'"
alias duckierun="ssh duckie@blueduckie.local 'docker exec dt-core bash -c \"cd /code/catkin_ws && source devel/setup.bash && roslaunch lane_follower advanced_lane_following.launch\"'"
```