# LeIsaac 🚀

https://github.com/user-attachments/assets/763acf27-d9a9-4163-8651-3ba0a6a185d7

This repository provides teleoperation functionality in [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html) using the SO101Leader ([LeRobot](https://github.com/huggingface/lerobot)), including data collection, data conversion, and subsequent policy training.

- 🤖 We use SO101Follower as the robot (and other related robot) in IsaacLab and provide relevant teleoperation method.
- 🔄 We offer scripts to convert data from HDF5 format to the LeRobot Dataset.
- 🧠 We utilize simulation-collected data to fine-tune [GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T) and deploy it on real hardware. And more policies will be supported.

> [!TIP]
> ***Welcome to the Lightwheel open-source community!***
>
> Join us, contribute, and help shape the future of AI and robotics. For questions or collaboration, contact [Zeyu](mailto:zeyu.hu@lightwheel.ai) or [Yinghao](mailto:yinghao.shuai@lightwheel.ai).

## Getting Started 📚

Please refer to our [documentation](https://lightwheelai.github.io/leisaac/) to learn how to use this repository. Follow these links to learn more about:

- [Installation and Setup](https://lightwheelai.github.io/leisaac/docs/getting_started/installation)
- [Extra Features](https://lightwheelai.github.io/leisaac/docs/features)
- [Policy Inference](https://lightwheelai.github.io/leisaac/docs/getting_started/policy_support)
- [Available Robots](https://lightwheelai.github.io/leisaac/resources/available_robots), [Environments](https://lightwheelai.github.io/leisaac/resources/available_env) and [Policy](https://lightwheelai.github.io/leisaac/resources/available_policy)

## Waypoint-Based Data Collection 🎯

For automated demonstration collection, we provide a waypoint-based control system. This allows you to define robot trajectories as sequences of target positions and automatically execute them with recording capabilities.

**Key Features:**
- Define trajectories as JSON waypoint sequences
- Multiple controller backends (DifferentialIK, OperationalSpaceController)
- Automatic data recording in HDF5 format
- Direct conversion to LeRobot format and HuggingFace upload

**Quick Example:**
```bash
# Collect demonstrations using waypoints
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/bi_arm_demo.json" \
    --controller_type=dik \
    --record \
    --dataset_file="datasets/demos.hdf5" \
    --num_demos=50 \
    --enable_cameras
```

📖 **[Full Documentation](scripts/environments/waypoints/README.md)** | **[한국어 문서](scripts/environments/waypoints/README_KOR.md)**


## Acknowledgements 🙏

We gratefully acknowledge [IsaacLab](https://github.com/isaac-sim/IsaacLab) and [LeRobot](https://github.com/huggingface/lerobot) for their excellent work, from which we have borrowed some code.

## Join Our Team! 💼

We're always looking for talented individuals passionate about AI and robotics! If you're interested in:

- 🤖 **Robotics Engineering**: Working with cutting-edge robotic systems and teleoperation
- 🧠 **AI/ML Research**: Developing next-generation AI models for robotics
- 💻 **Software Engineering**: Building robust, scalable robotics software
- 🔬 **Research & Development**: Pushing the boundaries of what's possible in robotics

**Join us at Lightwheel AI!** We offer:
- Competitive compensation and benefits
- Work with state-of-the-art robotics technology
- Collaborative, innovative environment
- Opportunity to shape the future of AI-powered robotics

**[Apply Now →](https://lightwheel.ai/career)** | **[Contact Now →](mailto:zeyu.hu@lightwheel.ai)** | **[Learn More About Us →](https://lightwheel.ai)**

---

**Let's build the future of robotics together! 🤝**

---
