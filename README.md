# Skilos

Skilos is a project to create a virtual quadruped similar to the homicidal robot dog in Black Mirror (S4:E5, Netflix), minus the psychopathic murdering tendencies.

![Metalhead Dog](https://pbs.twimg.com/media/DStingYWkAATwrx.jpg)

It is also a learning project for me to get to know OpenAI, TensorFlow, Reinforcement learning, and AI in general.

## Approach

The idea of using simulated environments to train real-world robotics is something I've been curious about for quite a while. I've done experiments before with XPlane to create control systems using Genetic Algorithms, and I've had this idea to create biologically-inspired robots using this method. With the rise of modern AI and Deep Learning, I feel the time has come to explore this further.

I'm using OpenAI Gym as a starting point, together with Tensorflow. There is a very useful environment called HalfCheetah, which is a model of... half a cheetah 😀:

![Half Cheetah](https://www.groundai.com/media/arxiv_projects/4068/figures/half_che.jpeg)

I has 6 actuators, 3 for each leg (thigh, shin, foot), modelled as Mujoco motors. The control of these motors will be trained to make the Cheetah cover the maximum distance possible, with some changes (see CHANGELOG.md)

Pat Coady submitted the [highest-ranking solution to this problem](https://gym.openai.com/envs/HalfCheetah-v1/), so I'm using his generously open sourced (MIT) code to begin with.

You can read more about his efforts on his blog:

[https://learningai.io/projects/2017/07/28/ai-gym-workout.html](https://learningai.io/projects/2017/07/28/ai-gym-workout.html)

And find his code for TRPO here:

[https://github.com/pat-coady/trpo](https://github.com/pat-coady/trpo)

## Do-it-yourself

I use pyenv and venv for virtual python environments. YMMV.

### Requirements

- Python 3.5.x
- Mujoco 1.3.0

### Create python virtual environment and install dependencies

```
$ python --version
Python 3.5.4
$ python -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

### Run the experiments

```
$ python ./src/train.py HalfCheetah-v1 -n 30000 -b 20
$ python ./src/train.py HalfCheetah-v2 -n 30000 -b 20
$ python ./src/train.py FullCheetah-v1 -n 30000 -b 20
```

Please log an issue on Github if you're trying to replicate this but aren't successful.
