# Tennis_agents: 
## 1. Project Overview: 

This an an implementation of deep reinforcemet multi-agents to solve a tennis game.

<img src="assets/tennis.png"/>

## 2.Task Description :

### 2.1 Environement :

For this project I am using the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) it simulates a tennis game.

With : 
- The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.
- there are two actions possible of a continuous nature :
    * Toward or away from the net.
    * Jumping .
    
For each agent :

- An agent receives a reward of +0.1 if it hits the ball over the net.
- An agent receives a reward of -0.01 if it hits the ball either on the ground or out of the bounds.

### 2.2 Solving the environement:
this task is episodic and the environment is considered to be solved if : we reach an average score (over 100 episodes) of at least +0.5.

## 3. Getting started :

#### 3.1 Clone this repository :
`
git clone https://github.com/ZSoumia/Tennis_agents
`
#### 3.2 Set up the environment :
Please follow instructions from this [repo](https://github.com/udacity/deep-reinforcement-learning#dependencies)

#### 3.3 Download the Unity Environment :

Select the Unity environement based on your opertaing system :

- Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)

- Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)

- Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)

- Windows (64-bit): click  [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Check out this [link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)


==> Place the downloaded file into your cloned project file .

## 4. Project structure : 

* The Agent.py file contains the general structure of the Reinforcement learning agent ( single agent) .
* The Actor.py contains the actor's network code .
* Critic.py contains the critic's network code.
* Multi_Agent.py is the code of the MADDPG algorithm .
* ReplayBuf.py : is the structure of the reply buffer object used. 
* *.pth are the models /weights checkpoints of my own implementations I provided them in case some one wants to reproduce my work.
* Tennis.ipynb is the notebook that I used for the training .
* Report.html is a detailed report about my approach.
Continuous control Report.md is a report about the different aspects of this project.

