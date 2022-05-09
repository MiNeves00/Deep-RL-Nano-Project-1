# Udacity Deep RL Nanodegree Projects

### <p style="text-align: center;">Miguel Carreira Neves</p>
<p style="text-align: center;">22/03/2022</p>

---

## Introduction

This repository is a collection of my personal solutions for the projects of the Udacity course - [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)

## Projects

### 1. Navigation

For this project, an agent was trained to navigate (and collect bananas!) in a large, square world. Its goal is to collect as many yellow bananas as possible in each episode while avoiding blue bananas.

<IMG SRC="./Project1-Navigation/imgs/BananaTestingCropped.gif" width = "600" >

### 2. Continuous Control

This project was made with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#reacher) environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 

Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

<IMG SRC="./Project2-ContinuousControl/imgs/ReacherTest2.gif" width = "600" >

### 3. Collaboration and Competition

This project was made with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

Due to the nature of the enviroment and how the reward function is designed, the **enviroment rewards cooperation** between the players. 

<IMG SRC="./Project3-CollaborationAndCompetition/imgs/results/test2.gif" width = "600" >

