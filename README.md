# Behavior-Guided-Reinforcement-Learning

{UPDATE: Some theory and explanation of algorithms can be found [in my blog](https://conflictednerd.github.io/blog/learning-to-score-behaviors/)}

Some time ago I read the paper [Learning to Score Behaviors for Guided Policy Optimization](https://arxiv.org/abs/1906.04349) and I really liked it. Unfortunately, the [official repo](https://github.com/behaviorguidedRL/BGRL) contains only a demo and doesn't provide a clean implementation that can be used to run extensive experiments with.  
I had some exciting ideas for possibly improving on the paper and adding some components to it. So, I decided to first implement the paper's methods myself and then continue from there.  
As if this wasn't challenging enough, I decided to write the code using [JAX](https://github.com/google/jax) (which I knew very little about at the time).  
When I was writing the code, I needed to implement a simple policy gradient algorithm. This reminded me of another old interest of mine: I always wanted to implement some of the RL algorithms from scratch, partly to understand them (and all their nitty-gritty implementation details) better and partly to get more confident with coding RL algorithms ([this repo](https://github.com/conflictednerd/Minesweeper-AI) was one previous attempt at this that I didn't continue). So again, I thought this is the perfect opportunity for that too. I don't want to get too fancy though, I just need some simple actor-critic methods.  

Long story short, this repo is created because I wanted to
1. implement and extend the methods in the aforementioned paper,
2. learn JAX,
3. implement some RL algorithms from scratch.

So far, it has been a successful experience. I now know much more about JAX, I have implemented an off-policy version of REINFORCE (an actor-critic method coming soon) and all of this has resulted in progress towards implementing the paper too.

A todo list (in no particular order):
- [x]  Implement a simple actor critic method to serve as a baseline (nothing fancy)
- [x]  (bit of a reach but) Implement PPO
- [x]  Use JAX to accelerate wherever possible (currently, experience collection is the bottleneck)
- [ ]  Implement some more advanced neural networks (I eventually want to have recurrent policies)
- [x]  Implement proper logging with Tensorboard (I'm always lazy for logging)
- [ ]  Find some interesting envs and get some initial results on them (current candidates are MiniGrid and some robotic envs)
- [ ]  Implement behavior guided policy gradient (BGPG) and evolutionary strategy (BGES) (the original plan)
- [ ]  Extend BGPG with some of my own ideas
