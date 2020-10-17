About
=====

This an experimental study of a gathering grid-world based environment under a multi-agent setup, where each agent is trained using different reinforcement learning algorithm (PPO and DQN). The agents get rewarded for consuming resources in an environment that has a limited set of resources per episode. 

The purpose of this experiment is to study how agents with policies derived using *different RL algorithms* behave  in a competitive environment.


RUNNING
=====

Create a virtualenv and install the required dependencies using  'requirements.txt'

In-order to train the agents run "train.py"

In-order to evaluate the agents run "evaluate.py" (Note you need to select a model checkpoint and episode count)

In-order to visualise the results and monitor training use the provided Jupiter notebook.

The environment is implemented using the open-AI GYM interface for multiple agents. To test the environment, 
use the  `env/test_arcade.py' script which runs a two player arcade game that simulates the environment.

TODO
=====

0.  Code needs clean-up, especially type-hints.
1.  Configurations are currently hard coded, create the appropriate interface
2.  Currently only DQN and PPO is implemented, use different RL algorithms.
