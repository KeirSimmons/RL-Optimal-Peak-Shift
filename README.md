# Reinforcement Learning Based Battery Control for Optimal Peak Shifting

This is the title of my thesis submitted for my MSc in Computational Statistics & Machine Learning at UCL, which received a distinction classification (highest possible). The `report.pdf` file contains the full thesis. This work was conducted at Informetis, in their Japan office, over a 3 month period.

I have made the code available here for reference only. It is not in a runnable state as all proprietary information (datasets included) have been removed and the directory structure has been slightly modified to improve legibility. This is the state within which I have received permission to distribute. [OpenAI baselines](https://github.com/openai/baselines) and [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) have also been used in this project, though they are not included in this repository. All commit messages have been removed (i.e. this is a simple copy paste from the original, private repository) to avoid leaking sensitive data. All code within has been written by me, unless explicitly stated otherwise.

## Abstract

This project explores the application of peak-shifting in tackling the problem of large peaks in the demand of electrical energy from the grid. These peak periods are a result of consumers generally using electricity at similar times and periods as each other, for example, turning lights on when returning home from work, or the widespread use of air conditioners during the summer. Without peak shifting, the grid's system operators are forced to use peaker plants to provide the additional energy, the operation of which is incredibly expensive and dangerous to the environment due to their high levels of carbon emissions. The use of a battery energy storage system to allow for the purchase and storage of energy during off-peak periods for later use, with the primary objective of achieving peak shifting, is explored. In addition, reduction in energy consumption and the lowering of consumer's utility bills are also sought, making this a multi-objective optimisation problem. Reinforcement learning methods are investigated to provide a solution to this problem by finding an optimal control policy which defines when is best to purchase and store energy with the objectives in mind.

Through the use of PPO, a popular policy gradient-based reinforcement learning method, a very strong policy has been found. This achieves over a 20% reduction in energy consumption and consumers' energy bills, as well as achieving perfect peak shifting, thereby removing peaking plants from the equation entirely. This result was obtained through the use of a simulator that the author has developed specifically for this task, which handles the model training, testing and evaluation process. Secondly, development of a novel technique, _automatic penalty shaping_, was also found to be crucial to the success of the learned model. This technique enabled automatic shaping of the reward signal, forcing the agent to pay equal attention to multiple individual signals, a necessity when applying reinforcement learning to multi-objective optimisation problems.

The policy does, however, attempt to overcharge the battery about 7% of the time, and promising methods to address this have been proposed as a direction for future research.

## More Information

Please see `report.pdf` for full information, details and results.

Please respect the license if you decide to use any code within this repository.

I am on [LinkedIn](https://www.linkedin.com/in/keirsimmons/) and always happy to hear about new job oppurtunities. Please see my [CV](https://github.com/KeirSimmons/CV/blob/master/CV.pdf) for more information :) .