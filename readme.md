# Project notes

- replay buffer sample time is too slow when on the cluster. How come? is it normal?
- the very first `train.step()` is too slow. how come?
- move reporting to a controller loop alone
- move evaluation loop to other processes

- distributing the gradient computation might bring other complexities in accessing the replay buffer for sampling 
