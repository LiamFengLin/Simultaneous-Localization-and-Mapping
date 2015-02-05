Member1 Name: Feng Lin
Member1 edX username: linfeng@berkeley.edu

Member2 Name: Hanyu Zhang
Member2 edX username: hzy@berkeley.edu

It works best for smallHunt, message and oneHunt layouts. For these two, it should well converge within 400 time steps, when tested under AutoSlAMAgent. 

Some cautions during use: 
1. Even though it has not occurred during many trials, please give it another run if it does not seem to be converging or crashes;
2. The layouts are all tested under AutoSlamAgent and Pacman appears to be lagging (and unresponsive) for keyboard manual control now because of the constraint on computation power. Some fixes to speed it up are suggested below.

For other larger layouts, it takes more time to converge and is slightly more prone to mistakes. Hence for larger layouts, if it doesn't converge the first time, please run it again and give it another chance! Sometimes walls seem to be wrongly recognized and labelled as non-walls. Under this implementation, the probabilities will never go to zero but will go to a very small probability. So wrongly labelled walls can recover later on. But this takes longer for larger layouts. 

It works worst for the bigHunt layout, even though it should converge given enough time steps. The reason is that it is computationally expensive to do map computation for every particle as each particle has its own map that needs to be inialized and updated. Hence, let N = number of particles and M = size of map. Time complexity is O(NM) with respect to N and M. 

I suggest a few fixes: 1. further optimization of the implementation. For example, for positions that are not in the path of the range model for a move, I could have done a selective update of the map instead of looping through all positions; 2. for better convergence behavior, the inverse range model can be more finetuned to include probabilities for positions beyond the noisy distances by inferring probabilities for the staff code that generates it; 3. since each particle is essentially independent from other particles, expensive operations such as map update can be parallelled for even larger layout; 4. to avoid duplicate map characteristics, an untested approach is to use DP-SLAM, which can store changes made by different particles in a global map using balanced trees rooted at each grid square. 