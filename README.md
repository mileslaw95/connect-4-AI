# connect-4-AI
Connect 4 AI using Neural Networks and Alpha Beta Pruning on a game tree

Create a Connect 4 AI using two methods, Neural Network and Alpha Beta Pruning on a game tree

Neural Network

Collect training data from UCI's connect-4 data set from http://archive.ics.uci.edu/ml/datasets/Connect-4. This consists of all legal 8-ply positions in the game of connect 4 where neither player has won yet. Each position is labelled with either a win or loss. Train an MLP regressor on these board positions and map each of them to its label (1 for loss, 0 for win). In game.py, there is also code where the AI plays against itself to create more data points for the neural network.

Alpha Beta Pruning on Game Tree

In game_tree.py, a game tree is implemented for connect-4. Each node has up to 7 children node corresponding to a move on each position on the board. Implemented a simple heurestic which counts how many 2-connected pieces a player has.  The more 2-connected pieces, the higher score, it is it assigned. For each 2-connected piece, we give the board position a (+1) score For a winning position, we give the board position an arbitrary (+100) AI makes decision by evaluating the game tree and choosing the best move using a Minimax algorithm. Also, implemented Alpha-Beta pruning to minimize the search space of the Minimax algorithm. The depth of the game tree is a variable that the user can specify.

E-mail milesklaw@yahoo.ca for inquiries on the code.
