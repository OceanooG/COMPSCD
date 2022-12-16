an implementation of a Chinese chess game that allows a player to play against an AI opponent using the Monte Carlo Tree Search (MCTS) algorithm. The code consists of several functions that perform various tasks related to the game, such as setting up the board, displaying the board, finding legal moves for a given piece, and evaluating the static value of a game state.

The main game loop is implemented in the play_game function, which handles player and AI turns, checks for the end of the game, and updates the game state. The MCTS algorithm is implemented in the mcts function, which takes in the current game state and uses simulations of possible game states to estimate the value of each move and select the best move for the AI to make. The playout function is used to simulate a game from the current state to the end, and the select_move function is used to select the next move during a simulation based on the estimated values of the available moves.

The code also includes several helper functions that perform tasks such as finding legal moves for a given piece, evaluating the static value of a game state, and checking for the end of the game. There is also a main function that sets up the Pygame window and calls the play_game function to start the game.

you can run the game by navigate to the provided directory in your command prompt, 'python ccmcts.py'

you can set the players by changing the players in the __init__() function 

you have three options valid for both side; mcts_playe, human_player, random_player

you can modify the iteration number and the max tree depth of the evaluation tree in the ccmcts.py file

ITERATIONS = 2 , MAX_DEPTH = 5