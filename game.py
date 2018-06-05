
# coding: utf-8

# # Code to Import Training Data from UCI DB

# In[1]:


import sys, numpy as np
#from global_vars import *

VERBOSE = True
DATA_NORMALIZE = False

STONE_BLANK = 0.0
STONE_HUMAN = 1.0
STONE_AI = -1.0

FIELD_WIDTH = 7
FIELD_HEIGHT = 6
CONNECT = 4

WIN = 0.0
DRAW = 0.5
LOSS = 1.0

DATA_NUM_ATTR = 43

def my_converter( x ):
    """ Converter for Numpys loadtxt function.
    Replace the strings with float values.
    """

    if x == 'x': return STONE_HUMAN
    elif x == 'o': return STONE_AI
    elif x == 'b': return STONE_BLANK
    elif x == "win": return WIN
    elif x == "loss": return LOSS
    elif x == "draw": return DRAW


def normalize( data ):
    """ Normalize given data. """

    sys.stdout.write( " Normalizing data ..." )
    sys.stdout.flush()

    data -= data.mean( axis = 0 )
    imax = np.concatenate(
        (
            data.max( axis = 0 ) * np.ones( ( 1, len( data[0] ) ) ),
            data.min( axis = 0 ) * np.ones( ( 1, len( data[0] ) ) )
        ), axis = 0 ).max( axis = 0 )
    data /= imax

    return data


def import_traindata( file_in ):
    """ Import the file with training data for the AI. """

    sys.stdout.write( "Importing training data ..." )
    sys.stdout.flush()

    # Assign converters for each attribute.
    # A dict where the key is the index for each attribute.
    # The value for each key is the same function to replace the string with a float.
    convs = dict( zip( range( DATA_NUM_ATTR + 1 ) , [my_converter] * DATA_NUM_ATTR ) )
    connectfour = np.loadtxt( file_in, delimiter = ',', converters = convs )

    cf_original = []
    f = open( file_in, "r" )
    for line in f:
        row = line.split( ',' )
        row[len( row ) - 1] = row[len( row ) - 1].replace( '\n', '' )
        cf_original.append( row )

    # Split in data and targets
    data = connectfour[:,:DATA_NUM_ATTR - 1]
    targets = connectfour[:,DATA_NUM_ATTR - 1:DATA_NUM_ATTR]

    if DATA_NORMALIZE:
        data = normalize( data )

    sys.stdout.write( " Done.\n\n" )

    return data, targets, cf_original


# # Node Class for Game Tree

# In[2]:


class GameNode(object):
    def __init__(self, child0, child1, child2, child3, child4, child5, child6, board, start, move):
        self.child0 = child0
        self.child1 = child1
        self.child2 = child2
        self.child3 = child3
        self.child4 = child4
        self.child5 = child5
        self.child6 = child6
        self.board = board
        self.start = start
        self.move = move


# # Alpha Beta Pruning Class

# In[3]:


class AlphaBeta:
    
    # print utility value of root node (assuming it is max)
    # print names of all nodes visited during search
    def __init__(self):
        #self.game_tree = game_tree  # GameTree
        #self.root = game_tree.root  # GameNode
        return

        
    def alpha_beta_search(self, node):
        infinity = float('inf')
        best_val = -infinity
        beta = infinity

        successors = self.getSuccessors(node)
        best_state = None
        for state in successors:
            value = self.min_value(state, best_val, beta)
            if value > best_val:
                best_val = value
                best_state = state
        #print "AlphaBeta:  Utility Value of Root Node: = " + str(best_val)
        #print "AlphaBeta:  Best State is: " + "\n" + str(best_state.board)
        return best_state.move

    def max_value(self, node, alpha, beta):
        #print "AlphaBeta-->MAX: Visited Node :: " + "\n" + str(node.move)
        if self.isTerminal(node):
            #print(str(self.getUtility(node)))
            return self.getUtility(node)
        infinity = float('inf')
        value = -infinity

        successors = self.getSuccessors(node)
        for state in successors:
            value = max(value, self.min_value(state, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, node, alpha, beta):
        #print "AlphaBeta-->MIN: Visited Node :: " + "\n" + str(node.move)
        if self.isTerminal(node):
            #print(str(self.getUtility(node)))
            return self.getUtility(node)
        infinity = float('inf')
        value = infinity

        successors = self.getSuccessors(node)
        for state in successors:
            value = min(value, self.max_value(state, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)

        return value
    #                     #
    #   UTILITY METHODS   #
    #                     #

    # successor states in a game tree are the child nodes...
    def getSuccessors(self, node):
        assert node is not None
        children = []
        if node.child0 != None:
            children.append(node.child0)
        if node.child1 != None:
            children.append(node.child1)
        if node.child2 != None:
            children.append(node.child2)
        if node.child3 != None:
            children.append(node.child3)
        if node.child4 != None:
            children.append(node.child4)
        if node.child5 != None:
            children.append(node.child5)
        if node.child6 != None:
            children.append(node.child6)      
        return children

    # return true if the node has NO children (successor states)
    # return false if the node has children (successor states)
    def isTerminal(self, node):
        assert node is not None
        if node.child0 == None and node.child1 == None and node.child2 == None and node.child3 == None and node.child4 == None and node.child5 == None and node.child6 == None:
            return True
        return False

    def getUtility(self, node):
        assert node is not None
        g = GameTree()
        utility = g.heurestic_1(node.board,node.start)
        return utility
    
    def test(self,x,board):
        g = GameTree()
        top = g.column_top(x,board)
        print(str(top))
        
        


# # GameTree class

# In[4]:


#!/usr/bin/python


import sys, random, numpy as np
#from global_vars import *

VERBOSE = True

STONE_BLANK = 0.0
STONE_HUMAN = 1.0
STONE_AI = -1.0

FIELD_WIDTH = 7
FIELD_HEIGHT = 6
CONNECT = 4

WIN = 0.0
DRAW = 0.5
LOSS = 1.0

DATA_NUM_ATTR = 43

train_data = []
train_labels = []

class GameTree:
    """ Play a game of Connect Four. """

    def __init__( self ):
        """ Constructor.
        ai -- Trained instance of game AI.
        """


    def _count_stones( self, column, row, stone, blank, ai, human ):
        """ Compares the value of the found stone with the existing stone types.
        Sets a stop flag if it is not possible anymore for either the human or the AI
        to connect enough stones.
        Returns a tuple in the form: ( position of a blank field, #ai, #human, flag_to_stop ).
        """

        flag_stop = False

        if stone == STONE_BLANK:
            # No danger/chance if there is more than one blank field in range
            if blank != -1: flag_stop = True
            # Save blank field. This is a candidate to place the stone.
            else: blank = { "col": column, "row": row }
        elif stone == STONE_AI:
            # If there has been at least one human stone already,
            # it is not possible for AI stones to have a connection of three and a blank field available.
            if human > 0: flag_stop = True
            else: ai += 1
        elif stone == STONE_HUMAN:
            # Same here, vice versa.
            if ai > 0: flag_stop = True
            else: human += 1

        return ( blank, ai, human, flag_stop )


    def _count_stones_up( self, x, y, board ):
        """ From position (x,y) count the types of the next stones upwards. """

        blank, ai, human = -1, 0, 0

        if y + CONNECT <= FIELD_HEIGHT:
            for i in range( CONNECT ):
                col, row = x, y + i
                stone = board[col][row]
                blank, ai, human, flag_stop = self._count_stones( col, row, stone, blank, ai, human )
                if flag_stop: break

        return ( blank, ai, human )


    def _count_stones_right( self, x, y, board ):
        """ From position (x,y) count the types of the next stones to the right. """

        blank, ai, human = -1, 0, 0

        if x + CONNECT <= FIELD_WIDTH:
            for i in range( CONNECT ):
                col, row = x + i, y
                stone = board[col][row]
                blank, ai, human, flag_stop = self._count_stones( col, row, stone, blank, ai, human )
                if flag_stop: break

        return ( blank, ai, human )


    def _count_stones_rightup( self, x, y, board ):
        """ From position (x,y) count the types of the next stones diagonal right up. """

        blank, ai, human = -1, 0, 0

        if x + CONNECT <= FIELD_WIDTH and y + CONNECT <= FIELD_HEIGHT:
            for i in range( CONNECT ):
                col, row = x + i, y + i
                stone = board[col][row]
                blank, ai, human, flag_stop = self._count_stones( col, row, stone, blank, ai, human )
                if flag_stop: break

        return ( blank, ai, human )


    def _count_stones_rightdown( self, x, y, board ):
        """ From position (x,y) count the types of the next stones diagonal right down. """

        blank, ai, human = -1, 0, 0

        if x + CONNECT <= FIELD_WIDTH and y - CONNECT + 1 >= 0:
            for i in range( CONNECT ):
                col, row = x + i, y - i
                stone = board[col][row]
                blank, ai, human, flag_stop = self._count_stones( col, row, stone, blank, ai, human )
                if flag_stop: break

        return ( blank, ai, human )

    def check_win( self, board ):
        """ Check the game board if someone has won.

        Returns the stone value of the winner or the value
        of a blank stone if there is no winner yet.
        """

        for x in range( FIELD_WIDTH ):
            for y in range( FIELD_HEIGHT ):
                # We only care about players, not blank fields
                if board[x][y] == STONE_BLANK:
                    continue
                    
                # Check: UP
                blank, ai, human = self._count_stones_up( x, y, board )
                if ai == CONNECT: return STONE_AI
                elif human == CONNECT: return STONE_HUMAN

                # Check: RIGHT
                blank, ai, human = self._count_stones_right( x, y, board )
                if ai == CONNECT: return STONE_AI
                elif human == CONNECT: return STONE_HUMAN

                # Check: DIAGONAL RIGHT UP
                blank, ai, human = self._count_stones_rightup( x, y , board)
                if ai == CONNECT: return STONE_AI
                elif human == CONNECT: return STONE_HUMAN

                # Check: DIAGONAL RIGHT DOWN
                blank, ai, human = self._count_stones_rightdown( x, y, board )
                if ai == CONNECT: return STONE_AI
                elif human == CONNECT: return STONE_HUMAN

        return STONE_BLANK


    def check_board_full( self, board ):
        """ Returns true if there are no more free fields, false otherwise. """
        
        if np.shape( np.where( board == STONE_BLANK ) )[1] == 0:
            return True
        return False
    
          
    
    #returns T or F depending on if the column is full
    def column_full(self, column, board):
        #if the 6th position is a blank, then the column is not full
        if board[column][5] == STONE_BLANK:
            return False
        else:
            return True
        
    #returns index of the top row containing a stone by either players
    #only call this function after column_full returns false
    def column_top(self,column,board):
        for x in range(FIELD_HEIGHT):
            if board[column][x] == STONE_BLANK:
                return x
    
    #move is the move that got the player to that state, eg. child0 = move 0 at column 0, child1 = move 1
    #move = -1, beginning of game
    #start is the starting player eg. if the starting player is HUMAN, we want to be evaluating the board's 
    #heurestic with respect to HUMAN
    #turn: (STONE_HUMAN, STONE_AI)
    def create_game_tree( self , board, depth, start, turn, move):
        if depth == 0:
            return None
        
        #if there is a connect 4 for this board, game is over
        if self.check_win(board) == 1 or self.check_win(board) == -1:
            WinningNode = GameNode(None,None,None,None,None,None,None,board,start,move)
            return WinningNode
        
        #Child0
        #if the first column is full, cannot insert a stone so no children
        if self.column_full(0,board):
            Child0 = None
        else:
            board_copy = np.copy(board)
            top = self.column_top(0,board)
            board_copy[0][top] = turn
            Child0 = self.create_game_tree(board_copy,depth-1,start,turn*-1,0)
        
        #Child1
        #if the first column is full, cannot insert a stone so no children
        if self.column_full(1,board):
            Child1 = None
        else:
            board_copy = np.copy(board)
            top = self.column_top(1,board)
            board_copy[1][top] = turn
            Child1 = self.create_game_tree(board_copy,depth-1,start,turn*-1,1)
            
        #Child2
        #if the first column is full, cannot insert a stone so no children
        if self.column_full(2,board):
            Child2 = None
        else:
            board_copy = np.copy(board)
            top = self.column_top(2,board)
            board_copy[2][top] = turn
            Child2 = self.create_game_tree(board_copy,depth-1,start,turn*-1,2)
            
        #Child3
        #if the first column is full, cannot insert a stone so no children
        if self.column_full(3,board):
            Child3 = None
        else:
            board_copy = np.copy(board)
            top = self.column_top(3,board)
            board_copy[3][top] = turn
            Child3 = self.create_game_tree(board_copy,depth-1,start,turn*-1,3)
            
        #Child4
        #if the first column is full, cannot insert a stone so no children
        if self.column_full(4,board):
            Child4 = None
        else:
            board_copy = np.copy(board)
            top = self.column_top(4,board)
            board_copy[4][top] = turn
            Child4 = self.create_game_tree(board_copy,depth-1,start,turn*-1,4)
            
        #Child5
        #if the first column is full, cannot insert a stone so no children
        if self.column_full(5,board):
            Child5 = None
        else:
            board_copy = np.copy(board)
            top = self.column_top(5,board)
            board_copy[5][top] = turn
            Child5 = self.create_game_tree(board_copy,depth-1,start,turn*-1,5)
            
        #Child6
        #if the first column is full, cannot insert a stone so no children
        if self.column_full(6,board):
            Child6 = None
        else:
            board_copy = np.copy(board)
            top = self.column_top(6,board)
            board_copy[6][top] = turn
            Child6 = self.create_game_tree(board_copy,depth-1,start,turn*-1,6)
            
        Node = GameNode(Child0,Child1,Child2,Child3,Child4,Child5,Child6,board,start,move)
        return Node
    
    
    #heurestic 1: Player A,B 
    #Lets say Player A's turn
    #The heurestic for Player Turn: #2s Player A - #2s Player B
    def heurestic_1(self,board,start):
        #if player won game, assign arbitrary big heurestic
        if self.check_win(board) == start:
            return 100
        if self.check_win(board) == start*-1:
            return -100
        #p1 has count of connected 2s for player turn
        p1 = 0
        p2 = 0
        for x in range(FIELD_WIDTH):
            for y in range(FIELD_HEIGHT):
                if board[x][y] == start:
                    #check if stone above is the same
                    if y < 5:
                        if board[x][y+1] == start:
                            p1 = p1 + 1
                    #check if stone top right is the same
                    if y < 5 and x < 6:
                        if board[x+1][y+1] == start:
                            p1 = p1 + 1
                    #check if the stone on the right is the same
                    if x < 6:
                        if board[x+1][y] == start:
                            p1 = p1 + 1
                    if y > 0 and x < 6:
                        if board[x+1][y-1] == start:
                            p1 = p1 + 1
                elif board[x][y] == start * -1:
                    #check if stone above is the same
                    if y < 5:
                        if board[x][y+1] == start*-1:
                            p1 = p1 + 1
                    #check if stone top right is the same
                    if y < 5 and x < 6:
                        if board[x+1][y+1] == start*-1:
                            p1 = p1 + 1
                    #check if the stone on the right is the same
                    if x < 6:
                        if board[x+1][y] == start*-1:
                            p1 = p1 + 1
                    if y > 0 and x < 6:
                        if board[x+1][y-1] == start*-1:
                            p1 = p1 + 1                
        heurestic = p1 - p2
        return heurestic
                    #check if the stone on bottom right is the same
                    

#if __name__ == "__main__":
#    # Small test
#    g = Game( None )
#    print g.board
#    g.print_board( g.board )
#    g.play()
#    print "Well, nothing crashed …"


# # Game Class

# In[7]:


#!/usr/bin/python


import sys, random, numpy as np
#from global_vars import *

VERBOSE = True

STONE_BLANK = 0.0
STONE_HUMAN = 1.0
STONE_AI = -1.0

FIELD_WIDTH = 7
FIELD_HEIGHT = 6
CONNECT = 4

WIN = 0.0
DRAW = 0.5
LOSS = 1.0

DATA_NUM_ATTR = 43

train_data = []
train_labels = []

class Game:
    """ Play a game of Connect Four. """

    def __init__( self, ai ):
        """ Constructor.
        ai -- Trained instance of game AI.
        """
        self.ai = ai
        
        self.count_ai_moves = 0
        self.last_move_human = 0

        self._init_field()

    def _init_field( self ):
        """ Init play field. """

        self.board = np.array( [[STONE_BLANK] * FIELD_HEIGHT] * FIELD_WIDTH )
        self.current_height = [0] * FIELD_WIDTH

    def input_validate( self, x ):
        """ Validates the chosen column x. """

        # Try to cast to a number
        try:
            x = int( x )
        except:
            return False

        # Outside of game board width.
        if x < 1 or x > FIELD_WIDTH:
            return False

        # Column is full.
        if self.current_height[x - 1] >= FIELD_HEIGHT:
            return False

        return True


    def _count_stones( self, column, row, stone, blank, ai, human ):
        """ Compares the value of the found stone with the existing stone types.
        Sets a stop flag if it is not possible anymore for either the human or the AI
        to connect enough stones.
        Returns a tuple in the form: ( position of a blank field, #ai, #human, flag_to_stop ).
        """

        flag_stop = False

        if stone == STONE_BLANK:
            # No danger/chance if there is more than one blank field in range
            if blank != -1: flag_stop = True
            # Save blank field. This is a candidate to place the stone.
            else: blank = { "col": column, "row": row }
        elif stone == STONE_AI:
            # If there has been at least one human stone already,
            # it is not possible for AI stones to have a connection of three and a blank field available.
            if human > 0: flag_stop = True
            else: ai += 1
        elif stone == STONE_HUMAN:
            # Same here, vice versa.
            if ai > 0: flag_stop = True
            else: human += 1

        return ( blank, ai, human, flag_stop )


    def _count_stones_up( self, x, y ):
        """ From position (x,y) count the types of the next stones upwards. """

        blank, ai, human = -1, 0, 0

        if y + CONNECT <= FIELD_HEIGHT:
            for i in range( CONNECT ):
                col, row = x, y + i
                stone = self.board[col][row]
                blank, ai, human, flag_stop = self._count_stones( col, row, stone, blank, ai, human )
                if flag_stop: break

        return ( blank, ai, human )


    def _count_stones_right( self, x, y ):
        """ From position (x,y) count the types of the next stones to the right. """

        blank, ai, human = -1, 0, 0

        if x + CONNECT <= FIELD_WIDTH:
            for i in range( CONNECT ):
                col, row = x + i, y
                stone = self.board[col][row]
                blank, ai, human, flag_stop = self._count_stones( col, row, stone, blank, ai, human )
                if flag_stop: break

        return ( blank, ai, human )


    def _count_stones_rightup( self, x, y ):
        """ From position (x,y) count the types of the next stones diagonal right up. """

        blank, ai, human = -1, 0, 0

        if x + CONNECT <= FIELD_WIDTH and y + CONNECT <= FIELD_HEIGHT:
            for i in range( CONNECT ):
                col, row = x + i, y + i
                stone = self.board[col][row]
                blank, ai, human, flag_stop = self._count_stones( col, row, stone, blank, ai, human )
                if flag_stop: break

        return ( blank, ai, human )


    def _count_stones_rightdown( self, x, y ):
        """ From position (x,y) count the types of the next stones diagonal right down. """

        blank, ai, human = -1, 0, 0

        if x + CONNECT <= FIELD_WIDTH and y - CONNECT + 1 >= 0:
            for i in range( CONNECT ):
                col, row = x + i, y - i
                stone = self.board[col][row]
                blank, ai, human, flag_stop = self._count_stones( col, row, stone, blank, ai, human )
                if flag_stop: break

        return ( blank, ai, human )


    def _check_proposed_col( self, pos ):
        """ Check if it is possible to place a stone in the given field.
        Returns True if possible, False otherwise.
        """

        if pos == -1:
            return False

        if pos["col"] >= 0 and pos["col"] < FIELD_WIDTH:
            # Check if it is possible to place the stone at the needed height
            if pos["row"] == 0 or self.board[pos["col"]][pos["row"] - 1] != STONE_BLANK:
                return True

        return False


    def _find_forced_move( self ):
        """ Check if the next move is a forced one and where to place the stone.
        A forced move occurs if the human player or the AI could win the game with the next move.
        Returns the position where to place the stone or -1 if not necessary.
        """

        force_x = -1

        for x in range( FIELD_WIDTH ):
            for y in range( FIELD_HEIGHT ):

                # Check: UP
                blank, ai, human = self._count_stones_up( x, y )
                # Evaluate: UP
                if blank != -1:
                    # If there is a chance to win: Do it!
                    if ai == 3: return blank["col"]
                    # Remember dangerous situation for now.
                    # Maybe there will be a chance to win somewhere else!
                    elif human == 3:
                        if VERBOSE: print "[human] could win UP with %d." % blank["col"]
                        force_x = blank["col"]

                # Check: RIGHT
                blank, ai, human = self._count_stones_right( x, y )
                # Evaluate: RIGHT
                if self._check_proposed_col( blank ):
                    if ai == 3: return blank["col"]
                    elif human == 3:
                        if VERBOSE: print "[human] could win RIGHT with %d." % blank["col"]
                        force_x = blank["col"]

                # Check: DIAGONAL RIGHT UP
                blank, ai, human = self._count_stones_rightup( x, y )
                # Evaluate: DIAGONAL RIGHT UP
                if self._check_proposed_col( blank ):
                    if ai == 3: return blank["col"]
                    elif human == 3:
                        if VERBOSE: print "[human] could win DIAGONAL RIGHT UP with %d." % blank["col"]
                        force_x = blank["col"]

                # Check: DIAGONAL RIGHT DOWN
                blank, ai, human = self._count_stones_rightdown( x, y )
                # Evaluate: DIAGONAL RIGHT DOWN
                if self._check_proposed_col( blank ):
                    if ai == 3: return blank["col"]
                    elif human == 3:
                        if VERBOSE: print "[human] could win DIAGONAL RIGHT DOWN with %d." % blank["col"]
                        force_x = blank["col"]

        return force_x


    def check_win( self ):
        """ Check the game board if someone has won.

        Returns the stone value of the winner or the value
        of a blank stone if there is no winner yet.
        """

        for x in range( FIELD_WIDTH ):
            for y in range( FIELD_HEIGHT ):
                # We only care about players, not blank fields
                if self.board[x][y] == STONE_BLANK:
                    continue

                # Check: UP
                blank, ai, human = self._count_stones_up( x, y )
                if ai == CONNECT: return STONE_AI
                elif human == CONNECT: return STONE_HUMAN

                # Check: RIGHT
                blank, ai, human = self._count_stones_right( x, y )
                if ai == CONNECT: return STONE_AI
                elif human == CONNECT: return STONE_HUMAN

                # Check: DIAGONAL RIGHT UP
                blank, ai, human = self._count_stones_rightup( x, y )
                if ai == CONNECT: return STONE_AI
                elif human == CONNECT: return STONE_HUMAN

                # Check: DIAGONAL RIGHT DOWN
                blank, ai, human = self._count_stones_rightdown( x, y )
                if ai == CONNECT: return STONE_AI
                elif human == CONNECT: return STONE_HUMAN

        return STONE_BLANK


    def check_board_full( self ):
        """ Returns true if there are no more free fields, false otherwise. """
        
        if np.shape( np.where( self.board == STONE_BLANK ) )[1] == 0:
            return True
        return False
    
    def ask_mlp( self ):
        #score closest to 1 is the best move
        move = -1
        max_score = -999
        for x in range(FIELD_WIDTH):
            y = y = self.current_height[x]
            if y >= FIELD_HEIGHT:
                continue
            if self.count_ai_moves == 0 and x == self.last_move_human:
                continue

            # Don't change the real game board
            board_copy = self.board.copy()
            board_copy[x][y] = STONE_AI
            ai_board_format = []
            
            for i in range( FIELD_WIDTH ):
                ai_board_format.extend( board_copy[i] )
            pred = self.ai.predict([ai_board_format])[0]
            
            #print("Move on column " + str(x) + " Pred Value is " + str(pred))
            if pred > max_score:
                max_score = pred
                move = x
            
        return move
                
    def create_samples(self, boards, winner):
        data = []
        labels = []
        final_boards = boards[-3:]
        for x in range(3):
            ai_board_format = []
            for i in range( FIELD_WIDTH ):
                ai_board_format.extend( final_boards[x][i] )
            data.append(ai_board_format)
            if winner == "AI":
                labels.append(1)
            elif winner == "Random":
                labels.append(0)
        return data,labels
        
    
    def self_play(self, player1, player2):
        #list_boards = []
        list_boards = []
        while 1:
            """
            Player 1 MOVE
            """
            if player1 == "random":
                move1 = self.rand_move()
                
            elif player1 == "mlp":
                forced_move = self._find_forced_move()
            
                # Forced move, don't ask the AI
                if forced_move > -1:
                    #if VERBOSE: print "forced_move = %d" % forced_move
                    move1 = forced_move
                else:
                    move1 = self.ask_mlp()
                    if move1 < 0: break
                        
            elif player1 == "gametree":
                forced_move = self._find_forced_move()
            
                # Forced move, don't ask the AI
                if forced_move > -1:
                    #if VERBOSE: print "forced_move = %d" % forced_move
                    move1 = forced_move
                else:
                    g = GameTree()
                    board_copy = np.copy(self.board)
                    root = g.create_game_tree(board_copy,3,STONE_HUMAN,STONE_HUMAN,-1)
                    b = AlphaBeta()
                    move1 = b.alpha_beta_search(root)
                    if move1 < 0: break
    
            #print player1 + " places stone in column " + str( move1 ) + "."
            self._place_stone(move1, STONE_HUMAN)
            #self.print_board(self.board)
            #list_boards.append(self.board)
            list_boards.append(self.board)
            winner = self.check_win()
            if winner == STONE_HUMAN:
                #data, labels = self.create_samples(list_boards, "Random")
                data, labels = self.create_samples(list_boards, "Random")
                train_data.extend(data)
                train_labels.extend(labels)
                #train_data.extend(data)
                #train_labels.extend(labels)
                #print("Data is " + str(data))
                #print("Labels is " + str(labels))
                #print "player1 won!"
                break
            elif winner == STONE_AI:
                data, labels = self.create_samples(list_boards, "AI")
                train_data.extend(data)
                train_labels.extend(labels)
                #data, labels = self.create_samples(list_boards, "AI")
                #train_data.extend(data)
                #train_labels.extend(labels)
                #print("Data is " + str(data))
                #print("Labels is " + str(labels))
                #print "player2 won!"
                break
                
            # Draw
            if self.check_board_full():
                #print "No more free fields. It's a draw!"
                break
                
            
            """
            Player 2 Move
            """
            if player2 == "random":
                    move2 = self.rand_move()

            elif player2 == "mlp":
                forced_move = self._find_forced_move()

                # Forced move, don't ask the AI
                if forced_move > -1:
                    #if VERBOSE: print "forced_move = %d" % forced_move
                    move2 = forced_move
                else:
                    move2 = self.ask_mlp()
                    if move2 < 0: break

            elif player2 == "gametree":
                forced_move = self._find_forced_move()
            
                # Forced move, don't ask the AI
                if forced_move > -1:
                    #if VERBOSE: print "forced_move = %d" % forced_move
                    move2 = forced_move
                else:
                    g = GameTree()
                    board_copy = np.copy(self.board)
                    root = g.create_game_tree(board_copy,2,STONE_AI,STONE_AI,-1)
                    b = AlphaBeta()
                    move2 = b.alpha_beta_search(root)
                    if move2 < 0: break
            
            #print player2 + " places stone in column " + str( move2 ) + "."
            self._place_stone(move2, STONE_AI)
            #self.print_board(self.board)
            #list_boards.append(self.board)
            list_boards.append(self.board)
            winner = self.check_win()
            if winner == STONE_HUMAN:
                data, labels = self.create_samples(list_boards, "Random")
                train_data.extend(data)
                train_labels.extend(labels)
                #print("Data is " + str(data))
                #print("Labels is " + str(labels))
                #print "player1 won!"
                break
            elif winner == STONE_AI:
                data, labels = self.create_samples(list_boards, "AI")
                train_data.extend(data)
                train_labels.extend(labels)
                #print("Data is " + str(data))
                #print("Labels is " + str(labels))
                #print "player2 won!"
                break
            # Draw
            if self.check_board_full():
                #print "No more free fields. It's a draw!"
                break
                
        #print "Game ended."            

    def play( self , ai ):
        """ Start playing. """
        while 1:
            try:
                x = raw_input( ">> Column: " )
            except EOFError: print; break
            except KeyboardInterrupt: print; break

            # Validate user input
            if not self.input_validate( x ):
                print "No valid field position!"
                continue
            x = int( x ) - 1

            # Place human stone
            self._place_stone( x, STONE_HUMAN )
            self.print_board( self.board )
            

            winner = self.check_win()
            if winner == STONE_HUMAN:
                print "You won!"
                break
            elif winner == STONE_AI:
                print "The AI won!"
                break

            # Draw
            if self.check_board_full():
                print "No more free fields. It's a draw!"
                break

            
            if ai == "random":
                ai_move = self.rand_move()
                
            elif ai == "mlp":
                forced_move = self._find_forced_move()
            
                # Forced move, don't ask the AI
                if forced_move > -1:
                    if VERBOSE: print "forced_move = %d" % forced_move
                    ai_move = forced_move
                else:
                    ai_move = self.ask_mlp()
                    if ai_move < 0: break
                        
            elif ai == "gametree":
                g = GameTree()
                board_copy = np.copy(self.board)
                root = g.create_game_tree(board_copy,5,STONE_AI,STONE_AI,x)
                b = AlphaBeta()
                ai_move = b.alpha_beta_search(root)
    

            print "AI places stone in column " + str( ai_move + 1 ) + "."

            self._place_stone( ai_move, STONE_AI )
            self.print_board( self.board )

            winner = self.check_win()
            if winner == STONE_HUMAN:
                print "You won!"
                break
            elif winner == STONE_AI:
                print "The AI won!"
                break

            # Draw
            if self.check_board_full():
                print "No more free fields. It's a draw!"
                break

        print "Game ended."


    def _place_stone( self, col, stone ):
        """ Place a stone in a column. """

        row = self.current_height[col]
        self.board[col][row] = stone
        self.current_height[col] += 1

        if stone == STONE_AI:
            self.count_ai_moves += 1
        elif stone == STONE_HUMAN:
            self.last_move_human = col


    #AI makes a random move at each iteration
    def rand_move( self ):    
        ran_field = range(FIELD_WIDTH)
        np.random.shuffle(ran_field)
        use_pos = -1
        for x in ran_field:
            if self.input_validate(x+1):
                use_pos = x
                break
        return use_pos

    

    def print_board( self, board ):
        """ Print the current game board. """

        for y in reversed( range( FIELD_HEIGHT ) ):
            sys.stdout.write( " | " )
            for x in range( FIELD_WIDTH ):
                field = ' '
                if board[x][y] == STONE_HUMAN:
                    field = 'x'
                elif board[x][y] == STONE_AI:
                    field = 'o'
                sys.stdout.write( field + " | " )
            print ''
        sys.stdout.write( ' ' )
        print '¯' * ( FIELD_WIDTH * 4 + 1 )

#if __name__ == "__main__":
#    # Small test
#    g = Game( None )
#    print g.board
#    g.print_board( g.board )
#    g.play()
#    print "Well, nothing crashed …"


# # Train MLP (Neural Network) with UCI data

# In[6]:


from sklearn.neural_network import MLPClassifier, MLPRegressor

data, targets, original_import = import_traindata("connect-4.data")
training_data = data[:60000,:]
training_targets = targets[:60000,:]
np.ravel(targets)

clf = MLPRegressor(solver='adam')
clf.fit(training_data, training_targets)


# In[8]:


g = Game( clf )
print g.board
g.print_board( g.board )
g.self_play("mlp","random")
print "Well, nothing crashed …"


# # Code to Test Validation Error and Create Data Samples

# In[52]:


from sklearn.neural_network import MLPClassifier, MLPRegressor

data, targets, original_import = import_traindata("connect-4.data")
training_data = data[:60000,:]
training_targets = targets[:60000,:]
np.ravel(targets)

clf = MLPRegressor(solver='adam')
clf.fit(training_data, training_targets)


# In[9]:


error = 0
for x in range(60000,data.shape[0]):
    a = data[x]
    #a.reshape(1,-1)
    pred = clf.predict([a])[0]
    error = error + np.square(targets[x] - pred)
    #print("TARGET IS " + str(targets[x]) + "PREDICT IS " + str(pred))
error/7557


# In[10]:


k = Game( clf )
for x in range(4000):
    k.self_play("mlp","random")
    k._init_field()


# In[17]:


from sklearn.neural_network import MLPClassifier, MLPRegressor

data, targets, original_import = import_traindata("connect-4.data")
training_data = data[:60000,:]
training_targets = targets[:60000,:]
train_data=np.asarray(train_data)
train_labels=np.asarray(train_labels)
train_labels = np.reshape(train_labels,(12003,1))
training_data = np.vstack((training_data,train_data))
training_targets = np.vstack((training_targets,train_labels))
np.ravel(targets)

clf2 = MLPRegressor(solver='adam')
clf2.fit(training_data, training_targets)


# In[18]:


error = 0
for x in range(60000,data.shape[0]):
    a = data[x]
    #a.reshape(1,-1)
    pred = clf2.predict([a])[0]
    error = error + np.square(targets[x] - pred)
    #print("TARGET IS " + str(targets[x]) + "PREDICT IS " + str(pred))
error/13557

