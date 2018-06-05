
# coding: utf-8

# # Node Class for Game Tree

# In[83]:


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

# In[84]:


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
        
        


# # Game Tree Class

# In[85]:


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


# In[86]:


def printTree(root):
    if root != None:
        print(root.board)
    else:
        return None

    printTree(root.child0)
    printTree(root.child1)
    printTree(root.child2)
    printTree(root.child3)
    printTree(root.child4)
    printTree(root.child5)
    printTree(root.child6)


# In[80]:


board = np.array( [[STONE_BLANK] * FIELD_HEIGHT] * FIELD_WIDTH )
board[0][0] = -1
board[1][0] = -1
board[2][0] = -1
board[0][1] = 1
board[6][0] = 1
board[6][1] = 1
g = GameTree()
root = g.create_game_tree(board,4,STONE_HUMAN,STONE_HUMAN,-10)
printTree(root)    


# In[82]:


b = AlphaBeta()
b.alpha_beta_search(root)

