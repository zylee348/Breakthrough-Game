import math
import utils
import time
import copy

class PlayerAI:
    ROW, COL = 6, 6
    def state_change_white(self, board, from_, to_):
        new_board = copy.deepcopy(board)
        new_board[from_[0]][from_[1]] = '_'
        new_board[to_[0]][to_[1]] = 'W'
        return new_board

    def generate_moves(self, board, maximizingPlayer):
        list_of_moves = []
        if maximizingPlayer:
            for r in range(self.ROW):
                for c in range(self.COL):
                    # check if B can move forward directly
                    if (r + 1) < self.ROW:
                        if board[r][c] == 'B' and board[r + 1][c] == '_':
                            src = [r, c]
                            dst = [r + 1, c]
                            list_of_moves.append((src, dst))

                    if c + 1 < self.COL and (r + 1) < self.ROW:
                        if board[r][c] == 'B' and (board[r + 1][c + 1] == '_' or board[r + 1][c + 1] == 'W'):
                            src = [r, c]
                            dst = [r + 1, c + 1]
                            list_of_moves.append((src, dst))

                    if c - 1 >= 0 and (r + 1) < self.ROW:
                        if board[r][c] == 'B' and (board[r + 1][c - 1] == '_' or board[r + 1][c - 1] == 'W'):
                            src = [r, c]
                            dst = [r + 1, c - 1]
                            list_of_moves.append((src, dst))

        elif not maximizingPlayer:
            for r in range(self.ROW):
                for c in range(self.COL):
                    # check if W can move forward directly
                    if r - 1 >= 0:
                        if board[r][c] == 'W' and board[r - 1][c] == '_':
                            src = [r, c]
                            dst = [r - 1, c]
                            list_of_moves.append((src, dst))

                    if c + 1 < self.COL and r - 1 >= 0:
                        if board[r][c] == 'W' and (board[r - 1][c + 1] == '_' or board[r - 1][c + 1] == 'B'):
                            src = [r, c]
                            dst = [r - 1, c + 1]
                            list_of_moves.append((src, dst))

                    if c - 1 >= 0 and r - 1 >= 0:
                        if board[r][c] == 'W' and (board[r - 1][c - 1] == '_' or board[r - 1][c - 1] == 'B'):
                            src = [r, c]
                            dst = [r - 1, c - 1]
                            list_of_moves.append((src, dst))

        return list_of_moves

    def heuristic(self, board):
        black_score = 0
        white_score = 0

        # Evaluate the position of the black pawn
        for row in range(self.ROW):
            for col in range(self.COL):
                if board[row][col] == 'B':
                    black_score += 1
                    if row + 1 == self.ROW:
                        black_score += 10
                    if row + 1 < self.ROW:
                        # blocked by white
                        if board[row + 1][col] == 'W':
                            black_score -= 1
                    else:
                        black_score += row

                if board[row][col] == 'W':
                    white_score += 1
                    if row - 1 == 0:
                        # Penalize white pawns on row index 1
                        white_score += 10
                    if row - 1 >= 0:
                        # blocking black
                        if board[row - 1][col] == 'B':
                            white_score += 1
                    else:
                        white_score += 5 - row

        # print(black_score - white_score)
        return black_score - white_score

    def minimax(self, board, depth, alpha, beta, maximizingPlayer):
        list_of_moves = self.generate_moves(board, maximizingPlayer)

        if depth == 0 or utils.is_game_over(board) or not list_of_moves:
            return None, self.heuristic(board)

        eval_move = None

        if maximizingPlayer:
            maxEval = -math.inf
            for move in list_of_moves:
                new_board = utils.state_change(board, move[0], move[1], False)
                _, value = self.minimax(new_board, depth - 1, alpha, beta, False)
                if value > maxEval:
                    eval_move = move
                    maxEval = value
                    # if (time.time() - start_time) > 2.97:
                    #     return maxEval_move, maxEval
                alpha = max(alpha, value)
                if alpha >= beta:
                    break

            return eval_move, maxEval

        else:
            minEval = math.inf
            for move in list_of_moves:
                new_board = self.state_change_white(board, move[0], move[1])
                _, value = self.minimax(new_board, depth - 1, alpha, beta, True)
                # print(value)
                if value < minEval:
                    eval_move = move
                    minEval = value
                    # if (time.time() - start_time) > 2.97:
                    #     return minEval_move, minEval
                beta = min(beta, value)
                if alpha >= beta:
                    break

            return eval_move, minEval

    def make_move(self, board):
        answer, value = self.minimax(board, 4, -math.inf, math.inf, True)
        return answer


import time
import copy
import utils


class PlayerAI2:
    weights = [[1, 2, 3, 3, 2, 1], [2, 3, 4, 4, 3, 2], [3, 4, 5, 5, 4, 3], [4, 5, 6, 6, 5, 4], [5, 6, 7, 7, 6, 5],
               [999, 999, 999, 999, 999, 999]]

    def make_move(self, board):
        rows = len(board)
        cols = len(board[0])
        bestEval = float('-inf')
        move = [0, 0], [0, 0]
        for r in range(rows):
            for c in range(cols):
                # check if B can move forward directly
                if board[r][c] == 'B':
                    src = [r, c]
                    dstForward = [r + 1, c]
                    dstDiagLeft = [r + 1, c - 1]
                    dstDiagRight = [r + 1, c + 1]
                    if utils.is_valid_move(board, src, dstForward):
                        newBoard = utils.state_change(board, src, dstForward, in_place=False)
                        forwardEval = self.miniMax(newBoard, 3, float('-inf'), float('inf'), maxPlayer=False)
                        if (forwardEval > bestEval):
                            bestEval = forwardEval
                            move = [r, c], [r + 1, c]
                    if utils.is_valid_move(board, src, dstDiagLeft):
                        newBoard = utils.state_change(board, src, dstDiagLeft, in_place=False)
                        diagLeftEval = self.miniMax(newBoard, 3, float('-inf'), float('inf'), maxPlayer=False)
                        if (diagLeftEval > bestEval):
                            bestEval = diagLeftEval
                            move = src, dstDiagLeft
                    if utils.is_valid_move(board, src, dstDiagRight):
                        newBoard = utils.state_change(board, src, dstDiagRight, in_place=False)
                        diagRightEval = self.miniMax(newBoard, 3, float('-inf'), float('inf'), maxPlayer=False)
                        if (diagRightEval > bestEval):
                            bestEval = diagRightEval
                            move = src, dstDiagRight
        print(move)
        return move  # invalid move

    def miniMax(self, board, depth, alpha, beta, maxPlayer=True, rows=6, cols=6):
        # Enumerate all possible steps from current state of board
        # Moves should only be forward, diagonally left, diagonally right
        # Optimisation here: Send entire tree so minimax only computes at most branch factor worth of leaves
        if depth == 0 or utils.is_game_over(board):
            return self.evaluation(board, depth)

        if maxPlayer:
            maxEval = float('-inf')
            for r in range(rows):
                for c in range(cols):
                    if board[r][c] == 'B':
                        # possibleStates = []
                        src = [r, c]
                        possibleDstFwd = [r + 1, c]
                        possibleDstDiagLeft = [r + 1, c - 1]
                        possibleDstDiagRight = [r + 1, c + 1]
                        if utils.is_valid_move(board, src, possibleDstFwd):
                            # possibleStates.append(possibleDstFwd)
                            newBoard = utils.state_change(board, src, possibleDstFwd, in_place=False)
                            eval = self.miniMax(newBoard, depth - 1, alpha, beta, maxPlayer=False)
                            maxEval = max(maxEval, eval)
                            alpha = max(alpha, maxEval)
                            if (beta <= alpha):
                                break
                        if utils.is_valid_move(board, src, possibleDstDiagLeft):
                            # possibleStates.append(possibleDstDiagLeft)
                            newBoard = utils.state_change(board, src, possibleDstDiagLeft, in_place=False)
                            eval = self.miniMax(newBoard, depth - 1, alpha, beta, maxPlayer=False)
                            maxEval = max(maxEval, eval)
                            alpha = max(alpha, maxEval)
                            if (beta <= alpha):
                                break
                        if utils.is_valid_move(board, src, possibleDstDiagRight):
                            # possibleStates.append(possibleDstDiagRight)
                            newBoard = utils.state_change(board, src, possibleDstDiagRight, in_place=False)
                            eval = self.miniMax(newBoard, depth - 1, alpha, beta, maxPlayer=False)
                            maxEval = max(maxEval, eval)
                            alpha = max(alpha, maxEval)
                            if (beta <= alpha):
                                break
            return maxEval
        else:
            minEval = float('inf')
            for r in range(rows):
                for c in range(cols):
                    if board[r][c] == 'W':
                        # possibleStates = []
                        src = [r, c]
                        possibleDstFwd = [r - 1, c]
                        possibleDstDiagLeft = [r - 1, c - 1]
                        possibleDstDiagRight = [r - 1, c + 1]
                        if self.is_valid_move_white(board, src, possibleDstFwd):
                            # possibleStates.append(possibleDstFwd)
                            newBoard = self.state_change_white(board, src, possibleDstFwd, in_place=False)
                            eval = self.miniMax(newBoard, depth - 1, alpha, beta, maxPlayer=True)
                            minEval = min(minEval, eval)
                            beta = min(beta, minEval)
                            if (beta <= alpha):
                                break
                        if self.is_valid_move_white(board, src, possibleDstDiagLeft):
                            # possibleStates.append(possibleDstDiagLeft)
                            newBoard = self.state_change_white(board, src, possibleDstDiagLeft, in_place=False)
                            eval = self.miniMax(newBoard, depth - 1, alpha, beta, maxPlayer=True)
                            minEval = min(minEval, eval)
                            beta = min(beta, eval)
                            if (beta <= alpha):
                                break
                        if self.is_valid_move_white(board, src, possibleDstDiagRight):
                            # possibleStates.append(possibleDstDiagRight)
                            newBoard = self.state_change_white(board, src, possibleDstDiagRight, in_place=False)
                            eval = self.miniMax(newBoard, depth - 1, alpha, beta, maxPlayer=True)
                            minEval = min(minEval, eval)
                            beta = min(beta, eval)
                            # Prune if beta <= alpha
                            if (beta <= alpha):
                                break
            return minEval

    def evaluation(self, board, depth):
        rows = len(board)
        cols = len(board[0])
        sum = 0
        black_count = 0
        white_count = 0
        weightedValues = 0

        for r in range(rows):
            for c in range(cols):
                if (board[r][c] == 'B'):
                    black_count += 1
                    sum += (6 - r)
                    weightedValues += self.weights[r][c]
                elif (board[r][c] == 'W'):
                    white_count += 1
        return (1 / sum) + (black_count - white_count) + black_count + depth + weightedValues

    def state_change_white(self, curr_board, from_, to_, in_place=True):
        ''' Updates the board configuration by modifying existing values if in_place is set to True, or creating a new board with updated values if in_place is set to False '''
        board = curr_board
        if not in_place:
            board = copy.deepcopy(curr_board)
        if self.is_valid_move_white(board, from_, to_):
            board[from_[0]][from_[1]] = '_'
            board[to_[0]][to_[1]] = 'W'
        return board

    # checks if a move made for white is valid or not. Move source: from_ [row, col], move destination: to_ [row, col]
    def is_valid_move_white(self, board, from_, to_):
        if board[from_[0]][from_[1]] != 'W':  # if move not made for white
            return False
        elif (to_[0] < 0 or to_[0] >= 6) or (to_[1] < 0 or to_[1] >= 6):  # if move takes pawn outside the board
            return False
        elif to_[0] != (from_[0] - 1):  # if move takes more than one step forward
            return False
        elif to_[1] > (from_[1] + 1) or to_[1] < (from_[1] - 1):  # if move takes beyond left/ right diagonal
            return False
        elif to_[1] == from_[1] and board[to_[0]][to_[1]] != '_':  # if pawn to the front, but still move forward
            return False
        elif ((to_[1] == from_[1] + 1) or (to_[1] == from_[1] - 1)) and board[to_[0]][
            to_[1]] == 'W':  # if white pawn to the diagonal or front, but still move forward
            return False
        else:
            return True


import time
import copy
import utils


class PlayerAI3:
    weights = [[0, 0, 0, 0, 0, 0], [2, 3, 4, 4, 3, 2], [3, 4, 5, 5, 4, 3], [4, 5, 6, 6, 5, 4], [5, 6, 7, 7, 6, 5],
               [999, 999, 999, 999, 999, 999]]
    initState = [
        ['B'] * 6, ['B'] * 6,  # 2 black rows
        ['_'] * 6, ['_'] * 6,  # 2 empty rows
        ['W'] * 6, ['W'] * 6,  # 2 white rows
    ]

    def make_move(self, board):
        rows = len(board)
        cols = len(board[0])
        bestEval = float('-inf')
        move = [0, 0], [0, 0]
        # if (board == self.initState):
        #     return [[1, 1], [2 , 1]]
        # else:
        for r in range(rows):
            for c in range(cols):
                # check if B can move forward directly
                if board[r][c] == 'B':
                    src = [r, c]
                    dstForward = [r + 1, c]
                    dstDiagLeft = [r + 1, c - 1]
                    dstDiagRight = [r + 1, c + 1]
                    if utils.is_valid_move(board, src, dstDiagLeft):
                        newBoard = utils.state_change(board, src, dstDiagLeft, in_place=False)
                        diagLeftEval = self.miniMax(newBoard, 3, float('-inf'), float('inf'), maxPlayer=False)
                        if (diagLeftEval > bestEval):
                            bestEval = diagLeftEval
                            move = src, dstDiagLeft

                    if utils.is_valid_move(board, src, dstDiagRight):
                        newBoard = utils.state_change(board, src, dstDiagRight, in_place=False)
                        diagRightEval = self.miniMax(newBoard, 3, float('-inf'), float('inf'), maxPlayer=False)
                        if (diagRightEval > bestEval):
                            bestEval = diagRightEval
                            move = src, dstDiagRight

                    if utils.is_valid_move(board, src, dstForward):
                        newBoard = utils.state_change(board, src, dstForward, in_place=False)
                        forwardEval = self.miniMax(newBoard, 3, float('-inf'), float('inf'), maxPlayer=False)
                        if (forwardEval > bestEval):
                            bestEval = forwardEval
                            move = [r, c], [r + 1, c]

        print(move)
        return move  # invalid move

    def miniMax(self, board, depth, alpha, beta, maxPlayer=True, rows=6, cols=6):
        # Enumerate all possible steps from current state of board
        # Moves should only be forward, diagonally left, diagonally right
        # Optimisation here: Send entire tree so minimax only computes at most branch factor worth of leaves
        if depth == 0 or utils.is_game_over(board):
            return self.evaluation(board, depth)

        if maxPlayer:
            maxEval = float('-inf')
            for r in range(rows):
                for c in range(cols):
                    if board[r][c] == 'B':
                        # possibleStates = []
                        src = [r, c]
                        possibleDstFwd = [r + 1, c]
                        possibleDstDiagLeft = [r + 1, c - 1]
                        possibleDstDiagRight = [r + 1, c + 1]
                        if utils.is_valid_move(board, src, possibleDstFwd):
                            # possibleStates.append(possibleDstFwd)
                            newBoard = utils.state_change(board, src, possibleDstFwd, in_place=False)
                            eval = self.miniMax(newBoard, depth - 1, alpha, beta, maxPlayer=False)
                            maxEval = max(maxEval, eval)
                            alpha = max(alpha, maxEval)
                            if (beta <= alpha):
                                break
                        if utils.is_valid_move(board, src, possibleDstDiagLeft):
                            # possibleStates.append(possibleDstDiagLeft)
                            newBoard = utils.state_change(board, src, possibleDstDiagLeft, in_place=False)
                            eval = self.miniMax(newBoard, depth - 1, alpha, beta, maxPlayer=False)
                            maxEval = max(maxEval, eval)
                            alpha = max(alpha, maxEval)
                            if (beta <= alpha):
                                break
                        if utils.is_valid_move(board, src, possibleDstDiagRight):
                            # possibleStates.append(possibleDstDiagRight)
                            newBoard = utils.state_change(board, src, possibleDstDiagRight, in_place=False)
                            eval = self.miniMax(newBoard, depth - 1, alpha, beta, maxPlayer=False)
                            maxEval = max(maxEval, eval)
                            alpha = max(alpha, maxEval)
                            if (beta <= alpha):
                                break
            return maxEval
        else:
            minEval = float('inf')
            for r in range(rows):
                for c in range(cols):
                    if board[r][c] == 'W':
                        # possibleStates = []
                        src = [r, c]
                        possibleDstFwd = [r - 1, c]
                        possibleDstDiagLeft = [r - 1, c - 1]
                        possibleDstDiagRight = [r - 1, c + 1]
                        if self.is_valid_move_white(board, src, possibleDstFwd):
                            # possibleStates.append(possibleDstFwd)
                            newBoard = self.state_change_white(board, src, possibleDstFwd, in_place=False)
                            eval = self.miniMax(newBoard, depth - 1, alpha, beta, maxPlayer=True)
                            minEval = min(minEval, eval)
                            beta = min(beta, minEval)
                            if (beta <= alpha):
                                break
                        if self.is_valid_move_white(board, src, possibleDstDiagLeft):
                            # possibleStates.append(possibleDstDiagLeft)
                            newBoard = self.state_change_white(board, src, possibleDstDiagLeft, in_place=False)
                            eval = self.miniMax(newBoard, depth - 1, alpha, beta, maxPlayer=True)
                            minEval = min(minEval, eval)
                            beta = min(beta, eval)
                            if (beta <= alpha):
                                break
                        if self.is_valid_move_white(board, src, possibleDstDiagRight):
                            # possibleStates.append(possibleDstDiagRight)
                            newBoard = self.state_change_white(board, src, possibleDstDiagRight, in_place=False)
                            eval = self.miniMax(newBoard, depth - 1, alpha, beta, maxPlayer=True)
                            minEval = min(minEval, eval)
                            beta = min(beta, eval)
                            # Prune if beta <= alpha
                            if (beta <= alpha):
                                break
            return minEval

    def evaluation(self, board, depth):
        rows = len(board)
        cols = len(board[0])
        sum = 0
        black_count = 0
        white_count = 0
        weightedValues = 0

        for r in range(rows):
            for c in range(cols):
                if (board[r][c] == 'B'):
                    black_count += 1
                    sum += (6 - r)
                    weightedValues += self.weights[r][c]
                elif (board[r][c] == 'W'):
                    white_count += 1 * (1 / ((r + 1) * (c + 1)))
        return (1 / sum) + (100 * (black_count - white_count)) + black_count + (100 * depth) + weightedValues

    def state_change_white(self, curr_board, from_, to_, in_place=True):
        ''' Updates the board configuration by modifying existing values if in_place is set to True, or creating a new board with updated values if in_place is set to False '''
        board = curr_board
        if not in_place:
            board = copy.deepcopy(curr_board)
        if self.is_valid_move_white(board, from_, to_):
            board[from_[0]][from_[1]] = '_'
            board[to_[0]][to_[1]] = 'W'
        return board

    # checks if a move made for white is valid or not. Move source: from_ [row, col], move destination: to_ [row, col]
    def is_valid_move_white(self, board, from_, to_):
        if board[from_[0]][from_[1]] != 'W':  # if move not made for white
            return False
        elif (to_[0] < 0 or to_[0] >= 6) or (to_[1] < 0 or to_[1] >= 6):  # if move takes pawn outside the board
            return False
        elif to_[0] != (from_[0] - 1):  # if move takes more than one step forward
            return False
        elif to_[1] > (from_[1] + 1) or to_[1] < (from_[1] - 1):  # if move takes beyond left/ right diagonal
            return False
        elif to_[1] == from_[1] and board[to_[0]][to_[1]] != '_':  # if pawn to the front, but still move forward
            return False
        elif ((to_[1] == from_[1] + 1) or (to_[1] == from_[1] - 1)) and board[to_[0]][
            to_[1]] == 'W':  # if white pawn to the diagonal or front, but still move forward
            return False
        else:
            return True
##########################
# Game playing framework #
##########################
if __name__ == "__main__":
    # # public test case 1
    # res1 = utils.test([['B', 'B', 'B', 'B', 'B', 'B'], ['_', 'B', 'B', 'B', 'B', 'B'], ['_', '_', '_', '_', '_', '_'],
    #                    ['_', 'B', '_', '_', '_', '_'], ['_', 'W', 'W', 'W', 'W', 'W'], ['W', 'W', 'W', 'W', 'W', 'W']],
    #                   PlayerAI())
    #
    # #print(res1)
    # assert (res1 == True)
    #
    # # public test case 2
    # res2 = utils.test([['_', 'B', 'B', 'B', 'B', 'B'], ['_', 'B', 'B', 'B', 'B', 'B'], ['_', '_', '_', '_', '_', '_'],
    #                    ['_', 'B', '_', '_', '_', '_'], ['W', 'W', 'W', 'W', 'W', 'W'], ['_', '_', 'W', 'W', 'W', 'W']],
    #                   PlayerAI())
    # # print(res2)
    # assert (res2 == True)
    #
    # # public test case 3
    # res3 = utils.test([['_', '_', 'B', 'B', 'B', 'B'], ['_', 'B', 'B', 'B', 'B', 'B'], ['_', '_', '_', '_', '_', '_'],
    #                    ['_', 'B', 'W', '_', '_', '_'], ['_', 'W', 'W', 'W', 'W', 'W'], ['_', '_', '_', 'W', 'W', 'W']],
    #                   PlayerAI())
    # #print(res3)
    # assert (res3 == True)

    # template code for question 2 and question 3
    # generates initial boardzzzzz
    board = utils.generate_init_state()
    # game play
    res = utils.play(PlayerAI(), PlayerAI(),
                     board)  # PlayerNaive() will be replaced by a baby agent in question 2, ozzzzr a base agent in question 3
    print(res)  # BLACK wins means your agent wins. Passing the test case on Coursemology means your agent wins.
