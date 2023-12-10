def my_agent(observation, configuration):
    import math
    import random
    import time
    import numpy as np
    global current_state 

    start_time = time.time()
    def update_board_state(board, col, mark, configuration):
        columns = configuration.columns
        rows = configuration.rows
        row = max([r for r in range(rows) if board[col + (r * columns)] == 0])
        board[col + (row * columns)] = mark

    def check_win(board, player, configuration):
        temp_board = board.copy()
        new_board = np.asarray(temp_board).reshape(configuration.rows, configuration.columns)

        is_win = False
        check_lists = ['h', 'v', 'p', 'n']

        for d in check_lists:

            if d == 'h':
                for row in range(configuration.rows):
                    for column in range(configuration.columns - (configuration.inarow - 1)):
                        temp_window = list(new_board[row, column:column + configuration.inarow])
                        if (temp_window.count(player) == 4 and temp_window.count(0) == configuration.inarow - 4):
                            return True

            elif d == 'v':
                for row in range(configuration.rows - (configuration.inarow - 1)):
                    for column in range(configuration.columns):
                        temp_window = list(new_board[row:row + configuration.inarow, column])
                        if (temp_window.count(player) == 4 and temp_window.count(0) == configuration.inarow - 4):
                            return True

            elif d == 'p':
                for row in range(configuration.rows - (configuration.inarow - 1)):
                    for column in range(configuration.columns - (configuration.inarow - 1)):
                        temp_window = list(new_board[row + i][column + i] for i in range(configuration.inarow))
                        if (temp_window.count(player) == 4 and temp_window.count(0) == configuration.inarow - 4):
                            return True

            elif d == 'n':
                for row in range(configuration.inarow - 1, configuration.rows):
                    for column in range(configuration.columns - (configuration.inarow - 1)):
                        temp_window =  list(new_board[row - i][column + i] for i in range(configuration.inarow))
                        if (temp_window.count(player) == 4 and temp_window.count(0) == configuration.inarow - 4):
                            return True

        return is_win

    def calculate_reward(board, mark, configuration):
        if check_win(board, mark, configuration):
            return (True, 1)
        if check_tie(board):
            return (True, 0.5)
        else:
            return (False, -1)

    def check_tie(board):
        return not (any(mark == 0 for mark in board))

    def random_choice(board, config):
        return random.choice([c for c in range(config.columns) if board[c] == 0])

    def func_ucb(node_wins, node_visits, parent_visits, explore_factor):
        if node_visits == 0:
            return math.inf
        else:
            value_estimate = node_wins / node_visits
            exploration = math.sqrt(
                2 * math.log(parent_visits) / (node_visits))
            ucb_score = value_estimate + explore_factor * exploration
            return ucb_score

    def policy(board, mark, configuration):
        cur_mark = mark
        board = board.copy()
        col = random_choice(board, configuration)
        update_board_state(board, col, mark, configuration)
        is_terminal, reward = calculate_reward(board, mark, configuration)
        while not is_terminal:
            mark = 3-mark
            col = random_choice(board, configuration)
            update_board_state(board, col, mark, configuration)
            is_terminal, reward = calculate_reward(board, mark, configuration)
        if mark == cur_mark:
            return reward
        opponent_reward = 1-reward
        return opponent_reward

    class Node():

        def __init__(self, configuration, board,player,parent = None,is_terminal=False, terminal_score=None,move=None):

            self.board = board.copy()
            self.possible_moves = [c for c in range(configuration.columns) if not board[c]]
            self.expandable_moves = self.possible_moves.copy()
            self.configuration = configuration
            self.player = player
            self.children = []
            self.total_score =0
            self.total_visits = 0
            self.is_terminal = is_terminal
            self.terminal_score = terminal_score
            self.move = move
            self.parent = parent
            self.explore_factor = 1

        def Best_Child_UCB(self):
            scores_list = [func_ucb(child.total_score, child.total_visits, self.total_visits, self.explore_factor) for
                           child in self.children]
            best_score = max(scores_list)
            best_child_idx = scores_list.index(best_score)
            best_child = self.children[best_child_idx]
            return best_child

        def Best_Child_Score(self):
            total_score_list = [child.total_score for child in self.children]
            max_score = max(total_score_list)
            best_child_indx = total_score_list.index(max_score)
            best_child = self.children[best_child_indx]
            return best_child

        def select_child(self,move):
            for child in self.children:
                if child.move == move:
                    return child
            return None

        def simulation(self):
            if self.is_terminal:
                return self.terminal_score
            scores = 1-policy(self.board,self.player,self.configuration)
            return scores

        def backpropagation(self,score):
            self.total_score += score
            self.total_visits += 1
            if self.parent is not None:
                self.parent.backpropagation(1-score)

        def expanded_and_simulation(self):
            col = random.choice(self.expandable_moves)
            new_board = self.board.copy()
            update_board_state(new_board, col, self.player, self.configuration)
            is_terminal, terminal_score = calculate_reward(new_board, self.player, self.configuration)
            self.children.append(Node(self.configuration, new_board, 3-self.player,
                                        parent=self,
                                       is_terminal=is_terminal,
                                       terminal_score=terminal_score,
                                       move=col
                                       ))
            simulation_score = self.children[-1].simulation()
            self.children[-1].backpropagation(simulation_score)
            self.expandable_moves.remove(col)

        def MCTS(self):
            if self.is_terminal:
                self.backpropagation(self.terminal_score)
                return
            if ((not self.is_terminal) and (len(self.expandable_moves)>0)):
                self.expanded_and_simulation()
                return
            self.Best_Child_UCB().MCTS()


    time_limit = configuration.timeout - 0.34
    board = observation.board
    player = observation.mark

    def opponent_action(cur_board,previous_board,configuration):
        for i, m in enumerate(cur_board):
            if m != previous_board[i]:
                return i%configuration.columns
        return -1
    try:
        current_state = current_state.select_child(opponent_action(board, current_state.board, configuration))
        current_state.parent = None
    except:
        current_state = Node(configuration,board, player, parent=None, is_terminal=False, terminal_score=None, move=None)

    while time.time() - start_time <= time_limit:
        current_state.MCTS()

    current_state = current_state.Best_Child_Score()
    return current_state.move
