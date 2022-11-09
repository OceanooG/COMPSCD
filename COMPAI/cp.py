import pygame
from time import sleep
import copy
from random import choice
import datetime
from math import log, sqrt, e, inf

SIDEWAYS_DIRECTIONS = [[0, 1], [1, 0], [0, -1], [-1, 0]]
DIAGONAL_DIRECTIONS = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
DOUBLE_DIAGONAL = [[2, 2], [2, -2], [-2, -2], [-2, 2]]
HORSE_DIRECTIONS = [[2, 1], [2, -1], [-2, -1], [-2, 1], [1, 2], [1, -2], [-1, -2], [-1, 2]]

# store board(9*10)
def setup_board():
    # create empty
    board = [['' for i in range(9)] for i in range(10)]
    # set blacks
    board[0] = ['b_c', 'b_m', 'b_x', 'b_s', 'b_j', 'b_s', 'b_x', 'b_m', 'b_c']
    board[2][1], board[2][7] = 'b_p', 'b_p'
    board[3][0], board[3][2], board[3][4],  board[3][6],  board[3][8] = 'b_z', 'b_z', 'b_z', 'b_z', 'b_z'
    # set reds
    board[9] = ['r_c', 'r_m', 'r_x', 'r_s', 'r_j', 'r_s', 'r_x', 'r_m', 'r_c']
    board[7][1], board[7][7] = 'r_p', 'r_p'
    board[6][0], board[6][2], board[6][4], board[6][6], board[6][8] = 'r_z', 'r_z', 'r_z', 'r_z', 'r_z'
    
    return board

def display_board_in_shell(board):
    for row in board:
        for square in row:
            print(square + ', ', end='')
        print('')


def display_board_with_gui(board):
    # clear screen
    screen.fill((0,0,0))
    # add background
    screen.blit(pictures['bg'], (6, 6))
    # add pieces
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] != '':
                screen.blit(pictures[board[i][j]], (j * 57 + 2, i * 57 + 2))
    # update screen
    pygame.display.update()


def other_side(side):
    if side == 'r':
        return 'b'
    return 'r'


def find_piece(board, piece):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == piece:
                return i, j
    # print('cannot find', board, piece)
    return False


def find_all_legal_moves(board, side_to_move, check_matters=True):
    all_legal_moves = []
    for mover_x in range(0, 10):
        for mover_y in range(0, 9):
            if side_to_move not in board[mover_x][mover_y]:
                continue
            moving_piece = board[mover_x][mover_y][-1]
            if 'z' == moving_piece:
                if side_to_move == 'b':
                    if mover_x + 1 <= 9 and side_to_move not in board[mover_x + 1][mover_y]:  # move forward
                        all_legal_moves.append([mover_x, mover_y, mover_x + 1, mover_y])

                    if not mover_x <= 4:  # if after river(sideways movement)
                        if 0 <= mover_y + 1 <= 8 and side_to_move not in board[mover_x][mover_y + 1]:
                            all_legal_moves.append([mover_x, mover_y, mover_x, mover_y + 1])
                        if 0 <= mover_y - 1 <= 8 and side_to_move not in board[mover_x][mover_y - 1]:
                            all_legal_moves.append([mover_x, mover_y, mover_x, mover_y - 1])

                if side_to_move == 'r':
                    if mover_x - 1 >= 0 and side_to_move not in board[mover_x - 1][mover_y]:  # move forward
                        all_legal_moves.append([mover_x, mover_y, mover_x - 1, mover_y])

                    if not mover_x >= 5:  # if after river(sideways movement)
                        if 0 <= mover_y + 1 <= 8 and side_to_move not in board[mover_x][mover_y + 1]:
                            all_legal_moves.append([mover_x, mover_y, mover_x, mover_y + 1])
                        if 0 <= mover_y - 1 <= 8 and side_to_move not in board[mover_x][mover_y - 1]:
                            all_legal_moves.append([mover_x, mover_y, mover_x, mover_y - 1])

            elif 'j' == moving_piece:
                for i in SIDEWAYS_DIRECTIONS:
                    new_coords = [mover_x + i[0], mover_y + i[1]]
                    if 3 <= new_coords[1] <= 5:
                        if side_to_move == 'b':
                            if 0 <= new_coords[0] <= 2 and 'b' not in board[new_coords[0]][new_coords[1]]:
                                all_legal_moves.append([mover_x, mover_y] + new_coords)

                        elif side_to_move == 'r':
                            if 7 <= new_coords[0] <= 9 and 'r' not in board[new_coords[0]][new_coords[1]]:
                                all_legal_moves.append([mover_x, mover_y] + new_coords)

            elif 's' == moving_piece:
                for i in DIAGONAL_DIRECTIONS:
                    new_coords = [mover_x + i[0], mover_y + i[1]]
                    if 3 <= new_coords[1] <= 5:
                        if side_to_move == 'b':
                            if 0 <= new_coords[0] <= 2 and 'b' not in board[new_coords[0]][new_coords[1]]:
                                all_legal_moves.append([mover_x, mover_y] + new_coords)

                        elif side_to_move == 'r':
                            if 7 <= new_coords[0] <= 9 and 'r' not in board[new_coords[0]][new_coords[1]]:
                                all_legal_moves.append([mover_x, mover_y] + new_coords)

            elif 'c' == moving_piece:
                for direction in SIDEWAYS_DIRECTIONS:
                    temp_pos = [mover_x + direction[0], mover_y + direction[1]]
                    while 0 <= temp_pos[0] <= 9 and 0 <= temp_pos[1] <= 8:
                        if side_to_move in board[temp_pos[0]][temp_pos[1]]:
                            break
                        elif other_side(side_to_move) in board[temp_pos[0]][temp_pos[1]]:
                            all_legal_moves.append([mover_x, mover_y, temp_pos[0], temp_pos[1]])
                        else:
                            all_legal_moves.append([mover_x, mover_y, temp_pos[0], temp_pos[1]])
                        temp_pos = [temp_pos[0] + direction[0], temp_pos[1] + direction[1]]

            elif 'p' == moving_piece:
                for direction in SIDEWAYS_DIRECTIONS:
                    temp_pos = [mover_x + direction[0], mover_y + direction[1]]
                    in_between_count = 0
                    while 0 <= temp_pos[0] <= 9 and 0 <= temp_pos[1] <= 8:
                        if '' == board[temp_pos[0]][temp_pos[1]] and in_between_count == 0:
                            all_legal_moves.append([mover_x, mover_y, temp_pos[0], temp_pos[1]])
                            temp_pos = [temp_pos[0] + direction[0], temp_pos[1] + direction[1]]
                            continue
                        elif side_to_move in board[temp_pos[0]][temp_pos[1]] and in_between_count == 1:
                            break
                        elif other_side(side_to_move) in board[temp_pos[0]][temp_pos[1]] and in_between_count == 1:
                            all_legal_moves.append([mover_x, mover_y, temp_pos[0], temp_pos[1]])
                            break

                        if board[temp_pos[0]][temp_pos[1]] != '':
                            if in_between_count == 0:
                                in_between_count = 1
                            else:
                                break

                        temp_pos = [temp_pos[0] + direction[0], temp_pos[1] + direction[1]]

            elif 'x' == moving_piece:
                for direction in HORSE_DIRECTIONS:
                    new_coords = [mover_x + direction[0], mover_y + direction[1]]
                    if 0 <= new_coords[0] <= 9 and 0 <= new_coords[1] <= 8:
                        if abs(direction[0]) == 2:
                            if side_to_move not in board[new_coords[0]][new_coords[1]] and board[int(mover_x + direction[0]/2)][mover_y] == '':
                                all_legal_moves.append([mover_x, mover_y, new_coords[0], new_coords[1]])
                        elif abs(direction[1]) == 2:
                            if side_to_move not in board[new_coords[0]][new_coords[1]] and board[mover_x][int(mover_y + direction[1]/2)] == '':
                                all_legal_moves.append([mover_x, mover_y, new_coords[0], new_coords[1]])
            
            elif 'm' == moving_piece:
                for direction in DOUBLE_DIAGONAL:
                    new_coords = [mover_x + direction[0], mover_y + direction[1]]
                    in_between_square = [int((new_coords[0] + mover_x) / 2), int((new_coords[1] + mover_y) / 2)]
                    if 0 <= new_coords[1] <= 8:
                        if side_to_move == 'b':
                            if 0 <= new_coords[0] <= 4 and 'b' not in board[new_coords[0]][new_coords[1]] and board[in_between_square[0]][in_between_square[1]] == '':
                                all_legal_moves.append([mover_x, mover_y] + new_coords)

                        elif side_to_move == 'r':
                            if 5 <= new_coords[0] <= 9 and 'r' not in board[new_coords[0]][new_coords[1]] and  board[in_between_square[0]][in_between_square[1]] == '':
                                all_legal_moves.append([mover_x, mover_y] + new_coords)

    # eliminate general opposition moves
    with_general_legal = []
    other_king_coords = find_piece(board, other_side(side_to_move) + '_j')
    for new in all_legal_moves:
        if other_king_coords[1] == new[3]:  # moved to the same column
            found_piece_in_between = False
            for i in range(0, 9):
                if (i <= other_king_coords[0] and i <= new[2]) or (
                        i >= other_king_coords[0] and i >= new[2]):
                    if board[i][other_king_coords[1]] != '':
                        found_piece_in_between = True
                        break
            if found_piece_in_between:
                with_general_legal.append(new)
        else:
            with_general_legal.append(new)

    # eliminate moves that are illegal because of check
    if check_matters:
        final = []
        for move in with_general_legal:
            board_after_move = apply_move_to_board(board, move)
            legal_moves_for_enemy_after_move = find_all_legal_moves(board_after_move, other_side(side_to_move), check_matters=False)

            my_king_x, my_king_y = find_piece(board_after_move, side_to_move + '_j')
            legal = True
            for m in legal_moves_for_enemy_after_move:
                if m[2] == my_king_x and m[3] == my_king_y:
                    legal = False
                    break
            if legal:
                final.append(move)
    else:
        final = with_general_legal

    return final


def apply_move_to_board(board, move):
    copied_board = copy.deepcopy(board)
    moving_piece = copied_board[move[0]][move[1]]
    copied_board[move[0]][move[1]] = ''
    copied_board[move[2]][move[3]] = moving_piece
    return copied_board


def static_evaluation(board):  # + good for red, - good for black
    value_dict = {
                    'c': 9.5,
                    'm': 2.5,
                    'x': 4,
                    's': 2,
                    'p': 4.5
                   }
    multiplicator_dict = {
                            'r': 1,
                            'b': -1
                          }
    positive_material, negative_material = 0, 0
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] != '':
                piece = board[i][j][-1]
                side = board[i][j][0]
                if piece not in ['j', 'z']:
                    piece_value = value_dict[piece]
                    if side == 'r':
                        positive_material += piece_value
                    elif side == 'b':
                        negative_material += piece_value
                elif piece == 'z':
                    if side == 'r':
                        if i >= 5:
                            positive_material += 1
                        else:
                            positive_material += 2
                    else:
                        if i <= 4:
                            negative_material += 1
                        else:
                            negative_material += 2
    return positive_material, negative_material


def random_player(board, side_to_move):
    move = choice(find_all_legal_moves(board, side_to_move))
    return move


def mcts_player(board, side_to_move):
    root = Node(board)
    result_move = mcts_pred(root, is_game_over(board, side_to_move), side_to_move)
    return result_move


class Node:
    def __init__(self, board):
        self.state = board
        self.action = ''
        self.children = set()
        self.parent = None
        self.N = 0
        self.n = 0
        self.v = 0

    def get_board(self):
        return self.state


def ucb1(curr_node):
    ans = curr_node.v + 2 * (sqrt(log(curr_node.N + e + (10 ** -6)) / (curr_node.n + (10 ** -10))))
    return ans


def rollout(curr_node, side_to_move, current_depth):
    if is_game_over(curr_node.state, side_to_move):
        if side_to_move == 'r':
            # print("h1")
            return 1, curr_node
        elif side_to_move == 'b':
            # print("h2")
            return -1, curr_node
    if current_depth >= 30:
        return sum(static_evaluation(curr_node.get_board())) / 55, curr_node

    all_moves = find_all_legal_moves(curr_node.get_board(), side_to_move)

    for move in all_moves:
        tmp_state = curr_node.get_board()
        tmp_state = apply_move_to_board(tmp_state, move)
        child = Node(tmp_state)
        child.parent = curr_node
        curr_node.children.add(child)
    rnd_state = choice(list(curr_node.children))

    return rollout(rnd_state, other_side(side_to_move), current_depth + 1)


def expand(curr_node, side_to_move):
    if len(curr_node.children) == 0:
        return curr_node
    max_ucb = -inf
    if side_to_move == 'r':
        idx = -1
        max_ucb = -inf
        sel_child = None
        for i in curr_node.children:
            tmp = ucb1(i)
            if (tmp > max_ucb):
                idx = i
                max_ucb = tmp
                sel_child = i

        return expand(sel_child, 'b')

    elif side_to_move == 'b':
        idx = -1
        min_ucb = inf
        sel_child = None
        for i in curr_node.children:
            tmp = ucb1(i)
            if (tmp < min_ucb):
                idx = i
                min_ucb = tmp
                sel_child = i

        return expand(sel_child, 'r')


def rollback(curr_node, reward):
    curr_node.n += 1
    curr_node.v += reward
    while (curr_node.parent != None):
        curr_node.N += 1
        curr_node = curr_node.parent
    return curr_node


def mcts_pred(curr_node, over, side_to_move, iterations=30):
    if (over):
        return -1
    all_moves = find_all_legal_moves(curr_node.get_board(), side_to_move)
    map_state_move = dict()

    for i in all_moves:
        temporary_board = curr_node.get_board()
        temporary_board = apply_move_to_board(temporary_board, i)
        child = Node(temporary_board)

        child.parent = curr_node
        curr_node.children.add(child)
        map_state_move[child] = i

    while (iterations > 0):
        if side_to_move == 'r':
            idx = -1
            max_ucb = -inf
            sel_child = None
            for i in curr_node.children:
                tmp = ucb1(i)
                if (tmp > max_ucb):
                    idx = i
                    max_ucb = tmp
                    sel_child = i
            ex_child = expand(sel_child, 'b')
            reward, state = rollout(ex_child, 'r', 1)
            curr_node = rollback(state, reward)
            iterations -= 1
        elif side_to_move == 'b':
            idx = -1
            min_ucb = inf
            sel_child = None
            for i in curr_node.children:
                tmp = ucb1(i)
                if (tmp < min_ucb):
                    idx = i
                    min_ucb = tmp
                    sel_child = i

            ex_child = expand(sel_child, 'r')

            reward, state = rollout(ex_child, 'b', 1)

            curr_node = rollback(state, reward)
            iterations -= 1
    if side_to_move == 'r':

        mx = -inf
        idx = -1
        selected_move = ''
        for i in (curr_node.children):
            tmp = ucb1(i)
            if (tmp > mx):
                mx = tmp
                selected_move = map_state_move[i]
        return selected_move
    elif side_to_move == 'b':
        mn = inf
        idx = -1
        selected_move = ''
        for i in (curr_node.children):
            tmp = ucb1(i)
            if (tmp < mn):
                mn = tmp
                selected_move = map_state_move[i]
        return selected_move


def is_game_over(board, side_to_move):
    if len(find_all_legal_moves(board, side_to_move)) == 0:
        return True
    else:
        return False


def main():
    board = setup_board()
    turn_counter = 0
    while not is_game_over(board, ['r', 'b'][turn_counter % 2]):
        time_before_move = datetime.datetime.now()
        display_board_with_gui(board)
        while True:
            try:
                if turn_counter % 2 == 0:
                    move = player1(board, 'r')
                else:
                    move = player2(board, 'b')
                break
            except:
                pass
        st_ev = static_evaluation(board)
        board = apply_move_to_board(board, move)

        print('Move', turn_counter)
        print('Played move:', move)
        print('Board after move:', board)
        print('Static evaluation:', st_ev[0] - st_ev[1])
        print('Time passed:', (datetime.datetime.now() - time_before_move).total_seconds(), 's\n')
        turn_counter += 1

    print('Game over. Winner:', other_side(['r', 'b'][turn_counter % 2]))


# running part!!!
# white = red
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode([510, 569])

    player1 = mcts_player  # red player
    player2 = random_player  # black player
    # c-chariot, m-elephant, x-horse, s-guard, j-king, z-pawn, p-cannon
    picture_names = ['b_c', 'b_j', 'b_m', 'b_p', 'b_s', 'b_x', 'b_z', 'bg', 'r_c', 'r_j', 'r_m', 'r_p', 'r_s', 'r_x', 'r_z']
    pictures = {}
    for name in picture_names:
        pictures[name] = pygame.image.load('imgs/s2/' + name + '.png')

    main()

    sleep(120)
