import pygame
from time import sleep
import copy
from random import choice
import datetime
from math import inf
import os

SIDEWAYS_DIRECTIONS = [[0, 1], [1, 0], [0, -1], [-1, 0]]
DIAGONAL_DIRECTIONS = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
DOUBLE_DIAGONAL = [[2, 2], [2, -2], [-2, -2], [-2, 2]]
HORSE_DIRECTIONS = [[2, 1], [2, -1], [-2, -1], [-2, 1], [1, 2], [1, -2], [-1, -2], [-1, 2]]

ITERATIONS = 2
MAX_DEPTH = 5

STATIC_EVAL_DIVISOR = 55

REPETITION_LIMIT = 3


piece_chosen_by_human = None


def create_filename():
    existing_logfiles = os.listdir('./saved_games/')
    highest_id = -1
    for logfile in existing_logfiles:

        id = int(logfile[logfile.index('e')+1:logfile.index('.')])

        if id > highest_id:
            highest_id = id
    return 'game' + str(highest_id + 1) + '.txt'


def save_to_file(filename, to_save):
    with open('./saved_games/' + filename, 'a') as logfile:
        logfile.write(to_save)


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
    print('cannot find', board, piece)
    return False


def is_check(board, side_to_protect):
    moves_for_enemy = find_all_legal_moves(board, other_side(side_to_protect), specials_matter=False)
    my_king_x, my_king_y = find_piece(board, side_to_protect + '_j')
    for enemy_move in moves_for_enemy:
        if enemy_move[2] == my_king_x and enemy_move[3] == my_king_y:
            return True
    return False


def find_all_legal_moves(board, side_to_move, specials_matter=True):
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
                            break
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

            elif 'm' == moving_piece:
                for direction in HORSE_DIRECTIONS:
                    new_coords = [mover_x + direction[0], mover_y + direction[1]]
                    if 0 <= new_coords[0] <= 9 and 0 <= new_coords[1] <= 8:
                        if abs(direction[0]) == 2:
                            if side_to_move not in board[new_coords[0]][new_coords[1]] and board[int(mover_x + direction[0]/2)][mover_y] == '':
                                all_legal_moves.append([mover_x, mover_y, new_coords[0], new_coords[1]])
                        elif abs(direction[1]) == 2:
                            if side_to_move not in board[new_coords[0]][new_coords[1]] and board[mover_x][int(mover_y + direction[1]/2)] == '':
                                all_legal_moves.append([mover_x, mover_y, new_coords[0], new_coords[1]])

            elif 'x' == moving_piece:
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
    final = []
    if specials_matter:
        for move in all_legal_moves:
            board_after_move = apply_move_to_board(board, move)

            # general opposition
            general_good = False
            my_king_coords = find_piece(board_after_move, side_to_move + '_j')
            other_king_coords = find_piece(board_after_move, other_side(side_to_move) + '_j')

            if other_king_coords[1] == my_king_coords[1]:  # moved to the same column
                found_piece_in_between = False
                for i in range(0, 9):
                    if (other_king_coords[0] > i > my_king_coords[0]) or (
                            other_king_coords[0] < i < my_king_coords[0]):
                        if board[i][other_king_coords[1]] != '':
                            found_piece_in_between = True

                            break
                if found_piece_in_between:
                    general_good = True
            else:
                general_good = True

            # sum
            if (not is_check(board_after_move, side_to_move)) and general_good:
                final.append(move)
        return final
    else:
        return all_legal_moves


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


def human_player(board, side_to_move, moves_so_far):
    moves_to_choose_from = find_all_legal_moves(board, side_to_move)
    global piece_chosen_by_human
    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                x = (pos[0] + 4) // 57
                y = (pos[1] + 3) // 57
                if piece_chosen_by_human is not None:
                    move_chosen = piece_chosen_by_human + [y, x]
                    if move_chosen in moves_to_choose_from:
                        piece_chosen_by_human = None
                        return move_chosen
                    else:
                        piece_chosen_by_human = [y, x]
                else:
                    piece_chosen_by_human = [y, x]


def random_player(board, side_to_move, moves_so_far):
    move = choice(find_all_legal_moves(board, side_to_move))
    return move


def mcts_player(board, side_to_move, moves_so_far):
    root = Node(board, moves_so_far)
    result_move = mcts_pred(root, side_to_move)
    return result_move


class Node:
    def __init__(self, board, moves_so_far):
        self.state = board
        self.moves_so_far = copy.deepcopy(moves_so_far)
        self.action = ''
        self.children = set()
        self.parent = None
        self.N = 0
        self.n = 0
        self.v = 0

    def get_board(self):
        return self.state


def ucb1(curr_node):
    if not curr_node.N == 0:
        ans = curr_node.v / curr_node.N
    else :
        return 0
    # ans = curr_node.v + 2 * (sqrt(log(curr_node.N + e + (10 ** -6)) / (curr_node.n + (10 ** -10))))
    return ans


def rollout(curr_node, side_to_move, current_depth):
    game_over, winner = is_game_over(curr_node.state, side_to_move, curr_node.moves_so_far)
    if game_over:
        if winner == 'r':
            # print("h1", current_depth)
            return 1, curr_node, current_depth
        elif winner == 'b':
            # print("h2", current_depth)
            return -1, curr_node, current_depth
    if current_depth >= MAX_DEPTH:
        # print('h3', current_depth)
        stat_ev = static_evaluation(curr_node.get_board())
        # return 0, curr_node, current_depth
        return (stat_ev[0] - stat_ev[1]) / STATIC_EVAL_DIVISOR, curr_node, current_depth

    all_moves = find_all_legal_moves(curr_node.get_board(), side_to_move)
    for move in all_moves:
        tmp_state = curr_node.get_board()
        tmp_state = apply_move_to_board(tmp_state, move)
        child = Node(tmp_state, curr_node.moves_so_far + [move])
        child.parent = curr_node
        curr_node.children.add(child)
    rnd_state = choice(list(curr_node.children))

    return rollout(rnd_state, other_side(side_to_move), current_depth + 1)


def expand(curr_node, side_to_move, real_starter_side_to_move):
    if len(curr_node.children) == 0:
        if side_to_move == real_starter_side_to_move:
            return curr_node
        else:
            return curr_node.parent

    if side_to_move == 'r':
        max_ucb = -inf
        selected_child = None
        for i in curr_node.children:
            tmp = ucb1(i)
            if (tmp > max_ucb):
                max_ucb = tmp
                selected_child = i

        return expand(selected_child, 'b', real_starter_side_to_move)

    elif side_to_move == 'b':
        min_ucb = inf
        selected_child = None
        for i in curr_node.children:
            tmp = ucb1(i)
            if (tmp < min_ucb):
                min_ucb = tmp
                selected_child = i

        return expand(selected_child, 'r', real_starter_side_to_move)


def rollback(curr_node, reward):
    curr_node.n += 1
    while (curr_node.parent != None):
        curr_node.v += reward
        curr_node.N += 1
        curr_node = curr_node.parent
    return curr_node


def mcts_pred(curr_node, side_to_move, iterations=ITERATIONS):
    all_moves = find_all_legal_moves(curr_node.get_board(), side_to_move)
    map_state_move = dict()

    # calculate children for current node
    for i in all_moves:
        temporary_board = curr_node.get_board()
        temporary_board = apply_move_to_board(temporary_board, i)

        child = Node(temporary_board, curr_node.moves_so_far + [i])

        child.parent = curr_node
        curr_node.children.add(child)
        map_state_move[child] = i

    while (iterations > 0):
        if side_to_move == 'r':
            for selected_child in curr_node.children:

                # get a leaf node from a previously created tree of nodes
                expanded_child = expand(selected_child, 'b', 'b')
                # rollout child

                reward, final_state, depth = rollout(expanded_child, 'b', 0)
                reward -= depth * 0.001
                # update values on the whole branch discovered by the rollout
                _ = rollback(final_state, reward)
            iterations -= 1


        elif side_to_move == 'b':
            for selected_child in curr_node.children:
                # get a leaf node from a previously created tree of nodes
                expanded_child = expand(selected_child, 'r', 'r')
                # rollout child

                reward, final_state, depth = rollout(expanded_child, 'r', 0)
                reward += depth * 0.001
                # update values on the whole branch discovered by the rollout
                _ = rollback(final_state, reward)
            iterations -= 1

    # choose best move from calculated children
    if side_to_move == 'r':
        mx = -inf
        selected_move = ''
        for i in curr_node.children:
            tmp = ucb1(i)
            # print(i.v, i.n, i.N, ucb1(i))
            if (tmp > mx):
                mx = tmp
                selected_move = map_state_move[i]
        return selected_move
    elif side_to_move == 'b':
        mn = inf
        selected_move = ''
        for i in (curr_node.children):
            tmp = ucb1(i)
            if (tmp < mn):
                mn = tmp
                selected_move = map_state_move[i]
        return selected_move


def is_game_over(my_board, side_to_move, moves_so_far):
    # over by repetition
    if len(moves_so_far) >= REPETITION_LIMIT * 4:
        can_be_repetitive = True
        last_4_moves = moves_so_far[-4:]
        for i in range(1, REPETITION_LIMIT):
            if last_4_moves != moves_so_far[-4+(i*-4):(i*-4)]:
                can_be_repetitive = False
        if can_be_repetitive:
            return True, side_to_move

    # over by checkmate/stalemate
    if len(find_all_legal_moves(my_board, side_to_move)) == 0:
        return True, other_side(side_to_move)
    else:
        return False, other_side(side_to_move)


def main():
    board = setup_board()
    moves_so_far = []
    '''
    board = [['' for i in range(9)] for i in range(10)]
    board[0][3] = 'b_j'
    board[9][4] = 'r_j'
    board[5][0] = 'r_c'
    '''

    turn_counter = 0
    game_over = False
    winner = None

    while not game_over:
        pygame.event.pump()
        time_before_move = datetime.datetime.now()
        display_board_with_gui(board)

        if turn_counter % 2 == 0:
            move = player1(board, 'r', copy.deepcopy(moves_so_far))
        else:
            move = player2(board, 'b', copy.deepcopy(moves_so_far))


        board = apply_move_to_board(board, move)
        moves_so_far.append(move)
        st_ev = static_evaluation(board)
        display_board_with_gui(board)

        # create output
        output_text = str('Move ' + str(turn_counter) + ',  played by ' + ['r', 'b'][turn_counter % 2] + '\n')
        output_text += str('Played move: ' + str(move) + '\n')
        output_text += str('Board after move: ' + str(board) + '\n')
        output_text += str('Legal moves for other player: ' + str(find_all_legal_moves(board, ['r', 'b'][(turn_counter + 1) % 2])) + '\n')
        output_text += str('Static evaluation: ' + str(st_ev[0] - st_ev[1]) + '\n')
        output_text += str('Check: ' + str(is_check(board, ['r', 'b'][(turn_counter + 1) % 2])) + '\n')
        output_text += str('Time passed: ' + str((datetime.datetime.now() - time_before_move).total_seconds()) + 's\n\n')

        # print and save output
        print(output_text)
        save_to_file(FILENAME, output_text)

        turn_counter += 1
        game_over, winner = is_game_over(board, ['r', 'b'][turn_counter % 2], moves_so_far)

    print('Game over. Winner:', winner)
    quit()


# running part!!!
# white = red




if __name__ == "__main__":
    FILENAME = create_filename()
    print('Results saved to', FILENAME)

    pygame.init()
    screen = pygame.display.set_mode([510, 569])

    player1 = mcts_player  # red player
    player2 = mcts_player  # black player
    # c-chariot, x-elephant, m-horse, s-guard, j-king, z-pawn, p-cannon
    picture_names = ['b_c', 'b_j', 'b_m', 'b_p', 'b_s', 'b_x', 'b_z', 'bg', 'r_c', 'r_j', 'r_m', 'r_p', 'r_s', 'r_x', 'r_z']
    pictures = {}
    for name in picture_names:
        pictures[name] = pygame.image.load('imgs/s2/' + name + '.png')

    main()
    sleep(120)
