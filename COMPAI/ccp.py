import pygame
from time import sleep

import copy
from random import choice


# store board(9*10)
def setup_board():
    # create empty
    board = [['' for i in range(9)] for i in range(10)]
    # set blacks
    board[0] = ['b_c', 'b_x', 'b_m', 'b_s', 'b_j', 'b_s', 'b_m', 'b_x', 'b_c']
    board[2][1], board[2][7] = 'b_p', 'b_p'
    board[3][0], board[3][2], board[3][4], board[3][6], board[3][8] = 'b_z', 'b_z', 'b_z', 'b_z', 'b_z'
    # set reds
    board[9] = ['r_c', 'r_x', 'r_m', 'r_s', 'r_j', 'r_s', 'r_m', 'r_x', 'r_c']
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
    screen.fill((0, 0, 0))
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
    return False


def is_move_legal(board, move, check_matters=True):
    moving_piece = board[move[0]][move[1]][-1]
    moving_side = board[move[0]][move[1]][0]
    # if target square is occupied by a piece of the same color
    if board[move[2]][move[3]] != '' and moving_side == board[move[2]][move[3]][0]:
        return False

    # if any of the squares are too big or too small
    if not (0 <= move[0] <= 9 and 0 <= move[2] <= 9 and 0 <= move[1] <= 8 and 0 <= move[3] <= 8):
        return False

    # chariot or cannon
    if moving_piece == 'c' or moving_piece == 'p':
        # loop through squares in the way and check if they are empty
        inbetween_counter = 0
        if move[1] == move[3]:
            for i in range(move[0], move[2], -1 if move[0] + 1 > move[2] else 1):
                if board[i][move[1]] != '' and i != move[0]:
                    inbetween_counter += 1
        elif move[0] == move[2]:
            for i in range(move[1], move[3], -1 if move[1] + 1 > move[3] else 1):
                if board[move[0]][i] != '' and i != move[1]:
                    inbetween_counter += 1
        else:
            return False
        if moving_piece == 'c' and inbetween_counter != 0:  # if chariot and line is not empty
            return False
        if moving_piece == 'p' and inbetween_counter != 1 and board[move[2]][
            move[3]] != '':  # if cannon captures and the number of inbetween pieces is not 1
            return False
        if moving_piece == 'p' and inbetween_counter != 0 and board[move[2]][
            move[3]] == '':  # if cannon does not capture and line is not empty
            return False

    # elephant
    elif moving_piece == 'm':
        if abs(move[0] - move[2]) == 2 and abs(move[1] - move[3]) == 2:
            if board[int((move[0] + move[2]) / 2)][int((move[1] + move[3]) / 2)] != '':
                return False
        else:
            return False

    # horse
    elif moving_piece == 'x':
        # valid direction and no blocking
        if abs(move[0] - move[2]) == 2 and abs(move[1] - move[3]) == 1:
            if board[int((move[0] + move[2]) / 2)][move[1]] != '':
                return False
        elif abs(move[0] - move[2]) == 1 and abs(move[1] - move[3]) == 2:
            if board[move[2]][int((move[1] + move[3]) / 2)] != '':
                return False
        else:
            return False
        # did not cross the river
        if moving_side == 'r':
            if move[2] <= 4:
                return False
        if moving_side == 'b':
            if move[2] >= 5:
                return False


    # guard
    elif moving_piece == 's':
        if abs(move[0] - move[2]) == 1 and abs(move[1] - move[3]) == 1:
            if moving_side == 'b':
                if not (move[2] <= 2 and 3 <= move[3] <= 5):
                    return False
            elif moving_side == 'r':
                if not (move[2] >= 7 and 3 <= move[3] <= 5):
                    return False
        else:
            return False

    # king
    elif moving_piece == 'j':
        if (abs(move[0] - move[2]) == 1 and abs(move[1] - move[3]) == 0) or (
                abs(move[0] - move[2]) == 0 and abs(move[1] - move[3]) == 1):
            if moving_side == 'b':
                if not (move[2] <= 2 and 3 <= move[3] <= 5):
                    return False
            elif moving_side == 'r':
                if not (move[2] >= 7 and 3 <= move[3] <= 5):
                    return False
        else:
            return False

    # pawn
    elif moving_piece == 'z':
        if moving_side == 'b':
            if move[0] <= 4:  # if before river
                if move[2] != move[0] + 1 or move[1] != move[3]:
                    return False
            else:  # if after river
                if move[2] != move[0] + 1 or abs(move[1] - move[3]) != 1:
                    return False
        if moving_side == 'r':
            if move[0] >= 5:  # if before river
                if move[2] != move[0] - 1 or move[1] != move[3]:
                    return False
            else:  # if after river
                if move[2] != move[0] - 1 or abs(move[1] - move[3]) != 1:
                    return False

    # filter moves that are illegal because of general opposition
    board_after_move = apply_move_to_board(board, move)
    pos_r, pos_b = [None, None], [None, None]
    for i in [0, 1, 2, 7, 8, 9]:
        for j in [3, 4, 5]:
            if board[i][j] == 'r_j':
                pos_r = [i, j]
            if board[i][j] == 'b_j':
                pos_b = [i, j]
    if pos_r[1] == pos_b[1]:
        something_is_between = False
        for i in range(pos_b[0], pos_r[0]):
            if board[i][pos_r[1]] != '':
                something_is_between = True
        if not something_is_between:
            return False

    # filter moves that are illegal because of check
    if check_matters:
        board_after_move = apply_move_to_board(board, move)
        legal_moves_for_enemy_after_move = find_all_legal_moves(board_after_move, other_side(moving_side),
                                                                check_matters=False)
        my_king_x, my_king_y = find_piece(board_after_move, moving_side + '_j')
        for m in legal_moves_for_enemy_after_move:
            if m[2] == my_king_x and m[3] == my_king_y:
                return False

    return True


def find_all_legal_moves(board, side_to_move, check_matters=True):
    all_legal_moves = []
    for mover_x in range(0, 10):
        for mover_y in range(0, 9):
            if side_to_move in board[mover_x][mover_y]:
                for target_x in range(0, 10):
                    for target_y in range(0, 9):
                        if is_move_legal(board, [mover_x, mover_y, target_x, target_y], check_matters=check_matters):
                            all_legal_moves.append([mover_x, mover_y, target_x, target_y])
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


def lack_of_material(board):
    return False


def random_player(board, side_to_move):
    move = choice(find_all_legal_moves(board, side_to_move))
    return move


def mcts_player(board, side_to_move):
    pass


def is_game_over(board, side_to_move):
    if len(find_all_legal_moves(board, side_to_move)) == 0 or lack_of_material(board):
        return True
    else:
        return False


def main():
    board = setup_board()
    turn_counter = 0
    while not is_game_over(board, ['r', 'b'][turn_counter % 2]):
        display_board_with_gui(board)
        # sleep(1)
        if turn_counter % 2 == 0:
            move = player1(board, 'r')
        else:
            move = player2(board, 'b')
        print('Move', turn_counter)
        print('Played move:', move)
        st_ev = static_evaluation(board)
        print('Static evaluation:', st_ev[0] - st_ev[1], '\n')
        board = apply_move_to_board(board, move)
        turn_counter += 1
    print('Game over.')


# running part!!!
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode([510, 569])

    player1 = random_player
    player2 = random_player
    # c-chariot, m-elephant, x-horse, s-guard, j-king, z-pawn, p-cannon
    picture_names = ['b_c', 'b_j', 'b_m', 'b_p', 'b_s', 'b_x', 'b_z', 'bg', 'r_c', 'r_j', 'r_m', 'r_p', 'r_s', 'r_x',
                     'r_z']
    pictures = {}
    for name in picture_names:
        pictures[name] = pygame.image.load('imgs/s2/' + name + '.png')

    main()
