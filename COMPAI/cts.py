import pygame

SCREEN_WIDTH=600
SCREEN_HEIGHT=650
s_x = 50
s_y = 50
l_s = 60

player1Color = 1
player2Color = 2
overColor = 3

BG_COLOR=pygame.Color(255, 255, 255)
Line_COLOR=pygame.Color(255, 255, 200)
TEXT_COLOR=pygame.Color(255, 0, 0)

# 定义颜色
BLACK = ( 0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = ( 0, 255, 0)
BLUE = ( 0, 0, 255)

repeat = 0

pieces_images = {
    'b_rook': pygame.image.load("imgs/s2/b_c.gif"),
    'b_elephant': pygame.image.load("imgs/s2/b_x.gif"),
    'b_king': pygame.image.load("imgs/s2/b_j.gif"),
    'b_knigh': pygame.image.load("imgs/s2/b_m.gif"),
    'b_mandarin': pygame.image.load("imgs/s2/b_s.gif"),
    'b_cannon': pygame.image.load("imgs/s2/b_p.gif"),
    'b_pawn': pygame.image.load("imgs/s2/b_z.gif"),

    'r_rook': pygame.image.load("imgs/s2/r_c.gif"),
    'r_elephant': pygame.image.load("imgs/s2/r_x.gif"),
    'r_king': pygame.image.load("imgs/s2/r_j.gif"),
    'r_knigh': pygame.image.load("imgs/s2/r_m.gif"),
    'r_mandarin': pygame.image.load("imgs/s2/r_s.gif"),
    'r_cannon': pygame.image.load("imgs/s2/r_p.gif"),
    'r_pawn': pygame.image.load("imgs/s2/r_z.gif"),
}

class MCTSMeta:
    EXPLORATION = 0.5
    RAVE_CONST = 300
    RANDOMNESS = 0.5
    POOLRAVE_CAPACITY = 10
    K_CONST = 10
    A_CONST = 0.25
    WARMUP_ROLLOUTS = 7

class GameMeta:
    PLAYERS = {'none': 0, 'player1Color': 1, 'player2Color': 2}
    INF = float('inf')
    GAME_OVER = -1
