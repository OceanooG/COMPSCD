import pygame
import cts

class  Pieces():
    def __init__(self, player,  x, y):
        self.imagskey = self.getImagekey()
        self.image = cts.pieces_images[self.imagskey]
        self.x = x
        self.y = y
        self.player = player
        self.rect = self.image.get_rect()
        self.rect.left = cts.s_x + x * cts.l_s - self.image.get_rect().width / 2
        self.rect.top = cts.s_y + y * cts.l_s - self.image.get_rect().height / 2

    def displaypieces(self,screen):
        #print(str(self.rect.left))
        self.rect.left = cts.s_x + self.x * cts.l_s - self.image.get_rect().width / 2
        self.rect.top = cts.s_y + self.y * cts.l_s - self.image.get_rect().height / 2
        screen.blit(self.image,self.rect);
        #self.image = self.images
        #MainGame.window.blit(self.image,self.rect)

    def canmove(self, arr, moveto_x, moveto_y):
        pass
    def getImagekey(self):
        return None
    def getScoreWeight(self,listpieces):
        return  None

class JU(Pieces):
    def __init__(self, player,  x, y):
        self.player = player
        super().__init__(player,  x, y)

    def getImagekey(self):
        if self.player == cts.player1Color:
            return "r_rook"
        else:
            return "b_rook"

    def canmove(self, arr, moveto_x, moveto_y):
        if self.x == moveto_x and self.y == moveto_y:
            return False
        if arr[moveto_x][moveto_y] ==self.player :
            return  False
        if self.x == moveto_x:
            step = -1 if self.y > moveto_y else 1
            for i in range(self.y +step, moveto_y, step):
                if arr[self.x][i] !=0 :
                    return False
            #print(" move y")
            return True

        if self.y == moveto_y:
            step = -1 if self.x > moveto_x else 1
            for i in range(self.x + step, moveto_x, step):
                if arr[i][self.y] != 0:
                    return False
            return True

    def getScoreWeight(self, listpieces):
        score = 11
        return score

class MA(Pieces):
    def __init__(self, player,  x, y):
        self.player = player
        super().__init__(player,  x, y)
    def getImagekey(self):
        if self.player == cts.player1Color:
            return "r_knigh"
        else:
            return "b_knigh"
    def canmove(self, arr, moveto_x, moveto_y):
        if self.x == moveto_x and self.y == moveto_y:
            return False
        if arr[moveto_x][moveto_y] == self.player:
            return False
        #print(str(self.x) +""+str(self.y))
        move_x = moveto_x-self.x
        move_y = moveto_y - self.y
        if abs(move_x) == 1 and abs(move_y) == 2:
            step = 1 if move_y > 0 else -1
            if arr[self.x][self.y + step] == 0:
                return True
        if abs(move_x) == 2 and abs(move_y) == 1:
            step = 1 if move_x >0 else -1
            if arr[self.x +step][self.y] ==0 :
                return  True

    def getScoreWeight(self, listpieces):
        score = 5
        return score

class XIANG(Pieces):
    def __init__(self, player, x, y):
        self.player = player
        super().__init__(player, x, y)
    def getImagekey(self):
        if self.player == cts.player1Color:
            return "r_elephant"
        else:
            return "b_elephant"
    def canmove(self, arr, moveto_x, moveto_y):
        if self.x == moveto_x and self.y == moveto_y:
            return False
        if arr[moveto_x][moveto_y] == self.player:
            return False
        if self.y <=4 and moveto_y >=5 or self.y >=5 and moveto_y <=4:
            return  False
        move_x = moveto_x - self.x
        move_y = moveto_y - self.y
        if abs(move_x) == 2 and abs(move_y) == 2:
            step_x = 1 if move_x > 0 else -1
            step_y = 1 if move_y > 0 else -1
            if arr[self.x + step_x][self.y + step_y] == 0:
                return True

    def getScoreWeight(self, listpieces):
        score = 2
        return score
class SHI(Pieces):

    def __init__(self, player,  x, y):
        self.player = player
        super().__init__(player,  x, y)

    def getImagekey(self):
        if self.player == cts.player1Color:
            return "r_mandarin"
        else:
            return "b_mandarin"
    def canmove(self, arr, moveto_x, moveto_y):
        if self.x == moveto_x and self.y == moveto_y:
            return False
        if arr[moveto_x][moveto_y] == self.player:
            return False
        if moveto_x <3 or moveto_x >5:
            return False
        if moveto_y > 2 and moveto_y < 7:
            return False
        move_x = moveto_x - self.x
        move_y = moveto_y - self.y
        if abs(move_x) == 1 and abs(move_y) == 1:
            return True
    def getScoreWeight(self, listpieces):
        score = 2
        return score

class JIANG(Pieces):
    def __init__(self, player, x, y):
        self.player = player
        super().__init__(player, x, y)
    def getImagekey(self):
        if self.player == cts.player1Color:
            return "r_king"
        else:
            return "b_king"

    def canmove(self, arr, moveto_x, moveto_y):
        if self.x == moveto_x and self.y == moveto_y:
            return False
        if arr[moveto_x][moveto_y] == self.player:
            return False
        if moveto_x < 3 or moveto_x > 5:
            return False
        if moveto_y > 2 and moveto_y < 7:
            return False
        move_x = moveto_x - self.x
        move_y = moveto_y - self.y
        if abs(move_x) + abs(move_y) == 1:
            return True
    def getScoreWeight(self, listpieces):
        score = 150
        return score
class PAO(Pieces):
    def __init__(self, player,  x, y):
        self.player = player
        super().__init__(player, x, y)
    def getImagekey(self):
        if self.player == cts.player1Color:
            return "r_cannon"
        else:
            return "b_cannon"

    def canmove(self, arr, moveto_x, moveto_y):
        if self.x == moveto_x and self.y == moveto_y:
            return False
        if arr[moveto_x][moveto_y] == self.player:
            return False
        overflag = False
        if self.x == moveto_x:
            step = -1 if self.y > moveto_y else 1
            for i in range(self.y + step, moveto_y, step):
                if arr[self.x][i] != 0:
                    if overflag:
                        return False
                    else:
                        overflag = True

            if overflag and arr[moveto_x][moveto_y] == 0:
                return False
            if not overflag and arr[self.x][moveto_y] != 0:
                return False

            return True

        if self.y == moveto_y:
            step = -1 if self.x > moveto_x else 1
            for i in range(self.x + step, moveto_x, step):
                if arr[i][self.y] != 0:
                    if overflag:
                        return False
                    else:
                        overflag = True

            if overflag and arr[moveto_x][moveto_y] == 0:
                return False
            if not overflag and arr[moveto_x][self.y] != 0:
                return False
            return True
    def getScoreWeight(self, listpieces):
        score = 6
        return score

class BING(Pieces):
    def __init__(self, player, x, y):
        self.player = player
        super().__init__(player,  x, y)
    def getImagekey(self):
        if self.player == cts.player1Color:
            return "r_pawn"
        else:
            return "b_pawn"

    def canmove(self, arr, moveto_x, moveto_y):
        if self.x == moveto_x and self.y == moveto_y:
            return False
        if arr[moveto_x][moveto_y] == self.player:
            return False
        move_x = moveto_x - self.x
        move_y = moveto_y - self.y

        if self.player == cts.player1Color:
            if self.y > 4  and move_x != 0 :
                return  False
            if move_y > 0:
                return  False
        elif self.player == cts.player2Color:
            if self.y <= 4  and move_x != 0 :
                return  False
            if move_y < 0:
                return False

        if abs(move_x) + abs(move_y) == 1:
            return True
    def getScoreWeight(self, listpieces):
        score = 2
        return score

def LPARRAY(piecesList):
    arr = [[0 for i in range(10)] for j in range(9)]
    for i in range(0, 9):
        for j in range(0, 10):
            if len(list(filter(lambda cm: cm.x == i and cm.y == j and cm.player == cts.player1Color,
                               piecesList))):
                arr[i][j] = cts.player1Color
            elif len(list(filter(lambda cm: cm.x == i and cm.y == j and cm.player == cts.player2Color,
                                 piecesList))):
                arr[i][j] = cts.player2Color

    return arr

