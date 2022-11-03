import pygame
import time
import cts
import pieces
import dbai
from button import Button
import MCTS

class MCC():
    window = None
    s_x = cts.s_x
    s_y = cts.s_y
    l_s = cts.l_s
    Max_X = s_x + 8 * l_s
    Max_Y = s_y + 9 * l_s

    player1Color = cts.player1Color
    player2Color = cts.player2Color
    Putdownflag = player1Color
    piecesSelected = None

    button_go = None
    piecesList = []

    def init_game(self):
        MCC.window = pygame.display.set_mode([cts.SCREEN_WIDTH, cts.SCREEN_HEIGHT])
        pygame.display.set_caption("MTCS Chinese chess")
        MCC.button_go = Button(MCC.window, "Start", cts.SCREEN_WIDTH - 100, 300)  # 创建开始按钮
        self.piecesInit()

        while True:
            time.sleep(0.1)
            # 获取事件
            MCC.window.fill(cts.BG_COLOR)
            self.drawChessboard()
            #MainGame.button_go.draw_button()
            self.piecesDisplay()
            self.WoF()
            self.dbai()
            self.getEvent()
            pygame.display.update()
            pygame.display.flip()

    def drawChessboard(self):
        mid_end_y = MCC.s_y + 4 * MCC.l_s
        min_s_y = MCC.s_y + 5 * MCC.l_s
        for i in range(0, 9):
            x = MCC.s_x + i * MCC.l_s
            if i==0 or i ==8:
                y = MCC.s_y + i * MCC.l_s
                pygame.draw.line(MCC.window, cts.BLACK, [x, MCC.s_y], [x, MCC.Max_Y], 1)
            else:
                pygame.draw.line(MCC.window, cts.BLACK, [x, MCC.s_y], [x, mid_end_y], 1)
                pygame.draw.line(MCC.window, cts.BLACK, [x, min_s_y], [x, MCC.Max_Y], 1)

        for i in range(0, 10):
            x = MCC.s_x + i * MCC.l_s
            y = MCC.s_y + i * MCC.l_s
            pygame.draw.line(MCC.window, cts.BLACK, [MCC.s_x, y], [MCC.Max_X, y], 1)

        speed_dial_start_x = MCC.s_x + 3 * MCC.l_s
        speed_dial_end_x = MCC.s_x + 5 * MCC.l_s
        speed_dial_y1 = MCC.s_y + 0 * MCC.l_s
        speed_dial_y2 = MCC.s_y + 2 * MCC.l_s
        speed_dial_y3 = MCC.s_y + 7 * MCC.l_s
        speed_dial_y4 = MCC.s_y + 9 * MCC.l_s

        pygame.draw.line(MCC.window, cts.BLACK, [speed_dial_start_x, speed_dial_y1], [speed_dial_end_x, speed_dial_y2], 1)
        pygame.draw.line(MCC.window, cts.BLACK, [speed_dial_start_x, speed_dial_y2],
                         [speed_dial_end_x, speed_dial_y1], 1)
        pygame.draw.line(MCC.window, cts.BLACK, [speed_dial_start_x, speed_dial_y3],
                         [speed_dial_end_x, speed_dial_y4], 1)
        pygame.draw.line(MCC.window, cts.BLACK, [speed_dial_start_x, speed_dial_y4],
                         [speed_dial_end_x, speed_dial_y3], 1)

    def piecesInit(self):
        MCC.piecesList.append(pieces.JU(MCC.player2Color, 0, 0))
        MCC.piecesList.append(pieces.JU(MCC.player2Color, 8, 0))
        MCC.piecesList.append(pieces.XIANG(MCC.player2Color, 2, 0))
        MCC.piecesList.append(pieces.XIANG(MCC.player2Color, 6, 0))
        MCC.piecesList.append(pieces.JIANG(MCC.player2Color, 4, 0))
        MCC.piecesList.append(pieces.MA(MCC.player2Color, 1, 0))
        MCC.piecesList.append(pieces.MA(MCC.player2Color, 7, 0))
        MCC.piecesList.append(pieces.PAO(MCC.player2Color, 1, 2))
        MCC.piecesList.append(pieces.PAO(MCC.player2Color, 7, 2))
        MCC.piecesList.append(pieces.SHI(MCC.player2Color, 3, 0))
        MCC.piecesList.append(pieces.SHI(MCC.player2Color, 5, 0))
        MCC.piecesList.append(pieces.BING(MCC.player2Color, 0, 3))
        MCC.piecesList.append(pieces.BING(MCC.player2Color, 2, 3))
        MCC.piecesList.append(pieces.BING(MCC.player2Color, 4, 3))
        MCC.piecesList.append(pieces.BING(MCC.player2Color, 6, 3))
        MCC.piecesList.append(pieces.BING(MCC.player2Color, 8, 3))

        MCC.piecesList.append(pieces.JU(MCC.player1Color, 0, 9))
        MCC.piecesList.append(pieces.JU(MCC.player1Color, 8, 9))
        MCC.piecesList.append(pieces.XIANG(MCC.player1Color, 2, 9))
        MCC.piecesList.append(pieces.XIANG(MCC.player1Color, 6, 9))
        MCC.piecesList.append(pieces.JIANG(MCC.player1Color, 4, 9))
        MCC.piecesList.append(pieces.MA(MCC.player1Color, 1, 9))
        MCC.piecesList.append(pieces.MA(MCC.player1Color, 7, 9))
        MCC.piecesList.append(pieces.PAO(MCC.player1Color, 1, 7))
        MCC.piecesList.append(pieces.PAO(MCC.player1Color, 7, 7))
        MCC.piecesList.append(pieces.SHI(MCC.player1Color, 3, 9))
        MCC.piecesList.append(pieces.SHI(MCC.player1Color, 5, 9))
        MCC.piecesList.append(pieces.BING(MCC.player1Color, 0, 6))
        MCC.piecesList.append(pieces.BING(MCC.player1Color, 2, 6))
        MCC.piecesList.append(pieces.BING(MCC.player1Color, 4, 6))
        MCC.piecesList.append(pieces.BING(MCC.player1Color, 6, 6))
        MCC.piecesList.append(pieces.BING(MCC.player1Color, 8, 6))

    def piecesDisplay(self):
        for item in MCC.piecesList:
            item.displaypieces(MCC.window)
            #MainGame.window.blit(item.image, item.rect)

    def getEvent(self):
        # 获取所有的事件
        eventList = pygame.event.get()
        for event in eventList:
            if event.type == pygame.QUIT:
                self.endGame()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                mouse_x = pos[0]
                mouse_y = pos[1]
                if (
                        mouse_x > MCC.s_x - MCC.l_s / 2 and mouse_x < MCC.Max_X + MCC.l_s / 2) and (
                        mouse_y > MCC.s_y - MCC.l_s / 2 and mouse_y < MCC.Max_Y + MCC.l_s / 2):
                    # print( str(mouse_x) + "" + str(mouse_y))
                    # print(str(MainGame.Putdownflag))
                    if MCC.Putdownflag != MCC.player1Color:
                        return

                    click_x = round((mouse_x - MCC.s_x) / MCC.l_s)
                    click_y = round((mouse_y - MCC.s_y) / MCC.l_s)
                    click_mod_x = (mouse_x - MCC.s_x) % MCC.l_s
                    click_mod_y = (mouse_y - MCC.s_y) % MCC.l_s
                    if abs(click_mod_x - MCC.l_s / 2) >= 5 and abs(
                            click_mod_y - MCC.l_s / 2) >= 5:
                        # print("有效点：x="+str(click_x)+" y="+str(click_y))
                        # 有效点击点
                        self.PutdownPieces(MCC.player1Color, click_x, click_y)
                else:
                    print("out")
                if MCC.button_go.is_click():
                    #self.restart()
                    print("button_go click")
                else:
                    print("button_go click out")

    def PutdownPieces(self, t, x, y):
        selectfilter=list(filter(lambda cm: cm.x == x and cm.y == y and cm.player == MCC.player1Color, MCC.piecesList))
        if len(selectfilter):
            MCC.piecesSelected = selectfilter[0]
            return

        if MCC.piecesSelected :
            #print("1111")

            arr = pieces.LPARRAY(MCC.piecesList)
            if MCC.piecesSelected.canmove(arr, x, y):
                self.PiecesMove(MCC.piecesSelected, x, y)
                MCC.Putdownflag = MCC.player2Color
        else:
            fi = filter(lambda p: p.x == x and p.y == y, MCC.piecesList)
            listfi = list(fi)
            if len(listfi) != 0:
                MCC.piecesSelected = listfi[0]

    def PiecesMove(self,pieces,  x , y):
        for item in  MCC.piecesList:
            if item.x ==x and item.y == y:
                MCC.piecesList.remove(item)
        pieces.x = x
        pieces.y = y
        print("move to " +str(x) +" "+str(y))
        return True

    def dbai(self):
        if MCC.Putdownflag == MCC.player2Color:
            print("AI's turn")
            computermove = dbai.getPlayInfo(MCC.piecesList)
            #if computer==None:
                #return
            piecemove = None
            for item in MCC.piecesList:
                if item.x == computermove[0] and item.y == computermove[1]:
                    piecemove= item

            self.PiecesMove(piecemove, computermove[2], computermove[3])
            MCC.Putdownflag = MCC.player1Color

    #win or lose
    def WoF(self):
        txt =""
        result = [MCC.player1Color, MCC.player2Color]
        for item in MCC.piecesList:
            if type(item) ==pieces.JIANG:
                if item.player == MCC.player1Color:
                    result.remove(MCC.player1Color)
                if item.player == MCC.player2Color:
                    result.remove(MCC.player2Color)

        if len(result)==0:
            return
        if result[0] == MCC.player1Color :
            txt = "U lost！"
        else:
            txt = "U Won！"
        MCC.window.blit(self.getTextSuface("%s" % txt), (cts.SCREEN_WIDTH - 100, 200))
        MCC.Putdownflag = cts.overColor

    def getTextSuface(self, text):
        pygame.font.init()
        # print(pygame.font.get_fonts())
        font = pygame.font.SysFont('kaiti', 18)
        txt = font.render(text, True, cts.TEXT_COLOR)
        return txt

    def endGame(self):
        print("exit")
        exit()

if __name__ == '__main__':
    MCC().init_game()

