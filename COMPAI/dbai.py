import cts
#import time
from pieces import LPARRAY

def getPlayInfo(listpieces):
    pieces = movedeep(listpieces ,1 ,cts.player2Color)
    return [pieces[0].x,pieces[0].y, pieces[1], pieces[2]]

def movedeep(listpieces, deepstep, player):
    arr = LPARRAY(listpieces)
    listMoveEnabel = []
    for i in range(0, 9):
        for j in range(0, 10):
            for item in listpieces:
                if item.player == player and item.canmove(arr, i, j):
                    piecesremove = None
                    for itembefore in listpieces:
                        if itembefore.x == i and itembefore.y == j:
                            piecesremove= itembefore
                            break
                    if piecesremove != None:
                        listpieces.remove(piecesremove)
                    move_x = item.x
                    move_y = item.y
                    item.x = i
                    item.y = j

                    print(str(move_x) + "," + str(move_y) + "," + str(item.x) + "  ,  " + str(item.y))
                    scoreplayer1 = 0
                    scoreplayer2 = 0
                    for itemafter in listpieces:
                        if itemafter.player == cts.player1Color:
                            scoreplayer1 += itemafter.getScoreWeight(listpieces)
                        elif  itemafter.player == cts.player2Color:
                            scoreplayer2 += itemafter.getScoreWeight(listpieces)

                    #print("score："+item.imagskey +", "+str(len(moveAfterListpieces))+","+str(i)+","+str(j)+"," +str(scoreplayer1) +"  ,  "+ str(scoreplayer2) )
                    print(str(deepstep))
                    #如果得子 判断对面是否可以杀过来，如果又被杀，而且子力评分低，则不干
                    arrkill = LPARRAY(listpieces)

                    if scoreplayer2 > scoreplayer1 :
                        for itemkill in listpieces:
                            if itemkill.player == cts.player1Color and itemkill.canmove(arrkill, i, j):
                                scoreplayer2=scoreplayer1

                    if deepstep > 0 :
                        nextplayer = cts.player1Color if player == cts.player2Color else cts.player2Color
                        nextpiecesbest= movedeep(listpieces, deepstep -1, nextplayer)
                        listMoveEnabel.append([item, i, j, nextpiecesbest[3], nextpiecesbest[4], nextpiecesbest[5]])
                    else:
                        print(str(len(listpieces)))
                        #print("得分：" + item.imagskey + ", " + str(len(listpieces)) + "," + str(move_x) + "," + str(move_y) + "," + str(i) + "  ,  " + str(j))
                        if player == cts.player2Color:
                            listMoveEnabel.append([item, i, j, scoreplayer1, scoreplayer2, scoreplayer1 - scoreplayer2])
                        else:
                            listMoveEnabel.append([item, i, j, scoreplayer1, scoreplayer2, scoreplayer2 - scoreplayer1])
                    print("score："+str(scoreplayer1))
                    item.x = move_x
                    item.y = move_y
                    if piecesremove != None:
                        listpieces.append(piecesremove)

    list_scorepalyer1 = sorted(listMoveEnabel, key=lambda tm: tm[5], reverse=True)
    piecesbest = list_scorepalyer1[0]
    if deepstep ==1 :
        print(list_scorepalyer1)
    return piecesbest

