import chess
import time as t
import numpy as np
from tensorflow.keras.models import model_from_json
import chess as c
import pandas as pd
from joblib import load
df=pd.read_csv("fen.csv")
df.columns=['fen','val']
def position_parser(position_string):
    piece_map = {'K': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'Q': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'R': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'B': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 'N': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                 'P': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                 'k': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 'q': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                 'r': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                 'b': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                 'n': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                 'p': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

    position_array = []

    ps = position_string.replace('/', '')

    for char in ps:
        position_array += 12 * int(char) * [0] if char.isdigit() else piece_map[char]

    # print("position_parser =>  position_array: {}".format(asizeof.asizeof(position_array)))

    return position_array


def fen_to_binary_vector(uci):
    # counter += 1
    # clear_output(wait=True)
    # print(str(counter)+"\n")
    board.push(uci)
    fen=board.fen()
    board.pop()
    fen_infos = fen.split()

    pieces_ = 0
    turn_ = 1
    castling_rights_ = 2
    en_passant_ = 3
    half_moves_ = 4
    moves_ = 5

    binary_vector = []

    binary_vector += ([1 if fen_infos[turn_] == 'w' else 0]
                      + [1 if 'K' in fen_infos[castling_rights_] else 0]
                      + [1 if 'Q' in fen_infos[castling_rights_] else 0]
                      + [1 if 'k' in fen_infos[castling_rights_] else 0]
                      + [1 if 'q' in fen_infos[castling_rights_] else 0]
                      + position_parser(fen_infos[pieces_])
                      )

    # print("fen_to_binary_vector =>  binary_vector: {}".format(asizeof.asizeof(binary_vector)))
    # clear_output(wait=True)

    return binary_vector
def eval():
    pawntable = [0, 0, 0, 0, 0, 0, 0, 0,5, 10, 10, -20, -20, 10, 10, 5,5, -5, -10, 0, 0, -10, -5, 5,0, 0, 0, 20, 20, 0, 0, 0,5, 5, 10, 25, 25, 10, 5, 5,10, 10, 20, 30, 30, 20, 10, 10,50, 50, 50, 50, 50, 50, 50, 50,0, 0, 0, 0, 0, 0, 0, 0]
    knightstable = [-50, -40, -30, -30, -30, -30, -40, -50,-40, -20, 0, 5, 5, 0, -20, -40,-30, 5, 10, 15, 15, 10, 5, -30,-30, 0, 15, 20, 20, 15, 0, -30,-30, 5, 15, 20, 20, 15, 5, -30,-30, 0, 10, 15, 15, 10, 0, -30,-40, -20, 0, 0, 0, 0, -20, -40,-50, -40, -30, -30, -30, -30, -40, -50]
    bishopstable = [-20, -10, -10, -10, -10, -10, -10, -20,-10, 5, 0, 0, 0, 0, 5, -10,-10, 10, 10, 10, 10, 10, 10, -10,-10, 0, 10, 10, 10, 10, 0, -10,-10, 5, 5, 10, 10, 5, 5, -10,-10, 0, 5, 10, 10, 5, 0, -10,-10, 0, 0, 0, 0, 0, 0, -10,-20, -10, -10, -10, -10, -10, -10, -20]
    rookstable = [0, 0, 0, 5, 5, 0, 0, 0,-5, 0, 0, 0, 0, 0, 0, -5,-5, 0, 0, 0, 0, 0, 0, -5,-5, 0, 0, 0, 0, 0, 0, -5,-5, 0, 0, 0, 0, 0, 0, -5,-5, 0, 0, 0, 0, 0, 0, -5,5, 10, 10, 10, 10, 10, 10, 5,0, 0, 0, 0, 0, 0, 0, 0]
    queenstable = [-20, -10, -10, -5, -5, -10, -10, -20,-10, 0, 0, 0, 0, 0, 0, -10,-10, 5, 5, 5, 5, 5, 0, -10,0, 0, 5, 5, 5, 5, 0, -5,-5, 0, 5, 5, 5, 5, 0, -5,-10, 0, 5, 5, 5, 5, 0, -10,-10, 0, 0, 0, 0, 0, 0, -10,-20, -10, -10, -5, -5, -10, -10, -20]
    kingstable = [20, 30, 10, 0, 0, 10, 30, 20,20, 20, 0, 0, 0, 0, 20, 20,-10, -20, -20, -20, -20, -20, -20, -10,-20, -30, -30, -40, -40, -30, -30, -20,-30, -40, -40, -50, -50, -40, -40, -30,-30, -40, -40, -50, -50, -40, -40, -30,-30, -40, -40, -50, -50, -40, -40, -30,-30, -40, -40, -50, -50, -40, -40, -30]
    if board.is_checkmate():
            if board.turn:
                return -9999
            else:
                return 9999
    if board.is_stalemate():
            return 0
    if board.is_insufficient_material():
            return 0
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))
    material = 100 * (wp - bp) + 320 * (wn - bn) + 330 * (wb - bb) + 500 * (wr - br) + 900 * (wq - bq)
    pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq = pawnsq + sum([-pawntable[chess.square_mirror(i)]
                           for i in board.pieces(chess.PAWN, chess.BLACK)])
    knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.KNIGHT, chess.BLACK)])
    bishopsq = sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq = bishopsq + sum([-bishopstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.BISHOP, chess.BLACK)])
    rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.ROOK, chess.BLACK)])
    queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queensq = queensq + sum([-queenstable[chess.square_mirror(i)]
                             for i in board.pieces(chess.QUEEN, chess.BLACK)])
    kingsq = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)])
    kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.KING, chess.BLACK)])
    eval = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq
    if board.turn:
        return eval
    else:
        return -eval
def moves():
    move_vec=list(map(fen_to_binary_vector,board.legal_moves))
    move_nvec=np.array(list(map(np.array,move_vec)))
    pred_eval=model.predict(move_nvec)
    return sorted(list(zip(pred_eval,board.legal_moves)),key=lambda x:x[0])[:1]
def selectmove(depth):
    bestMove = chess.Move.null()
    bestValue = -99999
    alpha = -100000
    beta = 100000
    for move in board.legal_moves:
        board.push(move)
        boardValue = -alphabeta(-beta, -alpha, depth - 1)
        if boardValue > bestValue:
            bestValue = boardValue
            bestMove = move
        if (boardValue > alpha):
            alpha = boardValue
        board.pop()
    return bestMove
def alphabeta(alpha, beta, depthleft):
    bestscore = -9999
    if (depthleft == 0):
        return quiesce(alpha, beta)
    for move in board.legal_moves:
        board.push(move)
        score = -alphabeta(-beta, -alpha, depthleft - 1)
        board.pop()
        if (score >= beta):
            return score
        if (score > bestscore):
            bestscore = score
        if (score > alpha):
            alpha = score
    return bestscore
def quiesce(alpha, beta):
    if board.fen() not in move_eval:
        stand_pat = eval()
        move_eval[board.fen()]=stand_pat
    else:
        stand_pat = move_eval[board.fen()]
    if (stand_pat >= beta):
        return beta
    if (alpha < stand_pat):
        alpha = stand_pat
    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -quiesce(-beta, -alpha)
            board.pop()
            if (score >= beta):
                return beta
            if (score > alpha):
                alpha = score
    return alpha
count = 0
movehistory = []
totalt2 = []
totalt1=[]
move_eval={}
model=model_from_json(open("model.json","r").read())
board=c.Board()
while not board.is_game_over(claim_draw=True):
    if not board.turn:
        s = t.time()
        move = selectmove(3)
        board.push(move)
        print(board)
        print(f"NegaMax plays the move {move}  and takes {t.time() - s}s time")
        totalt2.append(t.time() - s)
    else:
        s=t.time()
        move = moves()
        board.push(move[0][1])
        print(board)
        print(f"NeuralNet plays the move {move[0][1]} and takes {t.time() - s}s time")
        totalt1.append(t.time()-s)
print("NeuralNet Wins") if board.turn else print("NegaMax Wins")
print(f"The game ended in {len(board.move_stack())} moves")
print(f"NeuralNet takes total {sum(totalt1)}s time and NegaMax takes total {sum(totalt2)}s time")