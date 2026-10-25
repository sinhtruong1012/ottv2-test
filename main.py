"""OTT v2 bot: iterative-deepening alpha-beta with pondering. Standard library only.

* Position is kept in mutable arrays with make/unmake and a Zobrist hash.
* Search: negamax + PVS, transposition table, history ordering, and a
  quiescence search over captures / goal moves / forced goal defence.
* Pondering: while the opponent thinks, keep searching the position after our
  move (this CPU is not charged); results stay in the transposition table.
* Own-turn CPU use is a small fraction of the remaining budget.
"""

import json
import queue
import random
import sys
import threading
from time import monotonic, process_time

WIN = 100000
INF = 10 ** 9
OFF = 300000
TT_CAP = 160000
MAX_DEPTH = 24
MAT = (0, 350, 560, 700)
ADV = (0, 400, 190, 110, 70, 45, 30, 20, 12, 7, 3, 1, 0)
AVG_W = 18
KINDS = ("rock", "paper", "scissors")

SIDE_OF = (-1, 0, 0, 0, 1, 1, 1)
BEATS_T = []
for _p in range(7):
    _row = [False] * 7
    if _p:
        for _q in range(1, 7):
            if SIDE_OF[_p] != SIDE_OF[_q] and (_p - 1) % 3 == ((_q - 1) % 3 + 1) % 3:
                _row[_q] = True
    BEATS_T.append(tuple(_row))

_rng = random.Random(0x0771)
ZP = [[_rng.getrandbits(62) for _ in range(81)] for _ in range(7)]
ZB = [_rng.getrandbits(62) for _ in range(81)]
ZT = _rng.getrandbits(62)

# ---- mutable position -------------------------------------------------------
BOARD = [0] * 81
PCS = [set(), set()]
CNT = [0] * 7
BOMB = [False] * 81
NEI = [() for _ in range(81)]
GOAL = [0, 0]
DG = [[99] * 81, [99] * 81]
STATIC_KEY = None

TT = {}
HIST = [0] * (2 * 6561)
SEEN = {}
GAME_SEEN = {}
NODES = 0
BESTMV = -1
CPU_END = WALL_END = 0.0
PONDER = False
Q = queue.Queue()


class Abort(Exception):
    pass


def sq_index(name):
    return (int(name[1]) - 1) * 9 + ord(name[0]) - 97


def sq_name(i):
    return chr(97 + i % 9) + str(i // 9 + 1)


def setup_static(state):
    global NEI, GOAL, DG, STATIC_KEY
    obstacles = tuple(sorted(sq_index(p) for p in state["obstacles"]))
    goals = (sq_index(state["goals"]["A"]), sq_index(state["goals"]["B"]))
    if STATIC_KEY == (obstacles, goals):
        return
    STATIC_KEY = (obstacles, goals)
    blocked = set(obstacles)
    nei = []
    for sq in range(81):
        x, y = sq % 9, sq // 9
        row = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if (dx or dy) and 0 <= nx < 9 and 0 <= ny < 9 and ny * 9 + nx not in blocked:
                    row.append(ny * 9 + nx)
        nei.append(tuple(row))
    NEI = nei
    GOAL = list(goals)
    DG = []
    for side in (0, 1):
        dist = [99] * 81
        dist[goals[side]] = 0
        frontier = [goals[side]]
        while frontier:
            nxt = []
            for sq in frontier:
                for n in NEI[sq]:
                    if dist[n] == 99:
                        dist[n] = dist[sq] + 1
                        nxt.append(n)
            frontier = nxt
        DG.append(dist)
    TT.clear()


def load_position(state):
    setup_static(state)
    for i in range(81):
        BOARD[i] = 0
        BOMB[i] = False
    PCS[0].clear()
    PCS[1].clear()
    for i in range(7):
        CNT[i] = 0
    h = 0
    for pos, piece in state["board"].items():
        sq = sq_index(pos)
        code = (0 if piece["side"] == "A" else 3) + KINDS.index(piece["kind"]) + 1
        BOARD[sq] = code
        PCS[SIDE_OF[code]].add(sq)
        CNT[code] += 1
        h ^= ZP[code][sq]
    for pos in state["bombs"]:
        sq = sq_index(pos)
        BOMB[sq] = True
        h ^= ZB[sq]
    s = 0 if state["turn"] == "A" else 1
    if s:
        h ^= ZT
    return s, h


def snapshot():
    return BOARD[:], [set(PCS[0]), set(PCS[1])], CNT[:], BOMB[:]


def restore(snap):
    BOARD[:] = snap[0]
    PCS[0] = set(snap[1][0])
    PCS[1] = set(snap[1][1])
    CNT[:] = snap[2]
    BOMB[:] = snap[3]


# ---- moves ------------------------------------------------------------------
def make(s, st, en, h):
    p = BOARD[st]
    t = BOARD[en]
    bomb = BOMB[en]
    BOARD[st] = 0
    PCS[s].discard(st)
    h ^= ZP[p][st] ^ ZT
    if t:
        PCS[1 - s].discard(en)
        CNT[t] -= 1
        h ^= ZP[t][en]
    out = 0
    if bomb:
        BOMB[en] = False
        CNT[p] -= 1
        h ^= ZB[en]
        BOARD[en] = 0
        if CNT[p] == 0:
            out = -1
        elif t and CNT[t] == 0:
            out = 1
    else:
        BOARD[en] = p
        PCS[s].add(en)
        h ^= ZP[p][en]
        if en == GOAL[s] or (t and CNT[t] == 0):
            out = 1
    return h, p, t, bomb, out


def unmake(s, st, en, p, t, bomb):
    if bomb:
        BOMB[en] = True
        CNT[p] += 1
    else:
        PCS[s].discard(en)
    BOARD[st] = p
    PCS[s].add(st)
    BOARD[en] = t
    if t:
        PCS[1 - s].add(en)
        CNT[t] += 1


def gen(s, ttmv, tactical):
    out = []
    goal = GOAL[s]
    dg = DG[s]
    hb = s * 6561
    for st in PCS[s]:
        p = BOARD[st]
        beats = BEATS_T[p]
        dst = dg[st]
        row = st * 81
        for en in NEI[st]:
            t = BOARD[en]
            if t:
                if not beats[t]:
                    continue
                pri = 90000 if CNT[t] == 1 else (3200 if CNT[t] == 2 else 3000)
            else:
                if tactical and en != goal:
                    continue
                pri = HIST[hb + row + en]
            if en == goal:
                pri += 200000
            elif BOMB[en]:
                if tactical:
                    continue
                pri -= 6000
            pri += (dst - dg[en]) * 6
            if row + en == ttmv:
                pri += 1000000
            out.append((pri, st, en))
    out.sort(reverse=True)
    return out


def can_win_now(o):
    g = GOAL[o]
    t = BOARD[g]
    for st in NEI[g]:
        p = BOARD[st]
        if p and SIDE_OF[p] == o and (not t or BEATS_T[p][t]):
            return True
    return False


def evaluate(s):
    score = 0
    for side in (0, 1):
        b = side * 3
        v = MAT[CNT[b + 1]] + MAT[CNT[b + 2]] + MAT[CNT[b + 3]]
        dg = DG[side]
        best = 99
        total = 0
        pieces = PCS[side]
        for sq in pieces:
            d = dg[sq]
            total += d
            if d < best:
                best = d
        v += ADV[best if best < 12 else 12] - total * AVG_W // len(pieces)
        score += v if side == s else -v
    return score


# ---- search -----------------------------------------------------------------
def check_time():
    if PONDER:
        if not Q.empty():
            raise Abort
    elif monotonic() > WALL_END or process_time() > CPU_END:
        raise Abort


def qsearch(alpha, beta, ply, s, h, qd):
    global NODES
    NODES += 1
    if not NODES & 127:
        check_time()
    o = 1 - s
    threat = can_win_now(o)
    if threat:
        if qd < -3:
            return evaluate(s)
        best = -INF
        moves = gen(s, -1, False)
        if not moves:
            return -WIN + ply
    else:
        best = evaluate(s)
        if best >= beta:
            return best
        if best > alpha:
            alpha = best
        if qd <= 0:
            return best
        moves = gen(s, -1, True)
    for _, st, en in moves:
        nh, p, t, bomb, out = make(s, st, en, h)
        if out:
            score = (WIN - ply - 1) * out
        else:
            score = -qsearch(-beta, -alpha, ply + 1, o, nh, qd - 1)
        unmake(s, st, en, p, t, bomb)
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    return best


def negamax(depth, alpha, beta, ply, s, h):
    global NODES, BESTMV
    if depth <= 0:
        return qsearch(alpha, beta, ply, s, h, 3)
    NODES += 1
    if not NODES & 127:
        check_time()
    alpha0 = alpha
    ttmv = -1
    e = TT.get(h)
    if e is not None:
        ttmv = e & 8191
        e >>= 13
        if (e & 63) >= depth and ply:
            flag = (e >> 6) & 3
            sc = (e >> 8) - OFF
            if sc > WIN - 1000:
                sc -= ply
            elif sc < -WIN + 1000:
                sc += ply
            if flag == 0:
                return sc
            if flag == 1:
                if sc > alpha:
                    alpha = sc
            elif sc < beta:
                beta = sc
            if alpha >= beta:
                return sc
    moves = gen(s, ttmv, False)
    if not moves:
        return -WIN + ply
    o = 1 - s
    best = -INF
    bestmv = -1
    first = True
    for _, st, en in moves:
        nh, p, t, bomb, out = make(s, st, en, h)
        if out:
            score = (WIN - ply - 1) * out
        else:
            seen = SEEN.get(nh, 0)
            if seen >= 2:
                score = 0
            else:
                SEEN[nh] = seen + 1
                if first:
                    score = -negamax(depth - 1, -beta, -alpha, ply + 1, o, nh)
                else:
                    score = -negamax(depth - 1, -alpha - 1, -alpha, ply + 1, o, nh)
                    if alpha < score < beta:
                        score = -negamax(depth - 1, -beta, -alpha, ply + 1, o, nh)
                if seen:
                    SEEN[nh] = seen
                else:
                    del SEEN[nh]
        unmake(s, st, en, p, t, bomb)
        first = False
        if score > best:
            best = score
            bestmv = st * 81 + en
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if not t:
                        HIST[s * 6561 + bestmv] = min(HIST[s * 6561 + bestmv] + depth * depth, 2000)
                    break
    if not ply:
        BESTMV = bestmv
    flag = 0 if alpha0 < best < beta else (1 if best >= beta else 2)
    if len(TT) < TT_CAP or h in TT:
        sc = best
        if sc > WIN - 1000:
            sc += ply
        elif sc < -WIN + 1000:
            sc -= ply
        TT[h] = ((((sc + OFF) << 2 | flag) << 6 | depth) << 13) | (bestmv if bestmv >= 0 else 8191)
    return best


def iterate(s, h, max_depth, soft=None, started=0.0):
    """Iterative deepening from the current arrays; returns (move_code, score, depth)."""
    global BESTMV, SEEN
    snap = snapshot()
    SEEN = dict(GAME_SEEN)
    result = (-1, 0, 0)
    try:
        for depth in range(1, max_depth + 1):
            BESTMV = -1
            score = negamax(depth, -INF, INF, 0, s, h)
            if BESTMV >= 0:
                result = (BESTMV, score, depth)
            if abs(score) > WIN - 200:
                break
            if soft is not None and monotonic() - started > soft:
                break
    except Abort:
        pass
    restore(snap)
    return result


# ---- game loop --------------------------------------------------------------
def fallback_move(s):
    moves = gen(s, -1, False)
    pri, st, en = moves[0]
    return st * 81 + en


def decide(state):
    global PONDER, CPU_END, WALL_END
    s, h = load_position(state)
    GAME_SEEN[h] = GAME_SEEN.get(h, 0) + 1
    info = state.get("time") or {}
    cpu_left = info.get("cpu_remaining_seconds", {}).get("AB"[s], 5.0)
    wall_left = info.get("response_remaining_seconds", 1.0)
    cpu_slice = min(0.25, 0.05 * cpu_left) if cpu_left > 0.15 else 0.0
    wall_slice = max(0.05, min(0.55, wall_left - 0.3))
    code = -1
    if cpu_slice > 0:
        PONDER = False
        now = monotonic()
        CPU_END = process_time() + cpu_slice
        WALL_END = now + min(wall_slice, cpu_slice * 3 + 0.05)
        code, _, _ = iterate(s, h, MAX_DEPTH, soft=0.3 * min(cpu_slice * 1.5, wall_slice), started=now)
    if code < 0:
        e = TT.get(h)
        code = (e & 8191) if e is not None and (e & 8191) != 8191 else -1
        if code >= 0 and (BOARD[code // 81] == 0 or SIDE_OF[BOARD[code // 81]] != s):
            code = -1
    if code < 0:
        code = fallback_move(s)
    return s, h, code


def ponder(s, h, code):
    global PONDER
    st, en = divmod(code, 81)
    nh, p, t, bomb, out = make(s, st, en, h)
    GAME_SEEN[nh] = GAME_SEEN.get(nh, 0) + 1
    if out:
        return
    if len(TT) > TT_CAP * 0.7:
        TT.clear()
    for i in range(len(HIST)):
        HIST[i] >>= 1
    PONDER = True
    iterate(1 - s, nh, MAX_DEPTH)
    PONDER = False


def reader():
    for line in sys.stdin:
        if line.strip():
            Q.put(line)
    Q.put(None)


def main():
    threading.Thread(target=reader, daemon=True).start()
    print(json.dumps({"protocol_version": 1, "type": "ready"}), flush=True)
    while True:
        line = Q.get()
        if line is None:
            break
        state = json.loads(line)
        s = h = code = None
        try:
            s, h, code = decide(state)
            st, en = divmod(code, 81)
            move = {"from": sq_name(st), "to": sq_name(en)}
        except Exception:
            move = fallback_from_state(state)
        print(json.dumps(move), flush=True)
        if code is not None:
            try:
                load_position(state)
                if Q.empty():
                    ponder(s, h, code)
            except Exception:
                pass


def fallback_from_state(state):
    side = state["you"]
    board = state["board"]
    blocked = set(state["obstacles"])
    beats = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
    for start, piece in sorted(board.items()):
        if piece["side"] != side:
            continue
        x, y = ord(start[0]) - 97, int(start[1]) - 1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if (dx == dy == 0) or not (0 <= nx < 9 and 0 <= ny < 9):
                    continue
                end = chr(97 + nx) + str(ny + 1)
                target = board.get(end)
                if end in blocked or (target and (target["side"] == side
                                                  or beats[piece["kind"]] != target["kind"])):
                    continue
                return {"from": start, "to": end}
    own = next(iter(board))
    return {"from": own, "to": own}


if __name__ == "__main__":
    main()
