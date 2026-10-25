"""Standalone starter: READY once, then one flushed JSON move per state."""

import json
import sys

BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


def coordinate(pos):
    return ord(pos[0]) - 97, int(pos[1]) - 1


def choose(state):
    side, board = state["you"], state["board"]
    gx, gy = coordinate(state["goals"][side])
    moves = []
    for start, piece in sorted(board.items()):
        if piece["side"] != side:
            continue
        x, y = coordinate(start)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if (dx == dy == 0) or not (0 <= nx < 9 and 0 <= ny < 9):
                    continue
                end = chr(97 + nx) + str(ny + 1)
                if end in state["obstacles"]:
                    continue
                target = board.get(end)
                if target and (target["side"] == side or BEATS[piece["kind"]] != target["kind"]):
                    continue
                score = (1000 if end == state["goals"][side] else 0)
                score += 20 if target else 0
                score -= 2000 if end in state["bombs"] else 0
                score -= max(abs(nx - gx), abs(ny - gy))
                moves.append((score, start, end))
    if not moves:
        own = next(pos for pos, piece in board.items() if piece["side"] == side)
        return {"from": own, "to": own}  # No pass: retry until the original clock expires.
    _, start, end = max(moves)
    return {"from": start, "to": end}


def main():
    print(json.dumps({"protocol_version": 1, "type": "ready"}), flush=True)
    for line in sys.stdin:
        state = json.loads(line)
        print(json.dumps(choose(state)), flush=True)


if __name__ == "__main__":
    main()
