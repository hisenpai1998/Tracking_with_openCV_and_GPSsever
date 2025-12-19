from __future__ import annotations
import argparse
import json
import re
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray


@dataclass
class SpiralRow:
    id: int
    row: float
    col: float
    angle: float
    certainty: float


def rows_to_coords(rows: List[SpiralRow]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([[r.row, r.col] for r in rows], dtype=np.float32)


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=2)


def pairwise_distances_squared(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.einsum('ijk,ijk->ij', diff, diff, dtype=np.float32)


def detect_collisions(rows: List[SpiralRow], threshold: float, prev_map: Dict[int, SpiralRow] = None,
                      min_certainty: float = 0.0, return_suppressed: bool = False,
                      direction_mode: str = 'approaching',
                      active_pairs: set[Tuple[int, int]] | None = None,
                      hyst_factor: float = 1.05
                      ) -> Union[List[Tuple[int, int, float]], Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float, float, float]]]]:
    ids = [r.id for r in rows]
    certs = [r.certainty for r in rows]
    coords = rows_to_coords(rows)
    d2 = pairwise_distances_squared(coords)
    n = len(rows)
    collisions: List[Tuple[int, int, float]] = []
    suppressed: List[Tuple[int, int, float, float, float]] = []
    dir_vecs: List[Union[np.ndarray, None]] = [None] * n
    if prev_map is not None:
        for idx, r in enumerate(rows):
            prev = prev_map.get(r.id)
            if prev is not None:
                dx = r.row - prev.row
                dy = r.col - prev.col
                norm = math.hypot(dx, dy)
                dir_vecs[idx] = None if norm == 0 else np.array([dx / norm, dy / norm], dtype=np.float32)
    thr2 = float(threshold * threshold)
    thr_hyst2 = float((threshold * hyst_factor) * (threshold * hyst_factor))
    for i in range(n):
        for j in range(i + 1, n):
            dist2 = float(d2[i, j])
            a_id, b_id = ids[i], ids[j]
            pair = (min(a_id, b_id), max(a_id, b_id))
            sticky_inside = (direction_mode == 'sticky' and active_pairs is not None and pair in active_pairs and dist2 <= thr_hyst2)
            if dist2 < thr2 or sticky_inside:
                low_cert = (certs[i] < min_certainty) or (certs[j] < min_certainty)
                approaching = True
                if direction_mode == 'approaching':
                    vi = dir_vecs[i]
                    vj = dir_vecs[j]
                    if vi is not None and vj is not None:
                        r_vec = coords[j] - coords[i]
                        v_rel = vj - vi
                        approaching = float(np.dot(r_vec, v_rel)) < 0
                else:
                    approaching = True
                if approaching:
                    dist = math.sqrt(dist2)
                    if low_cert:
                        if return_suppressed:
                            suppressed.append((a_id, b_id, dist, certs[i], certs[j]))
                    else:
                        collisions.append((a_id, b_id, dist))
    return (collisions, suppressed) if return_suppressed else collisions


def pretty_distances(rows: List[SpiralRow]) -> str:
    if not rows:
        return "<no robots>"
    ids = [r.id for r in rows]
    coords = rows_to_coords(rows)
    d = pairwise_distances(coords)
    n = len(rows)
    header = f"Present IDs ({len(ids)}): {sorted(ids)}"
    if n < 2:
        return header + "\n<only one robot>"
    pairs: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(d[i, j]), ids[i], ids[j]))
    pairs.sort(key=lambda x: x[0])
    lines = [header, "Pairwise distances (sorted ascending):"]
    for dist, a, b in pairs:
        lines.append(f"  ({a}, {b}): {dist:.3f}")
    return "\n".join(lines)


def pretty_positions(rows: List[SpiralRow]) -> str:
    found: Dict[int, SpiralRow] = {r.id: r for r in rows if r.id >= 0}
    lines = ["Positions (ID, row, col, cert):"]
    for ident in range(10):
        r = found.get(ident)
        if r is None:
            lines.append(f"  {ident}: missing")
        else:
            lines.append(f"  {ident}: ({r.row:.3f}, {r.col:.3f})  cert={r.certainty:.2f}")
    return "\n".join(lines)


def pretty_table(rows: List[SpiralRow], header_time: bool = True) -> str:
    ts = time.strftime('%Y-%m-%d %H:%M:%S') if header_time else ''
    head = f"Time: {ts}" if header_time else "Positions and distances"
    sections = [head, pretty_positions(rows), pretty_distances(rows)]
    return "\n".join(sections)


def _as_spiral_rows_from_list(vals: List[float]) -> List[SpiralRow]:
    n = len(vals)
    rows: List[SpiralRow] = []
    if n % 5 != 0:
        return rows
    for i in range(0, n, 5):
        row, col, angle, rid, cert = (
            float(vals[i]),
            float(vals[i + 1]),
            float(vals[i + 2]),
            int(vals[i + 3]),
            float(vals[i + 4]),
        )
        if int(row) == -1 or rid == -1:
            continue
        rows.append(SpiralRow(rid, row, col, angle, cert))
    return rows


class MatrixCollisionNode(Node):
    def __init__(self, topic: str, msg_type: str, threshold: float, tick_hz: float, stale_sec: float,
                 min_certainty: float = 0.4, direction_mode: str = 'approaching', hyst_factor: float = 1.05,
                 history_file: str | None = None):
        super().__init__('matrix_gps_ros2')
        self.threshold = threshold
        self.tick_interval = 1.0 / max(1e-3, tick_hz)
        self.stale_sec = stale_sec
        self.min_certainty = min_certainty
        self.direction_mode = direction_mode
        self.hyst_factor = hyst_factor
        self.rows: Dict[int, SpiralRow] = {}
        self.last_seen: Dict[int, float] = {}
        self.prev_rows: Dict[int, SpiralRow] = {}
        self.active_collisions: set[Tuple[int, int]] = set()
        self.history_file = history_file
        self._dropped_rows: List[SpiralRow] = []
        if msg_type == 'string':
            self.sub = self.create_subscription(String, topic, self.cb_string, 10)
        else:
            self.sub = self.create_subscription(Float32MultiArray, topic, self.cb_array, 10)
        self.timer = self.create_timer(self.tick_interval, self.on_tick)

    def upsert(self, row: SpiralRow):
        if row.certainty < self.min_certainty:
            if self.history_file:
                self._dropped_rows.append(row)
            return
        prev = self.rows.get(row.id)
        if prev is not None:
            self.prev_rows[row.id] = prev
        self.rows[row.id] = row
        self.last_seen[row.id] = time.time()

    def cb_string(self, msg: String):
        s = msg.data
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                if obj and isinstance(obj[0], (list, tuple)):
                    flat: List[float] = []
                    for r in obj:
                        flat.extend([float(v) for v in r])
                    for row in _as_spiral_rows_from_list(flat):
                        self.upsert(row)
                elif obj and isinstance(obj[0], dict):
                    for r in obj:
                        row = float(r.get('row', r.get('r', r.get('Row', -1))))
                        col = float(r.get('col', r.get('c', r.get('Col', -1))))
                        angle = float(r.get('angle', r.get('Angle', 0.0)))
                        rid = int(r.get('id', r.get('ID', r.get('identity', -1))))
                        cert = float(r.get('certainty', r.get('conf', r.get('Conf', 1.0))))
                        if int(row) != -1 and rid != -1:
                            self.upsert(SpiralRow(rid, row, col, angle, cert))
                else:
                    vals = [float(v) for v in obj]
                    for row in _as_spiral_rows_from_list(vals):
                        self.upsert(row)
            elif isinstance(obj, dict):
                row = float(obj.get('row', obj.get('Row', -1)))
                col = float(obj.get('col', obj.get('Col', -1)))
                angle = float(obj.get('angle', obj.get('Angle', 0.0)))
                rid = int(obj.get('id', obj.get('ID', obj.get('identity', -1))))
                cert = float(obj.get('certainty', obj.get('Cert', 1.0)))
                if int(row) != -1 and rid != -1:
                    self.upsert(SpiralRow(rid, row, col, angle, cert))
            else:
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
                vals = [float(v) for v in nums]
                for row in _as_spiral_rows_from_list(vals):
                    self.upsert(row)
        except Exception:
            try:
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
                vals = [float(v) for v in nums]
                for row in _as_spiral_rows_from_list(vals):
                    self.upsert(row)
            except Exception:
                pass

    def cb_array(self, msg: Float32MultiArray):
        try:
            vals = list(msg.data)
            for row in _as_spiral_rows_from_list(vals):
                self.upsert(row)
        except Exception:
            pass

    def _append_history(self, rows: List[SpiralRow], collisions: List[Tuple[int, int, float]]):
        if not self.history_file:
            return
        try:
            with open(self.history_file, 'a', encoding='utf-8') as f:
                f.write(pretty_table(rows, header_time=True))
                f.write("\n")
                if self._dropped_rows:
                    f.write(f"Dropped (cert < {self.min_certainty:.2f}):\n")
                    for r in self._dropped_rows:
                        f.write(f"  {r.id}: ({r.row:.3f}, {r.col:.3f}) angle={r.angle:.3f} cert={r.certainty:.2f}\n")
                else:
                    f.write("Dropped (cert < min): none\n")
                if collisions:
                    f.write("Collisions:\n")
                    for a, b, d in collisions:
                        f.write(f"  {a} <-> {b} dist={d:.3f} (< {self.threshold})\n")
                else:
                    f.write("Collisions: none\n")
                f.write("----------\n")
        except Exception:
            pass
        finally:
            self._dropped_rows.clear()

    def on_tick(self):
        now = time.time()
        if self.stale_sec > 0:
            cutoff = now - self.stale_sec
            for rid in list(self.rows.keys()):
                if self.last_seen.get(rid, now) < cutoff:
                    self.rows.pop(rid, None)
                    self.last_seen.pop(rid, None)
                    self.prev_rows.pop(rid, None)
        for rid, row in list(self.rows.items()):
            if row.certainty < self.min_certainty:
                if self.history_file:
                    self._dropped_rows.append(row)
                self.rows.pop(rid, None)
                self.last_seen.pop(rid, None)
                self.prev_rows.pop(rid, None)
        rows = list(self.rows.values())
        cols = detect_collisions(rows, self.threshold, self.prev_rows,
                                 min_certainty=0.0,
                                 return_suppressed=False,
                                 direction_mode=self.direction_mode,
                                 active_pairs=self.active_collisions,
                                 hyst_factor=self.hyst_factor)
        if self.direction_mode == 'sticky':
            new_active = set(self.active_collisions)
            for a, b, _ in cols:
                new_active.add((min(a, b), max(a, b)))
            coords_map = {r.id: r for r in rows}
            to_remove = []
            for pair in new_active:
                a, b = pair
                if a in coords_map and b in coords_map:
                    ra, rb = coords_map[a], coords_map[b]
                    dist = math.hypot(ra.row - rb.row, ra.col - rb.col)
                    if dist > self.threshold * self.hyst_factor:
                        to_remove.append(pair)
                else:
                    to_remove.append(pair)
            for pair in to_remove:
                new_active.remove(pair)
            self.active_collisions = new_active
        if self.history_file:
            self._append_history(rows, cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', type=str,            default='robotPositions')
    ap.add_argument('--msg-type', type=str,         default='string', choices=['string', 'float32multiarray'])
    ap.add_argument('--threshold', type=float,      default=150.0)
    ap.add_argument('--tick-hz', type=float,        default=5.0)
    ap.add_argument('--stale-sec', type=float,      default=5.0)
    ap.add_argument('--min-cert', type=float,       default=0.25)
    ap.add_argument('--direction-mode', type=str,   default='approaching', choices=['approaching', 'any', 'sticky'])
    ap.add_argument('--hyst-factor', type=float,    default=1.05)
    ap.add_argument('--history-file', type=str,     default='./gps_history.log')
    args = ap.parse_args()
    rclpy.init()
    node = MatrixCollisionNode(
        args.topic, args.msg_type, args.threshold, args.tick_hz, args.stale_sec,
        args.min_cert, args.direction_mode, args.hyst_factor, history_file=args.history_file
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
