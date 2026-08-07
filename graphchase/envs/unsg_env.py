from __future__ import annotations

from pathlib import Path
from typing import Optional
try:
    from PIL import Image
except ImportError:  # Pillow may not be installed; fallback to numpy-only path
    Image = None

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import networkx as nx
import numpy as np

from graphchase.graph.game_settings import GameSettings


class UNSGEnv(gym.Env):
    """
    Continuous-time pursuit-evasion environment on a graph.
    State per agent: (start, end, dist_to_end).
      - On node: (node, node, 0.0)
      - On edge: (u, v, dist_to_v) meaning traveling u->v with dist_to_v remaining.
    Action: node id as direction. 0 or current node means stay.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, game_settings: GameSettings, render_mode: str | None = None):
        super().__init__()
        self.settings = game_settings
        self.graph = game_settings.graph
        self.num_attackers = len(game_settings.attacker_init)
        self.num_defenders = len(game_settings.defender_init)
        self.num_agents = self.num_attackers + self.num_defenders
        self.render_mode = render_mode
        num_nodes = self.graph.number_of_nodes()
        self.action_space = spaces.Dict(
            {
                "attacker_action": spaces.MultiDiscrete([num_nodes] * self.num_attackers),
                "defender_action": spaces.MultiDiscrete([num_nodes] * self.num_defenders),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "attacker_state": spaces.Box(low=0.0, high=np.inf, shape=(self.num_attackers, 3), dtype=np.float32),
                "defender_state": spaces.Box(low=0.0, high=np.inf, shape=(self.num_defenders, 3), dtype=np.float32),
            }
        )
        self.time_horizon = int(game_settings.time_horizon)
        self.exit_nodes = set(game_settings.exit_nodes)
        # attackers first, then defenders
        self.initial_nodes = list(game_settings.attacker_init) + list(game_settings.defender_init)
        self.positions: list[tuple[int, int, float]] = []
        self.remaining_time: float = 1.0
        self.decision_agents: list[int] = []
        self.step_count: int = 0
        self.cur_time: int = 0
        self._attacker_segments: list[tuple[np.ndarray, np.ndarray]] = []
        self._defender_segments: list[tuple[np.ndarray, np.ndarray]] = []
        self._layout = self._build_layout()
        self._fig = None
        self._ax = None
        self._attacker_icon = None
        self._defender_icon = None
        self._load_agent_icons()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.positions = [(node, node, 0.0) for node in self.initial_nodes]
        self.remaining_time = 1.0
        self.step_count = 0
        self.cur_time = 0
        self.decision_agents = list(range(self.num_agents))
        self._attacker_segments.clear()
        self._defender_segments.clear()
        obs = self.get_current_obs()
        info = {
            "decision_agents": self.decision_agents,
            "remaining_time": self.remaining_time,
            "cur_time": self.cur_time,
            "attacker_legal_action": self._legal_actions(is_attacker=True),
            "defender_legal_action": self._legal_actions(is_attacker=False),
        }
        return obs, info

    def step(self, action_dict: dict[str, object]):
        required_keys = {"attacker_action", "defender_action"}
        if not required_keys.issubset(action_dict.keys()):
            raise ValueError(f"Action dict must contain keys {required_keys}")
        attacker_action = action_dict["attacker_action"]
        defender_action = action_dict["defender_action"]
        if not isinstance(attacker_action, list) or not isinstance(defender_action, list):
            raise ValueError("attacker_action and defender_action must be lists")
        if len(attacker_action) != self.num_attackers or len(defender_action) != self.num_defenders:
            raise ValueError("Action list length does not match number of agents")
        remaining_time = float(self.remaining_time)
        actions = [int(a) for a in attacker_action + defender_action]
        prev_positions = list(self.positions)
        avail_dists = self._compute_avail_distances(actions, remaining_time)

        feasible = [d for d in avail_dists if d <= remaining_time]
        if feasible:
            move_dist = min(feasible)
        else:
            move_dist = remaining_time

        self._apply_moves(actions, move_dist)
        self._record_traversed_segments(prev_positions, move_dist)

        post_remaining = max(remaining_time - move_dist, 0.0)
        consumed_full_step = post_remaining <= 0.0
        if consumed_full_step:
            decision_agents = list(range(self.num_agents))
            self.cur_time += 1
        else:
            tol = 1e-8
            decision_agents = [i for i, dist in enumerate(avail_dists) if dist - move_dist <= tol]

        obs = self.get_current_obs()

        attacker_reward = 0.0
        defender_reward = 0.0
        terminated = False
        truncated = False

        attacker_positions = self.positions[: self.num_attackers]
        defender_positions = self.positions[self.num_attackers :]

        for a_pos in attacker_positions:
            for d_pos in defender_positions:
                if self._same_position(a_pos, d_pos):
                    attacker_reward = -1.0
                    defender_reward = 1.0
                    terminated = True
                    break
            if terminated:
                break

        if not terminated:
            for a_pos in attacker_positions:
                if self._on_exit(a_pos):
                    attacker_reward = 1.0
                    defender_reward = -1.0
                    terminated = True
                    break

        if consumed_full_step:
            self.step_count += 1
        if not terminated and consumed_full_step and self.step_count >= self.time_horizon:
            attacker_reward = -1.0
            defender_reward = 1.0
            truncated = True

        # start next step with updated remaining time budget
        self.remaining_time = 1.0 if consumed_full_step else post_remaining

        info = {
            "decision_agents": decision_agents,
            "remaining_time": self.remaining_time,
            "cur_time": self.cur_time,
            "attacker_legal_action": self._legal_actions(is_attacker=True),
            "defender_legal_action": self._legal_actions(is_attacker=False),
        }
        reward = {"attacker": attacker_reward, "defender": defender_reward}
        return obs, reward, terminated, truncated, info

    def _compute_avail_distances(self, actions: list[int], remaining_time: float) -> list[float]:
        if len(actions) != len(self.positions):
            raise ValueError("Action count does not match number of agents")
        avail = []
        for action, pos in zip(actions, self.positions):
            start, end, dist_to_end = pos
            if action == 0:
                avail.append(float(remaining_time))
                continue
            if start == end:
                if action not in self._neighbor_set(start):
                    raise ValueError(f"Action {action} not neighbor of node {start}")
                edge_weight = self._edge_weight(start, action)
                avail.append(float(edge_weight))
                continue
            if action == end:
                avail.append(float(dist_to_end))
                continue
            if action == start:
                edge_weight = self._edge_weight(start, end)
                back_distance = max(edge_weight - dist_to_end, 0.0)
                avail.append(float(back_distance))
                continue
            raise ValueError(f"Action {action} not on current edge ({start}, {end})")
        return avail

    def _apply_moves(self, actions: list[int], move_dist: float):
        new_positions = []
        for action, pos in zip(actions, self.positions):
            start, end, dist_to_end = pos
            if move_dist <= 0 or action == 0:
                new_positions.append(pos)
                continue
            if start == end:
                edge_weight = self._edge_weight(start, action)
                if move_dist >= edge_weight:
                    new_positions.append((action, action, 0.0))
                else:
                    remaining = edge_weight - move_dist
                    new_positions.append((start, action, remaining))
                continue

            edge_weight = self._edge_weight(start, end)
            if action == end:
                if move_dist >= dist_to_end:
                    new_positions.append((end, end, 0.0))
                else:
                    new_positions.append((start, end, dist_to_end - move_dist))
            elif action == start:
                back_distance = max(edge_weight - dist_to_end, 0.0)
                if move_dist >= back_distance:
                    new_positions.append((start, start, 0.0))
                else:
                    remaining_to_start = back_distance - move_dist
                    new_positions.append((end, start, remaining_to_start))
            else:
                raise ValueError(f"Action {action} not on current edge ({start}, {end})")
        self.positions = new_positions

    def _edge_weight(self, u: int, v: int) -> float:
        return self.settings.edge_weight(u, v)

    def _neighbor_set(self, node: int) -> set[int]:
        if self.settings.neighbor_map is not None:
            return self.settings.neighbor_map.get(node, set())
        return set(self.settings.neighbors(node))

    def _record_traversed_segments(self, prev_positions: list[tuple[int, int, float]], move_dist: float):
        if move_dist <= 0:
            return
        for idx, (prev_pos, new_pos) in enumerate(zip(prev_positions, self.positions)):
            if prev_pos == new_pos:
                continue
            start_xy = self._agent_xy(prev_pos)
            end_xy = self._agent_xy(new_pos)
            segment = (start_xy, end_xy)
            if idx < self.num_attackers:
                self._attacker_segments.append(segment)
            else:
                self._defender_segments.append(segment)

    def _legal_action_list(self, pos: tuple[int, int, float]) -> list[int]:
        start, end, _ = pos
        if start == end:
            neighbors = sorted(self._neighbor_set(start))
            return [0] + neighbors
        return [0, start, end]

    def _legal_actions(self, is_attacker: bool) -> list[list[int]]:
        if is_attacker:
            positions = self.positions[: self.num_attackers]
        else:
            positions = self.positions[self.num_attackers :]
        return [self._legal_action_list(pos) for pos in positions]

    def _same_position(self, pos_a: tuple[int, int, float], pos_b: tuple[int, int, float]) -> bool:
        return pos_a == pos_b

    def _on_exit(self, pos: tuple[int, int, float]) -> bool:
        start, end, dist = pos
        if start == end and dist == 0.0 and start in self.exit_nodes:
            return True
        return False

    def _build_layout(self):
        if isinstance(self.settings.metadata, dict) and "layout" in self.settings.metadata:
            return self.settings.metadata["layout"]
        pos_attrs = nx.get_node_attributes(self.graph, "pos")
        if pos_attrs:
            return pos_attrs
        return nx.spring_layout(self.graph, seed=0)

    def _agent_xy(self, pos: tuple[int, int, float]) -> np.ndarray:
        u, v, dist = pos
        coord_u = np.asarray(self._layout[u])
        if u == v:
            return coord_u
        coord_v = np.asarray(self._layout[v])
        weight = self._edge_weight(u, v)
        ratio = 1.0 - dist / weight if weight > 0 else 0.0
        ratio = min(max(ratio, 0.0), 1.0)
        return coord_u + ratio * (coord_v - coord_u)

    def _load_agent_icons(self):
        assets_dir = Path(__file__).resolve().parent.parent / "assets"
        attacker_path = assets_dir / "attacker_cartoon.png"
        defender_path = assets_dir / "defender_cartoon.png"

        def _read_icon(path: Path) -> np.ndarray | None:
            if not path.exists():
                return None
            img = mpimg.imread(path)
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            return self._resize_icon(img)

        self._attacker_icon = _read_icon(attacker_path)
        self._defender_icon = _read_icon(defender_path)

    def _layout_span(self) -> float:
        if not self._layout:
            return 1.0
        coords = np.array(list(self._layout.values()), dtype=float)
        if coords.size == 0:
            return 1.0
        return float(np.max(np.ptp(coords, axis=0)))

    def _resize_icon(self, img: np.ndarray, target_px: int = 32) -> np.ndarray:
        if Image is None:
            return img
        try:
            h, w = img.shape[:2]
            max_dim = max(h, w)
            if max_dim <= target_px:
                return img
            scale = target_px / float(max_dim)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            pil_img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8))
            pil_img = pil_img.resize((new_w, new_h), resample=Image.BILINEAR)
            arr = np.asarray(pil_img).astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            return arr
        except Exception:
            return img

    def _draw_icons(self, ax, coords: np.ndarray, icon: np.ndarray, zoom: float):
        for xy in coords:
            ab = AnnotationBbox(OffsetImage(icon, zoom=zoom), xy, frameon=False)
            ax.add_artist(ab)

    def render(self, mode=None):
        mode = mode or getattr(self, "render_mode", None) or "human"
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode {mode}")
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()
        ax = self._ax
        ax.clear()

        nx.draw_networkx_edges(self.graph, self._layout, ax=ax, edge_color="lightgray")
        for p0, p1 in self._attacker_segments:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="red", linewidth=3.5, solid_capstyle="round")
        for p0, p1 in self._defender_segments:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="blue", linewidth=2.0, solid_capstyle="round")
        regular_nodes = [n for n in self.graph.nodes if n not in self.exit_nodes]
        nx.draw_networkx_nodes(
            self.graph,
            self._layout,
            nodelist=regular_nodes,
            node_color="lightgray",
            edgecolors="black",
            ax=ax,
            node_size=300,
        )
        nx.draw_networkx_nodes(
            self.graph,
            self._layout,
            nodelist=list(self.exit_nodes),
            node_color="#b5ebb5",
            edgecolors="black",
            ax=ax,
            node_size=320,
        )
        nx.draw_networkx_labels(self.graph, self._layout, ax=ax, font_size=8)

        attacker_xy = np.array([self._agent_xy(p) for p in self.positions[: self.num_attackers]], dtype=float)
        defender_xy = np.array([self._agent_xy(p) for p in self.positions[self.num_attackers :]], dtype=float)
        attacker_plot = attacker_xy.copy()
        defender_plot = defender_xy.copy()

        span = self._layout_span()
        if len(attacker_xy) and len(defender_xy):
            offset = 0.02 * span if span > 0 else 0.02
            shifted_attackers: set[int] = set()
            shifted_defenders: set[int] = set()
            for a_idx, a_xy in enumerate(attacker_xy):
                for d_idx, d_xy in enumerate(defender_xy):
                    if np.allclose(a_xy, d_xy, atol=1e-6):
                        if a_idx not in shifted_attackers:
                            attacker_plot[a_idx, 0] -= offset
                            shifted_attackers.add(a_idx)
                        if d_idx not in shifted_defenders:
                            defender_plot[d_idx, 0] += offset
                            shifted_defenders.add(d_idx)

        # Match icon size roughly to node_size=320 (points^2) converted to pixels
        node_size = 320
        diameter_pts = 2 * np.sqrt(node_size / np.pi)
        target_px = diameter_pts * self._fig.dpi / 72.0
        icon_zoom = target_px / max(self._attacker_icon.shape[0] if self._attacker_icon is not None else 1,
                                    self._attacker_icon.shape[1] if self._attacker_icon is not None else 1,
                                    self._defender_icon.shape[0] if self._defender_icon is not None else 1,
                                    self._defender_icon.shape[1] if self._defender_icon is not None else 1)
        if len(attacker_xy):
            if self._attacker_icon is not None:
                self._draw_icons(ax, attacker_plot, self._attacker_icon, zoom=icon_zoom)
            else:
                ax.scatter(attacker_plot[:, 0], attacker_plot[:, 1], c="red", s=120, zorder=5)
        if len(defender_xy):
            if self._defender_icon is not None:
                self._draw_icons(ax, defender_plot, self._defender_icon, zoom=icon_zoom)
            else:
                ax.scatter(defender_plot[:, 0], defender_plot[:, 1], c="blue", s=120, zorder=5)

        ax.set_axis_off()

        if mode == "human":
            plt.pause(0.001)
            return None

        self._fig.canvas.draw()
        width, height = self._fig.canvas.get_width_height()
        buf = np.frombuffer(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buf.reshape(height, width, 4)[..., :3]

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig, self._ax = None, None

    def get_current_obs(self):
        attacker_state = np.array(
            [(s, e, float(d)) for (s, e, d) in self.positions[: self.num_attackers]], dtype=np.float32
        )
        defender_state = np.array(
            [(s, e, float(d)) for (s, e, d) in self.positions[self.num_attackers :]], dtype=np.float32
        )
        return {"attacker_state": attacker_state, "defender_state": defender_state}


# Backward-compatible alias
GraphEnv = UNSGEnv
