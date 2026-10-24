from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]


class AuctionTaskAllocator:
    def __init__(self) -> None:
        pass

    def allocate(
        self,
        robot_positions: Dict[str, Point],
        frontiers: List[Point],
    ) -> Dict[str, Point]:

        assignments: Dict[str, Point] = {}
        remaining_robots = set(robot_positions.keys())
        remaining_frontiers = list(frontiers)

        while remaining_robots and remaining_frontiers:
            best_pair: Optional[Tuple[str, Point, float]] = None
            for robot_id in remaining_robots:
                rx, ry = robot_positions[robot_id]
                for frontier in remaining_frontiers:
                    fx, fy = frontier
                    distance = ((rx - fx) ** 2 + (ry - fy) ** 2) ** 0.5
                    if best_pair is None or distance < best_pair[2]:
                        best_pair = (robot_id, frontier, distance)

            if best_pair is None:
                break

            robot_id, frontier, _distance = best_pair
            assignments[robot_id] = frontier
            remaining_robots.discard(robot_id)
            remaining_frontiers.remove(frontier)

        return assignments
