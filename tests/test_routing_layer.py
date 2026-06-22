from __future__ import annotations

import unittest

from rag_chatbot.routing_layer import (
    RoutingDecision,
    parse_routing_decision,
    route_query,
)


class FakeRouter:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def route(self, query: str) -> RoutingDecision:
        return RoutingDecision(mode=self.mode, reason="test decision")


class BrokenRouter:
    def route(self, query: str) -> RoutingDecision:
        raise RuntimeError("router failed")


class RoutingLayerTests(unittest.TestCase):
    def test_parses_valid_router_json(self) -> None:
        decision = parse_routing_decision(
            '{"mode": "keyword", "reason": "Exact section code."}'
        )

        self.assertEqual(decision.mode, "keyword")
        self.assertEqual(decision.reason, "Exact section code.")

    def test_uses_router_decision(self) -> None:
        decision = route_query("QM.1", router=FakeRouter("keyword"))

        self.assertEqual(decision.mode, "keyword")

    def test_falls_back_to_hybrid_when_router_fails(self) -> None:
        decision = route_query("quality duties", router=BrokenRouter())

        self.assertEqual(decision.mode, "hybrid")
        self.assertIn("Router unavailable", decision.reason)


if __name__ == "__main__":
    unittest.main()
