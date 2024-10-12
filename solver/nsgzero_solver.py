from solver.base_solver import BaseSolver
from agent.nsgzero.mcts_defender_runner import NsgzeroDefenderRunner

class NsgzeroSolver(BaseSolver):
    def __init__(self, env, defender_runner: NsgzeroDefenderRunner, evader_runner):
        super().__init__()
        self.env = env
        self.defender_runner = defender_runner
        self.evader_runner = evader_runner

    def solve(self,):
        self.defender_runner.train(self.evader_runner)