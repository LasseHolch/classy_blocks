import copy
import dataclasses
from typing import ClassVar, List

import scipy.optimize

from classy_blocks.mesh import Mesh
from classy_blocks.modify.clamps.clamp import ClampBase
from classy_blocks.modify.grid import Grid
from classy_blocks.modify.junction import Junction
from classy_blocks.util.constants import VBIG
from classy_blocks.util.tools import report


class NoJunctionError(Exception):
    """Raised when there's a clamp defined for a vertex that doesn't exist"""


class NoClampError(Exception):
    """Raised when there's no junction defined for a given Clamp"""


@dataclasses.dataclass
class IterationData:
    """Data about a single iteration's progress"""

    relaxation: float
    initial_quality: float
    final_quality: float = VBIG

    @property
    def improvement(self) -> float:
        return self.initial_quality - self.final_quality


class IterationDriver:
    """Bookkeeping: iterations, results, relaxation, quality and whatnot"""

    INITIAL_RELAXATION: ClassVar[float] = 0.5

    def __init__(self, max_iterations: int, relaxed_iterations: int, tolerance: float):
        self.max_iterations = max_iterations
        self.relaxed_iterations = relaxed_iterations
        self.tolerance = tolerance

        self.iterations: List[IterationData] = []

    def begin_iteration(self, quality: float) -> IterationData:
        iteration = IterationData(self.next_relaxation, quality)
        self.iterations.append(iteration)

        return iteration

    def end_iteration(self, quality: float) -> None:
        self.iterations[-1].final_quality = quality

    @property
    def last_quality(self) -> float:
        if len(self.iterations) < 1:
            return VBIG

        return self.iterations[-1].final_quality

    @property
    def next_relaxation(self) -> float:
        """Returns the relaxation factor for the next iteration"""
        step = (1 - self.INITIAL_RELAXATION) / self.relaxed_iterations
        iteration = len(self.iterations)

        return min(1, IterationDriver.INITIAL_RELAXATION + step * iteration)

    @property
    def initial_improvement(self) -> float:
        if len(self.iterations) < 1:
            return VBIG

        return self.iterations[0].improvement

    @property
    def last_improvement(self) -> float:
        if len(self.iterations) < 2:
            return self.initial_improvement

        return self.iterations[-1].improvement

    @property
    def converged(self) -> bool:
        if len(self.iterations) <= 1:
            # At least two iterations are needed
            # so that the result of the last can be compared with the first one
            return False

        if len(self.iterations) >= self.max_iterations:
            return True

        return self.last_improvement / self.initial_improvement < self.tolerance * self.initial_improvement


class Optimizer:
    """Provides tools for blocking optimization"""

    def __init__(self, mesh: Mesh, report: bool = True):
        self.mesh = mesh
        self.report = report

        self.grid = Grid(mesh)
        self.clamps: List[ClampBase] = []

    def release_vertex(self, clamp: ClampBase) -> None:
        self.clamps.append(clamp)

    def _get_junction(self, clamp: ClampBase) -> Junction:
        """Returns a Junction that corresponds to clamp"""
        for junction in self.grid.junctions:
            if junction.vertex == clamp.vertex:
                return junction

        raise NoJunctionError

    def _get_clamp(self, junction: Junction) -> ClampBase:
        """Returns a Clamp that corresponds to given Junction"""
        for clamp in self.clamps:
            if clamp.vertex == junction.vertex:
                return clamp

        raise NoClampError

    def optimize_clamp(self, clamp: ClampBase, iteration: IterationData) -> float:
        """Move clamp.vertex so that quality at junction is improved;
        rollback changes if grid quality decreased after optimization"""
        initial_grid_quality = self.grid.quality
        initial_params = copy.copy(clamp.params)
        junction = self._get_junction(clamp)

        def fquality(params):
            # move all vertices according to X
            clamp.update_params(params)

            if clamp.is_linked:
                return self.grid.quality
            else:
                return junction.quality

        scipy.optimize.minimize(
            fquality,
            clamp.params,
            bounds=clamp.bounds,
            method="L-BFGS-B",
            options={"maxiter": 20, "ftol": 1, "eps": junction.delta / 10},
        )
        # alas, works well with this kind of problem but does not support bounds
        # method="COBYLA",
        # options={"maxiter": 20, "tol": 1, "rhobeg": junction.delta / 10},

        current_grid_quality = self.grid.quality

        if current_grid_quality > initial_grid_quality:
            # rollback if quality is worse
            clamp.update_params(initial_params)
            msg = (
                f"  < Rollback at vertex {clamp.vertex.index}: {initial_grid_quality:.3e} < {current_grid_quality:.3e}"
            )
            report(msg)
            current_grid_quality = 1
        else:
            msg = "  > Optimized junction at vertex "
            msg += f"{clamp.vertex.index}: {initial_grid_quality:.3e} > {current_grid_quality:.3e}"
            report(msg)

            clamp.update_params(clamp.params, iteration.relaxation)

        return initial_grid_quality / current_grid_quality

    def optimize_iteration(self, iteration: IterationData) -> None:
        # gather points that can be moved with optimization
        for junction in self.grid.get_ordered_junctions():
            try:
                clamp = self._get_clamp(junction)
                self.optimize_clamp(clamp, iteration)
            except NoClampError:
                continue

    def optimize(self, max_iterations: int = 20, relaxed_iterations: int = 2, tolerance: float = 0.1) -> None:
        """Move vertices, defined and restrained with Clamps
        so that better mesh quality is obtained.

        Within each iteration, all vertices will be moved, starting with the one with the most influence on quality.
        Lower tolerance values"""
        driver = IterationDriver(max_iterations, relaxed_iterations, tolerance)

        while not driver.converged:
            iteration = driver.begin_iteration(self.grid.quality)
            self.optimize_iteration(iteration)
            driver.end_iteration(self.grid.quality)

        # for i in range(max_iterations):
        #     # use lower relaxation factor with first iterations, then increase
        #     # TODO: tests
        #     relaxation = 1 - (1 - initial_relaxation) * np.exp(-i)
        #     self.optimize_iteration(relaxation)

        #     this_quality = self.grid.quality

        #     report(f"Optimization iteration {i}: {prev_quality:.3e} > {this_quality:.3e} (relaxation: {relaxation})")

        #     if abs((prev_quality - this_quality) / (this_quality + VSMALL)) < tolerance:
        #         report("Tolerance reached, stopping optimization")
        #         break

        #     prev_quality = this_quality
