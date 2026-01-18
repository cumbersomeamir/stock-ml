"""Data pipeline orchestration with dependency tracking."""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("trading_lab.pipeline")


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineTask:
    """Represents a task in the pipeline."""
    
    name: str
    function: Callable
    dependencies: List[str]
    params: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    result: Any = None


class Pipeline:
    """
    Data pipeline with dependency tracking and orchestration.
    
    Ensures tasks run in correct order based on dependencies,
    with caching and error recovery.
    """

    def __init__(self, name: str = "pipeline"):
        """Initialize pipeline."""
        self.name = name
        self.tasks: Dict[str, PipelineTask] = {}
        self.results: Dict[str, Any] = {}

    def add_task(
        self,
        name: str,
        function: Callable,
        dependencies: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> "Pipeline":
        """
        Add a task to the pipeline.
        
        Args:
            name: Task name
            function: Function to execute
            dependencies: List of task names this task depends on
            params: Parameters to pass to function
            
        Returns:
            Self for method chaining
        """
        if name in self.tasks:
            raise ValueError(f"Task '{name}' already exists")
        
        self.tasks[name] = PipelineTask(
            name=name,
            function=function,
            dependencies=dependencies or [],
            params=params or {},
        )
        
        return self

    def run(
        self,
        skip_completed: bool = True,
        stop_on_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the pipeline.
        
        Args:
            skip_completed: Skip tasks that are already completed
            stop_on_error: Stop pipeline if a task fails
            
        Returns:
            Dictionary with task results
        """
        logger.info(f"Starting pipeline: {self.name}")
        
        # Validate dependencies
        self._validate_dependencies()
        
        # Get execution order
        execution_order = self._get_execution_order()
        
        # Execute tasks
        for task_name in execution_order:
            task = self.tasks[task_name]
            
            # Check if dependencies completed successfully
            dependencies_met = all(
                self.tasks[dep].status == TaskStatus.COMPLETED
                for dep in task.dependencies
            )
            
            if not dependencies_met:
                task.status = TaskStatus.SKIPPED
                logger.warning(f"Skipping task '{task_name}' - dependencies not met")
                continue
            
            # Execute task
            try:
                task.status = TaskStatus.RUNNING
                task.start_time = datetime.now()
                
                logger.info(f"Executing task: {task_name}")
                
                # Prepare parameters (include results from dependencies)
                params = task.params.copy()
                for dep in task.dependencies:
                    if dep in self.results:
                        params[f"{dep}_result"] = self.results[dep]
                
                # Execute
                result = task.function(**params)
                
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.end_time = datetime.now()
                self.results[task_name] = result
                
                duration = (task.end_time - task.start_time).total_seconds()
                logger.info(f"✓ Completed task '{task_name}' in {duration:.2f}s")
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.end_time = datetime.now()
                task.error = str(e)
                
                logger.error(f"✗ Task '{task_name}' failed: {e}")
                
                if stop_on_error:
                    raise
        
        # Log summary
        self._log_summary()
        
        return self.results

    def _validate_dependencies(self) -> None:
        """Validate that all dependencies exist."""
        for task_name, task in self.tasks.items():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    raise ValueError(
                        f"Task '{task_name}' depends on non-existent task '{dep}'"
                    )

    def _get_execution_order(self) -> List[str]:
        """Get execution order using topological sort."""
        visited = set()
        order = []
        
        def visit(task_name: str):
            if task_name in visited:
                return
            
            task = self.tasks[task_name]
            for dep in task.dependencies:
                visit(dep)
            
            visited.add(task_name)
            order.append(task_name)
        
        for task_name in self.tasks:
            visit(task_name)
        
        return order

    def _log_summary(self) -> None:
        """Log pipeline execution summary."""
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in self.tasks.values() if t.status == TaskStatus.SKIPPED)
        
        logger.info(
            f"Pipeline '{self.name}' completed: "
            f"{completed} completed, {failed} failed, {skipped} skipped"
        )


def create_standard_pipeline() -> Pipeline:
    """
    Create the standard trading lab pipeline.
    
    Returns:
        Configured pipeline
    """
    pipeline = Pipeline(name="trading_lab_standard")
    
    # Define tasks
    from trading_lab.unify import unify_prices
    from trading_lab.features.build_features import build_features
    from trading_lab.models.supervised.train_supervised import train_supervised
    from trading_lab.backtest.walk_forward import walk_forward_backtest
    from trading_lab.reports.generate_report import generate_report
    
    pipeline.add_task(
        name="unify_prices",
        function=lambda: unify_prices(force_refresh=False),
        dependencies=[],
        params={}
    )
    
    pipeline.add_task(
        name="build_features",
        function=build_features,
        dependencies=["unify_prices"],
        params={"force_refresh": False}
    )
    
    pipeline.add_task(
        name="train_models",
        function=train_supervised,
        dependencies=["build_features"],
        params={"model_name": "gradient_boosting", "force_refresh": False}
    )
    
    pipeline.add_task(
        name="backtest",
        function=walk_forward_backtest,
        dependencies=["train_models"],
        params={"strategy": "supervised_prob_threshold"}
    )
    
    pipeline.add_task(
        name="generate_report",
        function=generate_report,
        dependencies=["backtest"],
        params={}
    )
    
    return pipeline
