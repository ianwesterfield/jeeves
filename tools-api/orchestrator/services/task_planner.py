"""
Task Planner - Decomposition and Parallelization Detection

Expands batch operations into individual steps and detects
opportunities for parallel execution.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional
import glob
import uuid

from schemas.models import Step, WorkspaceContext


logger = logging.getLogger("orchestrator.planner")


class TaskPlanner:
    """
    Task decomposition and parallelization detection.
    
    Expands high-level batch operations into concrete parallel steps.
    """
    
    async def expand_batch(
        self,
        batch_step: Step,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> List[Step]:
        """
        Expand a batch step into individual parallelizable steps.
        
        Args:
            batch_step: Step with tool="batch" and pattern/operation params
            workspace_context: Current workspace for path resolution
            
        Returns:
            List of individual Step objects ready for parallel execution
        """
        params = batch_step.params
        pattern = params.get("pattern", "*")
        operation = params.get("operation", "read_file")
        batch_id = batch_step.batch_id or f"batch_{uuid.uuid4().hex[:8]}"
        
        # Resolve workspace root
        workspace_root = "."
        if workspace_context:
            workspace_root = workspace_context.cwd
        
        # Find matching files
        search_pattern = os.path.join(workspace_root, "**", pattern)
        matching_files = glob.glob(search_pattern, recursive=True)
        
        # Limit batch size for safety
        max_batch_size = 20
        if workspace_context:
            max_batch_size = min(
                workspace_context.max_parallel_tasks * 5,
                100  # Hard limit
            )
        
        if len(matching_files) > max_batch_size:
            logger.warning(
                f"Batch size {len(matching_files)} exceeds limit {max_batch_size}, "
                f"truncating to first {max_batch_size} files"
            )
            matching_files = matching_files[:max_batch_size]
        
        # Generate individual steps
        steps = []
        for i, file_path in enumerate(matching_files):
            step = Step(
                step_id=f"{batch_id}_{i}",
                tool=operation,
                params={"path": file_path},
                batch_id=batch_id,
                reasoning=f"Part of batch operation: {operation} on {pattern}",
            )
            steps.append(step)
        
        logger.info(f"Expanded batch to {len(steps)} steps")
        return steps
    
    def detect_parallelization(
        self,
        task: str,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> bool:
        """
        Detect if a task can be parallelized.
        
        Looks for patterns like:
        - "all files", "every file", "each file"
        - "multiple", "batch", "parallel"
        - File glob patterns (*.py, *.ts, etc.)
        
        Args:
            task: User's task description
            workspace_context: For checking if parallel is enabled
            
        Returns:
            True if task appears parallelizable
        """
        # Check if parallel execution is allowed
        if workspace_context and not workspace_context.parallel_enabled:
            return False
        
        # Keywords that suggest parallelization
        parallel_keywords = [
            "all files",
            "every file",
            "each file",
            "all python",
            "all typescript",
            "all javascript",
            "multiple files",
            "batch",
            "parallel",
            "*.py",
            "*.ts",
            "*.js",
            "*.ps1",
            "*.md",
        ]
        
        task_lower = task.lower()
        return any(kw in task_lower for kw in parallel_keywords)
    
    def estimate_task_complexity(self, task: str) -> int:
        """
        Estimate task complexity (number of steps).
        
        Used for progress reporting and timeout estimation.
        
        Returns:
            Estimated number of steps (1-10)
        """
        # Simple heuristic based on task length and keywords
        complexity = 1
        
        # Multi-step indicators
        if any(word in task.lower() for word in ["and then", "after that", "finally", "first"]):
            complexity += 2
        
        # Batch indicators
        if self.detect_parallelization(task):
            complexity += 3
        
        # Analysis indicators
        if any(word in task.lower() for word in ["analyze", "review", "examine", "check"]):
            complexity += 1
        
        # Write indicators
        if any(word in task.lower() for word in ["write", "create", "generate", "modify"]):
            complexity += 1
        
        return min(complexity, 10)
