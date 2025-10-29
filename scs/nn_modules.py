from __future__ import annotations

from flax import (
    nnx,
    struct,
)
import jax
import optax


class SimpleNNTrainingState(struct.PyTreeNode):
    """Training state container for Neural Network that can be passed through
    JAX transformations.

    Attributes:
        model_def: Network graph definition.
        model_state: Current model parameters.
        optimizer: Gradient transformation for optimization.
        optimizer_state: Current state of the optimizer.
    """

    model_def: nnx.GraphDef = struct.field(pytree_node=False)
    model_state: nnx.State
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    optimizer_state: optax.OptState

    def apply_gradients(self, grads: nnx.State) -> NNTrainingState:
        """Apply gradients to model parameters."""
        updates, optimizer_state = self.optimizer.update(grads, self.optimizer_state)
        model_state = optax.apply_updates(self.model_state, updates)
        return self.replace(
            model_state=model_state,
            optimizer_state=optimizer_state,
        )

    @classmethod
    def create(
        cls,
        model_def: nnx.GraphDef,
        model_state: nnx.State,
        optimizer: optax.GradientTransformation,
    ) -> SimpleNNTrainingState:
        """Create a new training state instance.

        Args:
            model_def: Network graph definition.
            model_state: Initial model parameters.
            optimizer: Gradient transformation for optimization.

        Returns:
            A new SimpleNNTrainingState instance.
        """
        optimizer_state = optimizer.init(model_state)
        return cls(
            model_def=model_def,
            model_state=model_state,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
        )


class NNTrainingState(struct.PyTreeNode):
    """Training state container for Neural Network that can be passed through
    JAX transformations and that allows for a target network with soft updates.

    Attributes:
        model_def: Network graph definition.
        model_state: Current model parameters.
        target_model_state: Target network parameters for stable training.
        tau: Weight for target network update (soft update parameter).
        optimizer: Gradient transformation for optimization.
        optimizer_state: Current state of the optimizer.
    """

    model_def: nnx.GraphDef = struct.field(pytree_node=False)
    model_state: nnx.State
    target_model_state: nnx.State
    tau: float = struct.field(pytree_node=False)
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    optimizer_state: optax.OptState

    def apply_gradients(self, grads: nnx.State) -> NNTrainingState:
        """Apply gradients to model parameters and update target network.

        Performs a gradient step and updates the target network using soft updates
        with the tau parameter.

        Args:
            grads: Gradients for the model parameters.

        Returns:
            Updated `NNTrainingState` instance.
        """
        updates, optimizer_state = self.optimizer.update(grads, self.optimizer_state)
        model_state = optax.apply_updates(self.model_state, updates)
        target_model_state = jax.tree.map(
            lambda tp, p: self.tau * p + (1 - self.tau) * tp,
            self.target_model_state,
            model_state,
        )
        return self.replace(
            model_state=model_state,
            target_model_state=target_model_state,
            optimizer_state=optimizer_state,
        )

    @classmethod
    def create(
        cls,
        model_def: nnx.GraphDef,
        model_state: nnx.State,
        target_model_state: nnx.State,
        tau: float,
        optimizer: optax.GradientTransformation,
    ) -> NNTrainingState:
        """Create a new training state instance.

        Args:
            model_def: Network graph definition.
            model_state: Initial model parameters.
            target_model_state: Initial target network parameters.
            tau: Target network update rate (soft update parameter).
            optimizer: Gradient transformation for optimization.

        Returns:
            A new NNTrainingState instance.
        """
        optimizer_state = optimizer.init(model_state)
        return cls(
            model_def=model_def,
            model_state=model_state,
            target_model_state=target_model_state,
            tau=tau,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
        )


@nnx.jit(static_argnums=(2,))
def soft_update_target_model(
    model: nnx.Module,
    model_target: nnx.Module,
    tau: float,
) -> nnx.Module:
    """Soft updates the target model..

    Updates the parameters of the target model using a convex combination, with
    weight `tau`, of the current model parameters and the target model parameters.
    """
    model_params = nnx.state(model)
    graph_def, target_params, batch_stats = nnx.split(
        model_target, nnx.Param, nnx.BatchStat
    )
    updated_params = jax.tree.map(
        lambda tp, p: tau * p + (1 - tau) * tp,
        target_params,
        model_params,
    )
    return nnx.merge(graph_def, updated_params, batch_stats)
