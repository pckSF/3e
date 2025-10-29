from __future__ import annotations

from flax import (
    nnx,
    struct,
)
import jax
import optax


class NNTrainingState(struct.PyTreeNode):
    """Training state container for a Neural Network that can be passed through
    JAX transformations.

    Attributes:
        model_def: The static graph definition of the neural network.
        model_state: The dynamic state of the model, including its parameters.
        optimizer: The Optax optimizer used for gradient updates.
        optimizer_state: The current state of the optimizer.
    """

    model_def: nnx.GraphDef = struct.field(pytree_node=False)
    model_state: nnx.State
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    optimizer_state: optax.OptState

    def apply_gradients(self, grads: nnx.State) -> NNTrainingState:
        """Applies gradients to the model parameters.

        Args:
            grads: The gradients to be applied.

        Returns:
            A new state with updated model and optimizer states.
        """
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
    ) -> NNTrainingState:
        """Creates a new `NNTrainingState` instance.

        Args:
            model_def: The static graph definition of the neural network.
            model_state: The initial state of the model.
            optimizer: The Optax optimizer to use.

        Returns:
            A new `NNTrainingState` instance.
        """
        optimizer_state = optimizer.init(model_state)
        return cls(
            model_def=model_def,
            model_state=model_state,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
        )


class NNTrainingStateSoftTarget(NNTrainingState):
    """Training state with a soft-updating target network that can be passed through
    JAX transformations.

    Attributes:
        target_model_state: The state of the target network.
        tau: The interpolation factor for the soft update.
    """

    target_model_state: nnx.State
    tau: float = struct.field(pytree_node=False)

    def apply_gradients(self, grads: nnx.State) -> NNTrainingStateSoftTarget:
        """Applies gradients and performs a soft update on the target network.

        Args:
            grads: The gradients to be applied to the main model.

        Returns:
            A new state with updated model, target model, and optimizer states.
        """
        # Apply gradients to the main model using the parent's method
        updated_base_state = super().apply_gradients(grads)

        # Perform the soft update for the target network
        target_model_state = jax.tree.map(
            lambda tp, p: self.tau * p + (1 - self.tau) * tp,
            self.target_model_state,
            updated_base_state.model_state,
        )
        return self.replace(
            model_state=updated_base_state.model_state,
            optimizer_state=updated_base_state.optimizer_state,
            target_model_state=target_model_state,
        )

    @classmethod
    def create(
        cls,
        model_def: nnx.GraphDef,
        model_state: nnx.State,
        tau: float,
        optimizer: optax.GradientTransformation,
    ) -> NNTrainingStateSoftTarget:
        """Creates a new `NNTrainingStateSoftTarget` instance.

        Args:
            model_def: The static graph definition of the neural network.
            model_state: The initial state of the model.
            tau: The interpolation factor for the soft update.
            optimizer: The Optax optimizer to use.

        Returns:
            A new `NNTrainingStateSoftTarget` instance.
        """
        optimizer_state = optimizer.init(model_state)
        return cls(
            model_def=model_def,
            model_state=model_state,
            target_model_state=model_state,  # Initialize target with same state
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
    """Performs a soft update on the parameters of a target model.

    Args:
        model: The source model (e.g., the online network).
        model_target: The target model to be updated.
        tau: The interpolation factor for the soft update.

    Returns:
        A new target model with updated parameters.
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
