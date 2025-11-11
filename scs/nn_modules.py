from __future__ import annotations

from typing import TYPE_CHECKING

from flax import (
    nnx,
    struct,
)
import jax
import optax

if TYPE_CHECKING:
    from scs.ppo.defaults import PPOConfig
    from scs.sac.defaults import SACConfig


def get_optimizer(
    config: PPOConfig | SACConfig, model: nnx.Module
) -> optax.GradientTransformation:
    """Create an optimizer based on the configuration and model class name.

    Args:
        config: The configuration object (PPOConfig or SACConfig).
        model: The model instance whose class name is used to extract the postfix.
            The class name is converted to lowercase to get the postfix
            (e.g., PolicyValue -> "policyvalue", Policy -> "policy").

    Returns:
        An Optax gradient transformation (optimizer with learning rate schedule).
    """
    # Extract postfix from model class name (e.g., "PolicyValue" -> "policyvalue")
    postfix = model.__class__.__name__.lower()

    optimizer_name = getattr(config, f"optimizer_{postfix}")
    lr_schedule_type = getattr(config, f"lr_schedule_{postfix}")
    lr_init = getattr(config, f"lr_{postfix}")
    lr_end = getattr(config, f"lr_end_value_{postfix}")
    lr_decay = getattr(config, f"lr_decay_{postfix}")

    if optimizer_name == "adam":
        optimizer = optax.adam
    elif optimizer_name == "sgd":
        optimizer = optax.sgd
    else:
        raise ValueError(
            f"Unsupported optimizer, expected 'adam' or 'sgd'; "
            f"received: {optimizer_name}"
        )

    if lr_schedule_type == "constant":
        lr_schedule = optax.constant_schedule(value=lr_init)
    elif lr_schedule_type == "linear":
        lr_schedule = optax.linear_schedule(
            init_value=lr_init,
            end_value=lr_end,
            transition_steps=(config.num_epochs * config.max_training_loops),
        )
    elif lr_schedule_type == "exponential":
        lr_schedule = optax.exponential_decay(
            init_value=lr_init,
            transition_steps=config.num_epochs * config.max_training_loops,
            decay_rate=lr_decay,
            end_value=lr_end,
        )
    else:
        raise ValueError(
            f"Unsupported learning rate schedule, expected 'constant', 'linear' or "
            f"'exponential'; received {lr_schedule_type}"
        )

    max_grad_norm = getattr(config, f"max_grad_norm_{postfix}", None)

    transforms: list[optax.GradientTransformation] = []
    if max_grad_norm is not None and max_grad_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(max_grad_norm))
    transforms.append(optimizer(learning_rate=lr_schedule))
    return optax.chain(*transforms)


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


class NNTrainingStateSoftTarget(struct.PyTreeNode):
    """Training state with a soft-updating target network that can be passed through
    JAX transformations.

    Attributes:
        model_def: The static graph definition of the neural network.
        model_state: The dynamic state of the model, including its parameters.
        target_model_state: The state of the target network.
        optimizer: The Optax optimizer used for gradient updates.
        optimizer_state: The current state of the optimizer.
        tau: The interpolation factor for the soft update.
    """

    model_def: nnx.GraphDef = struct.field(pytree_node=False)
    model_state: nnx.State
    target_model_state: nnx.State
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    optimizer_state: optax.OptState
    tau: float = struct.field(pytree_node=False)

    def apply_gradients(self, grads: nnx.State) -> NNTrainingStateSoftTarget:
        """Applies gradients and performs a soft update on the target network.

        Args:
            grads: The gradients to be applied to the main model.

        Returns:
            A new state with updated model, target model, and optimizer states.
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
            optimizer_state=optimizer_state,
            target_model_state=target_model_state,
        )

    @classmethod
    def create(
        cls,
        model_def: nnx.GraphDef,
        model_state: nnx.State,
        optimizer: optax.GradientTransformation,
        tau: float,
    ) -> NNTrainingStateSoftTarget:
        """Creates a new `NNTrainingStateSoftTarget` instance.

        Args:
            model_def: The static graph definition of the neural network.
            model_state: The initial state of the model.
            optimizer: The Optax optimizer to use.
            tau: The interpolation factor for the soft update.

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
    graph_def, target_params, batch_stats = nnx.split(  # type: ignore[misc]
        model_target, nnx.Param, nnx.BatchStat
    )
    updated_params = jax.tree.map(
        lambda tp, p: tau * p + (1 - tau) * tp,
        target_params,
        model_params,
    )
    return nnx.merge(graph_def, updated_params, batch_stats)
