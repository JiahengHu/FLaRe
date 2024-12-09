from architecture.models.allenact_transformer_models.allenact_dino_transformer import DinoLLAMATxNavActorCritic

class DinoLLAMATxNavActorCriticSeparate(DinoLLAMATxNavActorCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic_tsfm = DinoLLAMATxNavActorCritic(*args, **kwargs)

    def forward(self, *args, **kwargs):
        actor_output, memory = super().forward(*args, **kwargs)
        critic_output, critic_memory = self.critic_tsfm(*args, **kwargs)

        critic_output.distributions = actor_output.distributions

        return critic_output, critic_memory
