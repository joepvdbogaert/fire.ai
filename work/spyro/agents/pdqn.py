from __future__ import division
import copy
import numpy as np
import tensorflow as tf

from spyro.agents import DQNAgent


class OneStepPlanningDQNAgent(DQNAgent):

    def __init__(self, policy, memory, name="P1_DQN_Agent", always_plan=[0], reps=10,
                 plan_action_selection="all", n_plan_alternatives=50, batches_per_step=5,
                 *args, **kwargs):
        super().__init__(policy, memory, name=name, *args, **kwargs)
        self.always_plan = np.array(always_plan)
        self.reps = reps
        self.plan_action_selection = plan_action_selection
        self.n_plan_alternatives = n_plan_alternatives
        self.batches_per_step = batches_per_step

    def run_session(self):
        """Run a single episode / trial, possibly truncated by t_max."""
        self.state = np.array(self.env.reset(), dtype=np.float64)
        self.episode_step_counter = 0
        self.episode_reward = 0

        for i in range(self.tmax):

            snapshot = self.env.get_snapshot()
            actions, rewards, next_states, dones, value_estimates = self.plan(snapshot)

            # select and perform action
            self.action = self.policy.select_action(value_estimates.reshape(-1))
            new_state, self.reward, self.done, _ = self.env.step(self.action)

            # print("State: {}, qvalues: {}, action: {}, reward: {}, new_state: {}".format(self.state, qvalues, self.action, self.reward, new_state))
            # save experience
            states = np.array([copy.copy(self.state) for _ in range(len(actions))])
            self.memory.store_batch(states, actions, rewards, next_states, dones)

            # train and possibly soft-update the target network
            if (self.step_counter % self.train_frequency == 0) and (self.step_counter > self.warmup_steps):
                self.train()
                if self.tau < 1:
                    self.soft_update_target_network()

            # possibly hard-update the target network
            if self.use_target_network and (self.step_counter % self.tau == 0):
                self.hard_update_target_network()

            # bookkeeping
            self.step_counter += 1
            self.episode_reward += self.reward
            self.episode_step_counter += 1
            self.state = np.asarray(copy.copy(new_state), dtype=np.float64)

            # end of episode
            if self.done:
                break

        episode_summary = self.session.run(
            self.episode_summary,
            feed_dict={
                self.total_reward: self.episode_reward,
                self.mean_reward: self.episode_reward / self.episode_step_counter
            }
        )
        self.summary_writer.add_summary(episode_summary, self.episode_counter)
        print("\rFinished episode {}".format(self.episode_counter), end="")

    def _select_first_actions(self):
        """Select the actions for which to plan."""
        # plan for all actions
        if self.plan_action_selection == "all":
            return np.asarray(np.arange(0, self.n_actions, 1), dtype=np.int32)
        # plan a number of randomly selected actions
        elif self.plan_action_selection == "random":
            random_actions = np.random.randint(0, self.n_actions,
                                               self.n_plan_alternatives - len(actions))
            actions = np.append(self.always_plan, random_actions)
            return actions
        # invalid parameter
        else:
            raise ValueError("plan_action_selection must be one of ['all', 'random']")

    def plan(self, snapshot):
        actions = self._select_first_actions()
        rewards = np.zeros(len(actions))
        next_states = np.zeros((len(actions),) + self.obs_shape)
        dones = np.zeros(len(actions))
        value_estimates = np.zeros(len(actions))

        for a, action in enumerate(actions):
            action_rewards = np.zeros(self.reps)
            action_next_states = np.zeros((self.reps,) + self.obs_shape)
            action_dones = np.zeros(self.reps)

            for r in range(self.reps):
                self.env.set_snapshot(copy.deepcopy(snapshot))
                state = snapshot["state"]
                next_state, reward, done, _ = self.env.step(action)

                # log this repetition's results
                action_rewards[r] = reward
                action_next_states[r] = next_state
                action_dones[r] = done

            next_values = self.session.run(
                self.next_action_qvalue,
                feed_dict={self.next_states_ph: action_next_states.reshape(-1, *self.obs_shape)}
            )
            next_values = np.squeeze(next_values)
            assert len(next_values) == self.reps

            rewards[a] = np.mean(action_rewards)
            # find state value closest to mean
            chosen_state_idx = np.abs(next_values - np.mean(next_values)).argmin()
            next_states[a] = action_next_states[chosen_state_idx]
            dones[a] = action_dones[chosen_state_idx]
            value_estimates[a] = rewards[a] + next_values[chosen_state_idx]

        return actions, rewards, next_states, dones, value_estimates

    def train(self):
        """Perform a train step using a batch sampled from memory."""
        for _ in range(self.batches_per_step - 1):
            states, actions, rewards, next_states, dones = self.memory.sample(batch_size=self.batch_size)
            # reshape to minimum of two dimensions
            self.session.run(
                self.train_op,
                feed_dict={
                    self.states_ph: states.reshape(-1, *self.obs_shape),
                    self.actions_ph: actions.reshape(-1, *self.action_shape),
                    self.rewards_ph: rewards.reshape(-1, 1),
                    self.next_states_ph: next_states.reshape(-1, *self.obs_shape),
                    self.dones_ph: dones.reshape(-1, 1)
                }
            )

        # once more with summary data
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size=self.batch_size)
        # reshape to minimum of two dimensions
        step_summ, _ = self.session.run(
            [self.step_summary, self.train_op],
            feed_dict={
                self.states_ph: states.reshape(-1, *self.obs_shape),
                self.actions_ph: actions.reshape(-1, *self.action_shape),
                self.rewards_ph: rewards.reshape(-1, 1),
                self.next_states_ph: next_states.reshape(-1, *self.obs_shape),
                self.dones_ph: dones.reshape(-1, 1)
            }
        )
        self.summary_writer.add_summary(step_summ, self.step_counter)
